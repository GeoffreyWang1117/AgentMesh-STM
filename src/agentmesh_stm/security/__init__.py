"""
Security Module for AgentMesh-STM.

Provides security features including:
- Input validation and sanitization
- Rate limiting
- Authentication helpers
- Audit logging
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import re
import secrets
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from agentmesh_stm.utils.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# Input Validation
# =============================================================================

class ValidationError(Exception):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: str = "", code: str = "VALIDATION_ERROR"):
        super().__init__(message)
        self.message = message
        self.field = field
        self.code = code


@dataclass
class ValidationRule:
    """A validation rule."""

    name: str
    validator: Callable[[Any], bool]
    error_message: str


class InputValidator:
    """
    Validates and sanitizes input data.

    Provides protection against:
    - Path traversal attacks
    - Command injection
    - Overly large inputs
    - Invalid characters
    """

    # Dangerous patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.\\",
        r"%2e%2e",
        r"\.\.%2f",
        r"%2e%2e/",
    ]

    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$]",
        r"\$\(",
        r"`.*`",
        r"\|\|",
        r"&&",
    ]

    # Safe patterns
    SAFE_RESOURCE_ID_PATTERN = re.compile(r"^[\w\-./]+$")
    SAFE_IDENTIFIER_PATTERN = re.compile(r"^[\w\-]+$")

    def __init__(
        self,
        max_content_size: int = 10 * 1024 * 1024,  # 10MB
        max_path_length: int = 4096,
        max_identifier_length: int = 256,
    ):
        self.max_content_size = max_content_size
        self.max_path_length = max_path_length
        self.max_identifier_length = max_identifier_length

    def validate_resource_id(self, resource_id: str) -> str:
        """Validate and sanitize a resource ID (file path)."""
        if not resource_id:
            raise ValidationError("Resource ID cannot be empty", "resource_id")

        if len(resource_id) > self.max_path_length:
            raise ValidationError(
                f"Resource ID exceeds maximum length ({self.max_path_length})",
                "resource_id",
                "MAX_LENGTH_EXCEEDED",
            )

        # Check for path traversal
        for pattern in self.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, resource_id, re.IGNORECASE):
                raise ValidationError(
                    "Path traversal detected in resource ID",
                    "resource_id",
                    "PATH_TRAVERSAL",
                )

        # Check for valid characters
        if not self.SAFE_RESOURCE_ID_PATTERN.match(resource_id):
            raise ValidationError(
                "Resource ID contains invalid characters",
                "resource_id",
                "INVALID_CHARACTERS",
            )

        # Normalize path
        normalized = resource_id.replace("\\", "/")
        # Remove leading slashes for relative paths
        normalized = normalized.lstrip("/")

        return normalized

    def validate_content(self, content: str) -> str:
        """Validate content to be written."""
        if content is None:
            raise ValidationError("Content cannot be None", "content")

        if len(content) > self.max_content_size:
            raise ValidationError(
                f"Content exceeds maximum size ({self.max_content_size} bytes)",
                "content",
                "MAX_SIZE_EXCEEDED",
            )

        return content

    def validate_identifier(self, identifier: str, field_name: str = "identifier") -> str:
        """Validate a general identifier (transaction ID, agent ID, etc.)."""
        if not identifier:
            raise ValidationError(f"{field_name} cannot be empty", field_name)

        if len(identifier) > self.max_identifier_length:
            raise ValidationError(
                f"{field_name} exceeds maximum length",
                field_name,
                "MAX_LENGTH_EXCEEDED",
            )

        if not self.SAFE_IDENTIFIER_PATTERN.match(identifier):
            raise ValidationError(
                f"{field_name} contains invalid characters",
                field_name,
                "INVALID_CHARACTERS",
            )

        return identifier

    def validate_command(self, command: str) -> str:
        """Validate a shell command for safety."""
        if not command:
            raise ValidationError("Command cannot be empty", "command")

        for pattern in self.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, command):
                raise ValidationError(
                    "Potential command injection detected",
                    "command",
                    "COMMAND_INJECTION",
                )

        return command

    def sanitize_log_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize data for safe logging (remove sensitive info)."""
        sensitive_keys = {
            "password", "secret", "token", "api_key", "apikey",
            "authorization", "auth", "credential", "private_key",
        }

        sanitized = {}
        for key, value in data.items():
            key_lower = key.lower()
            if any(sensitive in key_lower for sensitive in sensitive_keys):
                sanitized[key] = "***REDACTED***"
            elif isinstance(value, dict):
                sanitized[key] = self.sanitize_log_data(value)
            elif isinstance(value, str) and len(value) > 1000:
                sanitized[key] = value[:100] + "...[truncated]"
            else:
                sanitized[key] = value

        return sanitized


# =============================================================================
# Rate Limiting
# =============================================================================

class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str, retry_after: float = 0):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_second: float = 10.0
    requests_per_minute: float = 100.0
    requests_per_hour: float = 1000.0
    burst_size: int = 20


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter.

    Allows burst traffic up to bucket size, then rate-limits
    to the configured rate.
    """

    def __init__(
        self,
        rate: float,  # tokens per second
        bucket_size: int = 10,
    ):
        self.rate = rate
        self.bucket_size = bucket_size
        self._tokens = float(bucket_size)
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens. Returns True if successful."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_update
            self._last_update = now

            # Add tokens based on elapsed time
            self._tokens = min(
                self.bucket_size,
                self._tokens + elapsed * self.rate,
            )

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True

            return False

    async def wait_and_acquire(self, tokens: int = 1, timeout: float = 30.0) -> bool:
        """Wait until tokens are available, with timeout."""
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            if await self.acquire(tokens):
                return True
            await asyncio.sleep(0.1)

        return False

    def time_until_available(self, tokens: int = 1) -> float:
        """Calculate time until tokens will be available."""
        if self._tokens >= tokens:
            return 0.0
        needed = tokens - self._tokens
        return needed / self.rate


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter.

    Tracks requests in time windows for more accurate limiting.
    """

    def __init__(
        self,
        limit: int,
        window_seconds: float,
    ):
        self.limit = limit
        self.window_seconds = window_seconds
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def is_allowed(self, key: str = "default") -> Tuple[bool, float]:
        """
        Check if request is allowed.

        Returns (allowed, retry_after_seconds).
        """
        async with self._lock:
            now = time.monotonic()
            window_start = now - self.window_seconds

            # Clean old requests
            self._requests[key] = [
                t for t in self._requests[key]
                if t > window_start
            ]

            if len(self._requests[key]) < self.limit:
                self._requests[key].append(now)
                return True, 0.0

            # Calculate retry after
            oldest = self._requests[key][0]
            retry_after = oldest + self.window_seconds - now
            return False, max(0, retry_after)

    async def reset(self, key: str = "default") -> None:
        """Reset rate limit for a key."""
        async with self._lock:
            self._requests[key] = []


class CompositeRateLimiter:
    """
    Combines multiple rate limiters.

    Enforces rate limits at different time scales.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None):
        config = config or RateLimitConfig()

        self._limiters = {
            "second": TokenBucketRateLimiter(
                rate=config.requests_per_second,
                bucket_size=config.burst_size,
            ),
            "minute": SlidingWindowRateLimiter(
                limit=int(config.requests_per_minute),
                window_seconds=60.0,
            ),
            "hour": SlidingWindowRateLimiter(
                limit=int(config.requests_per_hour),
                window_seconds=3600.0,
            ),
        }

    async def check(self, key: str = "default") -> Tuple[bool, str, float]:
        """
        Check all rate limits.

        Returns (allowed, limit_name, retry_after).
        """
        # Check burst limiter first
        if not await self._limiters["second"].acquire():
            retry_after = self._limiters["second"].time_until_available()
            return False, "second", retry_after

        # Check minute limit
        allowed, retry_after = await self._limiters["minute"].is_allowed(key)
        if not allowed:
            return False, "minute", retry_after

        # Check hour limit
        allowed, retry_after = await self._limiters["hour"].is_allowed(key)
        if not allowed:
            return False, "hour", retry_after

        return True, "", 0.0


def rate_limited(limiter: CompositeRateLimiter, key_func: Optional[Callable] = None):
    """Decorator to apply rate limiting to a function."""

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Determine the rate limit key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = "default"

            allowed, limit_name, retry_after = await limiter.check(key)
            if not allowed:
                raise RateLimitExceeded(
                    f"Rate limit exceeded ({limit_name})",
                    retry_after=retry_after,
                )

            return await func(*args, **kwargs)

        return wrapper
    return decorator


# =============================================================================
# Authentication Helpers
# =============================================================================

@dataclass
class APIKey:
    """An API key."""

    key_id: str
    key_hash: str
    name: str
    permissions: Set[str]
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    rate_limit: Optional[RateLimitConfig] = None


class APIKeyManager:
    """Manages API keys."""

    def __init__(self):
        self._keys: Dict[str, APIKey] = {}
        self._key_by_hash: Dict[str, str] = {}

    def generate_key(
        self,
        name: str,
        permissions: Optional[Set[str]] = None,
        expires_in: Optional[timedelta] = None,
        rate_limit: Optional[RateLimitConfig] = None,
    ) -> Tuple[str, APIKey]:
        """Generate a new API key."""
        # Generate random key
        key = secrets.token_urlsafe(32)
        key_id = secrets.token_urlsafe(8)
        key_hash = self._hash_key(key)

        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            permissions=permissions or {"read", "write"},
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + expires_in if expires_in else None,
            rate_limit=rate_limit,
        )

        self._keys[key_id] = api_key
        self._key_by_hash[key_hash] = key_id

        logger.info("API key generated", key_id=key_id, name=name)
        return f"{key_id}.{key}", api_key

    def validate_key(self, key: str) -> Optional[APIKey]:
        """Validate an API key and return the APIKey if valid."""
        if "." not in key:
            return None

        key_id, secret = key.split(".", 1)
        if key_id not in self._keys:
            return None

        api_key = self._keys[key_id]

        # Check if expired
        if api_key.expires_at and datetime.utcnow() > api_key.expires_at:
            logger.warning("Expired API key used", key_id=key_id)
            return None

        # Verify hash
        key_hash = self._hash_key(secret)
        if not hmac.compare_digest(key_hash, api_key.key_hash):
            logger.warning("Invalid API key", key_id=key_id)
            return None

        # Update last used
        api_key.last_used = datetime.utcnow()

        return api_key

    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        if key_id not in self._keys:
            return False

        api_key = self._keys.pop(key_id)
        self._key_by_hash.pop(api_key.key_hash, None)

        logger.info("API key revoked", key_id=key_id)
        return True

    def has_permission(self, api_key: APIKey, permission: str) -> bool:
        """Check if API key has a permission."""
        return permission in api_key.permissions or "*" in api_key.permissions

    @staticmethod
    def _hash_key(key: str) -> str:
        """Hash an API key."""
        return hashlib.sha256(key.encode()).hexdigest()


# =============================================================================
# Audit Logging
# =============================================================================

class AuditEventType(Enum):
    """Types of audit events."""

    # Authentication events
    AUTH_SUCCESS = auto()
    AUTH_FAILURE = auto()
    KEY_CREATED = auto()
    KEY_REVOKED = auto()

    # Transaction events
    TXN_BEGIN = auto()
    TXN_COMMIT = auto()
    TXN_ABORT = auto()

    # Resource events
    RESOURCE_READ = auto()
    RESOURCE_WRITE = auto()
    RESOURCE_DELETE = auto()

    # Admin events
    CONFIG_CHANGE = auto()
    PERMISSION_CHANGE = auto()


@dataclass
class AuditEvent:
    """An audit log event."""

    event_type: AuditEventType
    timestamp: datetime
    actor: str  # Who performed the action
    resource: Optional[str] = None  # What resource was affected
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    success: bool = True


class AuditLogger:
    """
    Records security-relevant events for compliance and debugging.
    """

    def __init__(self, max_events: int = 100000):
        self._events: List[AuditEvent] = []
        self._max_events = max_events
        self._lock = asyncio.Lock()
        self._validator = InputValidator()

    async def log(
        self,
        event_type: AuditEventType,
        actor: str,
        resource: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        success: bool = True,
    ) -> None:
        """Log an audit event."""
        # Sanitize details for logging
        safe_details = self._validator.sanitize_log_data(details or {})

        event = AuditEvent(
            event_type=event_type,
            timestamp=datetime.utcnow(),
            actor=actor,
            resource=resource,
            details=safe_details,
            ip_address=ip_address,
            success=success,
        )

        async with self._lock:
            self._events.append(event)

            # Trim old events if needed
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events:]

        # Also log to standard logger
        log_func = logger.info if success else logger.warning
        log_func(
            f"Audit: {event_type.name}",
            actor=actor,
            resource=resource,
            success=success,
            **safe_details,
        )

    async def get_events(
        self,
        event_type: Optional[AuditEventType] = None,
        actor: Optional[str] = None,
        resource: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Query audit events."""
        async with self._lock:
            events = self._events.copy()

        # Filter events
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if actor:
            events = [e for e in events if e.actor == actor]
        if resource:
            events = [e for e in events if e.resource == resource]
        if since:
            events = [e for e in events if e.timestamp >= since]

        # Return most recent first
        return list(reversed(events[-limit:]))

    async def export(self, format: str = "json") -> str:
        """Export audit log."""
        import json

        async with self._lock:
            events = self._events.copy()

        if format == "json":
            return json.dumps([
                {
                    "type": e.event_type.name,
                    "timestamp": e.timestamp.isoformat(),
                    "actor": e.actor,
                    "resource": e.resource,
                    "details": e.details,
                    "ip": e.ip_address,
                    "success": e.success,
                }
                for e in events
            ], indent=2)

        # CSV format
        lines = ["type,timestamp,actor,resource,success,ip"]
        for e in events:
            lines.append(
                f"{e.event_type.name},{e.timestamp.isoformat()},"
                f"{e.actor},{e.resource or ''},{e.success},{e.ip_address or ''}"
            )
        return "\n".join(lines)


# =============================================================================
# Global instances
# =============================================================================

_validator: Optional[InputValidator] = None
_rate_limiter: Optional[CompositeRateLimiter] = None
_api_key_manager: Optional[APIKeyManager] = None
_audit_logger: Optional[AuditLogger] = None


def get_validator() -> InputValidator:
    """Get the global input validator."""
    global _validator
    if _validator is None:
        _validator = InputValidator()
    return _validator


def get_rate_limiter() -> CompositeRateLimiter:
    """Get the global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = CompositeRateLimiter()
    return _rate_limiter


def get_api_key_manager() -> APIKeyManager:
    """Get the global API key manager."""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
