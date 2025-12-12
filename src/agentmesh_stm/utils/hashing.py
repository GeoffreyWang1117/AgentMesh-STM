"""Hashing utilities for content comparison and versioning."""

import hashlib
from typing import Union


def content_hash(content: Union[str, bytes]) -> str:
    """Compute SHA-256 hash of content.

    Args:
        content: String or bytes content to hash

    Returns:
        Hex-encoded SHA-256 hash
    """
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def compute_diff_hash(old_content: str, new_content: str) -> str:
    """Compute hash representing the difference between two contents.

    Args:
        old_content: Original content
        new_content: Modified content

    Returns:
        Hex-encoded hash of the concatenated hashes
    """
    old_hash = content_hash(old_content)
    new_hash = content_hash(new_content)
    return content_hash(f"{old_hash}:{new_hash}")


def resource_id_hash(resource_path: str, resource_type: str = "file") -> str:
    """Generate a unique identifier for a resource.

    Args:
        resource_path: Path or identifier of the resource
        resource_type: Type of resource (file, memory, etc.)

    Returns:
        Unique hash identifier for the resource
    """
    combined = f"{resource_type}:{resource_path}"
    return hashlib.md5(combined.encode("utf-8")).hexdigest()[:16]
