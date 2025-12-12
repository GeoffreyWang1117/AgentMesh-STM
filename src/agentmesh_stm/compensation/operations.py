"""
Concrete Compensable Operations for AgentMesh-STM.

This module provides implementations of compensable operations
for common side effects:
- File operations (write, delete, create)
- Git operations (commit, branch)
- API calls (with inverse operations)
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import aiofiles

from agentmesh_stm.compensation.manager import CompensableOperation, OperationType
from agentmesh_stm.utils.logging import get_logger

logger = get_logger(__name__)


class FileWriteCompensation(CompensableOperation):
    """Compensation for file write operations."""

    operation_type = OperationType.FILE_WRITE

    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a file write operation.

        Args:
            data: Must contain 'path' and 'content'

        Returns:
            Compensation data with original content
        """
        path = data["path"]
        content = data["content"]

        # Read original content for compensation
        original_content = None
        file_existed = os.path.exists(path)

        if file_existed:
            async with aiofiles.open(path, "r") as f:
                original_content = await f.read()

        # Write new content
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "w") as f:
            await f.write(content)

        logger.debug("File written", path=path)

        return {
            "path": path,
            "original_content": original_content,
            "file_existed": file_existed,
        }

    async def compensate(self, compensation_data: Dict[str, Any]) -> bool:
        """Restore original file content or delete if it didn't exist."""
        path = compensation_data["path"]
        original_content = compensation_data.get("original_content")
        file_existed = compensation_data.get("file_existed", True)

        try:
            if not file_existed:
                # File didn't exist before, delete it
                if os.path.exists(path):
                    os.remove(path)
                    logger.debug("Compensation: deleted new file", path=path)
            else:
                # Restore original content
                async with aiofiles.open(path, "w") as f:
                    await f.write(original_content or "")
                logger.debug("Compensation: restored file content", path=path)

            return True
        except Exception as e:
            logger.error("File write compensation failed", path=path, error=str(e))
            return False

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate file write parameters."""
        return "path" in data and "content" in data


class FileDeleteCompensation(CompensableOperation):
    """Compensation for file delete operations."""

    operation_type = OperationType.FILE_DELETE

    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a file delete operation.

        Args:
            data: Must contain 'path'

        Returns:
            Compensation data with file content and metadata
        """
        path = data["path"]

        # Read content for compensation
        content = None
        if os.path.exists(path):
            async with aiofiles.open(path, "r") as f:
                content = await f.read()

            # Delete the file
            os.remove(path)
            logger.debug("File deleted", path=path)

        return {
            "path": path,
            "content": content,
            "existed": content is not None,
        }

    async def compensate(self, compensation_data: Dict[str, Any]) -> bool:
        """Restore deleted file."""
        path = compensation_data["path"]
        content = compensation_data.get("content")
        existed = compensation_data.get("existed", False)

        if not existed:
            return True  # Nothing to restore

        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(path, "w") as f:
                await f.write(content or "")

            logger.debug("Compensation: restored deleted file", path=path)
            return True
        except Exception as e:
            logger.error("File delete compensation failed", path=path, error=str(e))
            return False

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate file delete parameters."""
        return "path" in data


class FileCreateCompensation(CompensableOperation):
    """Compensation for file create operations."""

    operation_type = OperationType.FILE_CREATE

    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a file create operation.

        Args:
            data: Must contain 'path' and 'content'

        Returns:
            Compensation data
        """
        path = data["path"]
        content = data.get("content", "")

        # Check if file already exists
        if os.path.exists(path):
            raise FileExistsError(f"File already exists: {path}")

        # Create file
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "w") as f:
            await f.write(content)

        logger.debug("File created", path=path)

        return {"path": path}

    async def compensate(self, compensation_data: Dict[str, Any]) -> bool:
        """Delete created file."""
        path = compensation_data["path"]

        try:
            if os.path.exists(path):
                os.remove(path)
                logger.debug("Compensation: deleted created file", path=path)
            return True
        except Exception as e:
            logger.error("File create compensation failed", path=path, error=str(e))
            return False

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate file create parameters."""
        return "path" in data


class GitCommitCompensation(CompensableOperation):
    """Compensation for git commit operations."""

    operation_type = OperationType.GIT_COMMIT

    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a git commit operation.

        Args:
            data: Must contain 'repo_path', 'message', optionally 'files'

        Returns:
            Compensation data with commit hash
        """
        import git

        repo_path = data["repo_path"]
        message = data["message"]
        files = data.get("files", [])

        repo = git.Repo(repo_path)

        # Stage files if specified
        if files:
            repo.index.add(files)
        else:
            repo.git.add(A=True)

        # Get HEAD before commit
        head_before = repo.head.commit.hexsha if repo.head.is_valid() else None

        # Create commit
        commit = repo.index.commit(message)

        logger.debug("Git commit created", commit=commit.hexsha, repo=repo_path)

        return {
            "repo_path": repo_path,
            "commit_hash": commit.hexsha,
            "head_before": head_before,
        }

    async def compensate(self, compensation_data: Dict[str, Any]) -> bool:
        """Revert git commit."""
        import git

        repo_path = compensation_data["repo_path"]
        commit_hash = compensation_data["commit_hash"]
        head_before = compensation_data.get("head_before")

        try:
            repo = git.Repo(repo_path)

            # Check if this is the latest commit
            if repo.head.commit.hexsha == commit_hash:
                # Reset to previous state
                if head_before:
                    repo.git.reset("--hard", head_before)
                else:
                    repo.git.reset("--hard", "HEAD~1")

                logger.debug("Compensation: reverted commit", commit=commit_hash)
            else:
                # Commit is not at HEAD, try to revert it
                repo.git.revert(commit_hash, "--no-commit")
                repo.index.commit(f"Revert {commit_hash[:8]}")
                logger.debug("Compensation: revert commit created", commit=commit_hash)

            return True
        except Exception as e:
            logger.error(
                "Git commit compensation failed",
                commit=commit_hash,
                error=str(e),
            )
            return False

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate git commit parameters."""
        return "repo_path" in data and "message" in data


class GitBranchCompensation(CompensableOperation):
    """Compensation for git branch operations."""

    operation_type = OperationType.GIT_BRANCH

    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a git branch operation.

        Args:
            data: Must contain 'repo_path', 'branch_name', optionally 'checkout'

        Returns:
            Compensation data
        """
        import git

        repo_path = data["repo_path"]
        branch_name = data["branch_name"]
        checkout = data.get("checkout", False)

        repo = git.Repo(repo_path)
        original_branch = repo.active_branch.name

        # Create branch
        new_branch = repo.create_head(branch_name)

        if checkout:
            new_branch.checkout()

        logger.debug("Git branch created", branch=branch_name, repo=repo_path)

        return {
            "repo_path": repo_path,
            "branch_name": branch_name,
            "original_branch": original_branch,
            "was_checked_out": checkout,
        }

    async def compensate(self, compensation_data: Dict[str, Any]) -> bool:
        """Delete created branch."""
        import git

        repo_path = compensation_data["repo_path"]
        branch_name = compensation_data["branch_name"]
        original_branch = compensation_data.get("original_branch")
        was_checked_out = compensation_data.get("was_checked_out", False)

        try:
            repo = git.Repo(repo_path)

            # Switch back to original branch if needed
            if was_checked_out and original_branch:
                repo.heads[original_branch].checkout()

            # Delete the branch
            repo.delete_head(branch_name, force=True)

            logger.debug("Compensation: deleted branch", branch=branch_name)
            return True
        except Exception as e:
            logger.error(
                "Git branch compensation failed",
                branch=branch_name,
                error=str(e),
            )
            return False

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate git branch parameters."""
        return "repo_path" in data and "branch_name" in data


class APICallCompensation(CompensableOperation):
    """Compensation for API call operations."""

    operation_type = OperationType.API_CALL

    async def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an API call operation.

        Args:
            data: Must contain 'url', 'method', optionally 'body', 'headers',
                  and 'compensation_endpoint' for rollback

        Returns:
            Compensation data with response and inverse operation
        """
        import aiohttp

        url = data["url"]
        method = data.get("method", "POST")
        body = data.get("body")
        headers = data.get("headers", {})
        compensation_endpoint = data.get("compensation_endpoint")
        compensation_method = data.get("compensation_method", "DELETE")

        async with aiohttp.ClientSession() as session:
            async with session.request(method, url, json=body, headers=headers) as response:
                response_data = await response.json() if response.content_type == "application/json" else await response.text()

        logger.debug("API call executed", url=url, method=method, status=response.status)

        return {
            "url": url,
            "compensation_endpoint": compensation_endpoint,
            "compensation_method": compensation_method,
            "response": response_data,
            "headers": headers,
        }

    async def compensate(self, compensation_data: Dict[str, Any]) -> bool:
        """Execute compensation API call."""
        import aiohttp

        compensation_endpoint = compensation_data.get("compensation_endpoint")
        if not compensation_endpoint:
            logger.warning("No compensation endpoint provided for API call")
            return True  # Consider it success if no compensation needed

        method = compensation_data.get("compensation_method", "DELETE")
        headers = compensation_data.get("headers", {})

        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method, compensation_endpoint, headers=headers
                ) as response:
                    if response.status < 400:
                        logger.debug(
                            "Compensation: API call succeeded",
                            endpoint=compensation_endpoint,
                        )
                        return True
                    else:
                        logger.error(
                            "Compensation: API call failed",
                            endpoint=compensation_endpoint,
                            status=response.status,
                        )
                        return False
        except Exception as e:
            logger.error(
                "API call compensation failed",
                endpoint=compensation_endpoint,
                error=str(e),
            )
            return False

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate API call parameters."""
        return "url" in data
