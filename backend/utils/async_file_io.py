"""
Async File I/O Utilities

Non-blocking file operations using asyncio.to_thread() or aiofiles.
Prevents blocking the event loop during file operations.

Usage:
    from backend.utils.async_file_io import async_read_json, async_write_json

    data = await async_read_json("/path/to/file.json")
    await async_write_json("/path/to/file.json", {"key": "value"})
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import structlog

logger = structlog.get_logger(__name__)


# ============================================================================
# Core Async File Operations
# ============================================================================

async def async_read_file(
    file_path: Union[str, Path],
    encoding: str = "utf-8"
) -> str:
    """
    Read file contents asynchronously.

    Args:
        file_path: Path to the file
        encoding: File encoding (default: utf-8)

    Returns:
        File contents as string

    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file cannot be read
    """
    def _read():
        with open(file_path, "r", encoding=encoding) as f:
            return f.read()

    return await asyncio.to_thread(_read)


async def async_write_file(
    file_path: Union[str, Path],
    content: str,
    encoding: str = "utf-8",
    create_dirs: bool = True
) -> None:
    """
    Write content to file asynchronously.

    Args:
        file_path: Path to the file
        content: Content to write
        encoding: File encoding (default: utf-8)
        create_dirs: Create parent directories if they don't exist

    Raises:
        IOError: If file cannot be written
    """
    def _write():
        path = Path(file_path)
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding=encoding) as f:
            f.write(content)

    await asyncio.to_thread(_write)


async def async_append_file(
    file_path: Union[str, Path],
    content: str,
    encoding: str = "utf-8",
    create_dirs: bool = True
) -> None:
    """
    Append content to file asynchronously.

    Args:
        file_path: Path to the file
        content: Content to append
        encoding: File encoding (default: utf-8)
        create_dirs: Create parent directories if they don't exist
    """
    def _append():
        path = Path(file_path)
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding=encoding) as f:
            f.write(content)

    await asyncio.to_thread(_append)


async def async_read_bytes(file_path: Union[str, Path]) -> bytes:
    """
    Read file contents as bytes asynchronously.

    Args:
        file_path: Path to the file

    Returns:
        File contents as bytes
    """
    def _read():
        with open(file_path, "rb") as f:
            return f.read()

    return await asyncio.to_thread(_read)


async def async_write_bytes(
    file_path: Union[str, Path],
    content: bytes,
    create_dirs: bool = True
) -> None:
    """
    Write bytes to file asynchronously.

    Args:
        file_path: Path to the file
        content: Bytes to write
        create_dirs: Create parent directories if they don't exist
    """
    def _write():
        path = Path(file_path)
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)

    await asyncio.to_thread(_write)


# ============================================================================
# JSON Utilities
# ============================================================================

async def async_read_json(
    file_path: Union[str, Path],
    default: Optional[Any] = None
) -> Any:
    """
    Read and parse JSON file asynchronously.

    Args:
        file_path: Path to the JSON file
        default: Default value if file doesn't exist or is invalid

    Returns:
        Parsed JSON data or default value
    """
    def _read():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return default
        except json.JSONDecodeError as e:
            logger.warning("json_parse_error", file=str(file_path), error=str(e))
            return default

    return await asyncio.to_thread(_read)


async def async_write_json(
    file_path: Union[str, Path],
    data: Any,
    indent: int = 2,
    create_dirs: bool = True
) -> None:
    """
    Write data to JSON file asynchronously.

    Args:
        file_path: Path to the JSON file
        data: Data to serialize to JSON
        indent: Indentation level for pretty-printing
        create_dirs: Create parent directories if they don't exist
    """
    def _write():
        path = Path(file_path)
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, default=str)

    await asyncio.to_thread(_write)


# ============================================================================
# File System Utilities
# ============================================================================

async def async_file_exists(file_path: Union[str, Path]) -> bool:
    """
    Check if file exists asynchronously.

    Args:
        file_path: Path to check

    Returns:
        True if file exists, False otherwise
    """
    return await asyncio.to_thread(lambda: Path(file_path).exists())


async def async_is_file(file_path: Union[str, Path]) -> bool:
    """
    Check if path is a file asynchronously.

    Args:
        file_path: Path to check

    Returns:
        True if path is a file, False otherwise
    """
    return await asyncio.to_thread(lambda: Path(file_path).is_file())


async def async_is_dir(dir_path: Union[str, Path]) -> bool:
    """
    Check if path is a directory asynchronously.

    Args:
        dir_path: Path to check

    Returns:
        True if path is a directory, False otherwise
    """
    return await asyncio.to_thread(lambda: Path(dir_path).is_dir())


async def async_list_dir(
    dir_path: Union[str, Path],
    pattern: str = "*"
) -> List[Path]:
    """
    List directory contents asynchronously.

    Args:
        dir_path: Path to directory
        pattern: Glob pattern (default: all files)

    Returns:
        List of Path objects
    """
    def _list():
        return list(Path(dir_path).glob(pattern))

    return await asyncio.to_thread(_list)


async def async_mkdir(
    dir_path: Union[str, Path],
    parents: bool = True,
    exist_ok: bool = True
) -> None:
    """
    Create directory asynchronously.

    Args:
        dir_path: Path to create
        parents: Create parent directories if needed
        exist_ok: Don't raise error if directory exists
    """
    def _mkdir():
        Path(dir_path).mkdir(parents=parents, exist_ok=exist_ok)

    await asyncio.to_thread(_mkdir)


async def async_remove(file_path: Union[str, Path]) -> bool:
    """
    Remove file asynchronously.

    Args:
        file_path: Path to file to remove

    Returns:
        True if removed, False if didn't exist
    """
    def _remove():
        path = Path(file_path)
        if path.exists():
            path.unlink()
            return True
        return False

    return await asyncio.to_thread(_remove)


async def async_copy_file(
    src: Union[str, Path],
    dst: Union[str, Path],
    create_dirs: bool = True
) -> None:
    """
    Copy file asynchronously.

    Args:
        src: Source file path
        dst: Destination file path
        create_dirs: Create destination parent directories
    """
    import shutil

    def _copy():
        if create_dirs:
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    await asyncio.to_thread(_copy)


# ============================================================================
# Environment File Utilities
# ============================================================================

async def async_read_env(file_path: Union[str, Path] = ".env") -> Dict[str, str]:
    """
    Read environment file asynchronously.

    Args:
        file_path: Path to .env file

    Returns:
        Dict of environment variables
    """
    def _read():
        env_vars = {}
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        # Remove quotes from value
                        value = value.strip().strip("'\"")
                        env_vars[key.strip()] = value
        except FileNotFoundError:
            pass
        return env_vars

    return await asyncio.to_thread(_read)


async def async_update_env(
    file_path: Union[str, Path],
    key: str,
    value: str
) -> None:
    """
    Update or add a key in environment file asynchronously.

    Args:
        file_path: Path to .env file
        key: Environment variable name
        value: Environment variable value
    """
    content = await async_read_file(file_path) if await async_file_exists(file_path) else ""

    lines = content.split("\n")
    found = False
    new_lines = []

    for line in lines:
        if line.strip().startswith(f"{key}="):
            new_lines.append(f"{key}={value}")
            found = True
        else:
            new_lines.append(line)

    if not found:
        new_lines.append(f"{key}={value}")

    await async_write_file(file_path, "\n".join(new_lines))


# ============================================================================
# Batch Operations
# ============================================================================

async def async_batch_read_json(file_paths: List[Union[str, Path]]) -> Dict[str, Any]:
    """
    Read multiple JSON files concurrently.

    Args:
        file_paths: List of file paths to read

    Returns:
        Dict mapping file path -> parsed data (or None if failed)
    """
    tasks = [async_read_json(fp) for fp in file_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return {
        str(fp): (None if isinstance(r, Exception) else r)
        for fp, r in zip(file_paths, results)
    }


async def async_batch_write_json(file_data: Dict[Union[str, Path], Any]) -> None:
    """
    Write multiple JSON files concurrently.

    Args:
        file_data: Dict mapping file path -> data to write
    """
    tasks = [async_write_json(fp, data) for fp, data in file_data.items()]
    await asyncio.gather(*tasks, return_exceptions=True)
