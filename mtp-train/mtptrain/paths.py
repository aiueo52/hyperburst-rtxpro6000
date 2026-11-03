"""Input boundaries for public dumps; JSON pointers never grant another root."""
from __future__ import annotations

from collections import deque
import os
from pathlib import Path
import stat


def _absolute(path) -> Path:
    if isinstance(path, str) and (not path.strip() or '\x00' in path):
        raise ValueError('invalid input path')
    path = Path(path).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def input_path(path, *, root=None) -> Path:
    """Resolve each symlink separately, checking even targets that link back out.

    Missing paths are allowed so callers retain their usual missing-file errors.
    With ``root``, every symlink target and traversal must stay inside it.
    """
    path = _absolute(path)
    root = input_path(root) if root is not None else None
    if root is not None and not path.is_relative_to(root):
        raise ValueError('input is outside the validated dump root')
    pending = deque(path.parts[1:])
    current = Path(path.anchor)
    hops = 0
    inside = False
    while pending:
        part = pending.popleft()
        candidate = current.parent if part == '..' else current / part
        if root is not None:
            if inside and not candidate.is_relative_to(root):
                raise ValueError('input traversal leaves the validated dump root')
            inside = inside or candidate == root
        try:
            is_link = stat.S_ISLNK(candidate.lstat().st_mode)
        except FileNotFoundError:
            is_link = False
        if is_link:
            hops += 1
            if hops > 40:
                raise ValueError('too many input symlink hops')
            target = Path(os.readlink(candidate))
            target = target if target.is_absolute() else current / target
            if root is not None and not target.is_relative_to(root):
                raise ValueError('input symlink leaves the validated dump root')
            pending.extendleft(reversed(target.parts[1:]))
            current = Path(target.anchor)
            inside = False
        else:
            current = candidate
    if root is not None and not current.is_relative_to(root):
        raise ValueError('input is outside the validated dump root')
    return current


def dump_root(path) -> str:
    root = input_path(path)
    if not root.is_dir():
        raise SystemExit(f'dump dir {os.fspath(path)!r}: no such directory')
    return str(root)


def dump_path(root, name) -> str:
    """A relative name or contained absolute path beneath a selected root."""
    if not isinstance(name, str) or not name or '\x00' in name:
        raise ValueError('invalid dump-relative input name')
    relative = Path(name)
    if '..' in relative.parts or relative == Path('.'):
        raise ValueError('dump input name must not contain traversal')
    if relative.is_absolute():
        return dump_input(root, name)
    return str(input_path(input_path(root) / relative, root=root))


def dump_input(root, path) -> str:
    """Explicit/legacy full paths are accepted only inside the selected root."""
    original_root = _absolute(root)
    path = _absolute(path)
    if '..' in path.parts:
        raise ValueError('dump input path must not contain traversal')
    resolved_root = input_path(original_root)
    if path.is_relative_to(original_root):
        path = resolved_root / path.relative_to(original_root)
    return str(input_path(path, root=resolved_root))
