#!/usr/bin/env python3
"""Gate public MTP writeback; never replace an existing model directory."""
from __future__ import annotations

import ctypes
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
INDEX = 'model.safetensors.index.json'
sys.dont_write_bytecode = True

from mtptrain.cli import StrictParser
from mtptrain.paths import input_path


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _json(path):
    return json.loads(_read_path(path).read_bytes(), object_pairs_hook=_unique)


def _path(value):
    if not str(value).strip():
        raise ValueError('empty path')
    return Path(os.path.abspath(Path(value).expanduser()))


def _read_path(value):
    return input_path(_path(value))


def writable_path(value):
    path = _path(value)
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError('output paths must not contain symlinks')
    return path


def _digests(path):
    hashes = (hashlib.sha256(), hashlib.blake2b(digest_size=12))
    with _read_path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            for digest in hashes:
                digest.update(block)
    return tuple(digest.hexdigest() for digest in hashes)


def _bound_path(value, root):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('invalid bound path')
    path = Path(value).expanduser()
    return _read_path(path if path.is_absolute() else root / path)


def gate_allows(gate, src, ckpt):
    """Re-evaluate training's measured reports; a saved passed flag is not enough."""
    sys.dont_write_bytecode = True
    from train_head import GATE_SCHEMA, SECONDARY_MAP_LIMITATION, evaluate_gate
    from mtptrain.weights import model_identity

    try:
        path, src, ckpt = _read_path(gate), _read_path(src), _read_path(ckpt)
        record = _json(path)
        if (not isinstance(record, dict) or type(record.get('schema')) is not int
                or record['schema'] != GATE_SCHEMA or record.get('passed') is not True):
            return False
        checkpoint = record.get('checkpoint', {})
        if not isinstance(checkpoint, dict):
            return False
        trained_path = Path(checkpoint['path']).expanduser()
        if not trained_path.is_absolute():
            trained_path = path.parent / trained_path
        if _read_path(trained_path).resolve(strict=True) != ckpt.resolve(strict=True):
            return False
        sha256, checkpoint_digest = _digests(ckpt)
        if checkpoint.get('sha256') != sha256:
            return False
        source = src.resolve(strict=True)
        if _read_path(record['source']).resolve(strict=True) != source:
            return False
        if (record.get('source_index_sha256') != _digests(source / INDEX)[0]
                or record.get('source_model_identity') != model_identity(str(source))):
            return False
        token_maps = record.get('token_maps')
        if not isinstance(token_maps, dict) or set(token_maps) != {'primary', 'secondary'}:
            return False
        primary, secondary = token_maps['primary'], token_maps['secondary']
        if (not isinstance(primary, dict) or not isinstance(secondary, dict)
                or not {'path', 'sha256', 'evaluated', 'limitation'} <= secondary.keys()
                or secondary.get('evaluated') is not False
                or secondary.get('limitation') != SECONDARY_MAP_LIMITATION):
            return False
        primary_sha256, primary_digest = _digests(_bound_path(primary['path'], path.parent))
        if primary.get('sha256') != primary_sha256:
            return False
        if secondary.get('path') is None:
            if secondary.get('sha256') is not None:
                return False
        elif secondary.get('sha256') != _digests(_bound_path(secondary['path'], path.parent))[0]:
            return False
        original = _json(path.parent / 'original-eval.json')
        best = _json(path.parent / 'best-eval.json')
        policy = record.get('renewal_policy')
        replay = isinstance(policy, dict) and policy.get('name') == 'argmax_label_replay'
        if replay:
            from regate import verify_evidence
            verify_evidence(policy, (original, best))
        decision = evaluate_gate(original, best, checkpoint, source=str(source), token_maps=token_maps,
                                 policy=policy if replay else None)
        if replay and record['renewal_policy'] != decision['renewal_policy']:
            return False
        if (decision['passed'] is not True or record.get('reason') != decision['reason']
                or record.get('relative_gain') != decision['relative_gain']
                or record.get('baseline') != decision['baseline']
                or record.get('best') != decision['best']
                or best.get('checkpoint_digest') != checkpoint_digest):
            return False
        for report in (original, best):
            if (report['renewal'].get('synthetic') is not False
                    or report['compare'].get('synthetic') is True
                    or report['compare'].get('token_map') != primary_digest
                    or report['compare'].get('token_map_sha256') != primary_sha256):
                return False
            for k in (3, 15):
                if not 1 <= report['summary'][f'accept_len@{k}'] <= k + 1:
                    return False
        return True
    except (OSError, ValueError, TypeError, KeyError, OverflowError, SystemExit):
        return False


def _vendor():
    # Load the public core by file to avoid this recipe's identical module name.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('_mtp_writeback_core',
                                                ROOT / 'writeback' / 'write_mtp.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _index(root, vendor):
    """Reject duplicate JSON keys and an ambiguous source tensor layout too."""
    from safetensors import safe_open

    record = _json(root / INDEX)
    weight_map = record.get('weight_map') if isinstance(record, dict) else None
    if (not isinstance(weight_map, dict) or not weight_map
            or any(not isinstance(key, str) or not key for key in weight_map)
            or any(not isinstance(value, str) for value in weight_map.values())):
        raise ValueError('invalid weight index')
    seen = {}
    for shard in sorted(set(weight_map.values())):
        vendor.check_name(shard)
        with safe_open(str(_read_path(root / shard)), framework='pt', device='cpu') as handle:
            for name in handle.keys():
                if name in seen or weight_map.get(name) != shard:
                    raise ValueError('tensor occurs twice, is unindexed, or is in the wrong shard')
                seen[name] = shard
    if seen != weight_map:
        raise ValueError('index references missing tensors')
    return weight_map


def _verify(src, dst, trained, vendor):
    """Vendor checks plus byte equality (including signed zero) in mixed shards."""
    import torch
    from safetensors import safe_open

    src_map = _index(src, vendor)
    dst_map = _index(dst, vendor)
    vendor.validate(str(src), trained, 'rewrite')
    report = vendor.verify(str(src), str(dst), trained)
    mixed = {shard for name, shard in src_map.items() if name.startswith('mtp.')}
    for shard in sorted(mixed):
        names = [name for name, value in src_map.items()
                 if value == shard and not name.startswith('mtp.')]
        if not names:
            continue
        with safe_open(str(src / shard), framework='pt', device='cpu') as before:
            for name in names:
                with safe_open(str(dst / dst_map[name]), framework='pt', device='cpu') as after:
                    left = before.get_tensor(name).reshape(-1).view(torch.uint8)
                    right = after.get_tensor(name).reshape(-1).view(torch.uint8)
                    if not torch.equal(left, right):
                        raise ValueError('non-MTP tensor bytes changed')
    return report


def _publish(staged, dst):
    """Linux no-replace rename, including a destination created during build."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, 'renameat2', None)
    if rename is None:
        raise OSError(errno.ENOSYS, 'atomic no-replace rename is unavailable')
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staged), -100, os.fsencode(dst), 1) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(dst))


def run(src, dst, ckpt, gate, *, strategy='rewrite', verify=False):
    if not gate_allows(gate, src, ckpt):
        return {'status': 'skipped', 'reason': 'renewal gate denied, missing, or stale'}
    src, ckpt = _read_path(src), _read_path(ckpt)
    dst = writable_path(dst)
    # The vendor also reads side files and symlinked shards: guard every
    # immediate source entry before its own index/stat/open operations.
    for path in src.iterdir():
        _read_path(path)
    vendor = _vendor()
    vendor.resolve_dirs(str(src), str(dst))
    if os.path.lexists(dst):
        raise ValueError('destination exists; writeback never overwrites')
    source_map = _index(src, vendor)
    trained = vendor.load_trained(str(ckpt))
    vendor.validate(str(src), trained, strategy)
    vendor.plan_sources(str(src), str(dst), sorted(set(source_map.values())))
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Vendor builds and verifies inside a private directory; only our final
    # no-replace rename publishes it. The vendor's --force is never exposed.
    with tempfile.TemporaryDirectory(prefix='.' + dst.name + '.writeback-', dir=dst.parent) as temporary:
        old_cache = os.environ.get('MTP_DIGEST_CACHE')
        os.environ['MTP_DIGEST_CACHE'] = str(Path(temporary) / 'digests.json')
        try:
            staged = Path(temporary) / 'model'
            report = vendor.build(str(src), str(staged), trained, strategy=strategy)
            report['verify'] = _verify(src, staged, trained, vendor)
            if not gate_allows(gate, src, ckpt):
                return {'status': 'skipped', 'reason': 'renewal gate changed during writeback'}
            writable_path(dst)
            _publish(staged, dst)
            from mtptrain.fileio import fsync_dir
            fsync_dir(str(dst.parent))
            if verify:
                report['verify'] = _verify(src, dst, trained, vendor)
            return {**report, 'dst': str(dst), 'status': 'written'}
        finally:
            if old_cache is None:
                os.environ.pop('MTP_DIGEST_CACHE', None)
            else:
                os.environ['MTP_DIGEST_CACHE'] = old_cache


def _write_status(path, report):
    """Publish a durable JSON receipt without replacing another run's receipt."""
    path = writable_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', prefix='.' + path.name + '.',
                                         suffix='.tmp', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, sort_keys=True, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        writable_path(path)
        _publish(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = StrictParser(description=__doc__, allow_abbrev=False)
    for option in ('src', 'dst', 'ckpt', 'gate'):
        parser.add_argument('--' + option, required=True)
    parser.add_argument('--strategy', choices=('rewrite', 'new-shard'), default='rewrite')
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--status-file', help='optional new atomic JSON receipt (never overwritten)')
    args = parser.parse_args(argv)
    status_path = None
    try:
        if args.status_file is not None:
            candidate = writable_path(args.status_file)
            if os.path.lexists(candidate):
                raise FileExistsError('status file already exists')
            if any(candidate.is_relative_to(_path(root).resolve()) for root in (args.src, args.dst)):
                raise ValueError('status file must be outside source and destination model directories')
            status_path = candidate
        report = run(args.src, args.dst, args.ckpt, args.gate,
                     strategy=args.strategy, verify=args.verify)
        if status_path is not None:
            _write_status(status_path, report)
    except (OSError, ValueError, RuntimeError, SystemExit) as exc:
        if status_path is not None:
            try:
                _write_status(status_path, {'status': 'failed', 'reason': type(exc).__name__})
            except (OSError, ValueError):
                pass
        parser.exit(2, f'writeback refused: {exc}\n')
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
