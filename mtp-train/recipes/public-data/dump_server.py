#!/usr/bin/env python3
"""Capture and execute the dump-enabled serving command without lifecycle control."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mtptrain.paths import input_path

REDACTED_KEY = '--admin-api-key=<redacted>'
CAPTURE_ARGS = ['bash', 'serve-fast.sh', 'w4', '--print', '--skip-tokenizer-init',
                '--disable-radix-cache', '--disable-cuda-graph', '--max-running-requests', '1',
                '--chunked-prefill-size', '2048', '--random-seed', '1729']


def without_extra_dumps(env):
    return {key: value for key, value in env.items()
            if key != 'SGLANG_MTP_DUMP_TAPS' and not key.startswith('SGLANG_MTP_DRAFT_DUMP_')}


def validate_startup(checkout, argv, env):
    if argv[1:5] != ['-P', '-B', str(checkout / 'server.py'), 'serve']:
        raise ValueError('server command must use the selected checkout')
    if env.get('PYTHONPATH') != f'{checkout / "flashinfer-vt"}:{checkout / "python"}':
        raise ValueError('server imports must use the selected checkout')
    if any(arg == '--chat-template' or arg.startswith('--chat-template=') for arg in argv):
        raise ValueError('chat template is forbidden for ID-only extraction')
    for flag in ('--skip-tokenizer-init', '--disable-radix-cache', '--disable-cuda-graph'):
        if argv.count(flag) != 1:
            raise ValueError('required extraction option is missing or repeated')
    expected = {'--context-length': '4096', '--chunked-prefill-size': '2048',
                '--max-running-requests': '1', '--model-path': env['TARGET_MODEL']}
    for flag, value in expected.items():
        matches = [argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == flag]
        if not matches or matches[-1] != value:
            raise ValueError('unexpected extraction server configuration')
    if (env.get('SGLANG_MTP_DUMP_TOPK') != '8' or env.get('SGLANG_MTP_DUMP_TAPS', '')
            or env.get('SGLANG_MTP_DUMP_MAX_GB') != '100'):
        raise ValueError('unexpected extraction dump configuration')


def prepare(checkout, model_dir, dump_dir, port=8001):
    checkout = input_path(checkout).resolve(strict=True)
    model_dir = input_path(model_dir).resolve(strict=True)
    if not checkout.is_dir() or not model_dir.is_dir():
        raise ValueError('checkout and model must be directories')
    for name in ('serve-fast.sh', 'python/sglang/srt/models/mtp_dump.py'):
        if not input_path(checkout / name, root=checkout).is_file():
            raise ValueError('checkout must contain the serving script and dump hook')
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError('invalid server port')
    dump_dir = Path(os.path.abspath(Path(dump_dir).expanduser()))
    if any(path.is_symlink() for path in (dump_dir, *dump_dir.parents)):
        raise ValueError('dump directory must not contain symlink ancestors')
    if dump_dir.exists() and (not dump_dir.is_dir() or any(dump_dir.iterdir())):
        raise ValueError('dump directory must be absent or empty')
    overrides = {'FLASH_NEXT_EDITION': 'normal', 'CONC': '1', 'CONTEXT_LENGTH': '4096',
                 'MAMBA_SLOTS': '10', 'MEM_FRACTION': '0.84',
                 'TOKEN_MAP': 'tokenmaps/public_49152.pt', 'TARGET_MODEL': str(model_dir),
                 'PORT': str(port), 'SGLANG_MTP_DUMP_DIR': str(dump_dir),
                 'SGLANG_MTP_DUMP_TOPK': '8', 'SGLANG_MTP_DUMP_CHUNK': '1024',
                 'SGLANG_MTP_DUMP_MAX_GB': '100', 'SGLANG_MTP_DUMP_TAG': 'public'}
    captured = subprocess.run(['bash', str(checkout / 'serve-fast.sh'), *CAPTURE_ARGS[2:]], cwd=checkout,
                              env=without_extra_dumps({**os.environ, **overrides}),
                              stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              check=True, timeout=30)
    try:
        resolved = json.loads(captured.stdout)
    except ValueError:
        raise ValueError('invalid serving command capture JSON') from None
    if not isinstance(resolved, dict):
        raise ValueError('invalid serving command capture')
    argv, exported = resolved.get('argv'), resolved.get('env')
    if (not isinstance(argv, list) or not argv or not argv[0]
            or any(not isinstance(arg, str) or '\0' in arg for arg in argv)
            or not isinstance(exported, dict)
            or any(not isinstance(key, str) or not key or '=' in key
                   or not isinstance(value, str) or '\0' in key + value
                   for key, value in exported.items())):
        raise ValueError('invalid serving command capture')
    filtered, i = [], 0
    while i < len(argv):
        arg = argv[i]
        if arg == '--chat-template':
            if i + 1 == len(argv) or not argv[i + 1] or argv[i + 1].startswith('--'):
                raise ValueError('chat template option requires a value')
            i += 2
            continue
        if arg.startswith('--chat-template='):
            if not arg.partition('=')[2]:
                raise ValueError('chat template option requires a value')
        else:
            filtered.append(arg)
        i += 1
    argv = filtered
    authentication = [arg for arg in argv
                      if arg == '--admin-api-key' or arg.startswith('--admin-api-key=')]
    if authentication != [REDACTED_KEY]:
        raise ValueError('exactly one redacted admin authentication option is required')
    additions = without_extra_dumps({**exported, **overrides})
    env = without_extra_dumps({**os.environ, **additions})
    validate_startup(checkout, argv, env)
    argv[argv.index(REDACTED_KEY)] = '--admin-api-key=' + secrets.token_hex(32)
    return checkout, argv, env, additions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--dump-dir', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--print', dest='print_only', action='store_true')
    args = parser.parse_args(argv)
    command = []
    try:
        checkout, command, env, additions = prepare(
            args.checkout, args.model_dir, args.dump_dir, args.port)
        if args.print_only:
            redacted = [REDACTED_KEY if arg.startswith('--admin-api-key=') else arg
                        for arg in command]
            print(json.dumps({'argv': redacted, 'env': additions}, sort_keys=True))
        else:
            dump_dir = Path(env['SGLANG_MTP_DUMP_DIR'])
            if any(path.is_symlink() for path in (dump_dir, *dump_dir.parents)):
                raise ValueError('dump directory must not contain symlink ancestors')
            if dump_dir.exists() and (not dump_dir.is_dir() or any(dump_dir.iterdir())):
                raise ValueError('dump directory must be absent or empty')
            dump_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            dump_dir.chmod(0o700)
            os.chdir(checkout)
            os.execvpe(command[0], command, env)
    except subprocess.CalledProcessError as error:
        tail = '\n'.join((error.stderr or '').rstrip('\n').split('\n')[-5:])
        reason = f'launcher exited with status {error.returncode}'
        if tail:
            reason += '\n' + tail
    except ValueError as error:
        reason = str(error)
    except (OSError, TypeError, subprocess.SubprocessError):
        reason = 'unable to access paths or execute the serving command'
    else:
        return 0
    for arg in command:
        if arg.startswith('--admin-api-key='):
            reason = reason.replace(arg.partition('=')[2], '<redacted>')
    parser.exit(2, 'dump server refused: ' + reason + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
