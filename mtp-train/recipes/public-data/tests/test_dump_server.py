"""Exercise command capture with a synthetic launcher and a replaced exec."""
import importlib.util
import json
from pathlib import Path
import re
import shlex
import stat
import sys

import pytest

spec = importlib.util.spec_from_file_location(
    'public_dump_server_test', Path(__file__).resolve().parents[1] / 'dump_server.py')
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)


def fixture_checkout(tmp_path, chat):
    checkout, model = tmp_path / 'checkout', tmp_path / 'model'
    checkout.mkdir()
    model.mkdir()
    hook = checkout / 'python/sglang/srt/models/mtp_dump.py'
    hook.parent.mkdir(parents=True)
    hook.write_text('# Synthetic hook marker.\n')
    packet = {'argv': [sys.executable, '-P', '-B', str(checkout / 'server.py'), 'serve',
                       '--model-path', str(model), '--context-length', '4096',
                       '--skip-tokenizer-init', '--disable-radix-cache', '--disable-cuda-graph',
                       '--max-running-requests', '1', '--chunked-prefill-size', '2048',
                       '--random-seed', '1729', *chat, recipe.REDACTED_KEY],
              'env': {'EXAMPLE_MERGED': 'captured', 'EXAMPLE_EXPORTED': 'value',
                      'PYTHONPATH': f'{checkout.resolve()}/flashinfer-vt:{checkout.resolve()}/python',
                      'SGLANG_MTP_DUMP_TOPK': '99', 'SGLANG_MTP_DUMP_TAPS': '3',
                      'SGLANG_MTP_DRAFT_DUMP_DIR': 'unused'}}
    return checkout, model, packet


def write_launcher(checkout, packet):
    expected = ' '.join(recipe.CAPTURE_ARGS[2:])
    (checkout / 'serve-fast.sh').write_text(
        '#!/bin/bash\nset -eu\n'
        'test "$0" = ' + shlex.quote(str(checkout / 'serve-fast.sh')) + '\n'
        'test "$*" = ' + shlex.quote(expected) + '\n'
        'test "$FLASH_NEXT_EDITION" = normal\n'
        'test "$CONC" = 1\n'
        'test "$CONTEXT_LENGTH" = 4096\n'
        'test "$MAMBA_SLOTS" = 10\n'
        'test "$MEM_FRACTION" = 0.84\n'
        'test "$TOKEN_MAP" = tokenmaps/public_49152.pt\n'
        'test "$SGLANG_MTP_DUMP_TOPK" = 8\n'
        'test -z "${SGLANG_MTP_DUMP_TAPS+x}"\n'
        'test -z "${SGLANG_MTP_DRAFT_DUMP_DIR+x}"\n'
        "printf '%s\\n' " + shlex.quote(json.dumps(packet)) + '\n')


@pytest.mark.parametrize('chat', [['--chat-template', 'synthetic-template'],
                                 ['--chat-template=synthetic-template']])
def test_capture_print_and_exec(tmp_path, monkeypatch, capsys, chat):
    checkout, model, packet = fixture_checkout(tmp_path, chat)
    write_launcher(checkout, packet)
    monkeypatch.setenv('EXAMPLE_MERGED', 'caller')
    monkeypatch.setenv('EXAMPLE_UNCHANGED', 'preserved')
    monkeypatch.setenv('SGLANG_MTP_DUMP_TAPS', '23')
    monkeypatch.setenv('SGLANG_MTP_DRAFT_DUMP_DIR', 'unused')
    secret = 'a' * 64
    monkeypatch.setattr(recipe.secrets, 'token_hex', lambda size: secret)
    executed, directories = [], []
    monkeypatch.setattr(recipe.os, 'chdir', lambda path: directories.append(path))
    monkeypatch.setattr(recipe.os, 'execvpe', lambda *args: executed.append(args))
    dump = tmp_path / 'dump'
    args = ['--checkout', str(checkout), '--model-dir', str(model), '--dump-dir', str(dump)]
    assert recipe.main([*args, '--print']) == 0
    output = capsys.readouterr()
    assert secret not in output.out + output.err
    printed = json.loads(output.out)
    assert printed['argv'] == [arg for arg in packet['argv'] if arg not in chat]
    assert printed['argv'].count(recipe.REDACTED_KEY) == 1
    assert printed['env'] == {
        'EXAMPLE_MERGED': 'captured', 'EXAMPLE_EXPORTED': 'value',
        'PYTHONPATH': f'{checkout.resolve()}/flashinfer-vt:{checkout.resolve()}/python',
        'FLASH_NEXT_EDITION': 'normal', 'CONC': '1', 'CONTEXT_LENGTH': '4096',
        'MAMBA_SLOTS': '10', 'MEM_FRACTION': '0.84', 'TOKEN_MAP': 'tokenmaps/public_49152.pt',
        'TARGET_MODEL': str(model), 'PORT': '8001', 'SGLANG_MTP_DUMP_DIR': str(dump),
        'SGLANG_MTP_DUMP_TOPK': '8', 'SGLANG_MTP_DUMP_CHUNK': '1024',
        'SGLANG_MTP_DUMP_MAX_GB': '100', 'SGLANG_MTP_DUMP_TAG': 'public'}
    assert not executed and not directories
    assert not dump.exists()
    assert recipe.main([*args, '--port', '8123']) == 0
    assert stat.S_IMODE(dump.stat().st_mode) == 0o700 and not list(dump.iterdir())
    assert directories == [checkout] and len(executed) == 1
    executable, command, env = executed[0]
    assert executable == sys.executable
    assert not any(arg.startswith('--chat-template') for arg in command)
    keys = [arg for arg in command if arg.startswith('--admin-api-key=')]
    assert len(keys) == 1 and re.fullmatch('--admin-api-key=[0-9a-f]{64}', keys[0])
    assert keys == ['--admin-api-key=' + secret]
    assert env['PYTHONPATH'] == packet['env']['PYTHONPATH']
    assert env['PORT'] == '8123' and env['EXAMPLE_MERGED'] == 'captured'
    assert env['EXAMPLE_UNCHANGED'] == 'preserved' and env['SGLANG_MTP_DUMP_TOPK'] == '8'
    assert 'SGLANG_MTP_DUMP_TAPS' not in env
    assert not any(key.startswith('SGLANG_MTP_DRAFT_DUMP_') for key in env)


@pytest.mark.parametrize(('invalid', 'reason'), [
    ('flag', 'required extraction option is missing or repeated'),
    ('hook', 'checkout must contain the serving script and dump hook'),
    ('nonempty', 'dump directory must be absent or empty'),
    ('argv-type', 'invalid serving command capture'),
    ('env-type', 'invalid serving command capture'),
    ('key', 'exactly one redacted admin authentication option is required'),
    ('server', 'server command must use the selected checkout'),
    ('pythonpath', 'server imports must use the selected checkout'),
])
def test_invalid_capture_or_inputs_refused(tmp_path, monkeypatch, capsys, invalid, reason):
    checkout, model, packet = fixture_checkout(tmp_path, [])
    dump = tmp_path / 'dump'
    if invalid == 'flag':
        packet['argv'].remove('--disable-cuda-graph')
    elif invalid == 'hook':
        (checkout / 'python/sglang/srt/models/mtp_dump.py').unlink()
    elif invalid == 'nonempty':
        dump.mkdir()
        (dump / 'keep').write_text('synthetic')
    elif invalid == 'argv-type':
        packet['argv'] = 'not an argv list'
    elif invalid == 'env-type':
        packet['env']['EXAMPLE_EXPORTED'] = 12
    elif invalid == 'key':
        packet['argv'].remove(recipe.REDACTED_KEY)
    elif invalid == 'server':
        packet['argv'][3] = str(tmp_path / 'other/server.py')
    elif invalid == 'pythonpath':
        packet['env']['PYTHONPATH'] = str(tmp_path / 'other/python')
    write_launcher(checkout, packet)
    secret = 'b' * 64
    monkeypatch.setattr(recipe.secrets, 'token_hex', lambda size: secret)
    monkeypatch.setattr(recipe.os, 'execvpe', lambda *args: pytest.fail('unexpected exec'))
    with pytest.raises(SystemExit) as error:
        recipe.main(['--checkout', str(checkout), '--model-dir', str(model), '--dump-dir', str(dump)])
    assert error.value.code == 2
    output = capsys.readouterr()
    assert output.err.startswith('dump server refused: ')
    assert reason in output.err
    assert not output.out and secret not in output.err
    if invalid == 'nonempty':
        assert (dump / 'keep').read_text() == 'synthetic'
    else:
        assert not dump.exists()


def test_exec_value_error_redacts_generated_key(tmp_path, monkeypatch, capsys):
    checkout, model, packet = fixture_checkout(tmp_path, [])
    write_launcher(checkout, packet)
    secret = 'd' * 64
    monkeypatch.setattr(recipe.secrets, 'token_hex', lambda size: secret)
    monkeypatch.setattr(recipe.os, 'chdir', lambda path: None)

    def refuse_exec(executable, command, env):
        assert '--admin-api-key=' + secret in command
        raise ValueError(f'synthetic exec refusal: {command}')

    monkeypatch.setattr(recipe.os, 'execvpe', refuse_exec)
    with pytest.raises(SystemExit) as error:
        recipe.main(['--checkout', str(checkout), '--model-dir', str(model),
                     '--dump-dir', str(tmp_path / 'dump')])
    assert error.value.code == 2
    output = capsys.readouterr()
    assert output.err.startswith('dump server refused: synthetic exec refusal: ')
    assert not output.out and secret not in output.err


def test_failed_launcher_reports_stderr_tail(tmp_path, monkeypatch, capsys):
    checkout, model, _ = fixture_checkout(tmp_path, [])
    secret = 'c' * 64
    lines = ['early --admin-api-key=' + secret, 'early diagnostic one', 'early diagnostic two',
             'tail diagnostic one', 'tail diagnostic two', 'tail diagnostic three',
             'tail diagnostic four', 'tail diagnostic five']
    (checkout / 'serve-fast.sh').write_text(
        '#!/bin/bash\n'
        "printf '%s\\n' " + ' '.join(shlex.quote(line) for line in lines) + ' >&2\n'
        'exit 17\n')
    monkeypatch.setattr(recipe.secrets, 'token_hex', lambda size: secret)
    monkeypatch.setattr(recipe.os, 'execvpe', lambda *args: pytest.fail('unexpected exec'))
    dump = tmp_path / 'dump'
    with pytest.raises(SystemExit) as error:
        recipe.main(['--checkout', str(checkout), '--model-dir', str(model), '--dump-dir', str(dump)])
    assert error.value.code == 2
    output = capsys.readouterr()
    assert output.err.startswith('dump server refused: ')
    assert re.search(r'\b17\b', output.err)
    assert all(line in output.err for line in lines[-5:])
    assert not any(line in output.err for line in lines[:-5])
    assert not output.out and secret not in output.err
    assert not dump.exists()
