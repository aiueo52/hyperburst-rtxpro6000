"""CPU-only snapshot cache advice using tiny temporary files."""
import os
from pathlib import Path
import sys

import pytest
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtptrain import fileio, train, weights


def test_drop_page_cache_flushes_advises_and_closes(tmp_path, monkeypatch):
    path = tmp_path / 'snapshot'
    path.write_bytes(b'weights')
    events = []
    real_open, real_close = os.open, os.close

    def open_file(value, flags):
        assert value == path
        assert flags & os.O_NOFOLLOW
        fd = real_open(value, flags)
        events.append(('open', fd))
        return fd

    monkeypatch.setattr(os, 'open', open_file)
    monkeypatch.setattr(os, 'fsync', lambda fd: events.append(('fsync', fd)))
    monkeypatch.setattr(os, 'posix_fadvise', lambda fd, offset, length, advice:
                        events.append(('advice', fd, offset, length, advice)))

    def close_file(fd):
        events.append(('close', fd))
        real_close(fd)

    monkeypatch.setattr(os, 'close', close_file)
    fileio.drop_page_cache(path)
    fd = events[0][1]
    assert events == [('open', fd), ('fsync', fd),
                      ('advice', fd, 0, 0, os.POSIX_FADV_DONTNEED), ('close', fd)]
    with pytest.raises(OSError):
        os.fstat(fd)


@pytest.mark.parametrize('operation', ['open', 'fsync', 'posix_fadvise', 'close'])
def test_drop_page_cache_failures_are_harmless(tmp_path, monkeypatch, operation):
    path = tmp_path / 'snapshot'
    path.write_bytes(b'weights')
    opened = []
    real_open, real_close = os.open, os.close

    def open_file(*args, **kwargs):
        fd = real_open(*args, **kwargs)
        opened.append(fd)
        return fd

    def fail(*args):
        if operation == 'close':
            real_close(args[0])
        raise OSError('simulated cache advice failure')

    monkeypatch.setattr(os, 'open', open_file)
    monkeypatch.setattr(os, operation, fail)
    fileio.drop_page_cache(path)
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)
    assert path.read_bytes() == b'weights'


def test_drop_page_cache_does_not_follow_symlink(tmp_path, monkeypatch):
    target = tmp_path / 'target'
    target.write_bytes(b'weights')
    link = tmp_path / 'link'
    link.symlink_to(target)
    monkeypatch.setattr(os, 'posix_fadvise', lambda *args: pytest.fail('followed symlink'))
    fileio.drop_page_cache(link)


@pytest.mark.parametrize('name', ['best-mtp.safetensors', 'mtp-step100.safetensors',
                                 'final-mtp.safetensors'])
def test_save_weights_advises_completed_atomic_snapshot(tmp_path, monkeypatch, name):
    path = tmp_path / name
    state = {'weight': torch.ones(2, dtype=torch.bfloat16, device='cpu')}
    model = object()

    def export(value):
        assert value is model
        return state

    monkeypatch.setattr(weights, 'export_state_dict', export)
    calls = []

    def advise(value):
        assert value == str(path)
        assert list(tmp_path.iterdir()) == [path]
        with safe_open(path, framework='pt', device='cpu') as snapshot:
            assert torch.equal(snapshot.get_tensor('weight'), state['weight'])
            assert snapshot.metadata()['step'] == '100'
            assert snapshot.metadata()['mtp_tensors_digest'] == train.tensors_digest(state)
        calls.append(value)

    monkeypatch.setattr(fileio, 'drop_page_cache', advise)
    train.save_weights_bf16(str(path), model, {'step': 100})
    assert calls == [str(path)]


def test_failed_atomic_snapshot_does_not_advise(tmp_path, monkeypatch):
    path = tmp_path / 'best-mtp.safetensors'
    path.write_bytes(b'previous snapshot')
    monkeypatch.setattr(weights, 'export_state_dict', lambda _: {})

    def fail(*args):
        raise OSError('simulated write failure')

    monkeypatch.setattr(train, 'write_atomic', fail)
    monkeypatch.setattr(fileio, 'drop_page_cache', lambda _: pytest.fail('failed snapshot advice'))
    with pytest.raises(OSError, match='write failure'):
        train.save_weights_bf16(str(path), object())
    assert path.read_bytes() == b'previous snapshot'
