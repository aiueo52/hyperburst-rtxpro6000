"""Synthetic CPU checks for renewal gates, retention and replay provenance."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_head
import regate
import writeback


def _reports():
    original = {
        'run_id': 'synthetic-check',
        'compare': {'eval_data': 'fixed', 'token_map_sha256': '1' * 64},
        'renewal': {'exact_greedy': True, 'documents': 2},
        'summary': {'rows': 40, 'ce': 1.5, 'agreement@1': 0.5,
                    'accept_len@3': 2.0, 'verifies@3': 20,
                    'accept_len@15': 4.0, 'verifies@15': 10},
    }
    return original, deepcopy(original)


def _maps():
    return {'primary': {'path': 'primary.pt', 'sha256': '1' * 64},
            'secondary': {'path': None, 'sha256': None, 'evaluated': False,
                          'limitation': train_head.SECONDARY_MAP_LIMITATION}}


@pytest.mark.parametrize('three,fifteen,passed', [
    (2.02, 3.96, True), (1.98, 4.04, True), (2.019, 4.0, False),
    (2.2, 3.959, False), (1.979, 4.4, False), (2.0, 4.0, False),
])
def test_gate_thresholds(three, fifteen, passed):
    original, best = _reports()
    best['summary'].update({'accept_len@3': three, 'accept_len@15': fifteen})
    gate = train_head.evaluate_gate(original, best, token_maps=_maps())
    assert gate['passed'] is passed


@pytest.mark.parametrize('field,value', [
    ('accept_len@3', None), ('accept_len@15', float('nan')),
    ('accept_len@3', float('inf')), ('accept_len@3', True),
    ('verifies@3', 0), ('verifies@15', None), ('rows', 0), ('ce', None),
])
def test_missing_measurements_fail_closed(field, value):
    original, best = _reports()
    best['summary']['accept_len@3'] = 3.0
    best['summary'][field] = value
    assert not train_head.evaluate_gate(original, best, token_maps=_maps())['passed']
    assert not train_head.evaluate_gate(None, best, token_maps=_maps())['passed']


def test_gate_requires_comparable_greedy_map_bound_reports():
    original, best = _reports()
    best['summary']['accept_len@3'] = 3.0
    assert not train_head.evaluate_gate(original, best)['passed']
    best['compare']['eval_data'] = 'changed'
    gate = train_head.evaluate_gate(original, best, token_maps=_maps())
    assert gate['reason'] == 'incomparable_evaluations'
    best['compare'] = deepcopy(original['compare'])
    best['renewal']['exact_greedy'] = False
    gate = train_head.evaluate_gate(original, best, token_maps=_maps())
    assert gate['reason'] == 'missing_greedy_renewal'


def test_guide_configuration_and_retention_flags(monkeypatch, tmp_path):
    monkeypatch.delenv('MTP_MODEL_DIR', raising=False)
    monkeypatch.delenv('MTP_DUMP_DIR', raising=False)
    args = train_head.build_parser().parse_args([
        '--model-dir', str(tmp_path / 'model'), '--dump-dir', str(tmp_path / 'dump'),
        '--output-dir', str(tmp_path / 'out'), '--token-map', str(tmp_path / 'map.pt')])
    argv = train_head.training_arguments(args)
    expected = {'--tokens-per-step': '8192', '--grad-accum': '3', '--max-len': '2048',
                '--prefetch': '6', '--soft-alpha': '0.5', '--rollout-k': '3',
                '--lr': '1e-5', '--warmup': '50', '--steps': '1000',
                '--snapshot-keep': '1', '--time-limit-seconds': '3600'}
    assert all(argv[argv.index(key) + 1] == value for key, value in expected.items())
    assert argv[argv.index('--renewal-ks') + 1:argv.index('--renewal-ks') + 3] == ['3', '15']
    assert '--no-latest' in argv and '--no-final-weights' in argv


def test_real_training_arguments_match_frozen_reference():
    args = SimpleNamespace(tiny=False, model_dir=Path('model'), dump_dir=Path('dump'),
                           eval_set=None, output_dir=Path('out'), device='cuda',
                           token_map=Path('primary.pt'), steps=None, max_seconds=3600)
    expected = [
        '--model-dir', 'model', '--dump-dir', 'dump', '--eval-set', 'dump/eval_fixed.json',
        '--out', 'out', '--device', 'cuda', '--bf16',
        '--tokens-per-step', '8192', '--grad-accum', '3', '--max-len', '2048', '--prefetch', '6',
        '--soft-alpha', '0.5', '--soft-temp', '1.0', '--soft-steps', 'first',
        '--rollout-k', '3', '--rollout-weight', '0.5', '--rollout-mask', 'accepted',
        '--epochs', '20', '--steps', '1000', '--lr', '1e-5', '--warmup', '50',
        '--min-lr-ratio', '0.1', '--grad-clip', '1.0', '--logit-chunk', '512',
        '--eval-every', '100', '--eval-batch', '8', '--eval-starts', '16', '--eval-batches', '20',
        '--chain-ks', '3', '15', '--eval-renewal', '--renewal-ks', '3', '15',
        '--renewal-docs', '48', '--renewal-batch', '32', '--renewal-min-gen', '32',
        '--snapshot-every-eval', '--snapshot-keep', '1', '--no-final-weights', '--no-latest',
        '--ckpt-every', '200', '--log-every', '20', '--eval-original-and-best',
        '--allow-missing-renewal', '--time-limit-seconds', '3600',
        '--token-map', 'primary.pt', '--renewal-dump-dir', 'renewal', '--renewal-greedy-only',
    ]
    assert train_head.training_arguments(args, Path('renewal'), greedy_only=True) == expected


def test_tiny_training_keeps_best_and_one_snapshot_without_optimizer_state(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.delenv('MTP_MODEL_DIR', raising=False)
    monkeypatch.delenv('MTP_DUMP_DIR', raising=False)
    monkeypatch.setenv('MTP_DIGEST_CACHE', str(tmp_path / 'digests.json'))
    out = tmp_path / 'run'
    args = train_head.build_parser().parse_args([
        '--tiny', '--device', 'cpu', '--out', str(out), '--steps', '3'])
    result = train_head.run_stage(args)
    assert result['training']['steps'] == 3
    assert (out / 'best-mtp.safetensors').is_file()
    assert not list(out.glob('*.pt'))
    assert len(list(out.glob('mtp-step*.safetensors'))) == 1
    assert not (out / 'final-mtp.safetensors').exists()
    gate = json.loads((out / 'gate.json').read_bytes())
    assert not gate['passed'] and gate['reason'] == 'synthetic_rehearsal_not_publishable'
    before = (out / 'best-mtp.safetensors').read_bytes()
    with pytest.raises(ValueError, match='empty'):
        train_head.run_stage(args)
    assert (out / 'best-mtp.safetensors').read_bytes() == before


def _replay_dump(path, *, offset=0, max_len=0):
    import torch
    from safetensors.torch import save_file
    from mtptrain.data import _hash_ids, is_holdout_key, samples_digest
    from mtptrain.select import load_selfgen_samples, select_selfgen_docs

    path.mkdir()
    manifest, index = [], {}
    key = next(f'synthetic-heldout-{i}' for i in range(10000)
               if is_holdout_key(f'synthetic-heldout-{i}', 19, 0.4)
               and not is_holdout_key(f'synthetic-heldout-{i}', 20260903, 0.1))
    for i in range(2):
        ids = torch.arange(54, dtype=torch.int32) + offset + i * 100
        labels = torch.cat((ids[1:], ids[-1:]))
        labels[[3, 14, 24, 34, 44]] += 1
        name = f'synthetic-{i}.safetensors'
        save_file({'input_ids': ids, 'target_argmax': labels,
                   'positions': torch.arange(54, dtype=torch.int32),
                   'hc_hidden': torch.zeros(54, 4, dtype=torch.bfloat16),
                   'seq_offsets': torch.tensor([0, 54], dtype=torch.int32)}, str(path / name))
        digest = _hash_ids(ids.tolist())
        manifest.append({'doc_hash': digest, 'split_key': key, 'mode': 'selfgen',
                         'temperature': 0.0, 'prompt_tokens': 4,
                         'generated_tokens': 50, 'bucket': 'synthetic'})
        index[digest] = {'file': name, 'seq': 0, 'tokens': 54, 'files': [name], 'seqs': [0]}
    (path / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in manifest))
    (path / 'index.json').write_text(json.dumps(index))
    chosen, _ = select_selfgen_docs(str(path), split_seed=19, holdout_frac=0.4,
                                    per_bucket=2, min_gen=32, seed=7, greedy_only=True, index=index)
    selected = load_selfgen_samples(str(path), chosen, index, max_len=max_len)
    data = samples_digest(selected.samples, selected.gen_start_rows + selected.buckets)
    rows, agree = (82, 74) if max_len == 44 else (100, 90)
    (path / 'extraction-result.json').write_text(json.dumps({
        'greedy_agreement': {'exact': False, 'documents': 2, 'rows': rows, 'agree': agree}}))
    return {'seed': 19, 'holdout': 0.4, 'synthetic': False,
            'renewal': {'docs': 2, 'buckets': 'synthetic', 'ks': [3, 15], 'batch': 32,
                        'max_len': max_len, 'min_gen': 32, 'greedy_only': True, 'seed': 7,
                        'data': data}}


def _replay_inputs(tmp_path, *, max_len=0):
    import torch
    from safetensors.torch import save_file
    from mtptrain.fileio import file_digest
    from mtptrain.weights import model_identity

    train, model, dump, output = (tmp_path / name for name in ('train', 'model', 'dump', 'replay'))
    train.mkdir()
    model.mkdir()
    save_file({'mtp.weight': torch.ones(2), 'model.weight': torch.zeros(2)},
              str(model / 'model.safetensors'))
    (model / 'model.safetensors.index.json').write_text(json.dumps({
        'weight_map': {'mtp.weight': 'model.safetensors', 'model.weight': 'model.safetensors'}}))
    for name in ('config.json', 'tokenizer.json'):
        (model / name).write_text('{"synthetic": 1}')
    checkpoint = train / 'best-mtp.safetensors'
    checkpoint.write_bytes(b'1234')
    primary, secondary = tmp_path / 'primary.pt', tmp_path / 'secondary.pt'
    primary.write_bytes(b'12')
    secondary.write_bytes(b'34')
    maps = _maps()
    maps['primary'] = {'path': str(primary), 'sha256': train_head._sha256(primary)}
    maps['secondary'].update(path=str(secondary), sha256=train_head._sha256(secondary))
    compare = _replay_dump(dump, max_len=max_len)
    reports = _reports()
    for report in reports:
        report['compare'].update(deepcopy(compare), token_map=file_digest(str(primary)),
                                 token_map_sha256=maps['primary']['sha256'])
        report['renewal'] = {'exact_greedy': False, 'sampled_diagnostic': True,
                             'synthetic': False, 'documents': 2}
    reports[1]['summary']['accept_len@3'] = 2.02
    reports[1]['checkpoint_digest'] = file_digest(str(checkpoint))
    for name, report in zip(('original-eval.json', 'best-eval.json'), reports):
        (train / name).write_text(json.dumps(report))
    (train / 'best.json').write_text('{}')
    binding = {'path': str(checkpoint), 'sha256': train_head._sha256(checkpoint)}
    gate = train_head.evaluate_gate(*reports, binding, source=str(model), token_maps=maps)
    gate['source_index_sha256'] = train_head._sha256(model / 'model.safetensors.index.json')
    gate['source_model_identity'] = model_identity(str(model))
    (train / 'gate.json').write_text(json.dumps(gate))
    return SimpleNamespace(train_from=train, output_dir=output, model_dir=model, dump_dir=dump,
                           token_map=primary, secondary_token_map=secondary), reports


def test_regate_retains_original_decision(tmp_path):
    args, reports = _replay_inputs(tmp_path)
    saved = (args.train_from / 'gate.json').read_bytes()
    policy = regate.collect_evidence(args.dump_dir, reports)
    assert policy['greedy_agreement'] == {'exact': False, 'agree': 90, 'rows': 100,
                                          'documents': 2, 'agreement': 0.9}
    assert policy['deviation'] == regate.DEVIATION and policy['min_agreement'] == 0.9
    result = regate.run(args)
    assert result['passed'] and result['renewal_policy'] == policy
    assert result['source_model_identity'] == json.loads(saved)['source_model_identity']
    assert (args.train_from / 'gate.json').read_bytes() == saved
    assert (args.output_dir / 'gate-exact-greedy.json').read_bytes() == saved
    checkpoint = args.output_dir / 'best-mtp.safetensors'
    assert os.path.samefile(args.train_from / checkpoint.name, checkpoint)
    assert writeback.gate_allows(args.output_dir / 'gate.json', args.model_dir, checkpoint)
    policy['greedy_agreement']['agree'] = 89
    denied = train_head.evaluate_gate(*reports, token_maps=result['token_maps'], policy=policy)
    assert not denied['passed'] and denied['reason'] == 'invalid_label_replay_policy'


@pytest.mark.parametrize('changed', ['model.safetensors', 'config.json', 'tokenizer.json', None])
def test_modified_or_unbound_source_denied(tmp_path, changed):
    args, _ = _replay_inputs(tmp_path)
    result = regate.run(args)
    approved = args.output_dir
    if changed is None:
        for directory in (args.train_from, approved):
            path = directory / 'gate.json'
            gate = json.loads(path.read_bytes())
            del gate['source_model_identity']
            path.write_text(json.dumps(gate))
    else:
        path = args.model_dir / changed
        stat, data = path.stat(), path.read_bytes()
        path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert result['source_index_sha256'] == train_head._sha256(
        args.model_dir / 'model.safetensors.index.json')
    args.output_dir = tmp_path / 'rejected'
    with pytest.raises(ValueError, match='source model identity'):
        regate.run(args)
    checkpoint = approved / 'best-mtp.safetensors'
    assert not writeback.gate_allows(approved / 'gate.json', args.model_dir, checkpoint)
    destination = tmp_path / 'published'
    assert writeback.run(args.model_dir, destination, checkpoint,
                         approved / 'gate.json')['status'] == 'skipped'
    assert not destination.exists() and not args.output_dir.exists()


def test_source_modified_during_training_denied(tmp_path, monkeypatch):
    import model_check
    import mtptrain.train
    from mtptrain.weights import model_identity

    inputs, reports = _replay_inputs(tmp_path)
    before = model_identity(str(inputs.model_dir))
    monkeypatch.setattr(model_check, 'validate_model_headers', lambda _: {})

    def synthetic_train(args, **kwargs):
        out = Path(args.out)
        (out / 'best-mtp.safetensors').write_bytes(b'1234')
        for name, report in zip(('original-eval.json', 'best-eval.json'), reports):
            (out / name).write_text(json.dumps(report))
        (inputs.model_dir / 'config.json').write_text('{"synthetic": 2}')
        return {'steps': 1, 'time_limit_reached': False}

    monkeypatch.setattr(mtptrain.train, 'train', synthetic_train)
    args = train_head.build_parser().parse_args([
        '--model-dir', str(inputs.model_dir), '--dump-dir', str(inputs.dump_dir),
        '--token-map', str(inputs.token_map), '--out', str(inputs.output_dir), '--device', 'cpu'])
    gate = train_head.run_stage(args)['gate']
    assert not gate['passed'] and gate['reason'] == 'source_model_changed'
    assert gate['source_model_identity'] == before


def test_unrelated_same_count_dump_denied_at_regate_and_writeback(tmp_path):
    args, reports = _replay_inputs(tmp_path)
    result = regate.run(args)
    approved = args.output_dir
    other = tmp_path / 'unrelated'
    other_compare = _replay_dump(other, offset=20)
    other_reports = deepcopy(reports)
    for report in other_reports:
        report['compare'].update(deepcopy(other_compare))
    other_policy = regate.collect_evidence(other, other_reports)
    assert other_policy['greedy_agreement'] == result['renewal_policy']['greedy_agreement']
    args.dump_dir, args.output_dir = other, tmp_path / 'rejected'
    with pytest.raises(ValueError, match='renewal data'):
        regate.run(args)
    assert not args.output_dir.exists()
    result['renewal_policy'] = other_policy
    gate_path = approved / 'gate.json'
    gate_path.write_text(json.dumps(result))
    checkpoint = approved / 'best-mtp.safetensors'
    assert not writeback.gate_allows(gate_path, args.model_dir, checkpoint)
    destination = tmp_path / 'published'
    assert writeback.run(args.model_dir, destination, checkpoint, gate_path)['status'] == 'skipped'
    assert not destination.exists()


def test_writeback_rechecks_dump_tensor_content(tmp_path):
    from safetensors.torch import load_file, save_file

    args, _ = _replay_inputs(tmp_path)
    regate.run(args)
    path = args.dump_dir / 'synthetic-0.safetensors'
    tensors = load_file(str(path))
    tensors['hc_hidden'][0, 0] = 1
    save_file(tensors, str(path))
    assert not writeback.gate_allows(args.output_dir / 'gate.json', args.model_dir,
                                     args.output_dir / 'best-mtp.safetensors')


@pytest.mark.parametrize('field,value', [('agree', 91), ('rows', 101), ('documents', 1),
                                         ('agree', True)])
def test_regate_checks_measured_receipt_counts(tmp_path, field, value):
    args, reports = _replay_inputs(tmp_path)
    path = args.dump_dir / 'extraction-result.json'
    receipt = json.loads(path.read_bytes())
    receipt['greedy_agreement'][field] = value
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='receipt agreement counts'):
        regate.collect_evidence(args.dump_dir, reports)


def test_regate_reconstructs_reported_truncation(tmp_path):
    args, _ = _replay_inputs(tmp_path, max_len=44)
    result = regate.run(args)
    assert result['passed']
    assert result['renewal_policy']['greedy_agreement']['rows'] == 82
    assert result['renewal_policy']['greedy_agreement']['agree'] == 74


def test_regate_requires_measurable_first_generated_token(tmp_path):
    from mtptrain.data import samples_digest
    from mtptrain.select import load_selfgen_samples, select_selfgen_docs

    args, reports = _replay_inputs(tmp_path)
    path = args.dump_dir / 'manifest.jsonl'
    manifest = [json.loads(line) for line in path.read_text().split('\n') if line]
    manifest[0]['prompt_tokens'] = 1
    path.write_text(''.join(json.dumps(r) + '\n' for r in manifest))
    index = json.loads((args.dump_dir / 'index.json').read_bytes())
    chosen, _ = select_selfgen_docs(str(args.dump_dir), split_seed=19, holdout_frac=0.4,
                                    per_bucket=2, seed=7, greedy_only=True, index=index)
    selected = load_selfgen_samples(str(args.dump_dir), chosen, index)
    digest = samples_digest(selected.samples, selected.gen_start_rows + selected.buckets)
    for report in reports:
        report['compare']['renewal']['data'] = digest
    with pytest.raises(ValueError, match='generation boundary'):
        regate.collect_evidence(args.dump_dir, reports)
