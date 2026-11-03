#!/usr/bin/env python3
"""CPU-only, explicit argmax-label replay of finished held-out evaluations."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mtptrain.cli import StrictParser, split_list
from mtptrain.paths import input_path
from train_head import (GATE_SCHEMA, LABEL_REPLAY_MIN_AGREEMENT, _sha256,
                        evaluate_gate, label_replay_agreement)

DEVIATION = ('post-hoc: exact greedy renewal impossible with this dump; gate evaluated '
             'on argmax-label replay of the same temperature-0 held-out documents')


def collect_evidence(dump, reports):
    """Reconstruct the reported selection and bind its full tensor content."""
    from safetensors import SafetensorError
    from mtptrain.data import samples_digest
    from mtptrain.select import load_selfgen_samples, select_selfgen_docs

    if (len(reports) != 2 or any(not isinstance(r, dict) for r in reports)
            or not isinstance(reports[0].get('compare'), dict)
            or reports[0]['compare'] != reports[1].get('compare')):
        raise ValueError('incomparable renewal reports')
    compare = reports[0]['compare']
    renewal = compare.get('renewal')
    if (not isinstance(renewal, dict)
            or any(type(renewal.get(k)) is not int or renewal[k] < 0
                   for k in ('docs', 'min_gen', 'max_len'))
            or type(renewal.get('seed')) is not int
            or type(renewal.get('greedy_only')) is not bool
            or not isinstance(renewal.get('buckets'), str)
            or type(compare.get('seed')) is not int
            or type(compare.get('holdout')) not in (int, float)
            or not math.isfinite(compare['holdout']) or not 0 <= compare['holdout'] <= 1):
        raise ValueError('invalid renewal selection settings')
    dump = input_path(dump).resolve(strict=True)
    receipt = input_path(dump / 'extraction-result.json')
    raw = receipt.read_bytes()
    extraction = json.loads(raw)
    index_path = input_path(dump / 'index.json')
    index_raw = index_path.read_bytes()
    try:
        chosen, index = select_selfgen_docs(
            str(dump), greedy_only=renewal['greedy_only'], split='eval',
            split_seed=compare['seed'], holdout_frac=compare['holdout'],
            buckets=split_list(renewal['buckets'], 'renewal buckets') or None,
            per_bucket=renewal['docs'], min_gen=renewal['min_gen'], seed=renewal['seed'],
            index=json.loads(index_raw))
        if not chosen or not all(type(row.get('temperature')) in (int, float)
                                 and row['temperature'] == 0 for row in chosen):
            raise ValueError('selected documents must have explicit temperature 0')
        selected = load_selfgen_samples(str(dump), chosen, index, max_len=renewal['max_len'])
    except (SystemExit, SafetensorError, RuntimeError, IndexError, AssertionError) as exc:
        raise ValueError('invalid renewal dump selection') from exc
    digest = samples_digest(selected.samples, list(selected.gen_start_rows) + list(selected.buckets))
    if any(report['compare']['renewal'].get('data') != digest for report in reports):
        raise ValueError('renewal data does not match evaluated samples')
    agreement = {'documents': len(selected), 'rows': 0, 'agree': 0}
    for sample, g in zip(selected.samples, selected.gen_start_rows):
        if g < 1 or g >= len(sample) - 1:
            raise ValueError('invalid renewal generation boundary')
        consistent = sample.greedy_consistent[g - 1:-1]
        agreement['rows'] += int(consistent.numel())
        agreement['agree'] += int(consistent.sum().item())
    if not agreement['documents'] or not agreement['rows']:
        raise ValueError('missing renewal agreement measurements')
    agreement['exact'] = agreement['agree'] == agreement['rows']
    measured = extraction.get('greedy_agreement') if isinstance(extraction, dict) else None
    if (not isinstance(measured, dict)
            or any(type(measured.get(k)) is not int or measured[k] != agreement[k]
                   for k in ('documents', 'rows', 'agree'))
            or measured.get('exact') is not agreement['exact']):
        raise ValueError('extraction receipt agreement counts do not match evaluated samples')
    agreement['agreement'] = agreement['agree'] / agreement['rows']
    return {'name': 'argmax_label_replay',
            'greedy_agreement': agreement,
            'extraction_result': {'path': str(receipt.resolve(strict=True)),
                                  'sha256': hashlib.sha256(raw).hexdigest()},
            'explicit_greedy_documents': len(selected),
            'manifest_sha256': _sha256(dump / 'manifest.jsonl'),
            'index_sha256': hashlib.sha256(index_raw).hexdigest(),
            'min_agreement': LABEL_REPLAY_MIN_AGREEMENT, 'deviation': DEVIATION}


def verify_evidence(policy, reports):
    """Rebind the receipt and evaluated sample tensors before writeback."""
    if (not label_replay_agreement(policy)
            or policy.get('min_agreement') != LABEL_REPLAY_MIN_AGREEMENT):
        raise ValueError('invalid label replay agreement')
    receipt = Path(policy['extraction_result']['path'])
    if not receipt.is_absolute() or receipt.name != 'extraction-result.json':
        raise ValueError('invalid extraction receipt path')
    if collect_evidence(receipt.parent, reports) != policy:
        raise ValueError('label replay evidence changed')


def run(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from mtptrain.fileio import file_digest, write_json_atomic
    from mtptrain.weights import model_identity

    train = input_path(args.train_from).resolve(strict=True)
    gate = json.loads(input_path(train / 'gate.json').read_bytes())
    checkpoint = input_path(train / 'best-mtp.safetensors')
    if (type(gate.get('schema')) is not int or gate['schema'] != GATE_SCHEMA
            or gate.get('passed') is not False or gate.get('reason') != 'missing_greedy_renewal'
            or gate.get('checkpoint', {}).get('sha256') != _sha256(checkpoint)):
        raise ValueError('requires a finished missing_greedy_renewal gate and matching checkpoint')
    output = Path(args.output_dir).expanduser().absolute()
    if (any(p.is_symlink() for p in (output, *output.parents))
            or output == train or output.is_relative_to(train)
            or (output.exists() and (not output.is_dir() or any(output.iterdir())))):
        raise ValueError('output directory must be new or empty and separate from training')
    model = input_path(args.model_dir).resolve(strict=True)
    if input_path(gate['source']).resolve(strict=True) != model:
        raise ValueError('source model does not match gate')
    index_sha = _sha256(model / 'model.safetensors.index.json')
    if gate.get('source_index_sha256') != index_sha:
        raise ValueError('source index changed')
    if gate.get('source_model_identity') != model_identity(str(model)):
        raise ValueError('source model identity changed or missing')
    maps = deepcopy(gate['token_maps'])
    for key, value in (('primary', args.token_map), ('secondary', args.secondary_token_map)):
        path = input_path(value).resolve(strict=True)
        if maps[key]['sha256'] != _sha256(path):
            raise ValueError('token map does not match gate')
        maps[key]['path'] = str(path)
    reports = [json.loads(input_path(train / name).read_bytes())
               for name in ('original-eval.json', 'best-eval.json')]
    if reports[1].get('checkpoint_digest') != file_digest(str(checkpoint)):
        raise ValueError('checkpoint evaluation digest mismatch')
    policy = collect_evidence(args.dump_dir, reports)
    # Provenance and thresholds are decided by evaluate_gate, including failures.
    output.mkdir(parents=True, exist_ok=True)
    os.link(checkpoint, output / checkpoint.name)  # EXDEV must fail; never copy weights.
    for name in ('original-eval.json', 'best-eval.json', 'best.json', 'history.json'):
        source = train / name
        if name != 'history.json' or source.exists():
            shutil.copyfile(input_path(source), output / name)
    shutil.copyfile(input_path(train / 'gate.json'), output / 'gate-exact-greedy.json')
    binding = {'path': str(output / checkpoint.name), 'sha256': _sha256(checkpoint)}
    result = evaluate_gate(*reports, binding, source=str(model), token_maps=maps, policy=policy)
    result['source_index_sha256'] = index_sha
    result['source_model_identity'] = deepcopy(gate['source_model_identity'])
    write_json_atomic(str(output / 'gate.json'), result)
    return result


def main(argv=None):
    parser = StrictParser(description=__doc__, allow_abbrev=False)
    for option in ('train-from', 'output-dir', 'dump-dir', 'model-dir', 'token-map', 'secondary-token-map'):
        parser.add_argument('--' + option, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (OSError, ValueError, TypeError, KeyError, OverflowError, SystemExit) as error:
        print('regate refused: ' + str(error), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
