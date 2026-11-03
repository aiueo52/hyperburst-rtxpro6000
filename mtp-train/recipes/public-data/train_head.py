#!/usr/bin/env python3
"""Bounded public MTP training and a fail-closed held-out renewal gate."""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mtptrain.paths import input_path

GATE_SCHEMA = 2
LABEL_REPLAY_MIN_AGREEMENT = 0.90
SECONDARY_MAP_LIMITATION = (
    'The secondary/recursive draft token map public_32768.pt was not evaluated; '
    'this gate covers only the supplied primary token map.'
)


def _number(value, *, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def label_replay_agreement(policy):
    """Validate count-based agreement without trusting the rounded receipt ratio."""
    if not isinstance(policy, dict) or policy.get('name') != 'argmax_label_replay':
        return False
    agreement = policy.get('greedy_agreement')
    if not isinstance(agreement, dict) or agreement.get('exact') is not False:
        return False
    agree, rows, docs = (agreement.get(k) for k in ('agree', 'rows', 'documents'))
    explicit = policy.get('explicit_greedy_documents')
    return (type(agree) is int and type(rows) is int and rows > 0 and 0 <= agree <= rows
            and type(docs) is int and docs > 0 and type(explicit) is int and explicit == docs
            and agree / rows >= LABEL_REPLAY_MIN_AGREEMENT)


def evaluate_gate(original, best, checkpoint=None, source=None, token_maps=None, policy=None):
    """Require measured, map-bound renewal with explicit provenance policy."""
    result = {'schema': GATE_SCHEMA, 'passed': False, 'reason': 'missing_measurements',
              'relative_gain': {'3': None, '15': None}, 'checkpoint': checkpoint,
              'source': source, 'baseline': None, 'best': None,
              'renewal_policy': 'exact_greedy' if policy is None else deepcopy(policy),
              'token_maps': token_maps or {
                  'primary': None, 'secondary': {'path': None, 'sha256': None,
                                                'evaluated': False,
                                                'limitation': SECONDARY_MAP_LIMITATION}}}
    token_map = result['token_maps'].get('primary') if isinstance(result['token_maps'], dict) else None
    if not isinstance(original, dict) or not isinstance(best, dict):
        return result
    result.update(baseline=original.get('summary'), best=best.get('summary'))
    if (not original.get('run_id') or original.get('run_id') != best.get('run_id')
            or not isinstance(original.get('compare'), dict)
            or original['compare'] != best.get('compare')):
        result['reason'] = 'incomparable_evaluations'
        return result
    if (not isinstance(token_map, dict) or not isinstance(token_map.get('path'), str)
            or not token_map['path'].strip()
            or not isinstance(token_map.get('sha256'), str) or len(token_map['sha256']) != 64
            or any(c not in '0123456789abcdef' for c in token_map['sha256'])
            or original['compare'].get('token_map_sha256') != token_map['sha256']):
        result['reason'] = 'missing_or_mismatched_token_map'
        return result
    if policy is not None and not label_replay_agreement(policy):
        result['reason'] = 'invalid_label_replay_policy'
        return result
    for report in (original, best):
        provenance = report.get('renewal', {})
        if policy is None:
            valid = (isinstance(provenance, dict) and provenance.get('exact_greedy') is True
                     and _number(provenance.get('documents'), positive=True))
        else:
            valid = (isinstance(provenance, dict) and provenance.get('exact_greedy') is False
                     and provenance.get('sampled_diagnostic') is True
                     and provenance.get('synthetic') is False
                     and type(provenance.get('documents')) is int
                     and provenance['documents'] == policy['explicit_greedy_documents'])
        if not valid:
            result['reason'] = 'missing_greedy_renewal'
            return result
        summary = report.get('summary')
        if (not isinstance(summary, dict) or not _number(summary.get('rows'), positive=True)
                or not _number(summary.get('ce'))
                or not _number(summary.get('agreement@1')) or summary['agreement@1'] > 1):
            return result
        for k in (3, 15):
            if (not _number(summary.get(f'accept_len@{k}'), positive=True)
                    or not 1 <= summary[f'accept_len@{k}'] <= k + 1
                    or type(summary.get(f'verifies@{k}')) is not int
                    or summary[f'verifies@{k}'] <= 0):
                return result
    gains = {}
    for k in (3, 15):
        a = original['summary'][f'accept_len@{k}']
        b = best['summary'][f'accept_len@{k}']
        gains[str(k)] = Decimal(str(b)) / Decimal(str(a)) - 1
    result['relative_gain'] = {k: float(v) for k, v in gains.items()}
    passed = max(gains.values()) >= Decimal('0.01') and min(gains.values()) >= Decimal('-0.01')
    result.update(passed=passed, reason='improved' if passed else 'renewal_threshold_not_met')
    return result


def training_arguments(args, renewal_dir=None, greedy_only=False):
    """Guide section 10: best plus one rolling snapshot; no latest or optimizer state."""
    if not args.tiny and args.token_map is None:
        raise ValueError('--token-map is required for real training and evaluation')
    if not args.tiny and (args.model_dir is None or args.dump_dir is None):
        raise ValueError('--model-dir and --dump-dir are required unless --tiny is used')
    argv = ['--out', str(args.output_dir), '--device', args.device, '--bf16',
            '--tokens-per-step', '8192', '--grad-accum', '3', '--max-len', '2048', '--prefetch', '6',
            '--soft-alpha', '0.5', '--soft-temp', '1.0', '--soft-steps', 'first',
            '--rollout-k', '3', '--rollout-weight', '0.5', '--rollout-mask', 'accepted',
            '--epochs', '20', '--steps', str(args.steps or (4 if args.tiny else 1000)),
            '--lr', '1e-5', '--warmup', '50', '--min-lr-ratio', '0.1',
            '--grad-clip', '1.0', '--logit-chunk', '512',
            '--eval-every', '100', '--eval-batch', '8', '--eval-starts', '16', '--eval-batches', '20',
            '--chain-ks', '3', '15', '--eval-renewal', '--renewal-ks', '3', '15',
            '--renewal-docs', '48', '--renewal-batch', '32', '--renewal-min-gen', '32',
            '--snapshot-every-eval', '--snapshot-keep', '1', '--no-final-weights', '--no-latest',
            '--ckpt-every', '200', '--log-every', '20', '--eval-original-and-best',
            '--allow-missing-renewal', '--time-limit-seconds', str(args.max_seconds)]
    if not args.tiny:
        argv = ['--model-dir', str(args.model_dir), '--dump-dir', str(args.dump_dir),
                '--eval-set', str(args.eval_set or Path(args.dump_dir) / 'eval_fixed.json')] + argv
    if args.token_map:
        argv += ['--token-map', str(args.token_map)]
    if args.tiny:
        # The synthetic teacher-forced split and the on-disk renewal split are
        # disjoint. Both use the public model/optimizer/evaluator code.
        argv += ['--tiny', '--synthetic', '4', '--tokens-per-step', '0', '--batch-size', '2',
                 '--grad-accum', '1', '--max-len', '24', '--prefetch', '1',
                 '--warmup', '1', '--eval-every', '1', '--eval-batch', '2',
                 '--eval-starts', '1', '--eval-batches', '1', '--renewal-docs', '1',
                 '--renewal-batch', '1', '--renewal-min-gen', '4', '--log-every', '1',
                 '--ckpt-every', '1']
    if renewal_dir:
        argv += ['--renewal-dump-dir', str(renewal_dir)]
    if greedy_only:
        argv += ['--renewal-greedy-only']
    return argv


def tiny_renewal_dump(path):
    """A small numeric-only fixture; no real model or corpus is accessed."""
    import torch
    from safetensors.torch import save_file
    from mtptrain.config import MTPConfig
    from mtptrain.data import _hash_ids, is_holdout_key
    from mtptrain.fileio import write_jsonl_atomic

    path.mkdir()
    config = MTPConfig.tiny()
    generator = torch.Generator().manual_seed(55)
    ids = torch.randint(config.vocab_size, (24,), generator=generator, dtype=torch.int32)
    labels = torch.cat((ids[1:], ids[-1:]))
    save_file({'input_ids': ids, 'positions': torch.arange(24, dtype=torch.int32),
               'target_argmax': labels,
               'hc_hidden': torch.randn(24, config.hc_size, generator=generator).to(torch.bfloat16),
               'seq_offsets': torch.tensor([0, 24], dtype=torch.int32),
               'prefix_lens': torch.tensor([0], dtype=torch.int32)}, str(path / 'tiny.safetensors'))
    key = next(f'tiny-heldout-{i}' for i in range(10000)
               if is_holdout_key(f'tiny-heldout-{i}', 20260903, 0.1))
    write_jsonl_atomic(str(path / 'manifest.jsonl'),
                       [{'doc_hash': _hash_ids(ids.tolist()), 'split_key': key,
                         'mode': 'selfgen', 'temperature': 0.0, 'prompt_tokens': 4,
                         'generated_tokens': 20, 'bucket': 'synthetic'}])


def renewal_policy(dump_dir, *, tiny=False):
    """Choose genuine greedy data if present, otherwise sampled diagnostics.

    Sampled documents are never relabelled as greedy. Missing boundaries or a
    missing selfgen manifest remain missing measurements, not zero acceptance.
    """
    from mtptrain.select import load_selfgen_samples, select_selfgen_docs

    options = dict(split='eval', split_seed=20260903, holdout_frac=0.1,
                   per_bucket=1 if tiny else 48, min_gen=4 if tiny else 32, seed=0)
    chosen, index = select_selfgen_docs(str(dump_dir), greedy_only=True, **options)
    explicit_greedy = bool(chosen) and all(type(r.get('temperature')) in (int, float)
                                          and r['temperature'] == 0 for r in chosen)
    if not explicit_greedy:
        chosen, index = select_selfgen_docs(str(dump_dir), greedy_only=False, **options)
    if not chosen:
        return False, {'exact_greedy': False, 'documents': 0,
                       'sampled_diagnostic': False, 'missing': True, 'synthetic': tiny}
    selected = load_selfgen_samples(str(dump_dir), chosen, index)
    # Temperature metadata alone is insufficient: verify the actual generated
    # continuation against dumped greedy argmax. The final row has no next ID.
    consistent = bool(len(selected)) and all(
        g >= 1 and len(s) - g > 1 and bool(s.greedy_consistent[g - 1:-1].all())
        for s, g in zip(selected.samples, selected.gen_start_rows))
    policy = {'exact_greedy': explicit_greedy and consistent,
              'documents': len(selected),
              'sampled_diagnostic': not (explicit_greedy and consistent),
              'synthetic': tiny}
    return explicit_greedy, policy


def _sha256(path):
    digest = hashlib.sha256()
    with input_path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run_stage(args, *, _clock=None):
    clock = _clock or time.monotonic
    started = clock()
    if not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 3600:
        raise ValueError('--max-seconds must be finite, positive and at most 3600')
    if args.steps is not None and not 1 <= args.steps <= 1000:
        raise ValueError('--steps must be between 1 and 1000')
    from mtptrain.fileio import file_digest, write_json_atomic
    from mtptrain.train import build_parser as trainer_parser, train
    from mtptrain.weights import model_identity
    from writeback import writable_path

    for key in ('model_dir', 'dump_dir', 'renewal_dump_dir', 'token_map', 'secondary_token_map', 'eval_set'):
        if getattr(args, key) is not None:
            setattr(args, key, input_path(getattr(args, key)))
    out = writable_path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError('output directory must be empty; previous best is never overwritten')
    out.mkdir(parents=True, exist_ok=True)
    args.output_dir = out
    gate_path = out / 'gate.json'
    write_json_atomic(str(gate_path), evaluate_gate(None, None))
    try:
        if not args.tiny and args.token_map is None:
            raise ValueError('--token-map is required for real training and evaluation')
        token_maps = {'primary': None,
                      'secondary': {'path': None, 'sha256': None, 'evaluated': False,
                                    'limitation': SECONDARY_MAP_LIMITATION}}
        for key, option in (('primary', 'token_map'), ('secondary', 'secondary_token_map')):
            path = getattr(args, option)
            if path is not None:
                if not path.is_file():
                    raise ValueError(f'--{option.replace("_", "-")} must be an existing file')
                binding = {'path': str(path), 'sha256': _sha256(path)}
                if key == 'secondary':
                    token_maps[key].update(binding)
                else:
                    token_maps[key] = binding
        source_identity = None
        source_index_sha256 = None
        if args.tiny:
            if args.device != 'cpu':
                raise ValueError('--tiny is CPU-only')
            renewal_dir = out / '.tiny-renewal'
            tiny_renewal_dump(renewal_dir)
        else:
            if args.model_dir is None or args.dump_dir is None:
                raise ValueError('--model-dir and --dump-dir are required unless --tiny is used')
            for path in (args.model_dir, args.dump_dir):
                if not Path(path).expanduser().is_dir():
                    raise ValueError('model and dump must be existing directories')
            from model_check import validate_model_headers

            validate_model_headers(args.model_dir)
            source_identity = model_identity(str(args.model_dir))
            source_index_sha256 = _sha256(args.model_dir / 'model.safetensors.index.json')
            renewal_dir = args.renewal_dump_dir or args.dump_dir
        greedy_only, policy = renewal_policy(renewal_dir, tiny=args.tiny)
        trainer_args = trainer_parser().parse_args(training_arguments(args, renewal_dir, greedy_only))
        # Leave up to five minutes for the final export evaluation.
        # The CLI watchdog enforces the full stage's hard deadline.
        reserve = 0 if args.tiny else min(300, args.max_seconds * 0.1)
        trainer_args.time_limit_seconds = max(1e-9, args.max_seconds - reserve - (clock() - started))
        stats = train(trainer_args, _clock=_clock)
        reports = []
        for name in ('original-eval.json', 'best-eval.json'):
            path = out / name
            report = json.loads(path.read_bytes())
            report['renewal'] = policy
            write_json_atomic(str(path), report)
            reports.append(report)
        checkpoint_path = out / 'best-mtp.safetensors'
        checkpoint = {'path': str(checkpoint_path), 'sha256': _sha256(checkpoint_path)}
        gate = evaluate_gate(*reports, checkpoint,
                             source=None if args.tiny else str(args.model_dir), token_maps=token_maps)
        for binding in token_maps.values():
            if binding is not None and binding['path'] is not None:
                if _sha256(binding['path']) != binding['sha256']:
                    gate.update(passed=False, reason='token_map_changed')
        gate['source_index_sha256'] = source_index_sha256
        gate['source_model_identity'] = source_identity
        if not args.tiny and model_identity(str(args.model_dir)) != source_identity:
            gate.update(passed=False, reason='source_model_changed')
        if reports[1].get('checkpoint_digest') != file_digest(str(checkpoint_path)):
            gate.update(passed=False, reason='checkpoint_digest_mismatch')
        if args.tiny:
            gate.update(passed=False, reason='synthetic_rehearsal_not_publishable')
        write_json_atomic(str(gate_path), gate)
        return {'gate': gate, 'training': stats}
    except BaseException:
        # A stale or partially completed run must not authorize write-back.
        write_json_atomic(str(gate_path), evaluate_gate(None, None))
        raise


def build_parser():
    from mtptrain.cli import StrictParser

    parser = StrictParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--dump-dir', type=Path, default=os.environ.get('MTP_DUMP_DIR'),
                        help='extracted dump (or env MTP_DUMP_DIR)')
    parser.add_argument('--model-dir', type=Path, default=os.environ.get('MTP_MODEL_DIR'),
                        help='source checkpoint (or env MTP_MODEL_DIR)')
    parser.add_argument('--output-dir', '--out', dest='output_dir', required=True, type=Path)
    parser.add_argument('--eval-set', type=Path)
    parser.add_argument('--renewal-dump-dir', type=Path,
                        help='optional separate, genuinely greedy held-out selfgen dump for the gate')
    parser.add_argument('--max-seconds', '--time-limit-seconds', dest='max_seconds', type=float, default=3600)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--token-map', type=Path,
                        help='required for real runs: primary serving hot-vocabulary map')
    parser.add_argument('--secondary-token-map', type=Path,
                        help='metadata only: hash the secondary map and explicitly record it as unevaluated')
    parser.add_argument('--tiny', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.tiny and args.device == 'cuda':
        args.device = 'cpu'
    if (not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 3600
            or args.steps is not None and not 1 <= args.steps <= 1000):
        print('training stage refused: invalid time or step limit', file=sys.stderr)
        return 2
    try:
        trainer_argv = training_arguments(args)
    except ValueError as exc:
        print(f'training stage refused: {exc}', file=sys.stderr)
        return 2
    if args.dry_run:
        print(json.dumps({'stage': 'train', 'max_seconds': args.max_seconds,
                          'argv': [sys.executable, '-m', 'mtptrain.train', *trainer_argv],
                          'gate': str(args.output_dir / 'gate.json'),
                          'greedy_renewal_required_for_gate': True}, sort_keys=True))
        return 0
    # A hard timeout cannot erase an atomic best; gate.json remains false
    # until all measurements and hashes are complete.
    watchdog = threading.Timer(args.max_seconds, lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    try:
        result = run_stage(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'training stage refused: {type(exc).__name__}', file=sys.stderr)
        return 2
    finally:
        watchdog.cancel()
    print(json.dumps({'stage': 'train', 'steps': result['training']['steps'],
                      'time_limit_reached': result['training']['time_limit_reached'],
                      'gate_passed': result['gate']['passed'], 'gate_reason': result['gate']['reason']}))
    # A denied gate is a successful measurement, not a failed training stage.
    # Stage 4 independently rechecks the reports and cleanly skips write-back.
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
