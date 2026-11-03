#!/usr/bin/env python3
"""Numbers-only A/B analysis using four server-start means per arm.

Pass eight fnbench genre JSONL files in start order. Acceptance length is
primary; decode t/s is secondary. Exact one-sided tests use all 70 labelings
of the eight starts, not the individual prompt rows.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

ORDER = ('A', 'H', 'H', 'A', 'A', 'H', 'H', 'A')
GENRES = ('code-edit', 'prose-en', 'prose-ja', 'agent-loop')


def _invalid(*args):
    raise ValueError('invalid analysis input')


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _invalid()
        result[key] = value
    return result


def _json(text):
    return json.loads(text, object_pairs_hook=_object, parse_constant=_invalid)


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        _invalid()
    return value


def _order(value):
    order = tuple(value.split(',')) if isinstance(value, str) else tuple(value)
    if len(order) != 8 or order.count('A') != 4 or order.count('H') != 4:
        _invalid()
    return order


def load(path):
    """Validate 32 native fnbench rows or numeric indexed genre rows."""
    rows = {}
    counts = dict.fromkeys(GENRES, 0)
    schema = None
    for line in Path(path).read_text(encoding='utf-8').split('\n'):
        if not line.strip():
            continue
        row = _json(line)
        if (not isinstance(row, dict) or not isinstance(row.get('server'), dict)
                or not isinstance(row.get('client'), dict)):
            _invalid()
        numeric = 'prompt_index' in row or 'workload_index' in row
        if schema is not None and schema != numeric:
            _invalid()
        schema = numeric
        if numeric:
            pid, workload = row.get('prompt_index'), row.get('workload_index')
            if (type(pid) is not int or not 0 <= pid < 32
                    or type(workload) is not int or not 0 <= workload < len(GENRES)
                    or workload != pid % len(GENRES)):
                _invalid()
            genre = GENRES[workload]
        else:
            genre, prompt_id = row.get('workload'), row.get('prompt_id')
            if (genre not in GENRES or not isinstance(prompt_id, str) or not prompt_id.strip()
                    or row.get('sampling_mode') != 'greedy'
                    or type(row.get('repeat')) is not int or row['repeat'] != 1):
                _invalid()
            # Keep identities for matching starts, never expose their text.
            pid = hashlib.sha256(prompt_id.encode('utf-8')).digest()
        if pid in rows:
            _invalid()
        acc = _number(row['server'].get('accept_length'))
        tps = _number(row['client'].get('decode_tps'))
        rows[pid] = (genre, acc, tps)
        counts[genre] += 1
    if len(rows) != 32 or any(count != 8 for count in counts.values()):
        _invalid()
    return rows


def mean(values):
    return math.fsum(value / len(values) for value in values)


def permutation(values_a, values_h):
    """Inclusive one-sided p(H>A), p(H<A), and the exact labeling count."""
    a, h = list(values_a), list(values_h)
    if len(a) != 4 or len(h) != 4:
        _invalid()
    pooled = [_number(value) for value in a + h]
    observed = mean(h) - mean(a)
    diffs = []
    for chosen in itertools.combinations(range(8), 4):
        diffs.append(mean([pooled[i] for i in chosen])
                     - mean([pooled[i] for i in range(8) if i not in chosen]))
    better = sum(diff >= observed - 1e-12 for diff in diffs) / 70
    worse = sum(diff <= observed + 1e-12 for diff in diffs) / 70
    return better, worse, len(diffs)


def analyze(paths, order=ORDER):
    order = _order(order)
    paths = [Path(path) for path in paths]
    if len(paths) != 8 or len({path.resolve() for path in paths}) != 8:
        _invalid()
    data = [load(path) for path in paths]
    identities = {pid: row[0] for pid, row in data[0].items()}
    if any({pid: row[0] for pid, row in rows.items()} != identities for rows in data):
        _invalid()
    per = {}
    for start, rows in enumerate(data):
        for genre in GENRES:
            selected = [row for row in rows.values() if row[0] == genre]
            per[start, genre] = (mean([row[1] for row in selected]),
                                 mean([row[2] for row in selected]))
    summary = {}
    for genre in GENRES:
        a, h = ([per[start, genre][0] for start, label in enumerate(order) if label == arm]
                for arm in ('A', 'H'))
        ta, th = ([per[start, genre][1] for start, label in enumerate(order) if label == arm]
                  for arm in ('A', 'H'))
        better, worse, total = permutation(a, h)
        summary[genre] = dict(a=a, h=h, ta=mean(ta), th=mean(th),
                              ratio=_number(mean(h) / mean(a)),
                              tps_ratio=_number(mean(th) / mean(ta)),
                              p_better=better, p_worse=worse, permutations=total)
    return per, summary


def conditions(summary):
    """Evaluate the preregistered good and bad criteria independently."""
    gain = any(summary[g]['ratio'] >= 1.05 and summary[g]['p_better'] <= .05
               for g in ('code-edit', 'agent-loop'))
    prose = all(summary[g]['ratio'] >= .98 for g in ('prose-en', 'prose-ja'))
    bad = any(row['ratio'] <= .97 and row['p_worse'] <= .05 for row in summary.values())
    return gain and prose, bad


def verdict(summary):
    """Use harm-priority labeling when both independent criteria are true."""
    good, bad = conditions(summary)
    return 'bad' if bad else 'good' if good else 'report numbers'


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        _invalid()


def main(argv=None):
    parser = _Parser(description=__doc__)
    parser.add_argument('--order', default=','.join(ORDER),
                        help='start labels: eight comma-separated A/H labels, four each')
    parser.add_argument('files', nargs=8, metavar='JSONL', help='one genre JSONL per server start')
    try:
        args = parser.parse_args(argv)
        order = _order(args.order)
        per, summary = analyze(args.files, order)
        good_condition, bad_condition = conditions(summary)
        decision = verdict(summary)
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError):
        # Never echo untrusted JSON, paths, exception text, prompts or outputs.
        print('cannot evaluate')
        return 1
    print('start arm genre        acc    t/s   n')
    for start, arm in enumerate(order):
        for genre in GENRES:
            acc, tps = per[start, genre]
            print(f'{start + 1:5d} {arm:3s} {genre:11s} {acc:6.2f} {tps:6.1f}   8')
    print('\ngenre        A acc  H acc    H/A p(H>A) p(H<A) | A t/s  H t/s H/A t/s (secondary) | starts A / H')
    for genre, row in summary.items():
        print(f"{genre:11s} {mean(row['a']):6.2f} {mean(row['h']):6.2f} {row['ratio']:6.3f}"
              f" {row['p_better']:6.3f} {row['p_worse']:6.3f} | {row['ta']:6.1f} {row['th']:6.1f}"
              f" {row['tps_ratio']:7.3f} | {[round(x, 2) for x in row['a']]} /"
              f" {[round(x, 2) for x in row['h']]} (perm n={row['permutations']})")
    print(f'\nPREREG conditions: good_condition={good_condition} bad_condition={bad_condition}'
          ' (label precedence: bad > good)')
    print('PREREG rule: ' + decision)
    return 0


if __name__ == '__main__':
    sys.exit(main())
