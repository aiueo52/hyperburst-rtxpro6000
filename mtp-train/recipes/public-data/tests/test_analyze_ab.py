import copy
import importlib.util
import json
from pathlib import Path


_spec = importlib.util.spec_from_file_location(
    'recipe_analyze_ab', Path(__file__).resolve().parents[1] / 'analyze_ab.py')
analysis = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analysis)


def test_synthetic_stats_thresholds_and_malformed_input(tmp_path, capsys):
    paths = [tmp_path / f'start-{i}.jsonl' for i in range(8)]

    def write(path, rows):
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows),
                        encoding='utf-8')

    def populate(native=False):
        for arm, path in zip(analysis.ORDER, paths):
            rows = []
            for i in range(32):
                genre = analysis.GENRES[i % 4]
                ratio = {'code-edit': 1.05, 'prose-en': .98, 'prose-ja': .98}.get(genre, 1)
                row = {'server': {'accept_length': 100 * (ratio if arm == 'H' else 1)},
                       'client': {'decode_tps': 1000 if arm == 'H' else 100},
                       'unused': 'synthetic-secret\u2028suffix'}
                row.update({'workload': genre, 'prompt_id': f'synthetic-{i}',
                            'sampling_mode': 'greedy', 'repeat': 1} if native else
                           {'prompt_index': i, 'workload_index': i % 4})
                rows.append(row)
            write(path, rows)
        return rows

    assert analysis.permutation([1] * 4, [2] * 4) == (1 / 70, 1, 70)
    assert analysis.permutation([2] * 4, [1] * 4) == (1, 1 / 70, 70)
    assert analysis.permutation([1] * 4, [1] * 4) == (1, 1, 70)
    for native in (False, True):
        populate(native)
        _, summary = analysis.analyze(paths)
        assert analysis.conditions(summary) == (True, False)
        assert summary['code-edit']['h'] == [105] * 4
        assert summary['code-edit']['tps_ratio'] == 10
        assert analysis.main(['--order', 'A,H,H,A,A,H,H,A', *map(str, paths)]) == 0
        output = capsys.readouterr().out
        assert 'PREREG rule: good\n' in output
        assert output.count('perm n=70') == 4
        assert 'synthetic-' not in output and str(tmp_path) not in output

    for genre, field, value, expected in (
            ('code-edit', 'ratio', 1.049, (False, False)),
            ('code-edit', 'p_better', .051, (False, False)),
            ('prose-ja', 'ratio', .979, (False, False)),
            ('agent-loop', 'ratio', .97, (True, False))):
        changed = copy.deepcopy(summary)
        changed[genre][field] = value
        assert analysis.conditions(changed) == expected
    summary['agent-loop'].update(ratio=.97, p_worse=.05)
    assert analysis.conditions(summary) == (True, True)
    assert analysis.verdict(summary) == 'bad'
    summary['agent-loop']['ratio'] = .971
    assert analysis.verdict(summary) == 'good'

    # Prompt rows are not replicates, and throughput cannot rescue acceptance.
    populate()
    for path, value in zip([paths[i] for i, arm in enumerate(analysis.ORDER) if arm == 'H'],
                           (110, 110, 110, 10)):
        rows = [json.loads(line) for line in path.read_text().strip().split('\n')]
        for row in rows:
            row['server']['accept_length'] = value
        write(path, rows)
    _, summary = analysis.analyze(paths)
    assert summary['code-edit']['ratio'] == .85
    assert summary['code-edit']['p_worse'] > .05
    assert analysis.verdict(summary) == 'report numbers'

    def invalid():
        assert analysis.main(list(map(str, paths))) == 1
        assert capsys.readouterr().out == 'cannot evaluate\n'

    base = populate()
    for rows in (base[:-1], base + [base[0]], [None] * 32):
        write(paths[0], rows)
        invalid()
    for group, field, value in ((None, 'prompt_index', True),
                                (None, 'workload_index', 1),
                                ('server', 'accept_length', float('nan')),
                                ('client', 'decode_tps', 0)):
        rows = copy.deepcopy(base)
        (rows[0] if group is None else rows[0][group])[field] = value
        write(paths[0], rows)
        invalid()
    for raw in ('synthetic-invalid\n', '{"prompt_index":0,"prompt_index":1}\n'):
        paths[0].write_text(raw)
        invalid()
    for field, value in (('prompt_id', 'synthetic-mismatch'),
                         ('sampling_mode', 'recommended'), ('repeat', True)):
        rows = populate(native=True)
        rows[0][field] = value
        write(paths[0], rows)
        invalid()
    populate()
    paths[-1].unlink()
    invalid()
