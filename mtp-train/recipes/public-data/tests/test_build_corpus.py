import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_corpus as corpus
from workload_hashes import sequence_hashes


def test_synthetic_provenance_exclusion_and_atomic_rebuild(tmp_path):
    model = tmp_path / 'model'
    model.mkdir()
    assets = {}
    for name in ('tokenizer.json', 'chat_template.jinja'):
        (model / name).write_text('{}', encoding='utf-8')
        assets[name] = corpus.sha256(model / name)
    prompt_ids = [[10, 11], [7] + list(range(103, 167)) + [8],
                  [20, 21], [30, 31], [30, 31]]
    completions = [[90], [91], [92], [93], [93]]
    prompts, outputs = [], []
    for number, ids in enumerate(prompt_ids):
        identity = {'prompt_id': f'{number:064x}', 'genre': 'en_chat',
                    'split': 'heldout' if number == 3 else 'build'}
        prompts.append({**identity, 'source': f'synthetic.jsonl#row={number}',
                        'license': 'MIT', 'input_ids': ids})
        outputs.append({**identity, 'output_ids': completions[number],
                        'prompt_tokens': len(ids), 'completion_tokens': 1, 'finish': 'stop'})
    prompts_path, outputs_path = tmp_path / 'prompts.jsonl', tmp_path / 'outputs.jsonl'
    for path, rows in ((prompts_path, prompts), (outputs_path, list(reversed(outputs)))):
        path.write_text(''.join(corpus.canonical(row) + '\n' for row in rows), encoding='utf-8')
    reference = {
        'total': 5, 'counts': {'en_chat': {'total': 5, 'build': 4, 'heldout': 1}},
        'sources': [{'local': 'synthetic.jsonl', 'license': 'MIT'}],
        'prompts': {'sha256': corpus.sha256(prompts_path)},
        'assets': [{'asset': name, 'sha256': digest} for name, digest in assets.items()],
        'selection_rule': {'render':
            'tokenizers; ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True); '
            'add_special_tokens=False; add_generation_prompt=True; other template defaults unchanged'},
    }
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(corpus.canonical(reference), encoding='utf-8')
    fixture = {'version': 1, 'encoding': 'int32-le', 'algorithm': 'sha256',
               'shingle_tokens': 64, 'shingle_stride_tokens': 1, 'assets': assets,
               'workloads': [{'name': 'code-edit-v1/synthetic',
                              'rendered': sequence_hashes([10, 11]),
                              'messages': [sequence_hashes(list(range(100, 180)))]}]}
    ab_hashes = tmp_path / 'ab-hashes.json'
    ab_hashes.write_text(corpus.canonical(fixture), encoding='utf-8')
    output_dir = tmp_path / 'corpus'
    arguments = dict(prompts=prompts_path, outputs=outputs_path, manifest=manifest,
                     model=model, ab_hashes=ab_hashes, output_dir=output_dir,
                     expected_ab_documents=1)
    report = corpus.prepare(**arguments)
    assert report['ready'] is True
    assert report['counts']['build']['documents'] == 1
    assert report['counts']['heldout']['documents'] == 1
    assert report['counts']['quarantine']['documents'] == 3
    excluded = list(corpus.read_jsonl(output_dir / 'quarantine.jsonl'))
    assert 'ab_exact_prompt' in excluded[0]['exclusion_reasons']
    assert 'ab_64_token_document' in excluded[1]['exclusion_reasons']
    assert 'heldout_document_hash' in excluded[2]['exclusion_reasons']
    retained = list(corpus.read_jsonl(output_dir / 'build.jsonl'))
    assert retained[0]['input_ids'] == prompt_ids[2] + completions[2]
    for split in ('build', 'heldout'):
        for row in corpus.read_jsonl(output_dir / (split + '.jsonl')):
            assert corpus.is_holdout_key(row['split_key']) == (split == 'heldout')
            assert row['doc_hash'] == corpus.hash_ids(row['input_ids'])
    before = {path.name: (path.stat().st_mtime_ns, path.read_bytes())
              for path in output_dir.iterdir()}
    assert corpus.prepare(**arguments) == report
    assert before == {path.name: (path.stat().st_mtime_ns, path.read_bytes())
                      for path in output_dir.iterdir()}
    reference['prompts']['sha256'] = hashlib.sha256(b'synthetic mismatch').hexdigest()
    manifest.write_text(corpus.canonical(reference), encoding='utf-8')
    with pytest.raises(ValueError, match='manifest hash mismatch'):
        corpus.prepare(**arguments)
    assert before == {path.name: (path.stat().st_mtime_ns, path.read_bytes())
                      for path in output_dir.iterdir()}
    reference['prompts']['sha256'] = corpus.sha256(prompts_path)
    manifest.write_text(corpus.canonical(reference), encoding='utf-8')
    fixture['workloads'][0]['input_ids'] = [1]
    with pytest.raises(ValueError):
        corpus.load_ab_hashes(fixture, assets=assets, expected_ab_documents=1)
    ab_hashes.unlink()
    report = corpus.prepare(**arguments)
    assert report['ready'] is False
    assert report['counts']['build']['documents'] == 0
    assert report['counts']['quarantine']['documents'] == 4
    assert (output_dir / 'heldout.jsonl').read_bytes() == before['heldout.jsonl'][1]
    assert json.loads((output_dir / 'report.json').read_text()) == report
