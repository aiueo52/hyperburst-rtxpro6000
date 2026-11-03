import importlib.util
import json
from pathlib import Path
import struct
import time
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location('public_extract', Path(__file__).parents[1] / 'extract.py')
extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract)


def test_synthetic_extraction_with_fake_client(tmp_path, monkeypatch):
    corpus, dump = tmp_path / 'corpus', tmp_path / 'dump'
    corpus.mkdir()
    dump.mkdir()
    files = {}
    for split, start in [('build', 1000), ('heldout', 2000)]:
        ids = list(range(start, start + 192))
        key = next(f'{i:x}' for i in range(10000)
                   if extract.is_holdout(f'{i:x}') == (split == 'heldout'))
        row = {'doc_id': split, 'split': split, 'split_key': key, 'input_ids': ids,
               'prompt_tokens': 160, 'completion_tokens': 32, 'token_count': len(ids),
               'doc_hash': extract.hash_ids(ids),
               'sha256': extract.hashlib.sha256(extract.ids_bytes(ids)).hexdigest()}
        path = corpus / f'{split}.jsonl'
        path.write_text(json.dumps(row) + '\n')
        files[split] = {'path': path.name, 'sha256': extract.sha256(path)}
    (corpus / 'report.json').write_text(json.dumps({
        'ready': True, 'split_policy': {'seed': extract.SEED, 'holdout_frac': extract.HOLDOUT},
        'files': files}))
    documents = extract.read_documents([corpus], require_ready=True)
    original = (corpus / 'build.jsonl').read_bytes()
    (corpus / 'build.jsonl').write_bytes(original + b'\n')
    with pytest.raises(ValueError, match='file hash mismatch'):
        extract.read_documents([corpus], require_ready=True)
    (corpus / 'build.jsonl').write_bytes(original)
    with pytest.raises(ValueError, match='split policy mismatch'):
        extract.read_documents([corpus], require_ready=True, seed=0)
    assert extract.PILOT_TOKENS == 200_000
    with pytest.raises(ValueError, match='smaller than the exact extraction pilot'):
        extract.plan_chunks(documents)
    monkeypatch.setattr(extract, 'PILOT_TOKENS', 32)
    chunks = extract.plan_chunks(documents, pilot_tokens=32)
    assert sum(len(row['ids']) for row in chunks if row['phase'] == 'pilot') == 32

    class FakeClient(extract.Client):
        def request(self, ids, *, generate=False):
            count = len(ids)
            arrays = {'input_ids': ids, 'positions': list(range(count)),
                      'prefix_lens': [0], 'seq_offsets': [0, count],
                      'target_argmax': ids[1:] + [0]}
            tensors = [(name, [len(values)], 'I32', extract.ids_bytes(values))
                       for name, values in arrays.items()]
            tensors += [('hc_hidden', [count, 10240], 'BF16', None),
                        ('topk_ids', [count, 8], 'I32', None),
                        ('topk_logits', [count, 8], 'F16', None),
                        ('lse', [count], 'F16', None)]
            header, offset = {}, 0
            for name, shape, dtype, payload in tensors:
                size = (4 if dtype == 'I32' else 2)
                for dimension in shape:
                    size *= dimension
                header[name] = {'shape': shape, 'dtype': dtype,
                                'data_offsets': [offset, offset + size]}
                offset += size
            encoded = json.dumps(header).encode()
            encoded += b' ' * (-len(encoded) % 8)
            path = dump / f'synthetic-{len(self.seen):04d}.safetensors'
            with path.open('xb') as stream:
                stream.write(struct.pack('<Q', len(encoded)) + encoded)
                for name, shape, dtype, payload in tensors:
                    if payload is not None:
                        stream.seek(8 + len(encoded) + header[name]['data_offsets'][0])
                        stream.write(payload)
                stream.truncate(8 + len(encoded) + offset)
            if generate:
                assert ids == documents[1]['input_ids'][:160]
            return list(range(3000, 3000 + extract.RENEWAL_TOKENS)) if generate else [0]

    deadline = time.monotonic() + 5400
    args = SimpleNamespace(endpoint='http://synthetic.invalid', dump_dir=dump,
                           seed=extract.SEED, holdout_frac=extract.HOLDOUT,
                           extract_only=False, max_seconds=5400)
    warmup = list(range(8))
    FakeClient(args.endpoint, dump, deadline).request(warmup)
    args.startup_dumps = extract.quarantine_startup(dump)
    assert len(args.startup_dumps) == 1
    baseline = dump / args.startup_dumps[0]['path']
    assert extract.sha256(baseline) == args.startup_dumps[0]['sha256']
    result = extract.run_extraction(args, documents, chunks, deadline,
                                    client=FakeClient(args.endpoint, dump, deadline))
    assert result['passed'] and result['postprocessed'] and result['identity_checked']
    assert result['greedy_agreement']['documents'] == 1
    assert result['greedy_agreement']['rows'] == 128
    assert result['greedy_agreement']['exact']
    assert json.loads((dump / 'extraction-result.json').read_bytes()) == result
    assert json.loads((dump / 'pilot-timing.json').read_bytes())['tokens'] == 32
    assert (dump / 'eval_fixed.json').is_file()
    assert extract.hash_ids(warmup) not in json.loads((dump / 'index.json').read_bytes())
    assert result['startup_dumps'] == args.startup_dumps and baseline.is_file()
    with pytest.raises(ValueError, match='prior stage data'):
        extract.quarantine_startup(dump)
    manifest = dump / 'manifest.jsonl'
    records = [json.loads(line) for line in manifest.read_text().split('\n') if line]
    assert records[-1]['temperature'] == 0 and records[-1]['split'] == 'heldout'
    assert records[-1]['generated_tokens'] == 128
    records[-1]['split'] = 'build'
    manifest.write_text(''.join(json.dumps(row) + '\n' for row in records))
    with pytest.raises(ValueError, match='renewal manifest identity/order mismatch'):
        extract.postprocess(args, chunks, deadline, documents)
