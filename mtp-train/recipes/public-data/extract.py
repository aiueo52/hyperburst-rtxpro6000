#!/usr/bin/env python3
"""Extract retained public IDs with an exact 200,000-token pilot from a running endpoint."""
from __future__ import annotations

import argparse
from array import array
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mtptrain.fileio import fsync_dir, write_json_atomic as atomic_json
from mtptrain.paths import input_path

PILOT_TOKENS = 200_000
MAX_LEN = 2048
SEED = 20260903
HOLDOUT = 0.1
RENEWAL_PROMPTS = 48
RENEWAL_TOKENS = 128
RENEWAL_MAX_LEN = MAX_LEN - 1
MAX_SECONDS = 5400
ASSUMED_GENERATION_TPS = 20
VERIFICATION_MARGIN_SECONDS = 120
EXTRACT_TOO_SLOW_RC = 74


class ExtractionTooSlow(RuntimeError):
    returncode = EXTRACT_TOO_SLOW_RC


def ids_bytes(ids):
    values = array('i', ids)
    if values.itemsize != 4:
        raise ValueError('32-bit int array required')
    if sys.byteorder != 'little':
        values.byteswap()
    return values.tobytes()


def hash_ids(ids):
    return hashlib.blake2b(ids_bytes(ids), digest_size=8).hexdigest()


def is_holdout(key, seed=SEED, holdout=HOLDOUT):
    digest = hashlib.blake2b(f'{seed}:{key}'.encode(), digest_size=8).hexdigest()
    return int(digest, 16) % 10000 < int(holdout * 10000)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _dump_path(root, name):
    root = input_path(root)
    if not isinstance(name, (str, Path)) or not str(name):
        raise ValueError('invalid dump input name')
    relative = Path(name)
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name in ('.', '..'):
        raise ValueError('dump inputs must be basenames within the selected root')
    path = root / relative
    resolved = input_path(path, root=root)
    if path.is_symlink() or not resolved.is_relative_to(root):
        raise ValueError('dump inputs must be materialized within the selected root')
    return resolved


def read_documents(paths, *, require_ready=False, seed=SEED, holdout=HOLDOUT):
    if not paths or not 0 < holdout < 1:
        raise ValueError('documents and a nonempty split policy are required')
    files, report = [], None
    for value in paths:
        path = input_path(value)
        if path.is_dir():
            if report is not None:
                raise ValueError('only one corpus directory is permitted')
            report = json.loads(_dump_path(path, 'report.json').read_bytes(), object_pairs_hook=_unique)
            if not isinstance(report, dict) or not isinstance(report.get('files'), dict):
                raise ValueError('invalid retained corpus report')
            policy = report.get('split_policy', {})
            if not isinstance(policy, dict):
                raise ValueError('invalid retained split policy')
            if policy.get('seed') != seed or policy.get('holdout_frac') != holdout:
                raise ValueError('corpus/extraction split policy mismatch')
            if require_ready and report.get('ready') is not True:
                raise ValueError('corpus is quarantined or incomplete; live extraction is forbidden')
            for split in ('build', 'heldout'):
                record = report.get('files', {}).get(split, {})
                if not isinstance(record, dict) or record.get('path') != split + '.jsonl':
                    raise ValueError('invalid retained corpus file name')
                source = path / record['path']
                if sha256(_dump_path(path, record['path'])) != record.get('sha256'):
                    raise ValueError('retained corpus file hash mismatch')
                files.append(source)
        else:
            files.append(path)
    if require_ready and report is None:
        # A sibling completion report binds explicit retained split files too.
        parents = {path.parent for path in files}
        if len(parents) != 1 or {path.name for path in files} != {'build.jsonl', 'heldout.jsonl'}:
            raise ValueError('live extraction requires report-verified retained build and heldout files')
        return read_documents(list(parents), require_ready=True, seed=seed, holdout=holdout)
    documents, identities = [], set()
    for source in files:
        with _dump_path(source.parent, source.name).open(encoding='utf-8') as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line, object_pairs_hook=_unique)
                except ValueError:
                    raise ValueError(f'invalid numeric corpus JSON at record {number}') from None
                if not isinstance(row, dict):
                    raise ValueError('corpus records must be objects')
                identity = row.get('doc_id', row.get('prompt_id'))
                split = row.get('split')
                ids = row.get('input_ids', row.get('token_ids'))
                if (not isinstance(identity, str) or re.fullmatch(r'[A-Za-z0-9_.:-]{1,200}', identity) is None
                        or identity in identities or split not in ('build', 'heldout')
                        or type(ids) is not list or len(ids) < 4
                        or any(type(token) is not int or not 0 <= token < 248320 for token in ids)):
                    raise ValueError(f'invalid numeric corpus schema at record {number}')
                if source.name in ('build.jsonl', 'heldout.jsonl') and source.stem != split:
                    raise ValueError('retained corpus filename/split mismatch')
                if row.get('exclusion_reasons'):
                    raise ValueError('quarantined documents cannot be extracted')
                payload = ids_bytes(ids)
                if ('sha256' in row and hashlib.sha256(payload).hexdigest() != row['sha256']
                        or 'doc_hash' in row and hash_ids(ids) != row['doc_hash']
                        or 'token_count' in row and row['token_count'] != len(ids)):
                    raise ValueError('corpus token identity/count mismatch')
                key = row.get('split_key')
                if key is None:
                    for nonce in range(100000):
                        key = hashlib.blake2b(f'public-data:{identity}:{nonce}'.encode(), digest_size=16).hexdigest()
                        if is_holdout(key, seed, holdout) == (split == 'heldout'):
                            break
                    else:
                        raise ValueError('cannot derive source-document split key')
                if (not isinstance(key, str) or re.fullmatch(r'[0-9a-f]{1,128}', key) is None
                        or is_holdout(key, seed, holdout) != (split == 'heldout')):
                    raise ValueError('source-document split key does not preserve heldout assignment')
                prompt_tokens, generated_tokens = row.get('prompt_tokens'), row.get('completion_tokens')
                if (type(prompt_tokens) is not int or type(generated_tokens) is not int
                        or not 4 <= prompt_tokens <= len(ids) or generated_tokens < 0
                        or prompt_tokens + generated_tokens != len(ids)):
                    raise ValueError('invalid corpus prompt/generation boundaries')
                metadata = {}
                for name in ('bucket', 'genre', 'source', 'license', 'workload', 'lang'):
                    if name in row:
                        if not isinstance(row[name], str) or len(row[name]) > 2048:
                            raise ValueError('invalid public corpus metadata')
                        metadata[name] = row[name]
                documents.append({**metadata, 'doc_id': identity, 'split': split, 'split_key': key,
                                  'input_ids': ids, 'document_hash': hash_ids(ids),
                                  'input_hash': hashlib.sha256(payload).hexdigest(),
                                  'prompt_tokens': prompt_tokens, 'generated_tokens': generated_tokens})
                identities.add(identity)
    if not documents or require_ready and {row['split'] for row in documents} != {'build', 'heldout'}:
        raise ValueError('retained corpus has no usable build/heldout documents')
    return documents


def _part(chunk, start, end):
    prompt_tokens = min(end - start, max(0, chunk.get('prompt_tokens', len(chunk['ids'])) - start))
    return {**chunk, 'offset': chunk['offset'] + start, 'ids': chunk['ids'][start:end],
            'prompt_tokens': prompt_tokens, 'generated_tokens': end - start - prompt_tokens}


def plan_chunks(documents, pilot_tokens=PILOT_TOKENS):
    chunks = []
    for row in documents:
        ids = row['input_ids']
        offset = 0
        while offset < len(ids):
            end = min(offset + MAX_LEN, len(ids))
            if 0 < len(ids) - end < 4:
                end = len(ids) - 4
            chunks.append(_part({**{key: value for key, value in row.items() if key != 'input_ids'},
                                 'offset': 0, 'ids': ids}, offset, end))
            offset = end
    if type(pilot_tokens) is not int or pilot_tokens < 4 or sum(len(row['ids']) for row in chunks) < pilot_tokens:
        raise ValueError('corpus is smaller than the exact extraction pilot')
    pilot, rest, remaining = [], [], pilot_tokens
    for chunk in chunks:
        size = len(chunk['ids'])
        if remaining == 0:
            rest.append(chunk)
            continue
        if size <= remaining:
            pilot.append(chunk)
            remaining -= size
            continue
        if remaining < 4:
            # The hook skips <4 rows. Borrow a four-token tail from an earlier
            # pilot chunk rather than losing IDs at the exact timing boundary.
            index = next((i for i in range(len(pilot) - 1, -1, -1) if len(pilot[i]['ids']) >= 8), None)
            if index is None:
                rest.append(chunk)
                continue
            previous = pilot[index]
            pilot[index] = _part(previous, 0, len(previous['ids']) - 4)
            rest.append(_part(previous, len(previous['ids']) - 4, len(previous['ids'])))
            remaining += 4
        if size - remaining < 4:
            # Defer a tiny boundary chunk; a subsequent full chunk can fill the
            # exact remaining budget without ever padding or dropping tokens.
            rest.append(chunk)
            continue
        pilot.append(_part(chunk, 0, remaining))
        rest.append(_part(chunk, remaining, size))
        remaining = 0
    if remaining:
        raise ValueError('cannot form an exact pilot with dumpable chunks')
    plan = [dict(row, phase=phase, doc_hash=hash_ids(row['ids']))
            for phase, group in (('pilot', pilot), ('rest', rest)) for row in group]
    splits = {}
    coverage = {}
    for chunk in plan:
        if not 4 <= len(chunk['ids']) <= MAX_LEN:
            raise ValueError('invalid extraction chunk length')
        digest, split = chunk['doc_hash'], chunk['split']
        if digest in splits and splits[digest] != split:
            raise ValueError('identical extraction chunks cross build/heldout; exclude at corpus stage')
        splits[digest] = split
        coverage.setdefault(chunk['doc_id'], []).append((chunk['offset'], len(chunk['ids'])))
    for row in documents:
        offset = 0
        for start, size in sorted(coverage[row['doc_id']]):
            if start != offset:
                raise ValueError('extraction chunk coverage has a gap or overlap')
            offset += size
        if offset != len(row['input_ids']):
            raise ValueError('extraction plan does not cover a whole document')
    return plan


def renewal_prompts(documents):
    """Use only original held-out chat-prompt IDs, never sampled continuations."""
    prompts, seen = [], set()
    heldout = sorted((row for row in documents if row['split'] == 'heldout'),
                     key=lambda row: hashlib.sha256(row['doc_id'].encode()).digest())
    for row in heldout:
        start = max(0, row['prompt_tokens'] - (RENEWAL_MAX_LEN - RENEWAL_TOKENS))
        ids = row['input_ids'][start:row['prompt_tokens']]
        digest = hash_ids(ids)
        if digest in seen:
            continue
        seen.add(digest)
        prompts.append({**{key: value for key, value in row.items() if key != 'input_ids'},
                        'ids': ids, 'offset': 0, 'source_prompt_offset': start,
                        'prompt_tokens': len(ids), 'generated_tokens': 0, 'phase': 'renewal'})
        if len(prompts) == RENEWAL_PROMPTS:
            break
    if not prompts:
        raise ValueError('genuine greedy renewal requires held-out prompts')
    return prompts


def renewal_chunk(prompt, output):
    ids = prompt['ids'] + output
    return {**prompt, 'ids': ids, 'doc_hash': hash_ids(ids), 'generated_tokens': len(output)}


def plan_summary(documents, chunks, seed=SEED, holdout=HOLDOUT):
    return {'documents': len(documents), 'chunks': len(chunks),
            'tokens': sum(len(row['ids']) for row in chunks),
            'pilot_tokens': sum(len(row['ids']) for row in chunks if row['phase'] == 'pilot'),
            'max_len': MAX_LEN, 'topk': 8, 'seed': seed, 'holdout_frac': holdout,
            'renewal': {'prompts': len(renewal_prompts(documents)), 'temperature': 0.0,
                        'generated_tokens_per_prompt': RENEWAL_TOKENS, 'max_len': RENEWAL_MAX_LEN,
                        'split': 'heldout'},
            'by_split': {split: {'documents': sum(row['split'] == split for row in documents),
                                 'tokens': sum(len(row['input_ids']) for row in documents if row['split'] == split)}
                         for split in ('build', 'heldout')}}


def _read_dump(path, expected_ids=None, *, with_arrays=False, root=None):
    """Read only numeric IDs and shape metadata, never the hidden-state payload."""
    path = Path(path)
    root = path.parent if root is None else Path(root)
    path = _dump_path(root, path.relative_to(root))
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('truncated safetensors file')
        length = struct.unpack('<Q', prefix)[0]
        if not 2 <= length <= 1024 * 1024:
            raise ValueError('invalid safetensors header length')
        header = json.loads(stream.read(length), object_pairs_hook=_unique)
        base = 8 + length
        size = path.stat().st_size - base
        def tensor(name, shape, dtype):
            entry = header.get(name, {})
            if entry.get('shape') != shape or entry.get('dtype') != dtype:
                raise ValueError('invalid dump tensor shape/dtype')
            offsets = entry.get('data_offsets')
            count = 1
            for value in shape:
                count *= value
            width = {'I32': 4, 'BF16': 2, 'F16': 2}[dtype]
            if (not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(value) is not int for value in offsets)
                    or not 0 <= offsets[0] <= offsets[1] <= size
                    or offsets[1] - offsets[0] != count * width):
                raise ValueError('invalid dump tensor offsets')
            return offsets
        shape = header.get('input_ids', {}).get('shape')
        if not isinstance(shape, list) or len(shape) != 1 or type(shape[0]) is not int or not 4 <= shape[0] <= MAX_LEN:
            return None
        count = shape[0]
        begin, end = tensor('input_ids', [count], 'I32')
        stream.seek(base + begin)
        payload = stream.read(end - begin)
        if len(payload) != end - begin:
            raise ValueError('truncated numeric dump tensor')
        arrays = {'input_ids': list(struct.unpack('<' + 'i' * count, payload))}
        if expected_ids is not None and arrays['input_ids'] != expected_ids:
            return None
        for name, dims in (('positions', [count]), ('prefix_lens', [1]),
                           ('seq_offsets', [2]), ('target_argmax', [count])):
            begin, end = tensor(name, dims, 'I32')
            stream.seek(base + begin)
            payload = stream.read(end - begin)
            if len(payload) != end - begin:
                raise ValueError('truncated numeric dump tensor')
            arrays[name] = list(struct.unpack('<' + 'i' * (len(payload) // 4), payload))
        tensor('hc_hidden', [count, 10240], 'BF16')
        tensor('topk_ids', [count, 8], 'I32')
        tensor('topk_logits', [count, 8], 'F16')
        tensor('lse', [count], 'F16')
        if (arrays['positions'] != list(range(count)) or arrays['prefix_lens'] != [0]
                or arrays['seq_offsets'] != [0, count]
                or any(not 0 <= value < 248320 for value in arrays['target_argmax'])):
            raise ValueError('dump is truncated, radix-reused or has invalid target IDs')
        return arrays if with_arrays else arrays['input_ids']



def sha256(path):
    digest = hashlib.sha256()
    with input_path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class Client:
    """ID-only requests; the caller manages the server and its dump directory."""
    def __init__(self, endpoint, dump_dir, deadline):
        self.url = urlsplit(endpoint)
        if (self.url.scheme not in ('http', 'https') or not self.url.hostname
                or self.url.username is not None or self.url.password is not None
                or self.url.query or self.url.fragment):
            raise ValueError('endpoint must be an HTTP(S) base URL without credentials or query')
        if self.url.port is not None and not 1 <= self.url.port <= 65535:
            raise ValueError('invalid endpoint port')
        self.dump = dump_dir
        self.deadline = deadline
        self.seen = set(dump_dir.glob('*.safetensors'))

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError('extraction stage deadline exceeded')

    def request(self, ids, *, generate=False):
        self.check()
        connection_type = (http.client.HTTPSConnection if self.url.scheme == 'https'
                           else http.client.HTTPConnection)
        connection = connection_type(self.url.hostname, self.url.port,
                                     timeout=min(180, max(0.1, self.deadline - time.monotonic())))
        try:
            sampling = {'temperature': 0.0, 'max_new_tokens': RENEWAL_TOKENS if generate else 1}
            if generate:
                sampling['ignore_eos'] = True
            payload = {'input_ids': ids, 'sampling_params': sampling,
                       'stream': False, 'return_logprob': False, 'return_text_in_logprobs': False}
            connection.request('POST', self.url.path.rstrip('/') + '/generate', json.dumps(payload),
                               {'Content-Type': 'application/json'})
            response = connection.getresponse()
            body = response.read(4 * 1024 * 1024 + 1)
            if response.status != 200 or len(body) > 4 * 1024 * 1024:
                raise RuntimeError('ID-only extraction request failed')
            packet = json.loads(body, object_pairs_hook=_unique)
            if not isinstance(packet, dict) or not isinstance(packet.get('meta_info'), dict):
                raise ValueError('invalid extraction response metadata')
            meta = packet['meta_info']
            if (type(meta.get('prompt_tokens')) is not int or meta['prompt_tokens'] != len(ids)
                    or type(meta.get('cached_tokens', 0)) is not int or meta.get('cached_tokens', 0) != 0):
                raise ValueError('server reused radix or received different input lengths')
            output = packet.get('output_ids')
            if (not isinstance(output, list) or len(output) != (RENEWAL_TOKENS if generate else 1)
                    or any(type(value) is not int or not 0 <= value < 248320 for value in output)
                    or type(meta.get('completion_tokens', len(output))) is not int
                    or meta.get('completion_tokens', len(output)) != len(output)):
                raise ValueError('server did not return the requested ID-only completion')
            return output
        finally:
            connection.close()

    def wait_dump(self, ids):
        # A successful HTTP reply does not imply the asynchronous writer committed.
        end = min(self.deadline, time.monotonic() + 120)
        while time.monotonic() < end:
            self.check()
            for path in sorted(set(self.dump.glob('*.safetensors')) - self.seen):
                if _read_dump(path, expected_ids=ids, root=self.dump) != ids:
                    raise ValueError('unexpected or incomplete extraction dump shard')
                self.seen.add(path)
                return path.name
            time.sleep(0.02)
        raise TimeoutError('request succeeded but its complete topk8 dump did not commit')

    def send(self, chunk):
        self.request(chunk['ids'])
        return self.wait_dump(chunk['ids'])

    def generate(self, prompt):
        output = self.request(prompt['ids'], generate=True)
        # Bind this auxiliary prompt dump, but never train/evaluate it as a document.
        return output, self.wait_dump(prompt['ids'])


def manifest_record(chunk, filename, generation_file=None):
    result = {key: value for key, value in chunk.items() if key != 'ids'}
    result.update(tokens=len(chunk['ids']), mode='selfgen',
                  temperature=0.0 if chunk['phase'] == 'renewal' else 0.6,
                  bucket=chunk.get('bucket', chunk.get('genre', 'public')),
                  name=chunk['doc_id'], dump_file=filename)
    if generation_file is not None:
        result['generation_dump_file'] = generation_file
    return result


def _dump_filename(filename):
    if not isinstance(filename, str) or re.fullmatch(r'[A-Za-z0-9_.-]+\.safetensors', filename) is None:
        raise ValueError('invalid dump filename in manifest')
    return filename


def quarantine_startup(dump):
    """Preserve only already-committed hook shards, outside the tools' scan root."""
    entries = sorted(dump.iterdir())
    for path in entries:
        if path.suffix != '.safetensors' or not path.is_file():
            raise ValueError('dump directory contains incomplete writer files or prior stage data')
        _dump_filename(path.name)
        _dump_path(dump, path.name)
    records = []
    for path in entries:
        before = path.stat()
        digest = sha256(path)
        after = path.stat()
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
            raise ValueError('startup dump changed while being bound')
        records.append({'path': 'startup/' + path.name + '.quarantined', 'sha256': digest})
    if set(dump.iterdir()) != set(entries):
        raise ValueError('startup writer is still active; wait for it to finish')
    if records:
        startup = dump / 'startup'
        startup.mkdir()
        # Final shards are atomic commits. Change the suffix too: tools scan recursively.
        for path in entries:
            path.rename(startup / (path.name + '.quarantined'))
        fsync_dir(str(startup))
        fsync_dir(str(dump))
    return records


def check_startup(dump):
    receipt = json.loads(_dump_path(dump, 'dump-result.json').read_bytes(), object_pairs_hook=_unique)
    records = receipt.get('startup_dumps', []) if isinstance(receipt, dict) else None
    if not isinstance(records, list):
        raise ValueError('invalid startup dump receipt')
    names = set()
    for record in records:
        if (not isinstance(record, dict) or set(record) != {'path', 'sha256'}
                or not isinstance(record['path'], str) or not record['path'].startswith('startup/')):
            raise ValueError('invalid startup dump receipt')
        name = record['path'][len('startup/'):]
        if not name.endswith('.quarantined'):
            raise ValueError('invalid startup dump quarantine suffix')
        _dump_filename(name[:-len('.quarantined')])
        if name in names:
            raise ValueError('duplicate startup dump receipt')
        names.add(name)
        startup = dump / 'startup'
        if startup.is_symlink() or sha256(_dump_path(startup, name)) != record['sha256']:
            raise ValueError('quarantined startup dump changed')
    startup = dump / 'startup'
    if startup.is_symlink():
        raise ValueError('startup dump directory must be materialized')
    if records:
        if {path.name for path in startup.iterdir()} != names:
            raise ValueError('unaccounted startup dump files')
    elif startup.exists():
        raise ValueError('unaccounted startup dump directory')


def postprocess(args, chunks, deadline, documents):
    check_startup(args.dump_dir)
    # verify_dump deliberately tolerates missing files. Bind every committed
    # manifest row to its expected request before invoking the public tools.
    with _dump_path(args.dump_dir, 'manifest.jsonl').open(encoding='utf-8') as stream:
        records = [json.loads(line, object_pairs_hook=_unique) for line in stream if line.strip()]
    if any(not isinstance(record, dict) for record in records):
        raise ValueError('dump manifest records must be objects')
    prompts = renewal_prompts(documents)
    if len(records) != len(chunks) + len(prompts):
        raise ValueError('dump manifest does not cover every requested chunk and renewal prompt')
    filenames, all_chunks = set(), list(chunks)
    hashes = {row['doc_hash'] for row in chunks}
    agreement = {'documents': 0, 'rows': 0, 'agree': 0}

    def bind(filename, ids):
        filename = _dump_filename(filename)
        if filename in filenames or _read_dump(args.dump_dir / filename, expected_ids=ids, root=args.dump_dir) != ids:
            raise ValueError('manifest dump is duplicated, truncated or has changed IDs')
        filenames.add(filename)

    for chunk, record in zip(chunks, records):
        filename = record.get('dump_file')
        if record != manifest_record(chunk, filename):
            raise ValueError('dump manifest identity/order mismatch')
        bind(filename, chunk['ids'])
    for prompt, record in zip(prompts, records[len(chunks):]):
        filename = _dump_filename(record.get('dump_file'))
        arrays = _read_dump(args.dump_dir / filename, with_arrays=True, root=args.dump_dir)
        if arrays is None:
            raise ValueError('invalid greedy renewal dump')
        ids, pt = arrays['input_ids'], len(prompt['ids'])
        if ids[:pt] != prompt['ids'] or len(ids) - pt != RENEWAL_TOKENS:
            raise ValueError('greedy renewal does not preserve its held-out prompt/generation bounds')
        chunk = renewal_chunk(prompt, ids[pt:])
        generation_file = _dump_filename(record.get('generation_dump_file'))
        if record != manifest_record(chunk, filename, generation_file) or chunk['doc_hash'] in hashes:
            raise ValueError('greedy renewal manifest identity/order mismatch or duplicate document')
        bind(filename, ids)
        bind(generation_file, prompt['ids'])
        hashes.add(chunk['doc_hash'])
        all_chunks.append(chunk)
        agreement['documents'] += 1
        agreement['rows'] += RENEWAL_TOKENS
        agreement['agree'] += sum(a == b for a, b in zip(arrays['target_argmax'][pt - 1:-1], ids[pt:]))
    if filenames != {path.name for path in args.dump_dir.glob('*.safetensors')} or list(args.dump_dir.glob('*.tmp')):
        raise ValueError('unaccounted or uncommitted extraction/generation dump shards')
    agreement['agreement'] = agreement['agree'] / agreement['rows']
    agreement['exact'] = agreement['agree'] == agreement['rows']
    tool_root = ROOT
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
               PYTHONNOUSERSITE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    commands = [
        ['extract/verify_dump.py', str(args.dump_dir)],
        ['extract/build_index.py', str(args.dump_dir)],
        ['extract/build_eval_set.py', str(args.dump_dir), '--rows', '200000',
         '--holdout', str(args.holdout_frac), '--seed', str(args.seed), '--max-len', str(MAX_LEN)],
    ]
    for command in commands:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('extraction postprocessing deadline exceeded')
        print(json.dumps({'event': 'postprocess', 'tool': command[0]}), flush=True)
        subprocess.run([sys.executable, '-P', '-B', str(tool_root / command[0]), *command[1:]],
                       env=env, stdin=subprocess.DEVNULL, check=True, timeout=remaining)
        if command[0].endswith('build_index.py'):
            index = json.loads(_dump_path(args.dump_dir, 'index.json').read_bytes(), object_pairs_hook=_unique)
            if any(row['doc_hash'] not in index or index[row['doc_hash']]['tokens'] != len(row['ids'])
                   for row in all_chunks):
                raise ValueError('public index does not cover every requested corpus/renewal chunk')
    evaluation = json.loads(_dump_path(args.dump_dir, 'eval_fixed.json').read_bytes(), object_pairs_hook=_unique)
    heldout = {row['doc_hash'] for row in all_chunks if row['split'] == 'heldout'}
    if not evaluation.get('doc_hashes') or not set(evaluation['doc_hashes']) <= heldout:
        raise ValueError('frozen evaluator includes non-heldout chunks')
    return agreement


def estimate_remaining(documents, total_tokens, pilot_seconds, deadline, max_seconds):
    if not math.isfinite(pilot_seconds) or pilot_seconds <= 0:
        raise ValueError('pilot clock did not advance')
    now = time.monotonic()
    rate = PILOT_TOKENS / pilot_seconds
    prompts = renewal_prompts(documents)
    prompt_tokens = sum(len(row['ids']) for row in prompts)
    generated_tokens = len(prompts) * RENEWAL_TOKENS
    combined_tokens = prompt_tokens + generated_tokens
    primary_seconds = (total_tokens - PILOT_TOKENS) / rate
    prefill_seconds = (prompt_tokens + combined_tokens) / rate
    generation_seconds = RENEWAL_PROMPTS * RENEWAL_TOKENS / ASSUMED_GENERATION_TPS
    estimated_seconds = primary_seconds + prefill_seconds + generation_seconds + VERIFICATION_MARGIN_SECONDS
    remaining_seconds = max(0, deadline - now)
    return {'tokens': PILOT_TOKENS, 'seconds': pilot_seconds, 'tokens_per_second': rate,
            'remaining_tokens': total_tokens - PILOT_TOKENS,
            'estimated_primary_seconds': primary_seconds,
            'renewal_prompts': len(prompts), 'renewal_prompt_tokens': prompt_tokens,
            'renewal_combined_tokens': combined_tokens, 'renewal_generated_tokens': generated_tokens,
            'estimated_renewal_prefill_seconds': prefill_seconds,
            'assumed_generation_prompts': RENEWAL_PROMPTS,
            'assumed_generation_tokens_per_second': ASSUMED_GENERATION_TPS,
            'estimated_renewal_generation_seconds': generation_seconds,
            'verification_margin_seconds': VERIFICATION_MARGIN_SECONDS,
            'estimated_remaining_seconds': estimated_seconds,
            'stage_max_seconds': max_seconds, 'stage_elapsed_seconds': max_seconds - (deadline - now),
            'remaining_stage_seconds': remaining_seconds,
            'fits_stage_clock': estimated_seconds <= remaining_seconds}


def run_extraction(args, documents, chunks, deadline, *, client=None):
    client = Client(args.endpoint, args.dump_dir, deadline) if client is None else client
    manifest = args.dump_dir / 'manifest.jsonl'
    summary = {**plan_summary(documents, chunks, args.seed, args.holdout_frac),
               'startup_dumps': getattr(args, 'startup_dumps', [])}
    atomic_json(args.dump_dir / 'extraction-plan.json', summary)
    pilot_start, pilot_count, completed = time.monotonic(), 0, 0
    with manifest.open('x', encoding='utf-8') as stream:
        for chunk in chunks:
            filename = client.send(chunk)
            stream.write(json.dumps(manifest_record(chunk, filename), sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            completed += len(chunk['ids'])
            print(json.dumps({'event': 'dump_committed', 'tokens': completed,
                              'chunk_tokens': len(chunk['ids'])}), flush=True)
            if chunk['phase'] == 'pilot':
                pilot_count += len(chunk['ids'])
                if pilot_count == PILOT_TOKENS:
                    elapsed = time.monotonic() - pilot_start
                    timing = estimate_remaining(documents, summary['tokens'], elapsed, deadline, args.max_seconds)
                    atomic_json(args.dump_dir / 'pilot-timing.json', timing)
                    print(json.dumps({'event': 'pilot_complete', **timing}), flush=True)
                    if not timing['fits_stage_clock']:
                        result = {**summary, 'passed': False, 'status': 'EXTRACT_TOO_SLOW',
                                  'event': 'EXTRACT_TOO_SLOW', 'rc': EXTRACT_TOO_SLOW_RC,
                                  'returncode': EXTRACT_TOO_SLOW_RC, 'pilot_timing': timing,
                                  'postprocessed': False, 'manifest': 'manifest.jsonl'}
                        atomic_json(args.dump_dir / 'extraction-result.json', result)
                        atomic_json(args.dump_dir / 'extraction-status.json', result)
                        print(json.dumps(result, sort_keys=True), flush=True)
                        raise ExtractionTooSlow('remaining extraction estimate exceeds the stage clock')
        hashes = {row['doc_hash'] for row in chunks}
        for prompt in renewal_prompts(documents):
            output, generation_file = client.generate(prompt)
            chunk = renewal_chunk(prompt, output)
            if chunk['doc_hash'] in hashes:
                raise ValueError('greedy renewal duplicates a corpus or renewal document')
            hashes.add(chunk['doc_hash'])
            filename = client.send(chunk)
            stream.write(json.dumps(manifest_record(chunk, filename, generation_file),
                                    sort_keys=True, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            print(json.dumps({'event': 'renewal_committed', 'prompt_tokens': len(prompt['ids']),
                              'generated_tokens': len(output)}), flush=True)
    if set(args.dump_dir.glob('*.safetensors')) != client.seen or list(args.dump_dir.glob('*.tmp')):
        raise ValueError('unaccounted or uncommitted extraction/generation dump shards')
    result = {**summary, 'passed': True, 'postprocessed': False, 'manifest': 'manifest.jsonl'}
    atomic_json(args.dump_dir / 'dump-result.json', result)
    if not args.extract_only:
        return finish_postprocess(args, documents, chunks, deadline)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def finish_postprocess(args, documents, chunks, deadline):
    summary = plan_summary(documents, chunks, args.seed, args.holdout_frac)
    dumped = json.loads(_dump_path(args.dump_dir, 'dump-result.json').read_bytes(), object_pairs_hook=_unique)
    if (not isinstance(dumped, dict) or dumped.get('passed') is not True
            or any(dumped.get(key) != value for key, value in summary.items())):
        raise ValueError('completed extraction does not match the retained corpus plan')
    agreement = postprocess(args, chunks, deadline, documents)
    result = {**dumped, 'postprocessed': True, 'identity_checked': True,
              'greedy_agreement_measured': True, 'greedy_agreement': agreement,
              'index': 'index.json', 'eval_set': 'eval_fixed.json'}
    atomic_json(args.dump_dir / 'extraction-result.json', result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main(argv=None):
    stage_start = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--endpoint', help='running dump-enabled HTTP(S) server base URL')
    parser.add_argument('--documents', nargs='+', type=Path, required=True,
                        help='ready corpus directory or its build.jsonl and heldout.jsonl')
    parser.add_argument('--dump-dir', type=Path, required=True,
                        help='serving SGLANG_MTP_DUMP_DIR; committed startup shards are preserved in startup/')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--holdout-frac', type=float, default=HOLDOUT)
    parser.add_argument('--max-seconds', type=float, default=MAX_SECONDS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--plan-only', action='store_true', help='validate the corpus and print the extraction plan')
    mode.add_argument('--extract-only', '--no-postprocess', dest='extract_only', action='store_true',
                      help='dump and time only; defer verification/index/eval')
    mode.add_argument('--postprocess-only', action='store_true', help='verify/index/eval a completed dump')
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0 or not 0 < args.holdout_frac < 1:
        parser.error('max-seconds must be finite and positive, holdout-frac in (0,1)')
    if not args.plan_only and not args.postprocess_only and not args.endpoint:
        parser.error('--endpoint is required for extraction')
    deadline = stage_start + args.max_seconds
    args.dump_dir = input_path(args.dump_dir)
    documents = read_documents(args.documents, require_ready=True, seed=args.seed, holdout=args.holdout_frac)
    chunks = plan_chunks(documents)
    if args.plan_only:
        result = plan_summary(documents, chunks, args.seed, args.holdout_frac)
        print(json.dumps(result, sort_keys=True))
        return result
    if args.postprocess_only:
        return finish_postprocess(args, documents, chunks, deadline)
    args.dump_dir.mkdir(parents=True, exist_ok=True)
    args.startup_dumps = quarantine_startup(args.dump_dir)
    return run_extraction(args, documents, chunks, deadline)


if __name__ == '__main__':
    try:
        main()
    except ExtractionTooSlow as error:
        print('EXTRACT_TOO_SLOW: ' + str(error), file=sys.stderr)
        raise SystemExit(EXTRACT_TOO_SLOW_RC)
    except (ValueError, OSError, RuntimeError, TimeoutError, http.client.HTTPException,
            subprocess.SubprocessError) as error:
        print('EXTRACTION FAILED: ' + str(error), file=sys.stderr)
        raise SystemExit(1)
