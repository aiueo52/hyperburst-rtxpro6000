#!/usr/bin/env python3
"""Generate ID-only continuations against a user-managed sampling server."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mtptrain.fileio import open_jsonl_append, read_jsonl, write_json_atomic
from mtptrain.paths import input_path

CONCURRENCY = 4
MAX_NEW_TOKENS = 3072
MAX_BODY = 4 * 1024 * 1024
SAMPLING_PARAMS = {'temperature': 0.6, 'top_p': 0.95, 'top_k': 20,
                   'max_new_tokens': MAX_NEW_TOKENS}
GENRES = {'en_chat', 'ja_chat', 'reasoning', 'tools', 'swe', 'code'}


def validate_ids(ids, allow_empty=False):
    if (type(ids) is not list or (not ids and not allow_empty)
            or any(type(token) is not int or not 0 <= token < 2**31 for token in ids)):
        raise ValueError('invalid numeric token IDs')
    return ids


def load_prompts(path):
    rows, seen = [], set()
    with input_path(path).open(encoding='utf-8') as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    for row in records:
        prompt_id = row.get('prompt_id') if isinstance(row, dict) else None
        if (not isinstance(prompt_id, str) or not re.fullmatch('[0-9a-f]{64}', prompt_id)
                or prompt_id in seen or row.get('genre') not in GENRES
                or row.get('split') not in ('build', 'heldout')):
            raise ValueError('invalid or duplicate prompt metadata')
        if len(validate_ids(row.get('input_ids'))) > 6000:
            raise ValueError('prompt exceeds 6000 tokens')
        rows.append({key: row[key] for key in ('genre', 'prompt_id', 'split', 'input_ids')})
        seen.add(prompt_id)
    if not rows:
        raise ValueError('no prompts')
    return rows


def parse_response(packet, prompt_length):
    if not isinstance(packet, dict) or 'error' in packet:
        raise ValueError('invalid generate response')
    meta = packet.get('meta_info')
    if (not isinstance(meta, dict) or type(meta.get('prompt_tokens')) is not int
            or meta['prompt_tokens'] != prompt_length):
        raise ValueError('prompt length mismatch')
    count, finish = meta.get('completion_tokens'), meta.get('finish_reason')
    if (type(count) is not int or not 0 <= count <= MAX_NEW_TOKENS
            or not isinstance(finish, dict) or finish.get('type') not in ('stop', 'length')
            or finish['type'] == 'length' and count != MAX_NEW_TOKENS):
        raise ValueError('invalid completion metadata')
    ids = validate_ids(packet.get('output_ids'), allow_empty=True)
    if len(ids) != count:
        raise ValueError('output length mismatch')
    return {'output_ids': ids, 'finish': finish['type'],
            'prompt_tokens': prompt_length, 'completion_tokens': count}


class SamplingClient:
    def __init__(self, endpoint, timeout=900):
        url = urlsplit(endpoint)
        if (url.scheme not in ('http', 'https') or not url.hostname or url.username is not None
                or url.password is not None or url.query or url.fragment
                or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError('invalid endpoint or timeout')
        self.host, self.port = url.hostname, url.port
        self.path = url.path.rstrip('/') + '/generate'
        self.timeout = timeout
        self.connection_type = (http.client.HTTPSConnection if url.scheme == 'https'
                                else http.client.HTTPConnection)

    def generate(self, ids):
        validate_ids(ids)
        payload = {'input_ids': ids, 'sampling_params': dict(SAMPLING_PARAMS), 'stream': False}
        connection = self.connection_type(self.host, self.port, timeout=self.timeout)
        try:
            connection.request('POST', self.path, json.dumps(payload),
                               {'Content-Type': 'application/json'})
            response = connection.getresponse()
            try:
                if (response.status != 200 or response.getheader('Content-Type', '').split(';', 1)[0]
                        .strip().lower() != 'application/json'):
                    raise ValueError('generate HTTP failure')
                raw = response.read(MAX_BODY + 1)
                if len(raw) > MAX_BODY:
                    raise ValueError('generate response too large')
                return parse_response(json.loads(raw), len(ids))
            finally:
                response.close()
        finally:
            connection.close()


def run(prompts, output_dir, client):
    prompts = input_path(prompts)
    rows = load_prompts(prompts)
    output_dir = Path(os.path.abspath(Path(output_dir).expanduser()))
    if any(path.is_symlink() for path in (output_dir, *output_dir.parents)):
        raise ValueError('output directory must not contain symlink ancestors')
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / 'outputs.jsonl'
    receipt = output_dir / 'generation.json'
    if output.is_symlink() or receipt.is_symlink():
        raise ValueError('output files must not be symlinks')
    digest = hashlib.sha256()
    with prompts.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    binding = {'schema': 1, 'prompts_sha256': digest.hexdigest(),
               'sampling_params': SAMPLING_PARAMS, 'concurrency': CONCURRENCY}
    if receipt.exists():
        previous = json.loads(receipt.read_bytes())
        if not isinstance(previous, dict) or any(previous.get(key) != value for key, value in binding.items()):
            raise ValueError('generation inputs or sampling policy changed')
    elif output.exists():
        raise ValueError('existing outputs require their generation receipt')
    by_id = {row['prompt_id']: row for row in rows}
    done, tokens = set(), 0
    if output.exists():
        for row in read_jsonl(str(output)):
            if not isinstance(row, dict):
                raise ValueError('invalid saved output')
            prompt_id = row.get('prompt_id')
            if not isinstance(prompt_id, str) or prompt_id not in by_id or prompt_id in done:
                raise ValueError('duplicate or unmatched saved output')
            prompt = by_id[prompt_id]
            if (row.get('error') or any(row.get(key) != prompt[key] for key in ('genre', 'split'))
                    or type(row.get('prompt_tokens')) is not int
                    or row['prompt_tokens'] != len(prompt['input_ids'])
                    or type(row.get('completion_tokens')) is not int):
                raise ValueError('saved output metadata mismatch')
            parse_response({'output_ids': row.get('output_ids'), 'meta_info': {
                'prompt_tokens': row['prompt_tokens'], 'completion_tokens': row['completion_tokens'],
                'finish_reason': {'type': row.get('finish')}}}, len(prompt['input_ids']))
            done.add(prompt_id)
            tokens += row['completion_tokens']
    write_json_atomic(str(receipt), {**binding, 'outputs': len(done), 'complete': False})
    remaining = iter(row for row in rows if row['prompt_id'] not in done)
    with open_jsonl_append(str(output)) as stream:
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            pending = {}

            def submit():
                item = next(remaining, None)
                if item is not None:
                    pending[executor.submit(client.generate, item['input_ids'])] = item

            for _ in range(CONCURRENCY):
                submit()
            try:
                while pending:
                    completed, _ = concurrent.futures.wait(
                        pending, return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in completed:
                        item = pending.pop(future)
                        generated = future.result()
                        # Validate injected clients exactly like HTTP responses.
                        parsed = parse_response({'output_ids': generated.get('output_ids'), 'meta_info': {
                            'prompt_tokens': generated.get('prompt_tokens'),
                            'completion_tokens': generated.get('completion_tokens'),
                            'finish_reason': {'type': generated.get('finish')}}}, len(item['input_ids']))
                        stream.write({**{key: item[key] for key in ('genre', 'prompt_id', 'split')}, **parsed})
                        os.fsync(stream._fh.fileno())
                        done.add(item['prompt_id'])
                        tokens += parsed['completion_tokens']
                    for _ in completed:
                        submit()
            finally:
                for future in pending:
                    future.cancel()
    report = {**binding, 'outputs': len(done), 'completion_tokens': tokens, 'complete': len(done) == len(rows)}
    write_json_atomic(str(receipt), report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--prompts', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--timeout', type=float, default=900)
    args = parser.parse_args(argv)
    try:
        report = run(args.prompts, args.output_dir, SamplingClient(args.endpoint, args.timeout))
    except (OSError, ValueError, TypeError, KeyError, http.client.HTTPException):
        parser.exit(2, 'generation failed; successful rows are retained for resume\n')
    print(json.dumps({key: report[key] for key in ('outputs', 'completion_tokens', 'complete')}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
