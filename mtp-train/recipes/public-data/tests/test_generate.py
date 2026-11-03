"""The sampling transport and resume path need no listening socket or model."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def test_sampling_and_resume(tmp_path):
    spec = importlib.util.spec_from_file_location(
        'public_generate_test', Path(__file__).resolve().parents[1] / 'generate.py')
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    requests = []

    class FakeResponse:
        status = 200

        def getheader(self, key, default=''):
            return 'application/json; charset=utf-8'

        def read(self, limit):
            return self.body[:limit]

        def close(self):
            pass

    class FakeConnection:
        def __init__(self, host, port, timeout):
            assert (host, port, timeout) == ('example.test', 8000, 900)

        def request(self, method, path, body, headers):
            packet = json.loads(body)
            assert (method, path) == ('POST', '/generate')
            assert packet['sampling_params'] == recipe.SAMPLING_PARAMS
            assert packet['stream'] is False
            requests.append(packet)
            self.length = len(packet['input_ids'])

        def getresponse(self):
            response = FakeResponse()
            response.body = json.dumps({'output_ids': [31, 32], 'meta_info': {
                'prompt_tokens': self.length, 'completion_tokens': 2,
                'finish_reason': {'type': 'stop'}}}).encode()
            return response

        def close(self):
            pass

    prompts = tmp_path / 'prompts.jsonl'
    rows = [{'genre': 'code', 'prompt_id': hashlib.sha256(str(i).encode()).hexdigest(),
             'split': 'build', 'input_ids': [i, i + 1]} for i in range(5)]
    prompts.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    client = recipe.SamplingClient('http://example.test:8000')
    client.connection_type = FakeConnection
    output = tmp_path / 'generated'
    result = recipe.run(prompts, output, client)
    assert result['complete'] is True and result['outputs'] == 5
    assert result['completion_tokens'] == 10 and len(requests) == 5
    saved = (output / 'outputs.jsonl').read_bytes()
    with (output / 'outputs.jsonl').open('ab') as stream:
        stream.write(b'{"interrupted":')
    assert recipe.run(prompts, output, client) == result
    assert len(requests) == 5 and (output / 'outputs.jsonl').read_bytes() == saved
    rows[0]['input_ids'] = [99]
    prompts.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match='changed'):
        recipe.run(prompts, output, client)
    with pytest.raises(ValueError, match='invalid'):
        recipe.parse_response({'output_ids': [True], 'meta_info': {
            'prompt_tokens': 1, 'completion_tokens': 1, 'finish_reason': {'type': 'stop'}}}, 1)
