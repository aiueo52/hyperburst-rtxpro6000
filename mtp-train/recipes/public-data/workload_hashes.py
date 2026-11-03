#!/usr/bin/env python3
"""Build a hash-only A/B fixture without displaying or persisting workload contents."""
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_corpus import (AB_DOMAINS, SHINGLE, argument_path, atomic_write, canonical,
                          ids_bytes, input_path, load_ab_hashes, output_path, sha256, validate_ids)

NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z')


class Renderer:
    """The public sandbox/tokenizers policy, with thinking explicitly enabled."""
    def __init__(self, tokenizer, template):
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

        def raise_exception(message):
            raise ValueError('invalid chat template invocation')

        environment.globals['raise_exception'] = raise_exception
        self.render = environment.from_string(template).render
        self.tokenizer = tokenizer

    @classmethod
    def from_model(cls, model):
        from tokenizers import Tokenizer

        model = input_path(model)
        template = input_path(model / 'chat_template.jinja', root=model)
        if not 0 < template.stat().st_size <= 1024 * 1024:
            raise ValueError('invalid chat template size')
        return cls(Tokenizer.from_file(str(input_path(model / 'tokenizer.json', root=model))),
                   template.read_text(encoding='utf-8'))

    def sequences(self, request):
        messages = request.get('messages')
        if (not isinstance(messages, list) or not messages
                or any(not isinstance(message, dict)
                       or message.get('role') not in ('system', 'user', 'assistant', 'tool')
                       or not isinstance(message.get('content'), str) for message in messages)):
            raise ValueError('unsupported server message shape')
        options = {'messages': messages, 'add_generation_prompt': True, 'enable_thinking': True}
        if 'tools' in request:
            if (not isinstance(request['tools'], list)
                    or any(not isinstance(tool, dict) for tool in request['tools'])):
                raise ValueError('invalid server tools')
            options['tools'] = request['tools']
        rendered = self.tokenizer.encode(self.render(**options), add_special_tokens=False).ids
        bodies = [self.tokenizer.encode(message['content'], add_special_tokens=False).ids
                  for message in messages]
        validate_ids(rendered)
        for ids in bodies:
            validate_ids(ids, allow_empty=True)
        return rendered, bodies


def fnbench_modules(bench):
    """Import only the explicitly selected public benchmark implementation."""
    bench = input_path(bench).resolve()
    for filename in ('__init__.py', 'models.py', 'workloads.py', 'http_client.py'):
        path = input_path(bench / 'fnbench' / filename, root=bench)
        if not path.resolve().is_relative_to(bench):
            raise ValueError('benchmark module escapes selected directory')
    name = '_public_fnbench'
    spec = importlib.util.spec_from_file_location(name, bench / 'fnbench/__init__.py',
                                                 submodule_search_locations=[str(bench / 'fnbench')])
    for loaded in list(sys.modules):
        if loaded == name or loaded.startswith(name + '.'):
            del sys.modules[loaded]
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return (importlib.import_module(name + '.workloads'),
            importlib.import_module(name + '.http_client'))


def validate_set_paths(root, bench):
    """Validate every loader input before fnbench opens any prompt file."""
    root, bench = input_path(root), input_path(bench)
    manifest = input_path(bench / 'workloads/manifest.json', root=bench)
    raw = json.loads(manifest.read_text(encoding='utf-8'))
    if not isinstance(raw, dict) or not isinstance(raw.get('workloads'), list):
        raise ValueError('invalid benchmark workload manifest')
    for domain in raw['workloads']:
        if not isinstance(domain, dict) or not isinstance(domain.get('prompt_file'), str):
            raise ValueError('invalid benchmark prompt path')
        input_path(manifest.parent / domain['prompt_file'], root=manifest.parent)
    for domain in AB_DOMAINS:
        manifest = input_path(root / (domain + '-v1') / 'manifest.json', root=root)
        raw = json.loads(manifest.read_text(encoding='utf-8'))
        if (not isinstance(raw, dict) or not isinstance(raw.get('prompts'), list)
                or len(raw['prompts']) != 8):
            raise ValueError('each A/B domain must contain eight prompts')
        for item in raw['prompts']:
            if not isinstance(item, dict) or not isinstance(item.get('file'), str):
                raise ValueError('invalid set prompt path')
            input_path(manifest.parent / item['file'], root=manifest.parent)
    return root


def server_request(client_module, workload):
    """Capture the real fnbench client payload in memory; never open a socket."""
    class Captured(Exception):
        pass

    class Session:
        def post(self, url, *, json, **kwargs):
            self.request = json
            raise Captured

    session = Session()
    client = client_module.OpenAIStreamClient('http://unused.invalid/v1', session=session)
    try:
        client.complete(prompt=workload.read_prompt(), max_tokens=workload.max_tokens,
                        sampling=workload.sampling, model='hash-only')
    except Captured:
        return session.request
    raise ValueError('benchmark client did not provide a request')


def sequence_hashes(ids):
    payload = ids_bytes(validate_ids(ids, allow_empty=True))
    return {'exact': hashlib.sha256(payload).hexdigest(),
            'shingles': sorted({hashlib.sha256(payload[offset:offset + SHINGLE * 4]).hexdigest()
                                for offset in range(0, len(payload) - SHINGLE * 4 + 1, 4)})}


def generate(bench=None, sets=None, model=None, output=None, renderer=None):
    bench = input_path(argument_path(bench, 'MTP_PUBLIC_BENCH_DIR'))
    sets = input_path(argument_path(sets, 'MTP_PUBLIC_AB_SETS', bench / 'workloads/sets'))
    model = input_path(argument_path(model, 'MTP_PUBLIC_MODEL_DIR'))
    output = output_path(argument_path(output, 'MTP_PUBLIC_AB_HASHES'))
    if any(output.is_relative_to(root) for root in (bench, sets, model)):
        raise ValueError('hash fixture output must not replace source inputs')
    try:
        root = validate_set_paths(sets, bench)
        workloads_module, client_module = fnbench_modules(bench)
        workloads = workloads_module.select_prompt_sets(','.join(AB_DOMAINS), root, version='v1')
        if len(workloads) != 32:
            raise ValueError('A/B suite must contain 32 prompts')
        assets = {name: sha256(input_path(model / name, root=model))
                  for name in ('tokenizer.json', 'chat_template.jinja')}
        renderer = renderer or Renderer.from_model(model)
        records = []
        for workload in workloads:
            if (workload.prompt_set != workload.name + '-v1'
                    or not isinstance(workload.prompt_id, str) or not NAME.fullmatch(workload.prompt_id)):
                raise ValueError('invalid A/B workload identity')
            rendered, bodies = renderer.sequences(server_request(client_module, workload))
            records.append({'name': workload.prompt_set + '/' + workload.prompt_id,
                            'rendered': sequence_hashes(rendered),
                            'messages': [sequence_hashes(ids) for ids in bodies]})
        fixture = {'version': 1, 'encoding': 'int32-le', 'algorithm': 'sha256',
                   'shingle_tokens': SHINGLE, 'shingle_stride_tokens': 1,
                   'assets': assets, 'workloads': records}
        # Validate before publishing, using exactly the corpus reader's schema.
        load_ab_hashes(fixture, assets=assets)
        atomic_write(output, canonical(fixture) + '\n')
    except Exception:
        # fnbench errors can embed prompt/manifest values; never propagate them.
        raise ValueError('invalid or unavailable public A/B inputs; contents suppressed') from None
    return {'documents': len(records), 'fixture_sha256': sha256(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('bench', 'sets', 'output'):
        parser.add_argument('--' + option, type=Path)
    parser.add_argument('--model-dir', '--model', dest='model', type=Path)
    args = parser.parse_args(argv)
    try:
        result = generate(**vars(args))
    except Exception:
        parser.exit(2, 'workload-hashes: invalid or unavailable public inputs; no contents displayed\n')
    print(canonical(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
