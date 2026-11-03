#!/usr/bin/env python3
"""Assemble public ID-only documents, excluding hash-only A/B suite overlaps."""
import argparse
from array import array
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mtptrain.paths import input_path
SEED = 20260903
HOLDOUT = 0.1
SHINGLE = 64
VOCAB = 248320
HEX64 = re.compile(r'[0-9a-f]{64}\Z')
GENRES = {'en_chat', 'ja_chat', 'reasoning', 'tools', 'swe', 'code'}
AB_DOMAINS = ('code-edit', 'prose-en', 'prose-ja', 'agent-loop')
AB_NAME = re.compile(r'(code-edit|prose-en|prose-ja|agent-loop)-v1/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z')
DOC_SCHEMA = {
    'version': 1,
    'files': {'build': 'build.jsonl', 'heldout': 'heldout.jsonl',
              'quarantine': 'quarantine.jsonl'},
    'fields': {
        'doc_id': 'public prompt_id', 'prompt_id': 'upstream sha256 prompt identity',
        'doc_hash': 'public mtptrain _hash_ids: blake2b-8 over little-endian int32 IDs',
        'split_key': 'deterministic hex key whose public hash split matches split',
        'split': 'original public build or heldout; never reassigned',
        'genre': 'upstream genre', 'source': 'upstream public source locator',
        'license': 'upstream source license, not a new license for model outputs',
        'input_ids': 'rendered prompt IDs followed by output IDs, without any extra tokens',
        'prompt_tokens': 'rendered prompt length', 'completion_tokens': 'output length',
        'token_count': 'total ID count', 'sha256': 'sha256 over little-endian int32 IDs',
        'bucket': 'genre', 'mode': 'corpus', 'workload': 'code, mixed or prose',
        'lang': 'ja for ja_chat, otherwise und', 'name': 'public prompt_id',
        'tokens': 'total ID count, for extraction manifest compatibility',
    },
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False)


def sha256(path):
    digest = hashlib.sha256()
    with input_path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def ids_bytes(ids):
    values = array('i', ids)
    if values.itemsize != 4:
        raise ValueError('32-bit int array required')
    if sys.byteorder != 'little':
        values.byteswap()
    return values.tobytes()


def hash_ids(ids):
    return hashlib.blake2b(ids_bytes(ids), digest_size=8).hexdigest()


def validate_ids(ids, allow_empty=False):
    if (type(ids) is not list or (not ids and not allow_empty)
            or any(type(token) is not int or not 0 <= token < VOCAB for token in ids)):
        raise ValueError('invalid numeric token IDs')
    return ids


def read_jsonl(path):
    with input_path(path).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                raise ValueError(f'invalid JSONL at record {number}') from None
            if not isinstance(row, dict):
                raise ValueError(f'expected JSON object at record {number}')
            yield row


def is_holdout_key(key, seed=SEED, holdout_frac=HOLDOUT):
    digest = hashlib.blake2b(f'{seed}:{key}'.encode(), digest_size=8).hexdigest()
    return int(digest, 16) % 10000 < int(holdout_frac * 10000)


def split_key(prompt_id, split, seed=SEED, holdout=HOLDOUT):
    if split not in ('build', 'heldout') or not 0 < holdout < 1:
        raise ValueError('invalid split policy')
    for nonce in range(100000):
        key = hashlib.blake2b(f'public-data:{prompt_id}:{nonce}'.encode(), digest_size=16).hexdigest()
        if is_holdout_key(key, seed, holdout) == (split == 'heldout'):
            return key
    raise ValueError('could not derive a matching split key')


def text_metadata(value):
    return (isinstance(value, str) and bool(value.strip()) and len(value) <= 4096
            and not any(ord(char) < 32 for char in value)
            and not value.startswith(('/', '~')) and '\\' not in value)


def identity(row):
    if (not isinstance(row, dict) or not isinstance(row.get('prompt_id'), str)
            or not HEX64.fullmatch(row['prompt_id']) or row.get('genre') not in GENRES
            or row.get('split') not in ('build', 'heldout')):
        raise ValueError('invalid public prompt identity')
    return row['prompt_id']


def source_name(locator):
    return locator.split('!', 1)[0].split('#', 1)[0]


def load_documents(prompts_path, outputs_path, manifest, seed=SEED, holdout=HOLDOUT):
    """Prompt IDs already include the exact public default template; never decode them."""
    prompts, outputs = {}, {}
    licenses = {}
    for source in manifest.get('sources', []):
        name, license_name = source.get('local'), source.get('license')
        if (not text_metadata(name) or not text_metadata(license_name) or name in licenses):
            raise ValueError('invalid public license manifest')
        licenses[name] = license_name
    if not licenses:
        raise ValueError('missing public license provenance')
    for row in read_jsonl(prompts_path):
        prompt_id = identity(row)
        if prompt_id in prompts:
            raise ValueError('duplicate public prompt ID')
        ids = validate_ids(row.get('input_ids'))
        if (len(ids) > 6000 or not text_metadata(row.get('source'))
                or row.get('license') != licenses.get(source_name(row['source']))):
            raise ValueError('invalid public prompt length or license provenance')
        prompts[prompt_id] = row
    if not prompts:
        raise ValueError('empty public prompt input')
    for row in read_jsonl(outputs_path):
        prompt_id = identity(row)
        if prompt_id in outputs or prompt_id not in prompts:
            raise ValueError('duplicate or unmatched public output ID')
        prompt = prompts[prompt_id]
        ids = validate_ids(row.get('output_ids'), allow_empty=True)
        if (row.get('error') or row.get('finish') not in ('stop', 'length')
                or any(row[key] != prompt[key] for key in ('genre', 'split'))
                or type(row.get('prompt_tokens')) is not int
                or row['prompt_tokens'] != len(prompt['input_ids'])
                or type(row.get('completion_tokens')) is not int
                or row['completion_tokens'] != len(ids) or len(ids) > 3072
                or row['finish'] == 'length' and len(ids) != 3072):
            raise ValueError('public output metadata mismatch or failed generation')
        outputs[prompt_id] = row
    if prompts.keys() != outputs.keys():
        raise ValueError('public outputs do not cover every prompt exactly once')
    actual_counts = defaultdict(Counter)
    documents = []
    for prompt_id, prompt in prompts.items():
        output = outputs[prompt_id]
        ids = prompt['input_ids'] + output['output_ids']
        payload = ids_bytes(ids)
        genre = prompt['genre']
        actual_counts[genre].update(total=1, **{prompt['split']: 1})
        documents.append({
            **{key: prompt[key] for key in ('prompt_id', 'split', 'genre', 'source', 'license')},
            'doc_id': prompt_id, 'doc_hash': hashlib.blake2b(payload, digest_size=8).hexdigest(),
            'sha256': hashlib.sha256(payload).hexdigest(),
            'split_key': split_key(prompt_id, prompt['split'], seed, holdout),
            'input_ids': ids, 'prompt_tokens': len(prompt['input_ids']),
            'completion_tokens': len(output['output_ids']), 'token_count': len(ids),
            'tokens': len(ids), 'bucket': genre, 'mode': 'corpus', 'name': prompt_id,
            'workload': ('code' if genre == 'code' else 'mixed' if genre in ('tools', 'swe') else 'prose'),
            'lang': 'ja' if genre == 'ja_chat' else 'und',
        })
    if (manifest.get('total') != len(documents)
            or {key: dict(value) for key, value in actual_counts.items()} != manifest.get('counts')):
        raise ValueError('public manifest prompt counts do not match')
    return documents


class OverlapIndex:
    """Hash-only exact and all-position 64-token shingle comparisons, no text."""
    def __init__(self, sequences=()):
        self.exact = set()
        self.shingles = set()
        for ids in sequences:
            self.add(ids)

    def add(self, ids):
        payload = ids_bytes(validate_ids(ids))
        self.exact.add(hashlib.sha256(payload).digest())
        for offset in range(0, len(payload) - SHINGLE * 4 + 1, 4):
            self.shingles.add(hashlib.sha256(payload[offset:offset + SHINGLE * 4]).digest())

    def matches(self, ids):
        payload = ids_bytes(ids)
        exact = hashlib.sha256(payload).digest() in self.exact
        shingle = any(hashlib.sha256(payload[offset:offset + SHINGLE * 4]).digest() in self.shingles
                      for offset in range(0, len(payload) - SHINGLE * 4 + 1, 4))
        return exact, shingle


def load_ab_hashes(fixture, assets=None, expected_ab_documents=32):
    """Strict, hash-only fixture reader; no token sequences or prompt text accepted."""
    if not isinstance(fixture, dict):
        fixture = json.loads(input_path(fixture).read_text(encoding='utf-8'))
    fields = {'version', 'encoding', 'algorithm', 'shingle_tokens',
              'shingle_stride_tokens', 'assets', 'workloads'}
    if (not isinstance(fixture, dict) or set(fixture) != fields
            or type(fixture['version']) is not int or fixture['version'] != 1
            or fixture['encoding'] != 'int32-le' or fixture['algorithm'] != 'sha256'
            or type(fixture['shingle_tokens']) is not int or fixture['shingle_tokens'] != SHINGLE
            or type(fixture['shingle_stride_tokens']) is not int or fixture['shingle_stride_tokens'] != 1
            or not isinstance(fixture['workloads'], list)
            or len(fixture['workloads']) != expected_ab_documents):
        raise ValueError('invalid hash-only A/B fixture schema or suite size')
    model_assets = fixture['assets']
    if (not isinstance(model_assets, dict) or set(model_assets) != {'tokenizer.json', 'chat_template.jinja'}
            or any(not isinstance(value, str) or not HEX64.fullmatch(value) for value in model_assets.values())
            or assets is not None and model_assets != assets):
        raise ValueError('A/B tokenizer/template provenance mismatch')
    index, names, domains = OverlapIndex(), set(), Counter()
    for record in fixture['workloads']:
        if (not isinstance(record, dict) or set(record) != {'name', 'rendered', 'messages'}
                or not isinstance(record['name'], str) or not AB_NAME.fullmatch(record['name'])
                or record['name'] in names or not isinstance(record['messages'], list)
                or not record['messages']):
            raise ValueError('invalid or duplicate A/B workload identity')
        names.add(record['name'])
        domains[record['name'].split('-v1/', 1)[0]] += 1
        for hashes in (record['rendered'], *record['messages']):
            if (not isinstance(hashes, dict) or set(hashes) != {'exact', 'shingles'}
                    or not isinstance(hashes['exact'], str) or not HEX64.fullmatch(hashes['exact'])
                    or not isinstance(hashes['shingles'], list)
                    or any(not isinstance(value, str) or not HEX64.fullmatch(value)
                           for value in hashes['shingles'])
                    or hashes['shingles'] != sorted(set(hashes['shingles']))):
                raise ValueError('invalid A/B exact or all-position shingle hashes')
            index.exact.add(bytes.fromhex(hashes['exact']))
            index.shingles.update(bytes.fromhex(value) for value in hashes['shingles'])
    if expected_ab_documents == 32 and domains != Counter({domain: 8 for domain in AB_DOMAINS}):
        raise ValueError('A/B fixture must cover all four eight-prompt domains')
    return index


def filter_documents(documents, ab, ab_complete=True):
    heldout = [doc for doc in documents if doc['split'] == 'heldout']
    held_hashes = {doc['doc_hash'] for doc in heldout}
    build, quarantine = [], []
    for doc in documents:
        if doc['split'] != 'build':
            continue
        reasons = []
        if not ab_complete:
            reasons.append('ab_hash_fixture_unavailable')
        if doc['doc_hash'] in held_hashes:
            reasons.append('heldout_document_hash')
        # Compare both prompt and combined document: an exact A/B prompt with a
        # completion is still overlap, even when that prompt is shorter than 64.
        for label, ids in (('document', doc['input_ids']),
                           ('prompt', doc['input_ids'][:doc['prompt_tokens']])):
            exact, shingle = ab.matches(ids)
            if exact:
                reasons.append('ab_exact_' + label)
            if shingle:
                reasons.append('ab_64_token_' + label)
        if reasons:
            quarantine.append({**doc, 'exclusion_reasons': reasons})
        else:
            build.append(doc)
    return build, heldout, quarantine


def summary(documents):
    return {'documents': len(documents),
            'prompt_tokens': sum(doc['prompt_tokens'] for doc in documents),
            'completion_tokens': sum(doc['completion_tokens'] for doc in documents),
            'tokens': sum(doc['token_count'] for doc in documents)}


def grouped_summary(documents, key):
    grouped = defaultdict(list)
    for doc in documents:
        grouped[key(doc)].append(doc)
    return {name: summary(group) for name, group in sorted(grouped.items())}


def output_path(path):
    """Allow caller-selected outputs, never symlink or traversal destinations."""
    path = Path(path).expanduser()
    if '..' in path.parts:
        raise ValueError('output path must not contain traversal')
    path = Path(os.path.abspath(path))
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError('outputs must not have symlink ancestors')
    return path


def argument_path(value, environment, default=None):
    value = value if value is not None else os.environ.get(environment, default)
    if value is None or isinstance(value, str) and not value.strip():
        raise ValueError('required input or output path is missing')
    return value


def atomic_write(path, content):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = content.encode('utf-8')
    if path.is_file() and path.read_bytes() == payload:
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name + '.',
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def verify_provenance(prompts, manifest, model):
    if sha256(prompts) != manifest.get('prompts', {}).get('sha256'):
        raise ValueError('public prompt manifest hash mismatch')
    expected_render = ('tokenizers; ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True); '
                       'add_special_tokens=False; add_generation_prompt=True; other template defaults unchanged')
    if manifest.get('selection_rule', {}).get('render') != expected_render:
        raise ValueError('unexpected public prompt rendering policy')
    assets = {}
    for asset in manifest.get('assets', []):
        if not isinstance(asset, dict):
            raise ValueError('invalid public asset provenance')
        name = asset.get('asset', asset.get('name'))
        if not text_metadata(name) or '/' in name or name in assets:
            raise ValueError('invalid or duplicate public asset identity')
        assets[name] = asset
    result = []
    for name in ('tokenizer.json', 'chat_template.jinja'):
        path = input_path(input_path(model) / name, root=input_path(model))
        digest = sha256(path)
        if digest != assets.get(name, {}).get('sha256'):
            raise ValueError('public prompt tokenizer/template hash mismatch')
        result.append({'asset': name, 'sha256': digest, 'bytes': path.stat().st_size})
    return result


def prepare(prompts=None, outputs=None, manifest=None, model=None, ab_hashes=None,
            output_dir=None, seed=SEED, holdout=HOLDOUT, expected_ab_documents=32):
    if (type(seed) is not int or not math.isfinite(holdout) or not 0 < holdout < 1
            or type(expected_ab_documents) is not int or expected_ab_documents < 1):
        raise ValueError('invalid split or A/B fixture policy')
    prompts = input_path(argument_path(prompts, 'MTP_PUBLIC_PROMPTS'))
    outputs = input_path(argument_path(outputs, 'MTP_PUBLIC_OUTPUTS'))
    manifest = input_path(argument_path(manifest, 'MTP_PUBLIC_MANIFEST', prompts.parent / 'manifest.json'))
    model = input_path(argument_path(model, 'MTP_PUBLIC_MODEL_DIR'))
    output_dir = output_path(argument_path(output_dir, 'MTP_PUBLIC_CORPUS_DIR'))
    ab_hashes = input_path(argument_path(ab_hashes, 'MTP_PUBLIC_AB_HASHES', output_dir / 'ab-hashes.json'))
    protected = {prompts, outputs, manifest, ab_hashes, Path(__file__).resolve(), HERE / 'build_prompts.py'}
    protected.update(input_path(model / name, root=model)
                     for name in ('tokenizer.json', 'chat_template.jinja'))
    for name in ('build.jsonl', 'heldout.jsonl', 'quarantine.jsonl', 'report.json'):
        if output_path(output_dir / name) in protected:
            raise ValueError('corpus output would replace an input')
    reference = json.loads(input_path(manifest).read_text(encoding='utf-8'))
    assets = verify_provenance(prompts, reference, model)
    documents = load_documents(prompts, outputs, reference, seed, holdout)
    ab_complete = ab_hashes.is_file()
    ab = (load_ab_hashes(ab_hashes, {asset['asset']: asset['sha256'] for asset in assets},
                         expected_ab_documents) if ab_complete else OverlapIndex())
    ab_provenance = ([{'sha256': sha256(ab_hashes), 'bytes': ab_hashes.stat().st_size,
                       'documents': expected_ab_documents}] if ab_complete else [])
    build, heldout, quarantine = filter_documents(documents, ab, ab_complete)
    original_build = [doc for doc in documents if doc['split'] == 'build']
    ab_excluded = [doc for doc in quarantine
                   if any(reason.startswith(('ab_exact_', 'ab_64_token_'))
                          for reason in doc['exclusion_reasons'])]
    original_tokens = sum(doc['token_count'] for doc in original_build)
    excluded_percent = 100 * sum(doc['token_count'] for doc in quarantine) / original_tokens if original_tokens else 0
    ab_excluded_percent = 100 * sum(doc['token_count'] for doc in ab_excluded) / original_tokens if original_tokens else 0
    reasons = Counter(reason for doc in quarantine for reason in doc['exclusion_reasons'])
    groups = {'input': documents, 'build': build, 'heldout': heldout, 'quarantine': quarantine}
    # Copy only portable public provenance fields; upstream manifests contain
    # machine-local paths that must never leak into the assembled report.
    source_fields = ('kind', 'repo', 'revision', 'tag', 'file', 'local', 'sha256', 'license', 'bytes')
    sources = [{**{key: source[key] for key in source_fields if key in source},
                'counts': {name: summary([doc for doc in rows
                                          if source_name(doc['source']) == source['local']])
                           for name, rows in groups.items()}}
               for source in reference['sources']]
    file_records = {}
    for split, rows in (('build', build), ('heldout', heldout), ('quarantine', quarantine)):
        content = ''.join(canonical(row) + '\n' for row in rows)
        payload = content.encode('utf-8')
        file_records[split] = {'path': split + '.jsonl', 'sha256': hashlib.sha256(payload).hexdigest(),
                               'bytes': len(payload), **summary(rows)}
        # Final report is the completion marker; fail closed while replacing any
        # previously complete generation, then publish retained build last.
        if split == 'build':
            build_content = content
        else:
            file_records[split]['_content'] = content
    report = {
        'version': 1, 'ready': ab_complete and bool(build) and bool(heldout), 'schema': DOC_SCHEMA,
        'blocking_reasons': ([] if ab_complete else ['A_B_hash_fixture_unavailable'])
                            + ([] if build else ['no_retained_build_documents'])
                            + ([] if heldout else ['no_heldout_documents']),
        'counts': {name: summary(rows) for name, rows in groups.items()},
        'by_genre': {name: grouped_summary(rows, lambda doc: doc['genre']) for name, rows in groups.items()},
        'by_license': {name: grouped_summary(rows, lambda doc: doc['license']) for name, rows in groups.items()},
        'sources': sources,
        'provenance': {'prompts': {'sha256': sha256(prompts), 'bytes': prompts.stat().st_size},
                       'outputs': {'sha256': sha256(outputs), 'bytes': outputs.stat().st_size},
                       'source_manifest': {'sha256': sha256(manifest), 'bytes': manifest.stat().st_size},
                       'assets': assets, 'assembler_sha256': sha256(Path(__file__)),
                       'renderer_sha256': sha256(HERE / 'build_prompts.py')},
        'rendering': {'policy': reference['selection_rule']['render'],
                      'method': 'preserve manifest-verified rendered prompt IDs verbatim; concatenate output IDs',
                      'decode': False, 'extra_tokens': 0},
        'generation': {'temperature': 0.6, 'top_p': 0.95, 'top_k': 20, 'max_new_tokens': 3072,
                       'license': 'Generated IDs; no independent output license inferred; review model and source terms.'},
        'split_policy': {'seed': seed, 'holdout_frac': holdout,
                         'rule': 'blake2b-8(seed + colon + split_key) mod 10000 < int(holdout_frac * 10000)',
                         'split_key': 'first blake2b-16(public-data:prompt_id:nonce) matching original public split',
                         'heldout_document_hashes_excluded_from_build': True},
        'exclusion': {'ab_complete': ab_complete, 'expected_ab_documents': expected_ab_documents,
                      'ab_documents': expected_ab_documents if ab_complete else 0, 'fixtures': ab_provenance,
                      'original_build': summary(original_build), 'excluded': summary(quarantine),
                      'excluded_by_source': grouped_summary(quarantine, lambda doc: source_name(doc['source'])),
                      'excluded_by_genre': grouped_summary(quarantine, lambda doc: doc['genre']),
                      'excluded_tokens_percent': excluded_percent,
                      'ab_excluded': summary(ab_excluded), 'ab_excluded_tokens_percent': ab_excluded_percent,
                      'over_ten_percent': ab_excluded_percent > 10,
                      'over_ten_percent_action': 'retain exclusions and continue; parent decides policy',
                      'shingle_tokens': SHINGLE, 'shingle_stride_tokens': 1,
                      'algorithm': 'sha256 over all-position contiguous 64-token int32-le windows and exact sequences',
                      'scope': 'compare rendered prompt and complete document against A/B rendered requests and message bodies',
                      'heldout_scope': 'unchanged; exclude build with identical public blake2b document hash',
                      'reasons': dict(sorted(reasons.items())),
                      'missing_fixture_action': 'quarantine all build; corpus never opens A/B text'},
    }
    contents = {name: record.pop('_content') for name, record in file_records.items() if '_content' in record}
    report['files'] = file_records
    serialized_report = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    # Identical runs are true no-ops, including file mtimes.
    desired = {'build.jsonl': build_content, 'report.json': serialized_report,
               **{name + '.jsonl': content for name, content in contents.items()}}
    if all((output_dir / name).is_file() and (output_dir / name).read_bytes() == content.encode('utf-8')
           for name, content in desired.items()):
        return report
    atomic_write(output_dir / 'report.json', '{"ready":false,"status":"assembling"}\n')
    atomic_write(output_dir / 'build.jsonl', '')
    for name, content in contents.items():
        atomic_write(output_dir / (name + '.jsonl'), content)
    atomic_write(output_dir / 'build.jsonl', build_content)
    atomic_write(output_dir / 'report.json', serialized_report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('prompts', 'outputs', 'manifest', 'output-dir'):
        parser.add_argument('--' + option, type=Path)
    parser.add_argument('--model-dir', '--model', dest='model', type=Path)
    parser.add_argument('--ab-hashes', type=Path,
                        help='hash-only fixture; missing fixture quarantines every build document')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--holdout', type=float, default=HOLDOUT)
    args = parser.parse_args(argv)
    try:
        report = prepare(**vars(args))
    except (OSError, ValueError, KeyError, TypeError):
        parser.exit(2, 'corpus: invalid or unavailable input/provenance; no input text displayed\n')
    print(canonical({'ready': report['ready'], 'counts': report['counts'],
                     'excluded_tokens_percent': report['exclusion']['excluded_tokens_percent'],
                     'over_ten_percent': report['exclusion']['over_ten_percent'],
                     'exclusion_reasons': report['exclusion']['reasons']}))
    return 0 if report['ready'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
