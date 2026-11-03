"""Validate the original checkpoint using only its index and tensor headers."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import stat
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mtptrain.cli import StrictParser
from mtptrain.paths import input_path

INDEX = 'model.safetensors.index.json'
SHARED_SHARD = 'model-bf16-00012.safetensors'
EMBED_KEY = 'model.language_model.embed_tokens.weight'
LM_HEAD_KEY = 'lm_head.weight'
SHARED_SHAPES = {EMBED_KEY: [248320, 2560], LM_HEAD_KEY: [248320, 2560]}
MTP_SHAPES = {
    'mtp.fc_embedding.weight': [2560, 2560],
    'mtp.fc_hidden.weight': [2560, 2560],
    'mtp.pre_fc_norm_embedding.weight': [2560],
    'mtp.pre_fc_norm_hidden.weight': [10240],
    'mtp.hyper_connection_mixer.hc_norm.weight': [10240],
    'mtp.hyper_connection_mixer.input_mix_weight_down.weight': [320, 10240],
    'mtp.hyper_connection_mixer.input_mix_weight_up.weight': [10240, 320],
    'mtp.layers.0.attn_hyper_connection.block_inject_weight.weight': [4, 10240],
    'mtp.layers.0.attn_hyper_connection.hc_norm.weight': [10240],
    'mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight': [320, 10240],
    'mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight': [10240, 320],
    'mtp.layers.0.mlp.experts.down_proj': [512, 2560, 640],
    'mtp.layers.0.mlp.experts.gate_up_proj': [512, 1280, 2560],
    'mtp.layers.0.mlp.gate.weight': [512, 2560],
    'mtp.layers.0.mlp.shared_expert.down_proj.weight': [2560, 640],
    'mtp.layers.0.mlp.shared_expert.gate_proj.weight': [640, 2560],
    'mtp.layers.0.mlp.shared_expert.up_proj.weight': [640, 2560],
    'mtp.layers.0.mlp.shared_expert_gate.weight': [1, 2560],
    'mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight': [4, 10240],
    'mtp.layers.0.mlp_hyper_connection.hc_norm.weight': [10240],
    'mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight': [320, 10240],
    'mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight': [10240, 320],
    'mtp.layers.0.self_attn.indexer.index_qk_proj.weight': [640, 2560],
    'mtp.layers.0.self_attn.indexer.k_layernorm.weight': [128],
    'mtp.layers.0.self_attn.indexer.q_layernorm.weight': [128],
    'mtp.layers.0.self_attn.k_norm.weight': [256],
    'mtp.layers.0.self_attn.k_proj.weight': [512, 2560],
    'mtp.layers.0.self_attn.o_proj.weight': [2560, 6144],
    'mtp.layers.0.self_attn.q_norm.weight': [256],
    'mtp.layers.0.self_attn.q_proj.weight': [12288, 2560],
    'mtp.layers.0.self_attn.v_proj.weight': [512, 2560],
}
MAX_HEADER_BYTES = 100_000_000  # safetensors' header limit; also bound the index.
_DTYPE_BYTES = {
    'BOOL': 1, 'U8': 1, 'I8': 1, 'F8_E4M3': 1, 'F8_E5M2': 1, 'F8_E8M0': 1,
    'I16': 2, 'U16': 2, 'F16': 2, 'BF16': 2,
    'I32': 4, 'U32': 4, 'F32': 4, 'I64': 8, 'U64': 8, 'F64': 8,
}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError('non-finite JSON number')


def _json(data):
    return json.loads(data.decode('utf-8'), object_pairs_hook=_unique,
                      parse_constant=_invalid_constant)


def _regular(path):
    path = input_path(path)
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError('checkpoint input is not a regular file')
    return path


def _header(path):
    with _regular(path).open('rb') as stream:
        size = os.fstat(stream.fileno()).st_size
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('truncated safetensors header length')
        length = struct.unpack('<Q', prefix)[0]
        if not 2 <= length <= min(MAX_HEADER_BYTES, size - 8):
            raise ValueError('invalid safetensors header size')
        data = stream.read(length)
        if len(data) != length or not data.startswith(b'{'):
            raise ValueError('truncated or invalid safetensors header')
    header = _json(data)
    if not isinstance(header, dict):
        raise ValueError('invalid safetensors header object')
    return header, length, size - 8 - length


def validate_model_headers(model_dir: str | os.PathLike) -> dict[str, int]:
    """Return numeric metadata, or raise ValueError/OSError on invalid input.

    Read every indexed shard's header, never its tensor payload. The original
    31 MTP tensors and both shared matrices must have the shipped BF16 layout.
    All index paths pass input_path, including their symlink targets.
    """
    if not str(model_dir).strip():
        raise ValueError('empty model directory')
    root = input_path(model_dir)
    with _regular(root / INDEX).open('rb') as stream:
        data = stream.read(MAX_HEADER_BYTES + 1)
    if len(data) > MAX_HEADER_BYTES:
        raise ValueError('weight index is too large')
    index = _json(data)
    weight_map = index.get('weight_map') if isinstance(index, dict) else None
    if (not isinstance(weight_map, dict) or not weight_map
            or any(not isinstance(name, str) or not name or name == '__metadata__'
                   for name in weight_map)):
        raise ValueError('invalid weight index')
    by_shard = {}
    for name, shard in weight_map.items():
        if (not isinstance(shard, str) or not shard.endswith('.safetensors')
                or any(char in shard for char in '/\\\0:')):
            raise ValueError('unsafe index shard path')
        by_shard.setdefault(shard, set()).add(name)
    if {name for name in weight_map if name.startswith('mtp.')} != set(MTP_SHAPES):
        raise ValueError('expected exactly the 31 original MTP tensor names')
    if any(weight_map.get(name) != SHARED_SHARD for name in SHARED_SHAPES):
        raise ValueError('shared tensors must be in the original BF16 shard')

    expected = {**MTP_SHAPES, **SHARED_SHAPES}
    header_bytes = tensor_bytes = 0
    for shard, names in by_shard.items():
        header, length, payload_size = _header(root / shard)
        metadata = header.pop('__metadata__', {})
        if (not isinstance(metadata, dict)
                or any(not isinstance(value, str) for value in metadata.values())):
            raise ValueError('invalid safetensors metadata')
        if set(header) != names:
            raise ValueError('header and index tensor names or shard assignments differ')
        spans = []
        for name, tensor in header.items():
            if (not isinstance(tensor, dict)
                    or set(tensor) != {'dtype', 'shape', 'data_offsets'}):
                raise ValueError('invalid tensor header')
            dtype, shape, offsets = tensor['dtype'], tensor['shape'], tensor['data_offsets']
            if (not isinstance(dtype, str) or dtype not in _DTYPE_BYTES
                    or not isinstance(shape, list)
                    or any(type(dim) is not int or dim < 0 for dim in shape)
                    or not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(offset) is not int for offset in offsets)):
                raise ValueError('invalid tensor dtype, shape or offsets')
            start, end = offsets
            if (not 0 <= start <= end <= payload_size
                    or end - start != math.prod(shape) * _DTYPE_BYTES[dtype]):
                raise ValueError('tensor offsets or byte size do not match the file')
            if name in expected and (dtype != 'BF16' or shape != expected[name]):
                raise ValueError('original tensor BF16 dtype or shape mismatch')
            spans.append((start, end))
        end = 0
        for start, stop in sorted(spans):
            if start != end:
                raise ValueError('overlapping tensor offsets or unindexed payload gap')
            end = stop
        if end != payload_size:
            raise ValueError('unindexed or truncated tensor payload')
        header_bytes += length
        tensor_bytes += payload_size
    # The shipped index's advisory total_size differs from its actual payloads.
    metadata = index.get('metadata', {})
    if (not isinstance(metadata, dict)
            or 'total_size' in metadata and (type(metadata['total_size']) is not int
                                            or metadata['total_size'] < 0)):
        raise ValueError('invalid index total tensor size')
    return {'tensor_count': len(weight_map), 'shard_count': len(by_shard),
            'header_bytes': header_bytes, 'tensor_bytes': tensor_bytes,
            'mtp_count': len(MTP_SHAPES),
            'mtp_bytes': 2 * sum(math.prod(shape) for shape in MTP_SHAPES.values()),
            'shared_count': len(SHARED_SHAPES),
            'shared_bytes': 2 * sum(math.prod(shape) for shape in SHARED_SHAPES.values()),
            'vocab_size': 248320, 'hidden_size': 2560}


def main(argv=None):
    parser = StrictParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--model-dir', type=Path, default=os.environ.get('MTP_MODEL_DIR'),
                        help='source checkpoint (or env MTP_MODEL_DIR)')
    args = parser.parse_args(argv)
    try:
        if args.model_dir is None:
            raise ValueError('--model-dir or MTP_MODEL_DIR is required')
        result = validate_model_headers(args.model_dir)
    except (OSError, ValueError) as exc:
        parser.exit(2, f'model check refused: {exc}\n')
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
