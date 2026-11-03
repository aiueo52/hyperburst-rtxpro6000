#!/usr/bin/env python3
"""Encode deterministic public prompts with a local tokenizer and chat template."""
import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import statistics
import sys
import tarfile
import tempfile

sys.dont_write_bytecode = True

SEED = "sgmap"
MAX_INPUT_TOKENS = 6000
COUNTS = {"en_chat": 300, "ja_chat": 300, "reasoning": 250,
          "tools": 300, "swe": 200, "code": 350}
FILES = {"en_chat": "ultrachat-test_sft.parquet", "ja_chat": "oasst2-33k-ja.jsonl",
         "reasoning": "openthoughts-00005.parquet", "tools": "hermes-func-calling.json",
         "swe": "swe-agent-traj-00000.parquet"}
ARCHIVES = ("cpython-v3.13.0.tar.gz", "go-go1.23.0.tar.gz", "react-v18.3.1.tar.gz",
            "tokio-1.40.0.tar.gz", "vuejs-core-v3.5.12.tar.gz", "curl-8_10_1.tar.gz")
# Match the companion metadata to the filenames used by its public fetch script.
SOURCE_KEYS = {
    "ultrachat-test_sft.parquet": ("hf-dataset", "HuggingFaceH4/ultrachat_200k",
                                  "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet"),
    "oasst2-33k-ja.jsonl": ("hf-dataset", "llm-jp/oasst2-33k-ja", "oasst2-33k-ja.jsonl"),
    "openthoughts-00005.parquet": ("hf-dataset", "open-thoughts/OpenThoughts-114k",
                                 "data/train-00005-of-00006.parquet"),
    "hermes-func-calling.json": ("hf-dataset", "NousResearch/hermes-function-calling-v1",
                                 "func-calling.json"),
    "swe-agent-traj-00000.parquet": ("hf-dataset", "nebius/SWE-agent-trajectories",
                                   "data/train-00000-of-00012.parquet"),
    "cpython-v3.13.0.tar.gz": ("github-tag", "python/cpython", "v3.13.0"),
    "go-go1.23.0.tar.gz": ("github-tag", "golang/go", "go1.23.0"),
    "react-v18.3.1.tar.gz": ("github-tag", "facebook/react", "v18.3.1"),
    "tokio-1.40.0.tar.gz": ("github-tag", "tokio-rs/tokio", "tokio-1.40.0"),
    "vuejs-core-v3.5.12.tar.gz": ("github-tag", "vuejs/core", "v3.5.12"),
    "curl-8_10_1.tar.gz": ("github-tag", "curl/curl", "curl-8_10_1"),
}
EXTENSIONS = {".py", ".go", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
              ".rs", ".vue", ".c", ".h", ".cc", ".cpp", ".hpp"}
EDIT_INSTRUCTIONS = (
    "Add useful docstrings or comments to this code without changing its behavior.",
    "Refactor this code for clarity while preserving its public behavior.",
    "Add appropriate error handling to this code, explaining any behavior changes.",
    "Write focused unit tests for this code, including edge cases.",
    "Identify and fix a likely bug in this code; explain the fix and a regression test.",
    "Convert this code to a consistent, idiomatic style without changing its behavior.",
)
TOOLS = [
    {"type": "function", "function": {
        "name": "list_files", "description": "List the entries of a workspace directory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "default": "."},
        }, "required": [], "additionalProperties": False},
    }},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read numbered lines, at most 300 lines and 24000 characters.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer", "minimum": 1, "default": 1},
            "end_line": {"type": "integer", "minimum": 1,
                         "description": "Inclusive; defaults to start_line + 299."},
        }, "required": ["path"], "additionalProperties": False},
    }},
    {"type": "function", "function": {
        "name": "search", "description": "Search file lines with a Python regular expression; at most 50 hits.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}, "path": {"type": "string", "default": "."},
        }, "required": ["pattern"], "additionalProperties": False},
    }},
    {"type": "function", "function": {
        "name": "edit_file", "description": "Replace exactly one occurrence, or leave the file unchanged on error.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "old_string": {"type": "string", "minLength": 1},
            "new_string": {"type": "string"},
        }, "required": ["path", "old_string", "new_string"], "additionalProperties": False},
    }},
    {"type": "function", "function": {
        "name": "write_file", "description": "Create or overwrite a UTF-8 file, creating parent directories if needed.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"},
        }, "required": ["path", "content"], "additionalProperties": False},
    }},
]


def cpu_environment():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def import_cpu_module(name):
    cpu_environment()
    return importlib.import_module(name)


def local_output(path):
    """Reject parent traversal and symlink destinations, including ancestors."""
    path = Path(path).expanduser().absolute()
    if ".." in path.parts:
        raise ValueError("parent traversal is not allowed")
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("symlink output paths are not allowed")
    return path


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def digest(purpose, value):
    return hashlib.sha256(f"{SEED}:{purpose}:{value}".encode("utf-8")).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_tools(tools):
    if not isinstance(tools, list) or not tools:
        raise ValueError("tools must be a nonempty list of function definitions")
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ValueError("invalid function definition")
        function = tool.get("function")
        if (not isinstance(function, dict) or not isinstance(function.get("name"), str)
                or not function["name"].strip() or not isinstance(function.get("parameters"), dict)):
            raise ValueError("invalid function schema")
    canonical(tools)
    return tools


class Renderer:
    def __init__(self, tokenizer, template):
        sandbox = import_cpu_module("jinja2.sandbox")
        environment = sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

        def raise_exception(message):
            raise ValueError("invalid local chat template invocation")

        environment.globals["raise_exception"] = raise_exception
        self.render = environment.from_string(template).render
        self.tokenizer = tokenizer

    @classmethod
    def from_model(cls, model_dir):
        model_dir = Path(model_dir).expanduser()
        template = model_dir / "chat_template.jinja"
        if not template.is_file() or template.stat().st_size > 1024 * 1024:
            raise ValueError("local chat template missing or too large")
        text = template.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError("empty local chat template")
        tokenizer = import_cpu_module("tokenizers").Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        return cls(tokenizer, text)

    def encode(self, content, tools=None):
        if not isinstance(content, str) or not content.strip():
            raise ValueError("empty or invalid user message")
        options = {"messages": [{"role": "user", "content": content}], "add_generation_prompt": True}
        if tools is not None:
            options["tools"] = validate_tools(tools)
        # Leave reasoning and all other template defaults unchanged.
        ids = self.tokenizer.encode(self.render(**options), add_special_tokens=False).ids
        if (type(ids) is not list or not ids
                or any(type(token) is not int or not 0 <= token < 2**31 for token in ids)):
            raise ValueError("tokenizer returned invalid input IDs")
        return ids


def first_user(messages):
    if not isinstance(messages, list):
        return None
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role", message.get("from")) in ("user", "human"):
            text = message.get("content", message.get("value", message.get("text")))
            return text.strip() if isinstance(text, str) else None
    return None


def hermes_tools(messages):
    if not isinstance(messages, list):
        raise ValueError("missing tool conversation")
    for message in messages:
        if not isinstance(message, dict) or message.get("role", message.get("from")) != "system":
            continue
        text = message.get("content", message.get("value", ""))
        if not isinstance(text, str):
            continue
        # The introductory prose itself mentions an empty <tools> </tools> pair.
        for section in text.split("<tools>")[1:]:
            if "</tools>" not in section:
                continue
            try:
                return validate_tools(json.loads(section.split("</tools>", 1)[0]))
            except (ValueError, TypeError):
                continue
    raise ValueError("missing or malformed Hermes function definitions")


def swe_issue(text):
    if not isinstance(text, str) or "ISSUE:\n" not in text:
        return None
    issue = text.split("ISSUE:\n", 1)[1]
    if "\nINSTRUCTIONS:\n" not in issue:
        return None
    # Exclude trajectory-specific terminal and command instructions.
    return issue.split("\nINSTRUCTIONS:\n", 1)[0].strip()


@dataclass
class Candidate:
    source: str
    content: str
    tools: object = None
    order_key: str = ""

    @property
    def rank(self):
        return digest("select", self.order_key or self.source), self.source


def data_candidates(genre, tools, skipped, raw):
    name = FILES[genre]
    path = Path(raw) / name
    if path.suffix == ".parquet":
        column = {"en_chat": "messages", "reasoning": "conversations", "swe": "trajectory"}[genre]
        parquet = import_cpu_module("pyarrow.parquet").ParquetFile(path)
        rows = (row for batch in parquet.iter_batches(batch_size=128, columns=[column])
                for row in batch.to_pylist())
    elif path.suffix == ".jsonl":
        def jsonl():
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    if line.strip():
                        yield json.loads(line)
        rows = jsonl()
        column = "conversations"
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("public JSON source must be a row list")
        column = "conversations"
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            skipped["malformed"] += 1
            continue
        messages = row.get(column)
        content = first_user(messages)
        row_tools = None
        if genre == "swe":
            content, row_tools = swe_issue(content), tools
        elif genre == "tools":
            try:
                row_tools = hermes_tools(messages)
            except ValueError:
                skipped["malformed_tools"] += 1
                continue
        if not content:
            skipped["empty_or_malformed"] += 1
            continue
        yield Candidate(f"{name}#row={index}", content, row_tools)


def source_member(member):
    path = PurePosixPath(member.name)
    return (member.isfile() and not path.is_absolute() and ".." not in path.parts
            and path.suffix.lower() in EXTENSIONS and 0 < member.size <= 2 * 1024 * 1024)


def code_chunk(text, key):
    lines = text.split("\n")
    if lines and not lines[-1]:
        lines.pop()
    if len(lines) < 40:
        return None
    seed = int(digest("chunk", key), 16)
    length = min(len(lines), 40 + seed % 211)
    start = (seed // 211) % (len(lines) - length + 1)
    chunk = "\n".join(lines[start:start + length])
    if not chunk.strip():
        return None
    instruction = EDIT_INSTRUCTIONS[int(digest("edit", key), 16) % len(EDIT_INSTRUCTIONS)]
    content = f"{instruction}\n\nSource: {key}, lines {start + 1}-{start + length}\n```\n{chunk}\n```"
    return Candidate(f"{key}#L{start + 1}-L{start + length}", content, order_key=key)


def code_candidates(skipped, raw):
    # Index names only, then read bounded SHA-ordered batches. Never extract archives.
    index = []
    for name in ARCHIVES:
        with tarfile.open(Path(raw) / name, "r|gz") as archive:
            seen_names = set()
            for member in archive:
                if not source_member(member):
                    continue
                if member.name in seen_names:
                    raise ValueError("duplicate source archive member")
                seen_names.add(member.name)
                key = f"{name}!{member.name}"
                index.append((digest("select", key), key, name, member.name))
    index.sort()
    for offset in range(0, len(index), 1024):
        batch = index[offset:offset + 1024]
        wanted = {(name, member): key for _, key, name, member in batch}
        candidates = {}
        for name in ARCHIVES:
            if not any(archive_name == name for archive_name, _ in wanted):
                continue
            with tarfile.open(Path(raw) / name, "r|gz") as archive:
                for member in archive:
                    key = wanted.get((name, member.name))
                    if key is None:
                        continue
                    if not source_member(member):
                        raise ValueError("source archive changed while reading")
                    stream = archive.extractfile(member)
                    with stream:
                        data = stream.read(2 * 1024 * 1024 + 1)
                    if len(data) != member.size:
                        raise ValueError("source archive member size mismatch")
                    try:
                        candidate = code_chunk(data.decode("utf-8"), key)
                    except UnicodeError:
                        candidate = None
                    if candidate is None:
                        skipped["short_empty_or_non_utf8"] += 1
                    else:
                        candidates[key] = candidate
        for _, key, _, _ in batch:
            if key in candidates:
                yield candidates[key]


def select(genre, candidates, count, renderer, licenses, seen, skipped, ordered=False):
    result = []
    population = candidates if ordered else sorted(candidates, key=lambda candidate: candidate.rank)
    for candidate in population:
        prompt_id = digest("prompt", canonical({"content": candidate.content, "tools": candidate.tools}))
        if prompt_id in seen:
            skipped["duplicate"] += 1
            continue
        ids = renderer.encode(candidate.content, candidate.tools)
        if len(ids) > MAX_INPUT_TOKENS:
            skipped["overlong"] += 1
            continue
        filename = candidate.source.split("!", 1)[0].split("#", 1)[0]
        result.append({"genre": genre, "prompt_id": prompt_id, "input_ids": ids,
                       "source": candidate.source, "license": licenses[filename]["license"]})
        seen.add(prompt_id)
        if len(result) == count:
            return result
    raise ValueError(f"insufficient valid {genre} prompts: {len(result)}/{count}")


def assign_splits(rows):
    grouped = {}
    seen = set()
    for row in rows:
        if row["prompt_id"] in seen:
            raise ValueError("duplicate prompt ID at split boundary")
        seen.add(row["prompt_id"])
        grouped.setdefault(row["genre"], []).append(row)
    for population in grouped.values():
        ordered = sorted(population, key=lambda row: (digest("split", row["prompt_id"]), row["prompt_id"]))
        heldout = {row["prompt_id"] for row in ordered[:len(ordered) // 10]}
        for row in population:
            row["split"] = "heldout" if row["prompt_id"] in heldout else "build"
    return rows


def length_summary(rows):
    values = sorted(len(row["input_ids"]) for row in rows)
    if not values:
        raise ValueError("cannot summarize an empty prompt set")
    quartiles = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else values * 3
    return {"min": values[0], "q1": quartiles[0], "median": quartiles[1],
            "q3": quartiles[2], "max": values[-1]}


def source_manifest(raw, sources):
    reference = json.loads(Path(sources).read_text(encoding="utf-8"))
    if (not isinstance(reference, dict) or not isinstance(reference.get("sources"), list)
            or any(not isinstance(source, dict) for source in reference["sources"])):
        raise ValueError("public sources metadata must contain a list of source objects")
    result = {}
    for name in (*FILES.values(), *ARCHIVES):
        kind, repo, ref = SOURCE_KEYS[name]
        field = "tag" if kind == "github-tag" else "file"
        matches = [source for source in reference["sources"]
                   if (source.get("kind"), source.get("repo"), source.get(field)) == (kind, repo, ref)]
        if len(matches) != 1:
            raise ValueError(f"missing or ambiguous public source: {name}")
        source = matches[0]
        if not isinstance(source.get("license"), str) or not source["license"].strip():
            raise ValueError(f"missing public source license: {name}")
        expected = source.get("sha256")
        if (not isinstance(expected, str) or len(expected) != 64
                or any(char not in "0123456789abcdef" for char in expected)):
            raise ValueError(f"invalid public source hash: {name}")
        path = Path(raw) / name
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"public source hash mismatch: {name}")
        metadata = {}
        for key in ("kind", "repo", "revision", "file", "tag", "license", "source"):
            if key in source:
                if not isinstance(source[key], str):
                    raise ValueError(f"invalid public source metadata: {name}")
                metadata[key] = source[key]
        result[name] = {**metadata, "local": name, "sha256": actual, "bytes": path.stat().st_size}
    return result


def atomic_write(path, text):
    path = local_output(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix="." + path.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare(raw, sources, model_dir, output_dir):
    cpu_environment()
    raw, sources, model_dir = (Path(path).expanduser() for path in (raw, sources, model_dir))
    output_dir = local_output(output_dir)
    inputs = [sources, model_dir / "tokenizer.json", model_dir / "chat_template.jinja",
              *(raw / name for name in (*FILES.values(), *ARCHIVES))]
    for name in ("prompts.jsonl", "manifest.json"):
        destination = local_output(output_dir / name)
        if any(destination.resolve() == path.resolve()
               or (destination.exists() and path.exists() and destination.samefile(path))
               for path in inputs):
            raise ValueError("output file collides with an input")
    licenses = source_manifest(raw, sources)
    renderer = Renderer.from_model(model_dir)
    tools = validate_tools(TOOLS)
    rows, seen, skipped = [], set(), {}
    for genre, count in COUNTS.items():
        skipped[genre] = Counter()
        candidates = (code_candidates(skipped[genre], raw) if genre == "code"
                      else data_candidates(genre, tools, skipped[genre], raw))
        rows.extend(select(genre, candidates, count, renderer, licenses, seen, skipped[genre],
                           ordered=genre == "code"))
    assign_splits(rows)
    counts = {genre: {"total": sum(row["genre"] == genre for row in rows),
                      **{split: sum(row["genre"] == genre and row["split"] == split for row in rows)
                         for split in ("build", "heldout")}}
              for genre in COUNTS}
    lengths = {genre: length_summary([row for row in rows if row["genre"] == genre]) for genre in COUNTS}
    assets = [{"name": name, "sha256": sha256(path), "bytes": path.stat().st_size}
              for name, path in (("tokenizer.json", model_dir / "tokenizer.json"),
                                 ("chat_template.jinja", model_dir / "chat_template.jinja"),
                                 ("sources.json", sources))]
    tool_bytes = canonical(TOOLS).encode("utf-8")
    assets.append({"name": "swe_tools", "sha256": hashlib.sha256(tool_bytes).hexdigest(),
                   "bytes": len(tool_bytes)})
    text = "".join(canonical(row) + "\n" for row in rows)
    manifest = {"version": 1, "seed": SEED, "counts": counts, "total": len(rows),
                "max_input_tokens": MAX_INPUT_TOKENS, "token_length_quartiles": lengths,
                "sources": list(licenses.values()), "assets": assets,
                "selection_rule": {
                    "order": "ascending sha256('sgmap:select:' + source row locator or archive!member), then locator",
                    "skip": "empty, malformed, duplicate full user-message/tools, or rendered input above 6000 tokens",
                    "prompt_id": "sha256('sgmap:prompt:' + canonical JSON of content and tools)",
                    "split": "per genre: first floor(N/10) in sha256('sgmap:split:' + prompt_id) order heldout; remainder build",
                    "render": "tokenizers; ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True); add_special_tokens=False; add_generation_prompt=True; other template defaults unchanged",
                    "swe": "first user ISSUE block, excluding INSTRUCTIONS; exact embedded TOOLS",
                    "code": {"extensions": sorted(EXTENSIONS), "max_member_bytes": 2 * 1024 * 1024,
                             "chunk": "one contiguous 40-250-line UTF-8 chunk per regular source member; sha256('sgmap:chunk:' + archive!member) determines length and start",
                             "edit": "sha256('sgmap:edit:' + archive!member) modulo instruction count",
                             "instructions": list(EDIT_INSTRUCTIONS)}},
                "skipped": skipped,
                "prompts": {"file": "prompts.jsonl", "rows": len(rows),
                            "bytes": len(text.encode("utf-8")),
                            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}}
    # Do not replace a complete set until all sources, encodings and counts succeed.
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    atomic_write(output_dir / "prompts.jsonl", text)
    atomic_write(output_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = prepare(args.raw, args.sources, args.model_dir, args.output_dir)
    for genre, count in report["counts"].items():
        print(genre, canonical({**count, "tokens": report["token_length_quartiles"][genre]}))
    print("total", report["total"])


if __name__ == "__main__":
    main()
