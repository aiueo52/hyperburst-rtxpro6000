"""Small synthetic, CPU-only selection, rendering and provenance check."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest


SPEC = importlib.util.spec_from_file_location(
    "public_prompt_recipe_test", Path(__file__).resolve().parents[1] / "build_prompts.py")
prompts = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prompts
SPEC.loader.exec_module(prompts)


def test_synthetic_selection_render_and_provenance(tmp_path, monkeypatch):
    prompts.cpu_environment()
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    assert prompts.SEED == "sgmap"
    assert prompts.COUNTS == {"en_chat": 300, "ja_chat": 300, "reasoning": 250,
                              "tools": 300, "swe": 200, "code": 350}
    assert sum(prompts.COUNTS.values()) == 1700
    assert prompts.MAX_INPUT_TOKENS == 6000
    assert [tool["function"]["name"] for tool in prompts.TOOLS] == [
        "list_files", "read_file", "search", "edit_file", "write_file"]

    raw, model, output = (tmp_path / name for name in ("raw", "model", "output"))
    raw.mkdir()
    model.mkdir()
    vocabulary = {word: index for index, word in enumerate([
        "[UNK]", "[BOS]", "[EOS]", "user", "Synthetic", "prompt", "tail",
        "no_tools", "list_files", "generation", "default",
    ])}
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]", special_tokens=[("[BOS]", 1), ("[EOS]", 2)])
    tokenizer.save(str(model / "tokenizer.json"))
    (model / "chat_template.jinja").write_text(
        "{{ messages[0].role }} {{ messages[0].content }} "
        "{{ tools[0].function.name if tools is defined else 'no_tools' }} "
        "{{ 'generation' if add_generation_prompt else 'no_generation' }} "
        "{{ 'default' if enable_thinking is not defined else 'overridden' }}",
        encoding="utf-8")

    filename, archive_name = prompts.FILES["ja_chat"], prompts.ARCHIVES[0]
    monkeypatch.setattr(prompts, "FILES", {"ja_chat": filename})
    monkeypatch.setattr(prompts, "ARCHIVES", (archive_name,))
    monkeypatch.setattr(prompts, "COUNTS", {"ja_chat": 10, "code": 1})
    contents = [f"Synthetic prompt {index}\u2028tail" for index in range(12)]
    records = [{"conversations": [{"role": "assistant", "content": "Ignore this turn"},
                                  {"from": "human", "value": f"  {content}  "}]}
               for content in contents]
    (raw / filename).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    source_code = "".join(f"value_{index} = {index}\n" for index in range(80)).encode()
    with tarfile.open(raw / archive_name, "w:gz") as archive:
        for name in ("project/sample.py", "../unsafe.py"):
            member = tarfile.TarInfo(name)
            member.size = len(source_code)
            archive.addfile(member, io.BytesIO(source_code))
    metadata = []
    for name in (filename, archive_name):
        kind, repo, ref = prompts.SOURCE_KEYS[name]
        metadata.append({"kind": kind, "repo": repo,
                         "tag" if kind == "github-tag" else "file": ref,
                         "license": "Synthetic-1.0", "sha256": prompts.sha256(raw / name),
                         "path": "omit-this-field"})
    sources = tmp_path / "sources.json"
    sources.write_text(json.dumps({"sources": metadata}), encoding="utf-8")

    report = prompts.prepare(raw, sources, model, output)
    rows = [json.loads(line) for line in (output / "prompts.jsonl").read_text(
        encoding="utf-8").split("\n") if line]

    def digest(purpose, value):
        return hashlib.sha256(f"sgmap:{purpose}:{value}".encode("utf-8")).hexdigest()

    selected = sorted(range(len(contents)),
                      key=lambda index: (digest("select", f"{filename}#row={index}"),
                                         f"{filename}#row={index}"))[:10]
    assert len(rows) == report["total"] == 11
    for row, index in zip(rows[:10], selected):
        expected = json.dumps({"content": contents[index], "tools": None},
                              ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        assert row["prompt_id"] == digest("prompt", expected)
        assert row["source"] == f"{filename}#row={index}"
        assert row["license"] == "Synthetic-1.0"
        assert row["input_ids"] == tokenizer.encode(
            f"user {contents[index]} no_tools generation default", add_special_tokens=False).ids
        assert set(row) == {"genre", "prompt_id", "input_ids", "source", "license", "split"}
    heldout = min(rows[:10], key=lambda row: (digest("split", row["prompt_id"]), row["prompt_id"]))
    assert [row["prompt_id"] for row in rows if row["split"] == "heldout"] == [heldout["prompt_id"]]
    assert report["counts"]["ja_chat"] == {"total": 10, "build": 9, "heldout": 1}
    assert rows[-1]["source"].startswith(f"{archive_name}!project/sample.py#L")
    assert rows[-1]["split"] == "build"
    renderer = prompts.Renderer.from_model(model)
    assert renderer.encode("Synthetic", prompts.TOOLS) == tokenizer.encode(
        "user Synthetic list_files generation default", add_special_tokens=False).ids

    # Ordered selection must skip duplicates and rendered inputs beyond the exact cap.
    skipped = prompts.Counter()
    licenses = {filename: {"license": "Synthetic-1.0"}}
    accepted = prompts.Candidate(f"{filename}#row=0", "Synthetic")
    accepted_id = prompts.digest("prompt", prompts.canonical({"content": "Synthetic", "tools": None}))
    result = prompts.select("ja_chat", [accepted,
                            prompts.Candidate(f"{filename}#row=1", "Synthetic " * 6001),
                            prompts.Candidate(f"{filename}#row=2", "Synthetic prompt")],
                            1, renderer, licenses, {accepted_id}, skipped, ordered=True)
    assert result[0]["source"].endswith("#row=2")
    assert skipped == {"duplicate": 1, "overlong": 1}

    manifest_text = (output / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)

    def contains_path(value):
        if isinstance(value, dict):
            return "path" in value or any(contains_path(item) for item in value.values())
        return isinstance(value, list) and any(contains_path(item) for item in value)

    assert not contains_path(manifest)
    assert str(tmp_path) not in manifest_text
    assert "omit-this-field" not in manifest_text
    assert {entry["local"] for entry in manifest["sources"]} == {filename, archive_name}
    assert all(entry["license"] == "Synthetic-1.0" for entry in manifest["sources"])
    assert manifest["prompts"]["sha256"] == prompts.sha256(output / "prompts.jsonl")
    assert {asset["name"] for asset in manifest["assets"]} == {
        "tokenizer.json", "chat_template.jinja", "sources.json", "swe_tools"}

    collision_dir = tmp_path / "collision"
    collision_dir.mkdir()
    collision_sources = collision_dir / "manifest.json"
    source_bytes = sources.read_bytes()
    collision_sources.write_bytes(source_bytes)
    with pytest.raises(ValueError, match="collides"):
        prompts.prepare(raw, collision_sources, model, collision_dir)
    assert collision_sources.read_bytes() == source_bytes
    assert not (collision_dir / "prompts.jsonl").exists()

    original = (output / "prompts.jsonl").read_bytes()
    with (raw / filename).open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        prompts.prepare(raw, sources, model, output)
    assert (output / "prompts.jsonl").read_bytes() == original
    assert (output / "manifest.json").read_text(encoding="utf-8") == manifest_text
    link = tmp_path / "linked-output"
    link.symlink_to(output, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        prompts.prepare(raw, sources, model, link)
