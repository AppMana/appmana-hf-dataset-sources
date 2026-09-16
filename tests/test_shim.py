import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture()
def fixture_repo(tmp_path):
    """A local bare git 'remote' holding a tiny imagefolder dataset."""
    work = tmp_path / "work"
    (work / "train" / "images").mkdir(parents=True)
    # 1x1 red png
    png = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108020000009077"
        "53de0000000c49444154089963f8cfc000000301010018dd8db00000000049"
        "454e44ae426082"
    )
    (work / "train" / "images" / "a.png").write_bytes(png)
    (work / "train" / "metadata.jsonl").write_text(
        json.dumps({"file_name": "images/a.png", "text": "a red pixel"}) + "\n"
    )
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=work, check=True, env=env)
    subprocess.run(["git", "add", "-A"], cwd=work, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=work, check=True, env=env)
    bare = tmp_path / "remote" / "mini.git"
    bare.parent.mkdir(parents=True)
    subprocess.run(["git", "clone", "-q", "--bare", str(work), str(bare)], check=True, env=env)
    return bare.parent


def _configure(monkeypatch, tmp_path, fixture_repo):
    monkeypatch.setenv("APPMANA_HF_GIT_MAP", json.dumps({"testorg/": fixture_repo.as_uri() + "/"}))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    import appmana_hf_git_datasets as shim

    shim._installed = False
    shim.install()
    return shim


def test_match_and_ensure_local(monkeypatch, tmp_path, fixture_repo):
    shim = _configure(monkeypatch, tmp_path, fixture_repo)
    assert shim.match("testorg/mini") is not None
    assert shim.match("other/mini") is None
    assert shim.match("testorg/nested/deep") is None
    local = shim.ensure_local("testorg/mini")
    assert (Path(local) / "train" / "metadata.jsonl").exists()
    # second call hits the cache
    assert shim.ensure_local("testorg/mini") == local


def test_load_dataset_transparent(monkeypatch, tmp_path, fixture_repo):
    _configure(monkeypatch, tmp_path, fixture_repo)
    datasets = pytest.importorskip("datasets")
    ds = datasets.load_dataset("testorg/mini", split="train")
    assert ds.num_rows == 1
    assert ds[0]["text"] == "a red pixel"
    assert ds[0]["image"].size == (1, 1)


def test_snapshot_download_transparent(monkeypatch, tmp_path, fixture_repo):
    shim = _configure(monkeypatch, tmp_path, fixture_repo)
    huggingface_hub = pytest.importorskip("huggingface_hub")
    path = huggingface_hub.snapshot_download("testorg/mini")
    assert path == shim.ensure_local("testorg/mini")


def test_unmapped_untouched(monkeypatch, tmp_path, fixture_repo):
    _configure(monkeypatch, tmp_path, fixture_repo)
    import appmana_hf_git_datasets as shim

    with pytest.raises(ValueError):
        shim.ensure_local("other/thing")


def test_noop_without_config(monkeypatch):
    monkeypatch.delenv("APPMANA_HF_GIT_MAP", raising=False)
    monkeypatch.delenv("APPMANA_HF_SOURCE_MAP", raising=False)
    import appmana_hf_git_datasets as shim

    shim._installed = False
    shim.install()
    assert shim._installed is False


def test_url_source_template_resolution(monkeypatch):
    monkeypatch.delenv("APPMANA_HF_GIT_MAP", raising=False)
    monkeypatch.setenv(
        "APPMANA_HF_SOURCE_MAP",
        json.dumps(
            {
                "public/": {
                    "url": "https://datasets.example/{name}.parquet",
                    "format": "parquet",
                }
            }
        ),
    )
    import appmana_hf_git_datasets as shim

    assert shim.resolve("public/captions") == {
        "type": "url",
        "url": "https://datasets.example/captions.parquet",
        "format": "parquet",
    }
    assert shim.resolve("public/nested/captions") is None
    assert shim.match("public/captions") is None


def test_url_source_infers_common_hf_format(monkeypatch):
    monkeypatch.delenv("APPMANA_HF_GIT_MAP", raising=False)
    monkeypatch.setenv(
        "APPMANA_HF_SOURCE_MAP",
        json.dumps({"public/captions": "https://datasets.example/captions.jsonl.gz?download=1"}),
    )
    import appmana_hf_git_datasets as shim

    assert shim.resolve("public/captions")["format"] == "json"


def test_url_source_dispatches_to_hf_format(monkeypatch):
    monkeypatch.delenv("APPMANA_HF_GIT_MAP", raising=False)
    monkeypatch.setenv(
        "APPMANA_HF_SOURCE_MAP",
        json.dumps(
            {
                "vendor/captions": {
                    "format": "json",
                    "data_files": {
                        "train": "https://datasets.example/train.jsonl",
                        "validation": "https://datasets.example/validation.jsonl",
                    },
                }
            }
        ),
    )
    import appmana_hf_git_datasets as shim

    calls = []

    def fake_load(path, *args, **kwargs):
        calls.append((path, args, kwargs))
        return "dataset"

    module = SimpleNamespace(load_dataset=fake_load, load_dataset_builder=fake_load)
    shim._patch_datasets(module)

    assert module.load_dataset("vendor/captions", split="train", revision="ignored") == "dataset"
    assert calls == [
        (
            "json",
            (),
            {
                "split": "train",
                "data_files": {
                    "train": "https://datasets.example/train.jsonl",
                    "validation": "https://datasets.example/validation.jsonl",
                },
            },
        )
    ]


def test_url_source_requires_explicit_hf_format(monkeypatch):
    monkeypatch.delenv("APPMANA_HF_GIT_MAP", raising=False)
    monkeypatch.setenv(
        "APPMANA_HF_SOURCE_MAP",
        json.dumps({"vendor/data": {"url": "https://datasets.example/data.zip"}}),
    )
    import appmana_hf_git_datasets as shim

    with pytest.raises(ValueError, match="requires a datasets format"):
        shim.resolve("vendor/data")


def test_poisoned_cache_recovers(monkeypatch, tmp_path, fixture_repo):
    shim = _configure(monkeypatch, tmp_path, fixture_repo)
    # simulate a failed prior clone: .git dir exists but no usable HEAD
    poisoned = shim._cache_root() / "testorg__mini"
    (poisoned / ".git").mkdir(parents=True)
    local = shim.ensure_local("testorg/mini")
    assert (Path(local) / "train" / "metadata.jsonl").exists()


def test_find_spec_probe_does_not_consume_hook(monkeypatch, tmp_path, fixture_repo):
    """A bare find_spec availability probe (as transformers does for datasets)
    must leave the post-import patch armed for the later real import."""
    import importlib.util
    import textwrap

    shim = _configure(monkeypatch, tmp_path, fixture_repo)
    modules_dir = tmp_path / "probe-modules"
    modules_dir.mkdir()
    (modules_dir / "probe_target_mod.py").write_text("value = 1\n")
    monkeypatch.syspath_prepend(str(modules_dir))
    patched = []
    monkeypatch.setitem(shim._PATCHES, "probe_target_mod", lambda m: patched.append(m.__name__))
    finder = shim._PostImportFinder()
    monkeypatch.setattr(sys, "meta_path", [finder] + sys.meta_path)

    spec = importlib.util.find_spec("probe_target_mod")  # probe only, never executed
    assert spec is not None
    assert patched == []
    import probe_target_mod  # noqa: F401

    assert patched == ["probe_target_mod"]
    del sys.modules["probe_target_mod"]
