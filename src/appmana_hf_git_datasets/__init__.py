"""Resolve Hugging Face repo ids to Git remotes or direct dataset URLs.

Configured entirely by environment variables; a no-op when unconfigured:

- ``APPMANA_HF_GIT_MAP``: JSON object mapping a repo-id prefix to a git base
  URL, e.g. ``{"appmana/": "https://github.com/AppMana/"}``. A repo id
  ``appmana/datasets-spellsource`` then resolves to
  ``https://github.com/AppMana/datasets-spellsource.git``.
- ``APPMANA_HF_SOURCE_MAP``: JSON object mapping an exact repo id or prefix to
  a source descriptor. URL descriptors have ``url`` and ``format`` keys, for
  example ``{"vendor/photos": {"url": "https://example/photos.zip",
  "format": "imagefolder"}}``. Templates may use ``{repo_id}`` and ``{name}``.
  Git descriptors use ``{"type": "git", "base_url": "..."}``.
- ``APPMANA_HF_GIT_TOKEN``: optional token for the git remote (GitHub PAT).
- ``APPMANA_GIT_LFS_URL`` / ``APPMANA_GIT_LFS_SECRET``: optional git-lfs
  endpoint override and its client secret. When unset, the repo's committed
  ``.lfsconfig`` endpoint is used with ``APPMANA_HF_GIT_TOKEN``-less anonymous
  access, so private LFS requires the secret.

Activation is automatic via a ``.pth`` at interpreter startup (see ``_auto``),
which registers post-import hooks so ``datasets.load_dataset`` and
``huggingface_hub.snapshot_download`` see mapped repo ids as local clones.
Clones are cached under ``$HF_HOME/git-datasets`` (or
``~/.cache/huggingface/git-datasets``).
"""

import importlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_ENV_MAP = "APPMANA_HF_GIT_MAP"
_ENV_SOURCE_MAP = "APPMANA_HF_SOURCE_MAP"
_ENV_TOKEN = "APPMANA_HF_GIT_TOKEN"
_ENV_LFS_URL = "APPMANA_GIT_LFS_URL"
_ENV_LFS_SECRET = "APPMANA_GIT_LFS_SECRET"

_installed = False


def repo_map() -> dict:
    raw = os.environ.get(_ENV_MAP, "")
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("%s is not valid JSON; ignoring", _ENV_MAP)
        return {}
    return {k: v for k, v in parsed.items() if isinstance(k, str) and isinstance(v, str)}


def source_map() -> dict:
    """Return validated entries from the generic source map."""
    raw = os.environ.get(_ENV_SOURCE_MAP, "")
    if not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("%s is not valid JSON; ignoring", _ENV_SOURCE_MAP)
        return {}
    if not isinstance(parsed, dict):
        logger.warning("%s must be a JSON object; ignoring", _ENV_SOURCE_MAP)
        return {}
    return {
        key: value
        for key, value in parsed.items()
        if isinstance(key, str) and isinstance(value, (str, dict))
    }


def _mapping_match(repo_id: str, mapping: dict):
    exact = mapping.get(repo_id)
    if exact is not None:
        return repo_id, "", exact
    matches = [key for key in mapping if key.endswith("/") and repo_id.lower().startswith(key.lower())]
    if not matches:
        return None
    prefix = max(matches, key=len)
    name = repo_id[len(prefix):]
    if not name or "/" in name:
        return None
    return prefix, name, mapping[prefix]


def _render(value, *, repo_id: str, name: str):
    if isinstance(value, str):
        return value.format(repo_id=repo_id, name=name)
    if isinstance(value, list):
        return [_render(item, repo_id=repo_id, name=name) for item in value]
    if isinstance(value, dict):
        return {key: _render(item, repo_id=repo_id, name=name) for key, item in value.items()}
    return value


def _infer_format(url: str):
    path = url.split("?", 1)[0].split("#", 1)[0].lower()
    for compression in (".gz", ".bz2", ".xz", ".zst"):
        if path.endswith(compression):
            path = path[: -len(compression)]
            break
    extension = path.rsplit(".", 1)[-1] if "." in path else ""
    return {
        "arrow": "arrow",
        "csv": "csv",
        "json": "json",
        "jsonl": "json",
        "parquet": "parquet",
        "txt": "text",
    }.get(extension)


def resolve(repo_id: str):
    """Return a normalized source descriptor for *repo_id*, or ``None``."""
    if not isinstance(repo_id, str) or "/" not in repo_id:
        return None
    matched = _mapping_match(repo_id, source_map())
    if matched is not None:
        _prefix, name, raw = matched
        if isinstance(raw, str):
            raw = {"url": raw}
        descriptor = _render(dict(raw), repo_id=repo_id, name=name)
        kind = descriptor.get("type")
        if not kind:
            kind = "url" if descriptor.get("url") or descriptor.get("data_files") else "git"
        descriptor["type"] = kind
        if kind == "git" and "url" not in descriptor:
            base = descriptor.get("base_url", "")
            if base:
                descriptor["url"] = base.rstrip("/") + "/" + name + ".git"
        if kind == "url" and not descriptor.get("format"):
            url = descriptor.get("url")
            if isinstance(url, str):
                descriptor["format"] = _infer_format(url)
            if not descriptor.get("format"):
                raise ValueError("URL source for %r requires a datasets format" % repo_id)
        if kind not in {"git", "url"} or (kind == "git" and not descriptor.get("url")):
            raise ValueError("invalid source descriptor for %r" % repo_id)
        return descriptor
    clone_url = _legacy_git_match(repo_id)
    if clone_url is not None:
        return {"type": "git", "url": clone_url}
    return None


def _legacy_git_match(repo_id: str):
    if not isinstance(repo_id, str) or "/" not in repo_id:
        return None
    matched = _mapping_match(repo_id, repo_map())
    if matched is None:
        return None
    _prefix, name, base = matched
    return base.rstrip("/") + "/" + name + ".git"


def match(repo_id: str):
    """Return the git clone URL for a mapped repo id, else None."""
    source = resolve(repo_id)
    if source is not None and source["type"] == "git":
        return source["url"]
    return None


def _cache_root() -> Path:
    hf_home = os.environ.get("HF_HOME") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "huggingface"
    )
    return Path(hf_home) / "git-datasets"


def _run(args, cwd=None, env=None, check=True):
    merged = dict(os.environ)
    if env:
        merged.update(env)
    # never let git prompt
    merged.setdefault("GIT_TERMINAL_PROMPT", "0")
    return subprocess.run(args, cwd=cwd, env=merged, check=check, capture_output=True, text=True)


def _authed_url(url: str) -> str:
    token = os.environ.get(_ENV_TOKEN, "")
    if token and url.startswith("https://") and "@" not in url:
        return "https://x-access-token:" + token + "@" + url[len("https://"):]
    return url


def _configure_lfs(clone_dir: Path):
    """Write repo-local LFS credentials (never global)."""
    cred_file = clone_dir / ".git" / "appmana-lfs-credentials"
    lines = []
    lfs_url = os.environ.get(_ENV_LFS_URL, "")
    lfs_secret = os.environ.get(_ENV_LFS_SECRET, "")
    if lfs_url:
        _run(["git", "config", "lfs.url", lfs_url], cwd=clone_dir)
    if lfs_secret:
        # host of the effective lfs endpoint: env override, repo git config,
        # else the committed .lfsconfig (git config does not include that file)
        effective = lfs_url
        if not effective:
            out = _run(["git", "config", "lfs.url"], cwd=clone_dir, check=False)
            effective = out.stdout.strip()
        if not effective and (clone_dir / ".lfsconfig").exists():
            out = _run(["git", "config", "--file", ".lfsconfig", "lfs.url"], cwd=clone_dir, check=False)
            effective = out.stdout.strip()
        if effective.startswith("https://"):
            host = effective[len("https://"):].split("/", 1)[0]
            lines.append("https://t:%s@%s" % (lfs_secret, host))
    token = os.environ.get(_ENV_TOKEN, "")
    if token:
        lines.append("https://x-access-token:%s@github.com" % token)
    if lines:
        cred_file.write_text("\n".join(lines) + "\n")
        _run(["git", "config", "credential.helper", "store --file=%s" % cred_file], cwd=clone_dir)


def _has_lfs_attributes(clone_dir: Path) -> bool:
    for attributes in clone_dir.rglob(".gitattributes"):
        try:
            if "filter=lfs" in attributes.read_text():
                return True
        except OSError:
            continue
    return False


def _healthy_clone(target: Path) -> bool:
    if not (target / ".git").exists():
        return False
    head = _run(["git", "rev-parse", "--verify", "HEAD"], cwd=target, check=False)
    return head.returncode == 0


def ensure_local(repo_id: str, revision=None) -> str:
    """Clone/update the mapped repo and return the local path."""
    import shutil

    clone_url = match(repo_id)
    if clone_url is None:
        raise ValueError("repo id %r is not mapped" % repo_id)
    target = _cache_root() / repo_id.replace("/", "__")
    # A failed prior clone (bad credentials, interrupted transfer) can leave a
    # directory with no usable HEAD; treat it as absent so it never poisons the
    # cache.
    if target.exists() and not _healthy_clone(target):
        shutil.rmtree(target)
    if not (target / ".git").exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        args = ["git", "clone", "--depth", "1"]
        if revision:
            args += ["--branch", str(revision)]
        args += [_authed_url(clone_url), str(target)]
        try:
            _run(args, env={"GIT_LFS_SKIP_SMUDGE": "1"})
        except subprocess.CalledProcessError as error:
            if target.exists():
                shutil.rmtree(target)
            # a sha revision cannot be cloned via --branch; fall back to full fetch
            if revision:
                _run(["git", "clone", _authed_url(clone_url), str(target)], env={"GIT_LFS_SKIP_SMUDGE": "1"})
            else:
                raise RuntimeError("git clone of %s failed: %s" % (repo_id, error.stderr[-500:])) from error
        if not _healthy_clone(target):
            if target.exists():
                shutil.rmtree(target)
            raise RuntimeError("git clone of %s produced no usable checkout" % repo_id)
        # keep the remote url credential-free
        _run(["git", "remote", "set-url", "origin", clone_url], cwd=target)
        _configure_lfs(target)
    else:
        _configure_lfs(target)
        _run(["git", "fetch", "--depth", "1", _authed_url(clone_url)], cwd=target, check=False)
    if revision:
        checkout = _run(["git", "checkout", str(revision)], cwd=target, check=False)
        if checkout.returncode != 0:
            _run(["git", "fetch", _authed_url(clone_url), str(revision)], cwd=target, check=False)
            _run(["git", "checkout", str(revision)], cwd=target)
    if _has_lfs_attributes(target):
        pull = _run(["git", "lfs", "pull"], cwd=target, check=False)
        if pull.returncode != 0:
            raise RuntimeError("git lfs pull failed for %s: %s" % (repo_id, (pull.stderr or pull.stdout)[-500:]))
    return str(target)


# ---------------------------------------------------------------------------
# patches

def _patch_datasets(module):
    original = module.load_dataset

    def load_dataset(path, *args, **kwargs):
        source = resolve(path)
        if source is not None and source["type"] == "git":
            local = ensure_local(path, revision=kwargs.pop("revision", None))
            kwargs.pop("token", None)
            kwargs.pop("use_auth_token", None)
            return original(local, *args, **kwargs)
        if source is not None and source["type"] == "url":
            kwargs.pop("revision", None)
            data_files = source.get("data_files", source.get("url"))
            kwargs.setdefault("data_files", data_files)
            if source.get("split") is not None:
                kwargs.setdefault("split", source["split"])
            return original(source["format"], *args, **kwargs)
        return original(path, *args, **kwargs)

    load_dataset.__wrapped__ = original
    module.load_dataset = load_dataset

    if hasattr(module, "load_dataset_builder"):
        original_builder = module.load_dataset_builder

        def load_dataset_builder(path, *args, **kwargs):
            source = resolve(path)
            if source is not None and source["type"] == "git":
                local = ensure_local(path, revision=kwargs.pop("revision", None))
                kwargs.pop("token", None)
                return original_builder(local, *args, **kwargs)
            if source is not None and source["type"] == "url":
                kwargs.pop("revision", None)
                kwargs.setdefault("data_files", source.get("data_files", source.get("url")))
                return original_builder(source["format"], *args, **kwargs)
            return original_builder(path, *args, **kwargs)

        load_dataset_builder.__wrapped__ = original_builder
        module.load_dataset_builder = load_dataset_builder
    logger.debug("appmana_hf_git_datasets patched datasets")


def _patch_huggingface_hub(module):
    original = module.snapshot_download

    def snapshot_download(repo_id=None, *args, **kwargs):
        source = resolve(repo_id) if repo_id is not None else None
        if source is not None and source["type"] == "git":
            return ensure_local(repo_id, revision=kwargs.get("revision"))
        return original(repo_id, *args, **kwargs)

    snapshot_download.__wrapped__ = original
    module.snapshot_download = snapshot_download
    logger.debug("appmana_hf_git_datasets patched huggingface_hub")


_PATCHES = {
    "datasets": _patch_datasets,
    "huggingface_hub": _patch_huggingface_hub,
}


class _PostImportFinder:
    """Minimal post-import hook: patches target modules right after they import.

    A target stays pending until its module actually executes: bare
    ``importlib.util.find_spec`` availability probes (transformers does this for
    ``datasets``) must not consume the hook, or the later real import runs
    unpatched.
    """

    def __init__(self):
        self._pending = set(_PATCHES)
        self._in_progress = set()

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self._pending or fullname in self._in_progress:
            return None
        import importlib.util

        self._in_progress.add(fullname)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._in_progress.discard(fullname)
        if spec is None or spec.loader is None:
            return None
        original_exec = spec.loader.exec_module

        def exec_module(module):
            original_exec(module)
            self._pending.discard(fullname)
            try:
                _PATCHES[fullname](module)
            except Exception:  # never break the host import
                logger.exception("appmana_hf_git_datasets failed to patch %s", fullname)

        spec.loader = _LoaderProxy(spec.loader, exec_module)
        return spec


class _LoaderProxy:
    def __init__(self, loader, exec_module):
        self._loader = loader
        self._exec_module = exec_module

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._exec_module(module)

    def __getattr__(self, item):
        return getattr(self._loader, item)


def install():
    """Activate the patches. Safe to call more than once; no-op if unconfigured."""
    global _installed
    if _installed or not (repo_map() or source_map()):
        return
    _installed = True
    for name, patch in _PATCHES.items():
        if name in sys.modules:
            try:
                patch(sys.modules[name])
            except Exception:
                logger.exception("appmana_hf_git_datasets failed to patch already-imported %s", name)
    sys.meta_path.insert(0, _PostImportFinder())
