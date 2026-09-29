# AppMana Hugging Face dataset sources

This small compatibility package lets existing code keep calling
`datasets.load_dataset("organization/name")` while the data comes from a Git
remote or an arbitrary HTTP(S) URL. It installs an import hook through a `.pth`
file, so consumers do not need pipeline-specific adapter code.

## URL sources

Set `APPMANA_HF_SOURCE_MAP` to a JSON object. Keys may be exact dataset IDs or
one-level prefixes ending in `/`. URL sources require the Hugging Face
`datasets` loader format for archives because an archive suffix does not
reliably identify its contents. The package infers `arrow`, `csv`, `json`,
`jsonl`, `parquet`, and `txt` URLs, including compressed forms such as
`.jsonl.gz`.

```shell
export APPMANA_HF_SOURCE_MAP='{
  "catalog/captions": {
    "format": "parquet",
    "url": "https://data.example/captions.parquet"
  },
  "images/": {
    "format": "imagefolder",
    "url": "https://data.example/{name}.zip"
  }
}'
```

Then load the mapped IDs normally:

```python
from datasets import load_dataset

captions = load_dataset("catalog/captions", split="train")
photos = load_dataset("images/product-photos", split="train")
```

Descriptors may use `data_files` instead of `url`, including a split mapping:

```json
{
  "catalog/text": {
    "format": "json",
    "data_files": {
      "train": "https://data.example/train.jsonl",
      "validation": "https://data.example/validation.jsonl"
    }
  }
}
```

Both `url` and `data_files` values may contain `{repo_id}` and `{name}`. The
package delegates downloads, caching, archive extraction, and conversion to
Arrow to Hugging Face `datasets`.

## Git sources

The original `APPMANA_HF_GIT_MAP` remains supported:

```shell
export APPMANA_HF_GIT_MAP='{"appmana/":"https://github.com/AppMana/"}'
```

The package clones matching repositories into `$HF_HOME/git-datasets`, fetches
Git LFS objects, and redirects both `datasets.load_dataset` and
`huggingface_hub.snapshot_download` to the checkout. `APPMANA_HF_GIT_TOKEN`,
`APPMANA_GIT_LFS_URL`, and `APPMANA_GIT_LFS_SECRET` configure private access.

When datasets need different LFS secrets in one process, give each repository
its own. Either map repository IDs (or `prefix/`) to secrets, used with each
repository's committed `.lfsconfig`:

```shell
export APPMANA_GIT_LFS_SECRETS='{"example/first":"<secret>","example/second":"<secret>"}'
```

or point at a directory with one subdirectory per repository, holding a
`client-secret` file and optionally a `url` file that overrides the committed
endpoint. Mounting one Kubernetes Secret with those keys per repository at
`<dir>/<owner>/<name>` produces exactly this layout:

```shell
export APPMANA_GIT_LFS_CREDENTIALS_DIR=/var/run/secrets/git-lfs
# /var/run/secrets/git-lfs/example/first/client-secret
# /var/run/secrets/git-lfs/example/second/client-secret
```

A repository found in either place uses its own secret; the global
`APPMANA_GIT_LFS_URL`/`APPMANA_GIT_LFS_SECRET` pair applies to the others.

The generic map also supports explicit Git descriptors:

```json
{
  "appmana/": {
    "type": "git",
    "base_url": "https://github.com/AppMana"
  }
}
```

## Development and release

Run tests with an existing uv environment:

```shell
uv run --no-sync pytest
```

Build with `uv build`. The GitHub release workflow builds the sdist and wheel,
checks both with Twine, and publishes to PyPI through trusted publishing. The
PyPI project must first trust the `publish.yml` workflow in the GitHub
repository environment named `pypi`.
