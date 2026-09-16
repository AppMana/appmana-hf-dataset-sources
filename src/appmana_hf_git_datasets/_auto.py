"""Imported by the installed .pth at interpreter startup. Must never raise."""

try:
    from appmana_hf_git_datasets import install

    install()
except Exception:  # noqa: BLE001 - a broken hook must not break every python start
    pass
