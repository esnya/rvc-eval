# Partial security update for the legacy runtime

**Torch and Hydra remain unresolved. Python 3.10 is end-of-life. This is not a
fully remediated environment.** The README records the existing contract:

> - Python: 3.10.x
> - PyTorch: 2.0.0+cu118
> - Start from the checked-in `Pipfile.lock`. Regenerating the lock file with a newer interpreter or toolchain is a separate compatibility task.

`setup.cfg` also restricts Python to `==3.10.*`. This patch does not change
Python, Torch, TorchAudio, fairseq, Hydra, OmegaConf, NumPy, or Numba versions.
It does not change the model format or inference API.

The independent updates are lxml 6.1.0, setuptools 78.1.1, Jinja2 3.1.6, and
development-only GitPython 3.2.1. These are official fixed releases compatible
with Python 3.10. Setuptools is kept on the minimum release fixing its current
high-severity advisories, retaining the legacy build surface. PyWorld and SciPy,
already imported by the evaluation pipeline, are added to Pipenv explicitly.
The unused Pillow package requirement is removed: no application module or
the pinned submodule's inference-pack sources imports it.

`tools/refresh_security_lock.py` refreshes only those six public release
entries and their official PyPI SHA-256 hashes, then calculates the Pipfile
hash. All other lock entries are preserved. This is a targeted lock refresh,
not a full dependency solve or a successful legacy environment installation.

The legacy resolver constraint is concrete: official fairseq 0.12.2 requires
`hydra-core>=1.0.7,<1.1` and `omegaconf<2.1`; upstream main retains those caps.
Hydra's logging fix is 1.3.6, and the subsequent policy-mutation fix is 1.3.7.
Those releases cannot satisfy fairseq's published requirements. OmegaConf
2.0.6 additionally declares the invalid requirement `PyYAML (>=5.1.*)`, which
modern pip rejects. No package metadata is rewritten and no dependency cap is
forced to make this appear solved.

Torch's weights-only loading bypass is fixed in 2.6.0, while this repository's
legacy generator uses `torch.load` without that protection. Fairseq's HuBERT
loader also calls `torch.load` without an explicit mode. Upgrading Torch alone
changes that default and can reject legacy Namespace/DictConfig checkpoints.
Do not treat the existing model-loading APIs as safe for untrusted model files.

Two migration candidates remain **unapplied**: a matching Torch/TorchAudio
2.7.1 CUDA 11.8 pair plus a reviewed fairseq state-loading adapter, or a move to
converted Transformers HuBERT artifacts on Python 3.11+. The former requires
an explicit decision about trusted legacy pickle loading; the latter changes
the model contract and requires migration of existing artifacts. Neither is
activated by this patch, and no unsafe-load opt-in is made the default.

The new CI workflow installs an explicit public-wheel CPU smoke environment
on Python 3.10. It tests native Praat/PyWorld pitch extraction, the actual VC
tensor pipeline with small generated feature/generator fixtures, lxml default
external-entity handling, local GitPython operations, and Jinja escaping. It
does not install fairseq, initialize the large submodule, download models,
accept dataset licenses, contact an audit service, or run deployment/release
steps. It is deliberately named a partial legacy smoke test.

Python 3.10.22 is the final release and reached end-of-life on 2026-10-01:
[official Python release](https://www.python.org/downloads/release/python-31022/).
Other primary evidence is the
[fairseq release metadata](https://pypi.org/pypi/fairseq/0.12.2/json),
[OmegaConf release metadata](https://pypi.org/pypi/omegaconf/2.0.6/json),
[Hydra logging advisory](https://github.com/advisories/GHSA-c3wx-c55w-pxjq), and
[Torch advisory](https://github.com/advisories/GHSA-53q9-r3pm-6pq6).
