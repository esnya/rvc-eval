"""Refresh only independent public artifacts in the frozen legacy lock.

This does not resolve or rebuild fairseq's invalid legacy OmegaConf dependency
metadata. It preserves all other entries, including the Python/Torch contract.
"""
import json
from pathlib import Path
from urllib.request import urlopen

from pipenv.vendor.plette import Pipfile

ROOT = Path(__file__).resolve().parents[1]
UPDATES = {
    "default": {
        "lxml": ("6.1.0", "3.8"),
        "setuptools": ("78.1.1", "3.9"),
        "jinja2": ("3.1.6", "3.7"),
        "pyworld": ("0.3.5", None),
        "scipy": ("1.15.3", "3.10"),
    },
    "develop": {"gitpython": ("3.2.1", "3.8")},
}


def main():
    lock_path = ROOT / "Pipfile.lock"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for section, packages in UPDATES.items():
        for name, (version, minimum_python) in packages.items():
            with urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=30) as response:
                metadata = json.load(response)
            if metadata["info"]["version"] != version:
                raise ValueError(f"Unexpected version for {name}")
            hashes = sorted({
                "sha256:" + artifact["digests"]["sha256"]
                for artifact in metadata["urls"] if not artifact.get("yanked")
            })
            if not hashes:
                raise ValueError(f"No release artifacts for {name}=={version}")
            entry = {"hashes": hashes, "index": "pypi", "version": "==" + version}
            if minimum_python:
                entry["markers"] = f"python_version >= '{minimum_python}'"
            lock[section][name] = entry
        lock[section] = dict(sorted(lock[section].items()))
    with (ROOT / "Pipfile").open(encoding="utf-8") as manifest:
        lock["_meta"]["hash"]["sha256"] = Pipfile.load(manifest).get_hash().value
    lock_path.write_text(json.dumps(lock, indent=4) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
