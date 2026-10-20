"""Private runtime I/O and provenance for the revision analyses."""
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'inputs.json').read_text())
    for relative, digest in manifest.items():
        assert sha(directory / relative) == digest, relative
    return manifest
