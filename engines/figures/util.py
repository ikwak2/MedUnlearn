"""Private runtime I/O; exported results contain aggregates only."""
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def link(source,target):
    source,target=Path(source),Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        assert os.path.samefile(source,target) or sha(source)==sha(target),str(target)
    else:os.link(source,target)


def verify(root):
    root=Path(root)
    for name,digest in json.loads((root/'inputs.json').read_text()).items():
        assert sha(root/name)==digest,'Changed input: '+name
