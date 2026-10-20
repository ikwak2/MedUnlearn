"""Fetch the exact optional official UEraser source used by the AA experiments."""
from pathlib import Path
import subprocess
from .policy import REPO

AA_COMMIT='6076a275a521b5bcefd3ae4a698b7e6958e2ae7a'


def fetch_aa():
    root=REPO/'third_party/UEraser'
    if not root.exists():
        subprocess.run(['git','clone','https://github.com/lafeat/ueraser.git',str(root)],check=True)
    subprocess.run(['git','checkout','--detach',AA_COMMIT],cwd=root,check=True)
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if actual!=AA_COMMIT:raise ValueError('Unexpected UEraser version')
    return root
