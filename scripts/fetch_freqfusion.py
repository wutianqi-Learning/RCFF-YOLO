"""Fetch the pinned upstream source without modifying it."""
import hashlib
from pathlib import Path
from urllib.request import urlopen

COMMIT = '3fb0c70637a3c194fb74294d3ce4681958b26241'
SHA256 = '5852efce406681ce6f3f7763ff8e034cab44f93cb0fdda0114130bc4c8c97e42'
URL = f'https://raw.githubusercontent.com/Linwei-Chen/FreqFusion/{COMMIT}/FreqFusion.py'


def main():
    target = Path(__file__).resolve().parents[1] / 'third_party/freqfusion/FreqFusion.py'
    if target.exists():
        payload = target.read_bytes()
    else:
        with urlopen(URL, timeout=60) as response:
            payload = response.read()
    if hashlib.sha256(payload).hexdigest() != SHA256:
        raise RuntimeError('Downloaded source failed its SHA-256 check')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    print(f'FreqFusion ready at {target}')


if __name__ == '__main__':
    main()
