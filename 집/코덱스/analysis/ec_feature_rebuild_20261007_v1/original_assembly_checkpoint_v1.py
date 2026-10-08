"""Single writer and fresh-object equality resume; no model or score gate."""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import time
from checkpoint_v1 import atomic
from checkpoint_v2 import start_ticks


def blob(obj):
    return json.dumps(obj, ensure_ascii=False, allow_nan=False)


def canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def checked_existing(path, expected):
    path = Path(path)
    if not path.exists(): return None
    before = path.read_bytes()
    value = json.loads(before.decode('utf-8'))
    if canonical(value) != canonical(expected):
        raise ValueError('fresh recomputation differs; existing artifact preserved: ' + str(path))
    if path.read_bytes() != before: raise ValueError('artifact changed during resume check')
    return before


def persist_pair(folder, predictions, audit_factory):
    """Preflight BOTH existing files before writing either missing one."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if {p.name for p in folder.glob('*.json')} - {'predictions.json', 'audit.json'}:
        raise ValueError('unexpected JSON artifact; preserve and investigate')
    pp, ap = folder / 'predictions.json', folder / 'audit.json'
    existing = checked_existing(pp, predictions)
    predblob = existing if existing is not None else blob(predictions).encode('utf-8')
    audit = audit_factory(hashlib.sha256(predblob).hexdigest())
    checked_existing(ap, audit)
    for path, obj in ((pp, predictions), (ap, audit)):
        if not path.exists(): atomic(path, blob(obj))
        checked_existing(path, obj)
    if hashlib.sha256(pp.read_bytes()).hexdigest() != audit['predictions_sha256']:
        raise ValueError('prediction bytes changed after pair publication')
    return audit


def persist_complete(path, fresh):
    checked_existing(path, fresh)
    if not Path(path).exists(): atomic(path, blob(fresh))
    checked_existing(path, fresh)


@contextmanager
def single_writer(root, registration_sha):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / 'RUN_WRITER_LOCK.json'
    token = {'pid': os.getpid(), 'start_ticks': start_ticks(os.getpid()),
             'created_at': time.time(), 'registration_sha256': registration_sha}
    try: fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        raise RuntimeError('writer lock exists; audit owner before manual recovery, no automatic deletion')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as out:
            json.dump(token, out); out.flush(); os.fsync(out.fileno())
        yield token
    finally:
        if lock.exists() and json.loads(lock.read_text(encoding='utf-8')) == token:
            lock.unlink()
