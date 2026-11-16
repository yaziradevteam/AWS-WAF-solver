import hashlib
import json
import os
import shutil
import threading
import time

DATA_DIR = "dataset"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
LABELS_FILE = os.path.join(DATA_DIR, "labels.json")
LOG_FILE = os.path.join(DATA_DIR, "log.jsonl")
REVIEW_DIR = os.path.join(DATA_DIR, "review")
REVIEW_DONE_DIR = os.path.join(DATA_DIR, "review_done")

TARGETS = ["bag", "curtain", "bed", "chair", "bucket", "hat", "clock"]

_lock = threading.RLock()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def store_images(paths):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    hashes = []
    for p in paths:
        h = sha256_file(p)
        dst = os.path.join(IMAGES_DIR, h + ".jpg")
        if not os.path.exists(dst):
            shutil.copy(p, dst)
        hashes.append(h)
    return hashes


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    if not data:
        return {}
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        obj, _ = json.JSONDecoder().raw_decode(data)
        return obj


def load_labels():
    with _lock:
        if os.path.exists(LABELS_FILE):
            try:
                return _read_json(LABELS_FILE)
            except Exception:
                pass
        return _rebuild_from_log()


def save_labels(labels):
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = LABELS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(labels, f)
    os.replace(tmp, LABELS_FILE)


def _append_event(ev):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(ev) + "\n")


def _rebuild_from_log():
    labels = {}
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                hashes = ev.get("h")
                target = ev.get("t")
                if not hashes or not target:
                    continue
                picked = set(ev.get("p", []))
                for i, h in enumerate(hashes):
                    rec = labels.setdefault(h, {})
                    is_pos = i in picked
                    rec[f"has_{target}"] = is_pos
                    if is_pos:
                        rec["class"] = target
                    rec["seen"] = rec.get("seen", 0) + 1
    return labels


def record_solve(image_paths, target, picked, success):
    hashes = store_images(image_paths)
    with _lock:
        if success:
            _append_event({"t": target, "h": hashes, "p": picked, "ok": True, "ts": time.time()})
            labels = load_labels()
            picked_set = set(picked)
            changed = 0
            for i, h in enumerate(hashes):
                rec = labels.get(h, {})
                is_pos = i in picked_set
                key = f"has_{target}"
                if rec.get(key) != is_pos:
                    rec[key] = is_pos
                    changed += 1
                if is_pos:
                    rec["class"] = target
                rec["seen"] = rec.get("seen", 0) + 1
                labels[h] = rec
            save_labels(labels)
            return {"hashes": hashes, "labeled": changed}
        else:
            rid = sha256_bytes("".join(hashes).encode())[:12]
            entry = {"id": rid, "target": target, "hashes": hashes, "picked": picked, "time": time.time()}
            os.makedirs(REVIEW_DIR, exist_ok=True)
            with open(os.path.join(REVIEW_DIR, rid + ".json"), "w", encoding="utf-8") as f:
                json.dump(entry, f)
            return {"hashes": hashes, "queued_review": rid}


def apply_review_labels(rid, target, correct_indices):
    path = os.path.join(REVIEW_DIR, rid + ".json")
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        entry = json.load(f)
    with _lock:
        _append_event({"t": target, "h": entry["hashes"], "p": correct_indices, "ok": True, "review": rid, "ts": time.time()})
        labels = load_labels()
        correct = set(correct_indices)
        for i, h in enumerate(entry["hashes"]):
            rec = labels.get(h, {})
            is_pos = i in correct
            rec[f"has_{target}"] = is_pos
            if is_pos:
                rec["class"] = target
            labels[h] = rec
        save_labels(labels)
    os.makedirs(REVIEW_DONE_DIR, exist_ok=True)
    os.replace(path, os.path.join(REVIEW_DONE_DIR, rid + ".json"))
    return True


def lookup_grid(hashes, target):
    labels = load_labels()
    return [labels.get(h, {}).get(f"has_{target}") for h in hashes]


def known_fraction(hashes, target):
    labels = load_labels()
    n = sum(1 for h in hashes if f"has_{target}" in labels.get(h, {}))
    return n / len(hashes) if hashes else 0.0


def stats():
    labels = load_labels()
    n_images = len(os.listdir(IMAGES_DIR)) if os.path.isdir(IMAGES_DIR) else 0
    per_class = {t: 0 for t in TARGETS}
    for rec in labels.values():
        c = rec.get("class")
        if c in per_class:
            per_class[c] += 1
    return {"images": n_images, "labeled_records": len(labels), "per_class": per_class}
