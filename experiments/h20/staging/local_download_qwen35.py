import os, time, json, hashlib, sys
from pathlib import Path

OUT = Path(r"D:/go work/tmp/Qwen3.5-0.8B")
RECEIPT = Path(r"D:/go work/tmp/qwen35_local_download_receipt.json")
OUT.mkdir(parents=True, exist_ok=True)

os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "15"

from huggingface_hub import HfApi, snapshot_download

REPO = "Qwen/Qwen3.5-0.8B"
for attempt in range(1, 51):
    RECEIPT.write_text(json.dumps({"status": "DOWNLOADING", "repo": REPO, "attempt": attempt}))
    try:
        snapshot_download(REPO, local_dir=str(OUT), max_workers=4, token=False)
        break
    except Exception as e:
        print("DOWNLOAD_RETRY", attempt, type(e).__name__, flush=True)
        if attempt == 50:
            RECEIPT.write_text(json.dumps({"status": "FAIL", "repo": REPO, "error_type": type(e).__name__, "attempts": 50}))
            raise
        time.sleep(8)

RECEIPT.write_text(json.dumps({"status": "CHECKING_HASHES", "repo": REPO}))
info = HfApi().model_info(REPO, files_metadata=True, token=False)
checks = []
for entry in info.siblings:
    p = OUT / entry.rfilename
    assert p.is_file(), entry.rfilename
    if entry.size is not None:
        assert p.stat().st_size == entry.size, "size mismatch %s" % entry.rfilename
    lfs = entry.lfs
    if lfs:
        expected = getattr(lfs, "sha256", None) or lfs["sha256"]
        h = hashlib.sha256()
        with p.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(block)
        assert h.hexdigest() == expected, "sha256 mismatch %s" % entry.rfilename
        checks.append({"file": entry.rfilename, "sha256": expected, "bytes": p.stat().st_size})
RECEIPT.write_text(json.dumps({
    "status": "PASS",
    "repo": REPO,
    "resolved_commit": info.sha,
    "path": str(OUT),
    "official_files": len(info.siblings),
    "lfs_hash_checks": checks,
}, indent=2))
print("LOCAL_QWEN35_ALL_HASHES_PASS", flush=True)
