#!/usr/bin/env python3
"""
Fast parallel PRADAN downloader using urllib.
Runs 10 parallel HTTP workers with short timeouts and cookie refresh.
"""

import os, sys, json, time, hashlib
import urllib.request
import concurrent.futures
from pathlib import Path

COOKIE_FILE = "/tmp/pradan_cookie.txt"
SOLEXS_LIST = "/tmp/solexs_all_urls.txt"
HEL1OS_LIST = "/tmp/hel1os_all_urls.txt"
OUT = Path("data/raw")
WORKERS = 10
TIMEOUT = 30
BATCH_LOG = 100  # log every N files


def get_cookie():
    try:
        with open(COOKIE_FILE) as f:
            return f.read().strip()
    except:
        return None


def download_one(url, dest_dir):
    """Download a single file. Returns (status, filename)."""
    fname = url.split("/")[-1].split("?")[0]
    dpath = dest_dir / fname
    if dpath.exists() and dpath.stat().st_size > 1000:
        return "exists", fname

    cookie = get_cookie()
    if not cookie:
        return "no-cookie", fname

    req = urllib.request.Request(url)
    req.add_header("Cookie", cookie)
    req.add_header("User-Agent", "Mozilla/5.0")

    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = resp.read()
            if len(data) > 1000:
                with open(dpath, "wb") as f:
                    f.write(data)
                return "ok", fname
            return f"small:{len(data)}", fname
    except Exception as e:
        return f"err:{type(e).__name__}", fname


def download_batch(url_file, dest_dir, label):
    """Download all URLs from file with parallel workers."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    with open(url_file) as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"[{label}] {len(urls)} files, {WORKERS} workers")

    done = len(list(dest_dir.glob("*.zip")))
    stats = {"ok": 0, "exists": done, "fail": 0}

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        fut_map = {ex.submit(download_one, url, dest_dir): url for url in urls}

        for i, fut in enumerate(concurrent.futures.as_completed(fut_map), 1):
            status, fname = fut.result()

            if status == "ok":
                stats["ok"] += 1
            elif status == "exists":
                stats["exists"] += 1
            else:
                stats["fail"] += 1

            if i % BATCH_LOG == 0 or i == len(urls):
                total = stats["ok"] + stats["exists"]
                pct = 100.0 * total / len(urls)
                print(
                    f"  [{label}] {i}/{len(urls)} ({pct:.0f}%) | OK:{stats['ok']} EXIST:{stats['exists']} FAIL:{stats['fail']}"
                )

    print(f"[{label}] DONE: {stats['ok'] + stats['exists']}/{len(urls)} files")
    return stats


def main():
    print("╔══════════════════════════════════════╗")
    print("║  PRADAN Fast Parallel Downloader     ║")
    print(f"║  Workers: {WORKERS}                          ║")
    print("╚══════════════════════════════════════╝")

    cookie = get_cookie()
    if not cookie:
        print("[ERROR] No cookie found in", COOKIE_FILE)
        sys.exit(1)
    print(f"[COOKIE] Loaded ({len(cookie)} bytes)")

    t0 = time.time()

    s = download_batch(SOLEXS_LIST, OUT / "solexs", "SoLEXS")
    h = download_batch(HEL1OS_LIST, OUT / "hel1os", "HEL1OS")

    elapsed = time.time() - t0
    total = s["ok"] + s["exists"] + h["ok"] + h["exists"]
    print(f"\n{'=' * 50}")
    print(f"  TIME: {elapsed:.0f}s")
    print(f"  SoLEXS: {s['ok'] + s['exists']}/{len(open(SOLEXS_LIST).readlines())}")
    print(f"  HEL1OS: {h['ok'] + h['exists']}/{len(open(HEL1OS_LIST).readlines())}")
    print(f"  TOTAL: {total} files")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
