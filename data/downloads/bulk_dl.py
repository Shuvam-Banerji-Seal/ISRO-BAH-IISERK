#!/usr/bin/env python3
"""
Bulk download SoLEXS & HEL1OS data with cookie refresh support.
Reads cookies from /tmp/pradan_cookies.json, refreshes every 60s if file changes.
"""

import os, sys, json, time, hashlib
from pathlib import Path
import requests

COOKIE_FILE = "/tmp/pradan_cookies.json"
URLS_SOLEXS = "/tmp/solexs_all_urls.txt"
URLS_HEL1OS = "/tmp/hel1os_all_urls.txt"
OUT_DIR = Path("data/raw")
POLL_INTERVAL = 60  # check for cookie refresh every 60s


def load_cookies():
    if not os.path.exists(COOKIE_FILE):
        print("[COOKIE] No cookie file found. Waiting...")
        return None, ""
    with open(COOKIE_FILE) as f:
        data = json.load(f)
    return data, hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


def create_session(cookie_dict):
    s = requests.Session()
    s.cookies.update(cookie_dict)
    s.headers.update(
        {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
    )
    return s


def check_session(session):
    """Verify session is still valid."""
    try:
        r = session.get(
            "https://pradan1.issdc.gov.in/al1/protected/browse.xhtml?id=solexs",
            timeout=15,
            allow_redirects=False,
        )
        return r.status_code == 200
    except:
        return False


def download_file(session, url, dest, timeout=120):
    fname = url.split("/")[-1].split("?")[0]
    dpath = dest / fname
    if dpath.exists() and dpath.stat().st_size > 1000:
        return "exists"
    try:
        r = session.get(url, stream=True, timeout=timeout, allow_redirects=True)
        if r.status_code == 200 and len(r.content) > 1000:
            with open(dpath, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)
            return "ok"
        elif r.status_code == 302:
            return "redirect"
        else:
            return f"HTTP {r.status_code}"
    except Exception as e:
        return f"err:{e}"


def download_batch(session, urls, dest, label):
    dest.mkdir(parents=True, exist_ok=True)
    total = len(urls)
    done = len(list(dest.glob("*.zip")))

    for i, url in enumerate(urls):
        result = download_file(session, url, dest)
        if result == "ok":
            done += 1
        elif result == "redirect":
            return False  # session expired
        if (i + 1) % 20 == 0 or result == "ok":
            print(f"  [{label}] {i + 1}/{total} | {done} files | last: {result}")

    return True


def main():
    print("[DOWNLOAD] Starting bulk downloader with cookie refresh")
    print(f"[DOWNLOAD] Polling {COOKIE_FILE} every {POLL_INTERVAL}s")

    session = None
    cookie_hash = ""
    retry_wait = 5

    for attempt in range(999):  # run forever
        new_data, new_hash = load_cookies()

        if new_data and new_hash != cookie_hash:
            session = create_session(new_data)
            cookie_hash = new_hash
            if check_session(session):
                print(f"[COOKIE] Session valid (hash={new_hash[:8]}...)")
                retry_wait = POLL_INTERVAL
            else:
                print("[COOKIE] Session invalid despite new cookies, waiting...")
                session = None
                time.sleep(10)
                continue

        if session is None:
            print("[WAIT] No valid session yet...")
            time.sleep(10)
            continue

        # Download SoLEXS
        urls = Path(URLS_SOLEXS).read_text().strip().split("\n")
        print(f"\n>>> SoLEXS: {len(urls)} files")
        ok = download_batch(session, urls, OUT_DIR / "solexs", "SoLEXS")
        if not ok:
            print("[SESSION] Expired during SoLEXS download!")
            cookie_hash = ""  # force cookie refresh
            time.sleep(5)
            continue
        print(f"<<< SoLEXS done: {len(list((OUT_DIR / 'solexs').glob('*.zip')))} files")

        # Download HEL1OS
        urls = Path(URLS_HEL1OS).read_text().strip().split("\n")
        print(f"\n>>> HEL1OS: {len(urls)} files")
        ok = download_batch(session, urls, OUT_DIR / "hel1os", "HEL1OS")
        if not ok:
            print("[SESSION] Expired during HEL1OS download!")
            cookie_hash = ""
            time.sleep(5)
            continue
        print(f"<<< HEL1OS done: {len(list((OUT_DIR / 'hel1os').glob('*.zip')))} files")

        # Both complete
        s_count = len(list((OUT_DIR / "solexs").glob("*.zip")))
        h_count = len(list((OUT_DIR / "hel1os").glob("*.zip")))
        print(f"\n{'=' * 50}")
        print(f"  ALL DONE: SoLEXS {s_count} + HEL1OS {h_count} = {s_count + h_count}")
        print(f"{'=' * 50}")
        break

    # Keep running for cookie refresh
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
