#!/usr/bin/env python3
"""
PRADAN Cookie Grabber + Download Manager

Extracts fresh cookies from the Playwright browser's CDP endpoint
and saves them for parallel wget workers to use.

Usage:
  python3 data/downloads/cookie_grabber.py            # one-shot grab
  python3 data/downloads/cookie_grabber.py --watch     # keep refreshing every 60s
"""

import os, sys, json, time, subprocess, re, hashlib
from pathlib import Path
from urllib.request import urlopen, Request

COOKIE_FILE = "/tmp/pradan_cookie.txt"
BROWSER_WS_ENDPOINT_FILE = "/tmp/pradan_browser_ws.txt"


# Try to find the Playwright browser's CDP/websocket endpoint
def find_browser_ws_endpoint():
    """Find the running Chromium CDP endpoint."""
    # Method 1: Check common Playwright socket paths
    sock_paths = [
        "/tmp/playwright-*",
        "/tmp/.chromium-dev-*",
        "/tmp/.org.chromium.Chromium.*",
    ]

    # Method 2: Look for Chrome/chromium process with --remote-debugging-port
    try:
        output = subprocess.check_output(["ps", "aux"], text=True, timeout=5)
        # Look for --remote-debugging-port or --ws-endpoint
        for line in output.split("\n"):
            if "chromium" in line.lower() or "chrome" in line.lower():
                # Check for debugging port
                m = re.search(r"--remote-debugging-port=(\d+)", line)
                if m:
                    port = m.group(1)
                    return f"http://127.0.0.1:{port}"
                m = re.search(r"--ws-endpoint=([^\s]+)", line)
                if m:
                    return m.group(1)
    except:
        pass

    # Method 3: Read from saved file
    if os.path.exists(BROWSER_WS_ENDPOINT_FILE):
        with open(BROWSER_WS_ENDPOINT_FILE) as f:
            return f.read().strip()

    return None


def get_cookies_via_cdp(ws_endpoint):
    """Extract cookies via Chrome DevTools Protocol."""
    import json
    import asyncio
    import websockets

    async def _get():
        async with websockets.connect(ws_endpoint, max_size=2**20) as ws:
            # Send Cookies.getAllCookies
            msg_id = 1
            await ws.send(
                json.dumps(
                    {"id": msg_id, "method": "Network.getAllCookies", "params": {}}
                )
            )
            resp = await asyncio.wait_for(ws.recv(), timeout=10)
            result = json.loads(resp)

            if "result" in result and "cookies" in result["result"]:
                cookies = result["result"]["cookies"]
                # Filter for PRADAN-related cookies
                relevant = {}
                for c in cookies:
                    if any(d in c["domain"] for d in ["pradan1", "issdc", "idp"]):
                        relevant[c["name"]] = c["value"]

                # Build cookie header string
                keep_order = [
                    "FGTServer",
                    "KEYCLOAK_IDENTITY",
                    "KEYCLOAK_IDENTITY_LEGACY",
                    "KEYCLOAK_SESSION",
                    "KEYCLOAK_SESSION_LEGACY",
                    "AUTH_SESSION_ID",
                    "AUTH_SESSION_ID_LEGACY",
                    "JSESSIONID",
                    "OAuth_Token_Request_State",
                ]
                parts = []
                for k in keep_order:
                    if k in relevant:
                        parts.append(f"{k}={relevant[k]}")
                return "; ".join(parts)
            return None

    try:
        return asyncio.run(_get())
    except Exception as e:
        print(f"[CDP] Error: {e}", file=sys.stderr)
        return None


def verify_cookie(cookie_str):
    """Test if the cookie still works."""
    import urllib.request

    test_url = "https://pradan1.issdc.gov.in/al1/protected/downloadData/solexs/level1/2026/06/N00_0000/AL1_SLX_L1_20260622_v1.0.zip?solexs"

    req = Request(test_url, method="GET")
    req.add_header("Cookie", cookie_str)
    req.add_header("User-Agent", "Mozilla/5.0")

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read()
            return len(content) > 10000  # zip files are >10KB
    except:
        return False


def save_cookie(cookie_str):
    """Save cookie to file."""
    with open(COOKIE_FILE, "w") as f:
        f.write(cookie_str)
    size = os.path.getsize(COOKIE_FILE)
    print(f"[SAVED] Cookie file: {size} bytes")
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PRADAN Cookie Grabber")
    parser.add_argument(
        "--watch", action="store_true", help="Watch mode: refresh every 60s"
    )
    parser.add_argument("--verify", action="store_true", help="Verify existing cookie")
    args = parser.parse_args()

    if args.verify:
        if os.path.exists(COOKIE_FILE):
            with open(COOKIE_FILE) as f:
                cookie = f.read().strip()
            valid = verify_cookie(cookie)
            print(f"[VERIFY] Cookie valid: {valid}")
            if not valid:
                print("[VERIFY] Need fresh cookie!")
            return
        else:
            print("[VERIFY] No cookie file found")
            return

    # One-shot mode
    print("[GRABBER] Looking for browser...")
    ws = find_browser_ws_endpoint()

    if ws:
        print(f"[GRABBER] Found browser at {ws[:60]}...")
        cookie = get_cookies_via_cdp(ws)
        if cookie:
            save_cookie(cookie)
            valid = verify_cookie(cookie)
            print(f"[GRABBER] Cookie valid: {valid}")
            if valid:
                print(f"[GRABBER] Ready for downloads!")
                return
        print("[GRABBER] CDP approach failed, trying manual...")
    else:
        print("[GRABBER] No browser endpoint found via CDP")

    # Manual mode: ask user to paste cookie
    print("\n" + "=" * 60)
    print("  COOKIE NEEDED!")
    print("  Run this in your conversation with the AI:")
    print("    playwright_browser_evaluate: () => {")
    print("      const c = await page.context().cookies();")
    print("      const u = {}; c.forEach(c => { u[c.name] = c.value; });")
    print("      const k = ['FGTServer','KEYCLOAK_IDENTITY','KEYCLOAK_SESSION',")
    print(
        "                 'AUTH_SESSION_ID','JSESSIONID','OAuth_Token_Request_State'];"
    )
    print("      return k.filter(x => u[x]).map(x => x+'='+u[x]).join('; ');")
    print("    }")
    print("=" * 60)

    if args.watch:
        print("[WATCH] Will retry every 60s...")
        while True:
            time.sleep(60)
            ws = find_browser_ws_endpoint()
            if ws:
                cookie = get_cookies_via_cdp(ws)
                if cookie and verify_cookie(cookie):
                    save_cookie(cookie)


if __name__ == "__main__":
    main()
