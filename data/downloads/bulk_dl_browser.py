#!/usr/bin/env python3
"""
Browser-based bulk downloader using Playwright.
Maintains the browser's authenticated session for all downloads.
"""

import asyncio
import json
import os
from pathlib import Path
from playwright.async_api import async_playwright

SOLEXS_URLS = "/tmp/solexs_all_urls.txt"
HEL1OS_URLS = "/tmp/hel1os_all_urls.txt"
OUT_DIR = Path("data/raw")
BATCH_SIZE = 20
MAX_CONCURRENT = 5  # parallel downloads


async def download_file(page, url, dest_dir, semaphore):
    fname = url.split("/")[-1].split("?")[0]
    dpath = dest_dir / fname
    if dpath.exists() and dpath.stat().st_size > 1000:
        return "exists"

    async with semaphore:
        try:
            response = await page.evaluate(f"""
                async () => {{
                    const resp = await fetch('{url}');
                    if (!resp.ok) return {{ok: false, status: resp.status}};
                    const blob = await resp.blob();
                    const reader = new FileReader();
                    return await new Promise((resolve) => {{
                        reader.onload = () => resolve({{ok: true, data: Array.from(new Uint8Array(reader.result))}});
                        reader.readAsArrayBuffer(blob);
                    }});
                }}
            """)
            if response.get("ok"):
                data = bytes(response["data"])
                with open(dpath, "wb") as f:
                    f.write(data)
                return "ok"
            else:
                return f"HTTP {response.get('status')}"
        except Exception as e:
            return f"err:{e}"


async def download_batch(page, urls, dest_dir, label):
    dest_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    total = len(urls)
    done = len(list(dest_dir.glob("*.zip")))

    for i in range(0, total, BATCH_SIZE):
        batch = urls[i : i + BATCH_SIZE]
        tasks = [download_file(page, url, dest_dir, semaphore) for url in batch]
        results = await asyncio.gather(*tasks)

        for r in results:
            if r == "ok":
                done += 1

        if (i + BATCH_SIZE) % 100 == 0 or i == 0:
            print(f"  [{label}] {min(i + BATCH_SIZE, total)}/{total} | {done} files")

        # Small delay between batches
        await asyncio.sleep(0.5)

    return done


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        )
        page = await context.new_page()

        # Log in using existing cookies from JSON
        if os.path.exists("/tmp/pradan_cookies.json"):
            cookies = json.load(open("/tmp/pradan_cookies.json"))
            # Convert to Playwright cookie format
            pw_cookies = []
            for name, value in cookies.items():
                pw_cookies.append(
                    {
                        "name": name,
                        "value": value,
                        "domain": ".pradan1.issdc.gov.in",
                        "path": "/",
                    }
                )
            await context.add_cookies(pw_cookies)

        # Verify session
        await page.goto(
            "https://pradan1.issdc.gov.in/al1/protected/browse.xhtml?id=solexs",
            wait_until="networkidle",
            timeout=30000,
        )
        if "Sign in" in await page.title():
            print(
                "[ERROR] Session expired! Refresh cookies in /tmp/pradan_cookies.json"
            )
            await browser.close()
            return

        print("[SESSION] Authenticated!")

        # Download SoLEXS
        solexs_urls = Path(SOLEXS_URLS).read_text().strip().split("\n")
        print(f"\n>>> SoLEXS: {len(solexs_urls)} files")
        s_count = await download_batch(page, solexs_urls, OUT_DIR / "solexs", "SoLEXS")
        print(f"<<< SoLEXS: {s_count} files")

        # Download HEL1OS
        hel1os_urls = Path(HEL1OS_URLS).read_text().strip().split("\n")
        print(f"\n>>> HEL1OS: {len(hel1os_urls)} files")
        h_count = await download_batch(page, hel1os_urls, OUT_DIR / "hel1os", "HEL1OS")
        print(f"<<< HEL1OS: {h_count} files")

        print(
            f"\n=== TOTAL: SoLEXS {s_count} + HEL1OS {h_count} = {s_count + h_count} files ==="
        )
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
