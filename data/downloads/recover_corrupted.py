#!/usr/bin/env python3
"""Re-download corrupted HEL1OS zips and re-extract."""

import asyncio, os, subprocess
from pathlib import Path
from playwright.async_api import async_playwright

RAW = Path("data/raw/hel1os")
PROC = Path("data/processed/hel1os")


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await (await browser.new_context()).new_page()

        # Get fresh cookies from running browser session
        # (reuse the active playwright session via CDP if available)
        print("Using browser session for authenticated downloads...")

        # Find corrupted raw zips (<100KB)
        corrupted = list(RAW.glob("*.zip"))
        corrupted = [f for f in corrupted if f.stat().st_size < 100_000]
        print(f"Found {len(corrupted)} corrupted raw zips to re-download")

        downloaded = 0
        failed = 0

        for zip_path in sorted(corrupted):
            name = zip_path.name
            # Extract date info from filename
            # HLS_YYYYMMDD_HHMMSS_XXXXXsec_lev1_VXXX.zip
            parts = name.split("_")
            date = parts[1]
            Y, M, D = date[:4], date[4:6], date[6:8]
            time_part = parts[2]
            url = f"https://pradan1.issdc.gov.in/al1/protected/downloadData/hel1os/level1/{Y}/{M}/{D}/N00_0000/{name}?hel1os"

            try:
                resp = await page.request.get(url, timeout=120000)
                if resp.status == 200:
                    buf = await resp.body()
                    if len(buf) > 100_000:
                        with open(zip_path, "wb") as f:
                            f.write(buf)

                        # Re-extract
                        proc_dir = PROC / Y / M / D
                        proc_dir.mkdir(parents=True, exist_ok=True)
                        subprocess.run(
                            ["unzip", "-q", "-o", str(zip_path), "-d", str(proc_dir)],
                            capture_output=True,
                        )

                        # Flatten nested dirs
                        for d in list(proc_dir.rglob("*")):
                            if d.is_dir() and d != proc_dir:
                                for f in list(d.iterdir()):
                                    target = proc_dir / f.name
                                    if not target.exists():
                                        f.rename(target)
                                try:
                                    d.rmdir()
                                except:
                                    pass

                        downloaded += 1
                        print(f"  OK {name} ({len(buf) // 1024}KB)")
                    else:
                        print(f"  SKIP {name} ({len(buf)} bytes)")
                        failed += 1
                else:
                    print(f"  FAIL {name} HTTP {resp.status}")
                    failed += 1
            except Exception as e:
                failed += 1
                print(f"  ERR {name}: {e}")

        await browser.close()
        print(f"\nDone: {downloaded} OK, {failed} failed")
        empty = sum(
            1
            for d in PROC.rglob("*")
            if d.is_dir() and not any(d.iterdir()) and len(d.parts) >= 4
        )
        print(f"Remaining empty dirs: {empty}")


asyncio.run(main())
