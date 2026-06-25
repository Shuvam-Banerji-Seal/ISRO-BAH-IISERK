#!/bin/bash
# Bulk download script — run in background
# Usage: bash data/downloads/bulk_download.sh
set -e

COOKIE_FILE="/tmp/pradan_fresh_cookie.txt"
BASE_DIR="data/raw"
mkdir -p "$BASE_DIR/solexs" "$BASE_DIR/hel1os"

echo "[$(date)] Starting SoLEXS download (749 files)..."
wget -nc --content-disposition --tries=3 --timeout=30 \
    --load-cookies /tmp/pradan_cookies_netscape.txt \
    --save-cookies /tmp/pradan_cookies_netscape.txt \
    --keep-session-cookies \
    -P "$BASE_DIR/solexs" \
    -i /tmp/solexs_all_urls.txt 2>&1 | tail -5
echo "[$(date)] SoLEXS done. Files: $(ls $BASE_DIR/solexs/*.zip 2>/dev/null | wc -l)"

echo "[$(date)] Starting HEL1OS download (2536 files)..."
wget -nc --content-disposition --tries=3 --timeout=30 \
    --load-cookies /tmp/pradan_cookies_netscape.txt \
    --save-cookies /tmp/pradan_cookies_netscape.txt \
    --keep-session-cookies \
    -P "$BASE_DIR/hel1os" \
    -i /tmp/hel1os_all_urls.txt 2>&1 | tail -5
echo "[$(date)] HEL1OS done. Files: $(ls $BASE_DIR/hel1os/*.zip 2>/dev/null | wc -l)"

echo "[$(date)] All downloads complete."
echo "SoLEXS: $(ls $BASE_DIR/solexs/*.zip 2>/dev/null | wc -l) files"
echo "HEL1OS: $(ls $BASE_DIR/hel1os/*.zip 2>/dev/null | wc -l) files"
