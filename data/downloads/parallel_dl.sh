#!/bin/bash
# Parallel PRADAN downloader — runs multiple wget workers simultaneously
# Usage: FRESH_COOKIE="..." bash data/downloads/parallel_dl.sh
set -euo pipefail

COOKIE="${COOKIE:-$(cat /tmp/pradan_cookie.txt 2>/dev/null || true)}"
export COOKIE
[ -z "$COOKIE" ] && { echo "Need COOKIE env var or /tmp/pradan_cookie.txt"; exit 1; }

WORKERS=8           # parallel downloads
SOLEXS_LIST="/tmp/solexs_all_urls.txt"
HEL1OS_LIST="/tmp/hel1os_all_urls.txt"
OUT_DIR="data/raw"
mkdir -p "$OUT_DIR/solexs" "$OUT_DIR/hel1os"

# ── Split URL file into N chunks ──
split_urls() {
    local input=$1 output_prefix=$2 chunks=$3
    local total
    total=$(wc -l < "$input")
    local lines_per_chunk=$(( (total + chunks - 1) / chunks ))
    split -l "$lines_per_chunk" "$input" "${output_prefix}_"
}

# ── Download worker ──
worker() {
    local url_file=$1 dest=$2 worker_id=$3
    local cookie="$COOKIE"
    local ok=0 fail=0
    
    while IFS= read -r url; do
        [ -z "$url" ] && continue
        
        # Refresh cookie from file every 30 downloads
        if [ $((ok + fail)) -gt 0 ] && [ $(( (ok + fail) % 30 )) -eq 0 ]; then
            local new_cookie
            new_cookie=$(cat /tmp/pradan_cookie.txt 2>/dev/null || echo "$COOKIE")
            [ -n "$new_cookie" ] && cookie="$new_cookie"
        fi
        
        if wget -q -nc --content-disposition --tries=2 --timeout=30 --no-cookies \
            --header "Cookie: $cookie" \
            -P "$dest" "$url" 2>/dev/null; then
            ok=$((ok + 1))
        else
            fail=$((fail + 1))
        fi
    done < "$url_file"
    
    echo "[Worker $worker_id] OK:$ok FAIL:$fail"
}

# ── Main ──
echo "╔══════════════════════════════════════════╗"
echo "║  Parallel PRADAN Downloader              ║"
echo "║  Workers: $WORKERS                           ║"
echo "╚══════════════════════════════════════════╝"
echo ""

echo "[$(date)] Starting SoLEXS ($(wc -l < $SOLEXS_LIST) files)..."
split_urls "$SOLEXS_LIST" "/tmp/solexs_chunk" "$WORKERS"

pids=""
idx=0
for f in /tmp/solexs_chunk_*; do
    worker "$f" "$OUT_DIR/solexs" "$idx" &
    pids="$pids $!"
    idx=$((idx + 1))
done
wait $pids
rm -f /tmp/solexs_chunk_*
echo "[$(date)] SoLEXS done: $(ls $OUT_DIR/solexs/*.zip 2>/dev/null | wc -l) files"
echo ""

echo "[$(date)] Starting HEL1OS ($(wc -l < $HEL1OS_LIST) files)..."
split_urls "$HEL1OS_LIST" "/tmp/hel1os_chunk" "$WORKERS"

pids=""
idx=0
for f in /tmp/hel1os_chunk_*; do
    worker "$f" "$OUT_DIR/hel1os" "$idx" &
    pids="$pids $!"
    idx=$((idx + 1))
done
wait $pids
rm -f /tmp/hel1os_chunk_*
echo "[$(date)] HEL1OS done: $(ls $OUT_DIR/hel1os/*.zip 2>/dev/null | wc -l) files"

echo ""
echo "══════════════════════════════════════════"
echo "  FINAL: $(ls $OUT_DIR/solexs/*.zip 2>/dev/null | wc -l) SoLEXS + $(ls $OUT_DIR/hel1os/*.zip 2>/dev/null | wc -l) HEL1OS files"
echo "══════════════════════════════════════════"
