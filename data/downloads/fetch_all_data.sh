#!/bin/bash
# Download ALL SoLEXS and HEL1OS Level-1 data from PRADAN
# Usage: COOKIE="session_cookie" bash data/downloads/fetch_all_data.sh
set -euo pipefail

COOKIE="${COOKIE:?Set COOKIE env var after logging in}"
BASE="https://pradan1.issdc.gov.in"
ROWS=100
OUT="data/raw"
mkdir -p "$OUT/solexs" "$OUT/hel1os" /tmp/pradan

# Shared curl opts for AJAX calls
CURL="curl -s --connect-timeout 30 --max-time 60"
WGET_OPTS="-N --content-disposition --tries=3 --no-cookies --header Cookie:${COOKIE}"

# ── fetch all URLs for one instrument via AJAX pagination ──
fetch_all_urls() {
    local instr=$1       # "solexs" or "hel1os"
    local total_pages=$2
    local url_file=$3

    echo "[${instr}] Fetching page 1..."

    # Get first page HTML to extract ViewState + initial download links
    $CURL -H "Cookie: $COOKIE" "${BASE}/al1/protected/browse.xhtml?id=${instr}" \
        > "/tmp/pradan/${instr}_p1.html"

    # Extract ViewState from hidden field
    VS=$(grep -oP 'javax\.faces\.ViewState" value="\K[^"]+' "/tmp/pradan/${instr}_p1.html" \
         | head -1 | sed 's/&amp;/\&/g')
    echo "[${instr}] ViewState: ${VS:0:30}..."

    # Extract download URLs from page 1 HTML
    grep -oP "/al1/protected/downloadData/${instr}/[^\"]+\.zip\?${instr}" \
        "/tmp/pradan/${instr}_p1.html" | sort -u > "$url_file"
    echo "[${instr}] Page 1: $(wc -l < "$url_file") URLs"

    # Pages 2..N via AJAX
    for ((pg=1; pg<total_pages; pg++)); do
        local first=$((pg * ROWS))
        local np=$((pg + 1))

        $CURL -H "Cookie: $COOKIE" \
            -H "Faces-Request: partial/ajax" \
            -H "X-Requested-With: XMLHttpRequest" \
            -H "Content-Type: application/x-www-form-urlencoded; charset=UTF-8" \
            -X POST "${BASE}/al1/protected/browse.xhtml?id=${instr}" \
            -d "javax.faces.partial.ajax=true" \
            -d "javax.faces.source=tableForm:lazyDocTable" \
            -d "javax.faces.partial.execute=tableForm:lazyDocTable" \
            -d "javax.faces.partial.render=tableForm:lazyDocTable" \
            -d "tableForm:lazyDocTable=tableForm:lazyDocTable" \
            -d "tableForm:lazyDocTable_pagination=true" \
            -d "tableForm:lazyDocTable_first=${first}" \
            -d "tableForm:lazyDocTable_rows=${ROWS}" \
            -d "tableForm:lazyDocTable_skipChildren=true" \
            -d "tableForm:lazyDocTable_encodeFeature=true" \
            -d "tableForm=tableForm" \
            -d "tableForm:lazyDocTable_rppDD=${ROWS}" \
            -d "javax.faces.ViewState=${VS}" \
            > "/tmp/pradan/${instr}_p${np}.xml"

        # Extract from XML response
        grep -oP "/al1/protected/downloadData/${instr}/[^\"]+\.zip\?${instr}" \
            "/tmp/pradan/${instr}_p${np}.xml" | sort -u >> "$url_file"
        local pc
        pc=$(wc -l < "$url_file")
        echo "[${instr}] Page ${np}/${total_pages}: total ${pc} unique URLs"
    done

    sort -u -o "$url_file" "$url_file"
    echo "[${instr}] DONE: $(wc -l < "$url_file") unique files"
}

# ── batch download from URL list ──
batch_dl() {
    local instr=$1
    local url_file=$2
    local dest=$3
    echo "[${instr}] Starting download of $(wc -l < "$url_file") files..."

    # Use wget with -nc (no-clobber) to skip already-downloaded files
    wget -nc --content-disposition --tries=3 --no-cookies \
        --header "Cookie: ${COOKIE}" \
        -P "$dest" \
        -i "$url_file" 2>&1 | tail -3

    local done_c
    done_c=$(find "$dest" -name '*.zip' -type f 2>/dev/null | wc -l)
    echo "[${instr}] Downloaded: ${done_c} files"
}

# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════

echo "╔══════════════════════════════════════╗"
echo "║  PRADAN Bulk Data Downloader         ║"
echo "║  SoLEXS (750 files, 8 pages)         ║"
echo "║  HEL1OS (~2537 files, 26 pages)      ║"
echo "╚══════════════════════════════════════╝"
echo ""

fetch_all_urls "solexs" 8 "/tmp/solexs_urls.txt"
echo ""
batch_dl "solexs" "/tmp/solexs_urls.txt" "$OUT/solexs"
echo ""

fetch_all_urls "hel1os" 26 "/tmp/hel1os_urls.txt"
echo ""
batch_dl "hel1os" "/tmp/hel1os_urls.txt" "$OUT/hel1os"
echo ""

echo "═══════════════════════════════════════"
echo "FINAL SUMMARY"
echo "  SoLEXS: $(find $OUT/solexs -name '*.zip' | wc -l) files"
echo "  HEL1OS: $(find $OUT/hel1os -name '*.zip' | wc -l) files"
echo "  Total:  $(find $OUT -name '*.zip' | wc -l) files"
echo "═══════════════════════════════════════"
