#!/bin/bash
# Automated PRADAN download manager
# Run: bash data/downloads/download_manager.sh
# Cookie file /tmp/pradan_cookie.txt is refreshed externally

COOKIE_FILE="/tmp/pradan_cookie.txt"
SOLEXS_LIST="/tmp/solexs_all_urls.txt"
HEL1OS_LIST="/tmp/hel1os_all_urls.txt"
OUT_DIR="data/raw"
mkdir -p "$OUT_DIR/solexs" "$OUT_DIR/hel1os"

get_cookie() {
    cat "$COOKIE_FILE" 2>/dev/null || echo ""
}

download_batch() {
    local list=$1
    local dest=$2
    local label=$3
    local url_list
    
    url_list=$(cat "$list" 2>/dev/null)
    [ -z "$url_list" ] && return 0
    
    local total
    total=$(echo "$url_list" | wc -l)
    
    echo "[$(date)] $label: $total files to download"
    
    # Use wget with --load-cookies approach - read cookie fresh for each invocation
    local cookie
    cookie=$(get_cookie)
    [ -z "$cookie" ] && { echo "[ERROR] No cookie available"; return 1; }
    
    # Download in batches of 50 to avoid cookie timeout
    local batch_size=50
    local i=0
    
    echo "$url_list" | while read -r url; do
        [ -z "$url" ] && continue
        i=$((i + 1))
        
        wget -nc --content-disposition --tries=2 --timeout=60 --no-cookies \
            --header "Cookie: $cookie" \
            -P "$dest" \
            "$url" 2>/dev/null
        
        # Every batch_size files, check if cookie needs refresh
        if [ $((i % batch_size)) -eq 0 ]; then
            local done_count
            done_count=$(find "$dest" -name '*.zip' -type f 2>/dev/null | wc -l)
            echo "[$(date)] $label: $i/$total | $done_count files done"
            
            # Refresh cookie
            cookie=$(get_cookie)
            [ -z "$cookie" ] && { echo "[ERROR] Cookie expired!"; return 1; }
        fi
    done
    
    local final_count
    final_count=$(find "$dest" -name '*.zip' -type f 2>/dev/null | wc -l)
    echo "[$(date)] $label: FINAL - $final_count files"
    return 0
}

echo "╔══════════════════════════════════════════╗"
echo "║  PRADAN Download Manager                  ║"
echo "║  Cookie: $COOKIE_FILE                      ║"
echo "║  Refresh cookie every ~60s via grabber     ║"
echo "╚══════════════════════════════════════════╝"

# Main loop - retry if cookie expires
while true; do
    cookie=$(get_cookie)
    if [ -z "$cookie" ]; then
        echo "[WAIT] No cookie yet..."
        sleep 10
        continue
    fi
    
    echo "[START] $(date)"
    
    download_batch "$SOLEXS_LIST" "$OUT_DIR/solexs" "SoLEXS"
    local s_exit=$?
    
    download_batch "$HEL1OS_LIST" "$OUT_DIR/hel1os" "HEL1OS"
    local h_exit=$?
    
    s_count=$(find "$OUT_DIR/solexs" -name '*.zip' -type f | wc -l)
    h_count=$(find "$OUT_DIR/hel1os" -name '*.zip' -type f | wc -l)
    echo "[DONE] SoLEXS: $s_count | HEL1OS: $h_count | Total: $((s_count + h_count))"
    
    # Check if all files downloaded
    solexs_total=$(wc -l < "$SOLEXS_LIST")
    hel1os_total=$(wc -l < "$HEL1OS_LIST")
    
    if [ "$s_count" -ge "$solexs_total" ] && [ "$h_count" -ge "$hel1os_total" ]; then
        echo "ALL DONE!"
        break
    fi
    
    echo "[WAIT] Resuming in 30s..."
    sleep 30
done
