#!/bin/bash
# Decompress SoLEXS and HEL1OS zip files into structured directory
# Usage: bash data/downloads/decompress.sh [solexs|hel1os]
set -euo pipefail

MODE="${1:-all}"
WORKERS=8
RAW="data/raw"
OUT="data/processed"

decompress_solexs() {
    local src="$RAW/solexs"
    local dst="$OUT/solexs"
    mkdir -p "$dst"
    
    echo "[SoLEXS] Processing $(ls $src/*.zip 2>/dev/null | wc -l) files..."
    
    for zip in "$src"/*.zip; do
        [ -f "$zip" ] || continue
        local fname
        fname=$(basename "$zip" .zip)
        
        # Extract date from filename: AL1_SLX_L1_YYYYMMDD_v1.0.zip
        local date_part="${fname#*_L1_}"
        date_part="${date_part%_v*}"
        local Y="${date_part:0:4}"
        local M="${date_part:4:2}"
        local D="${date_part:6:2}"
        
        local outdir="$dst/$Y/$M/$D"
        mkdir -p "$outdir"
        
        # Skip if already extracted
        if [ -f "$outdir/.done" ]; then
            continue
        fi
        
        # Extract zip
        unzip -q -o "$zip" -d /tmp/solexs_extract_$$ 2>/dev/null
        
        # Find the extracted directory
        local extracted
        extracted=$(find /tmp/solexs_extract_$$ -mindepth 1 -maxdepth 1 -type d | head -1)
        
        if [ -n "$extracted" ]; then
            # Copy SDD1 and SDD2
            for det in SDD1 SDD2; do
                if [ -d "$extracted/$det" ]; then
                    mkdir -p "$outdir/$det"
                    for gzfile in "$extracted/$det"/*.gz; do
                        [ -f "$gzfile" ] || continue
                        local outname
                        outname=$(basename "$gzfile" .gz)
                        gunzip -c "$gzfile" > "$outdir/$det/$outname"
                    done
                fi
            done
            touch "$outdir/.done"
        fi
        
        rm -rf /tmp/solexs_extract_$$
    done
    
    echo "[SoLEXS] Done: $(find $dst -name '.done' | wc -l) days extracted"
}

decompress_hel1os() {
    local src="$RAW/hel1os"
    local dst="$OUT/hel1os"
    mkdir -p "$dst"
    
    echo "[HEL1OS] Processing $(ls $src/*.zip 2>/dev/null | wc -l) files..."
    
    for zip in "$src"/*.zip; do
        [ -f "$zip" ] || continue
        
        # Extract date from path inside zip: YYYY/MM/DD/
        local Y="${zip##*level1/}"
        Y="${Y:0:4}"
        local M="${zip##*level1/????/}"
        M="${M:0:2}"
        local D=$(basename "$zip" | grep -oP '\d{8}' | head -1)
        D="${D:6:2}"
        
        [ -n "$Y" ] && [ -n "$M" ] && [ -n "$D" ] || continue
        
        local outdir="$dst/$Y/$M/$D"
        mkdir -p "$outdir"
        
        # Check marker
        if [ -f "$outdir/.done" ]; then
            continue
        fi
        
        # Extract only light curves and aux (skip evt.fits which is huge)
        unzip -q -o "$zip" \
            "*lightcurve*.fits" \
            "*spectra*.fits" \
            "*.hk.fits" \
            "*gti*.fits" \
            "*.txt" \
            -d "$outdir" 2>/dev/null || true
        
        # Remove the deeply nested directory structure
        # Files come out as YYYY/MM/DD/HLS_xxx/... so flatten
        find "$outdir" -type f -name "*.fits" -o -name "*.txt" | while read -r f; do
            local parent
            parent=$(dirname "$f")
            if [ "$parent" != "$outdir" ]; then
                mv "$f" "$outdir/"
            fi
        done
        
        # Clean up empty subdirectories
        find "$outdir" -type d -empty -delete 2>/dev/null || true
        
        touch "$outdir/.done"
    done
    
    echo "[HEL1OS] Done: $(find $dst -name '.done' | wc -l) days extracted"
}

echo "╔══════════════════════════════════════╗"
echo "║  Data Decompression                  ║"
echo "╚══════════════════════════════════════╝"

case "$MODE" in
    solexs)
        decompress_solexs
        ;;
    hel1os)
        decompress_hel1os
        ;;
    all)
        decompress_solexs
        echo ""
        decompress_hel1os
        ;;
esac

echo ""
echo "=== Summary ==="
echo "SoLEXS: $(find $OUT/solexs -name '.done' | wc -l) days"
echo "HEL1OS: $(find $OUT/hel1os -name '.done' | wc -l) days"
echo "Total space: $(du -sh $OUT 2>/dev/null | cut -f1)"
