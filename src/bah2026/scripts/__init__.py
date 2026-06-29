"""CLI scripts for the BAH 2026 pipeline.

Scripts
-------
run_pipeline          — Full pipeline: nowcast → features → forecast (GPU)
analyze_unused_data   — 10-phase comprehensive data utilization analysis
extract_aux_files     — Extract HK/GTI/evt/dispix from raw HEL1OS zips
build_goes_catalog    — Build GOES flare catalogue from netCDF
audit_data_usage      — Audit which data sources/attributes are used
verify_pipeline       — End-to-end pipeline verification
analysis_deep         — Deep analysis of specific flare events
investigate_czt2      — Investigate CZT2 detector behavior
investigate_detection_failures — Debug missed flare detections
"""
