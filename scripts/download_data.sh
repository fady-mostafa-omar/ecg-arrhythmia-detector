#!/usr/bin/env bash
# Download the MIT-BIH Arrhythmia Database (about 100 MB) into data/mitdb (or the folder given as $1).
# Uses PhysioNet's official open-data copy on AWS (fast); falls back to physionet.org through wfdb.
set -e
OUT="${1:-data/mitdb}"
mkdir -p "$OUT"
if ! command -v aws >/dev/null 2>&1; then pip install -q awscli; fi
aws s3 sync --no-sign-request --only-show-errors s3://physionet-open/mitdb/1.0.0/ "$OUT" || \
  python -c "import sys, wfdb; wfdb.dl_database('mitdb', sys.argv[1])" "$OUT"
echo "MIT-BIH records in $OUT: $(ls "$OUT"/*.dat | wc -l)"
