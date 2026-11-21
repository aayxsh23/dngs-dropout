#!/usr/bin/env bash
# Build the CSD361 submission zip from committed files only: code, results, docs and the report (LaTeX + PDF).
# git archive ships exactly the tracked tree, so data/, output*/, checkpoints, archives and .git never enter the zip.
#   bash scripts/make_submission_zip.sh [git ref, default HEAD] [report PDF, default report/main.pdf] [output zip]
set -euo pipefail
NAME=CSD361_2410110012_2410110128
REF=${1:-HEAD}; PDF=${2:-report/main.pdf}; OUT=${3:-$PWD/$NAME.zip}
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
git archive --format=tar --prefix="$NAME/" "$REF" | tar -x -C "$tmp"
mkdir -p "$tmp/$NAME/report"   # report/ may not be in the archived ref yet
if [ -f "$PDF" ]; then cp "$PDF" "$tmp/$NAME/report/${NAME}_report.pdf"; else echo "WARNING: no report PDF at $PDF; the zip has the LaTeX source only" >&2; fi
bad=$(find "$tmp/$NAME" \( -name '*.pth' -o -name '*.pt' -o -name '*.ply' -o -name '*.tar' -o -name '*.tgz' -o -name '*.zip' \) -print)
if [ -n "$bad" ]; then echo "refusing: forbidden files in the tree:" >&2; echo "$bad" >&2; exit 1; fi
rm -f "$OUT"
python3 - "$tmp" "$NAME" "$OUT" <<'PY'
import os, sys, zipfile
root, name, out = sys.argv[1:]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for d, _, fs in os.walk(os.path.join(root, name)):
        for f in sorted(fs):
            p = os.path.join(d, f)
            z.write(p, os.path.relpath(p, root))
PY
sha256sum "$OUT"
du -h "$OUT" | cut -f1
