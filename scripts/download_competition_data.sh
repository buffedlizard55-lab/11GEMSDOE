#!/usr/bin/env bash
# Fetch the three official competition rasters into data/ and verify them by SHA-256.
#
# WHAT THIS SCRIPT DOES
# ---------------------
# 1. For each pinned file, tries the Dropbox mirror linked from the official
#    competition data tab:
#       https://www.drivendata.org/competitions/306/competition-doe-gems/data/
#    (those mirror URLs are recorded verbatim in config/data_pins.json).
# 2. If that host is unreachable from this machine, falls back to a sparse,
#    read-only checkout of the public transport bridge published by the team's
#    sibling repository and reassembles the byte parts.
# 3. Verifies the SHA-256 of every file against config/data_pins.json and ABORTS
#    on any mismatch. A drifted or truncated download is never left in data/.
#
# USAGE
#   bash scripts/download_competition_data.sh              # competition rasters
#   bash scripts/download_competition_data.sh --with-aux   # + auxiliary channels
#   bash scripts/download_competition_data.sh --dest /mnt/data
#
# EXIT CODES
#   0  every requested file is present and hash-verified
#   1  a download failed or a hash did not match
#
# LIMITATION (documented, not hidden)
# --------------------------------------
# The official DrivenData data tab requires an enrolled account. In restricted
# environments the script therefore falls back to a team-published mirror of the
# same bytes. The bytes are hash-verified, but the *origin* is then a Git mirror
# rather than a fresh pull from DrivenData. Re-run this script on a machine with
# DrivenData credentials and against the official portal to re-anchor the lineage.
# See docs/SOURCES.md entries S4 and S15.

set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PINS="$ROOT/config/data_pins.json"
DEST="$ROOT/data"
WITH_AUX=0
BRIDGE_DIR="${BRIDGE_DIR:-$DEST/.bridge}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-aux) WITH_AUX=1; shift ;;
    --dest) DEST="$2"; shift 2 ;;
    --bridge) BRIDGE_DIR="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$DEST"
[[ -f "$PINS" ]] || { echo "FATAL: missing pin file $PINS" >&2; exit 1; }

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || { echo "FATAL: $PY not found" >&2; exit 1; }

read_pin() { # read_pin <jq-ish python expr name>
  "$PY" - "$PINS" "$1" <<'PYEOF'
import json, sys
pins = json.load(open(sys.argv[1]))
key = sys.argv[2]
rows = pins["files"] if key == "files" else pins["auxiliary"]
idx = int(sys.argv[3])
print(rows[idx][key])
PYEOF
}

n_files=$("$PY" -c "import json,sys;print(len(json.load(open('$PINS'))['files']))")
bridge_ok=0
bridge_repo=""
bridge_commit=""

ensure_bridge() {
  [[ $bridge_ok -eq 1 ]] && return 0
  bridge_repo=$("$PY" -c "import json;print(json.load(open('$PINS'))['bridge_repo'])")
  bridge_commit=$("$PY" -c "import json;print(json.load(open('$PINS'))['bridge_commit'])")
  echo "  fetching transport bridge $bridge_repo @ $bridge_commit"
  rm -rf "$BRIDGE_DIR"
  git init -q "$BRIDGE_DIR"
  git -C "$BRIDGE_DIR" remote add origin "$bridge_repo"
  git -C "$BRIDGE_DIR" config core.sparseCheckout true
  mkdir -p "$BRIDGE_DIR/.git/info"
  printf 'data/bridge/\ndata/aux_bridge/\n' > "$BRIDGE_DIR/.git/info/sparse-checkout"
  git -C "$BRIDGE_DIR" fetch -q --depth 1 origin "$bridge_commit"
  git -C "$BRIDGE_DIR" checkout -q FETCH_HEAD
  bridge_ok=1
}

sha256_of() { sha256sum "$1" | cut -d' ' -f1; }

verify() { # verify <path> <expected_sha> <expected_bytes>
  local path="$1" exp_sha="$2" exp_bytes="$3"
  local got_bytes; got_bytes=$(stat -c%s "$path")
  local got_sha; got_sha=$(sha256_of "$path")
  if [[ "$got_bytes" != "$exp_bytes" ]]; then
    echo "  FAIL size $path: $got_bytes != $exp_bytes" >&2; return 1
  fi
  if [[ "$got_sha" != "$exp_sha" ]]; then
    echo "  FAIL sha256 $path: $got_sha != $exp_sha" >&2; return 1
  fi
  echo "  OK   $(basename "$path")  $got_bytes bytes  sha256 ${got_sha:0:12}..."
  return 0
}

status=0
for ((i = 0; i < n_files; i++)); do
  name=$("$PY" -c "import json,sys;print(json.load(open('$PINS'))['files'][$i]['canonical_name'])")
  url=$("$PY" -c "import json,sys;print(json.load(open('$PINS'))['files'][$i]['primary_url'])")
  bridge_name=$("$PY" -c "import json,sys;print(json.load(open('$PINS'))['files'][$i]['bridge_name'])")
  n_parts=$("$PY" -c "import json,sys;print(json.load(open('$PINS'))['files'][$i]['n_parts'])")
  exp_sha=$("$PY" -c "import json,sys;print(json.load(open('$PINS'))['files'][$i]['sha256'])")
  exp_bytes=$("$PY" -c "import json,sys;print(json.load(open('$PINS'))['files'][$i]['bytes'])")
  target="$DEST/$name"

  if [[ -f "$target" ]] && verify "$target" "$exp_sha" "$exp_bytes" >/dev/null 2>&1; then
    echo "[$name] already present and verified"
    continue
  fi

  echo "[$name] downloading from the official mirror..."
  got=0
  if command -v curl >/dev/null 2>&1 && curl -fsSL --connect-timeout 20 --max-time 3600 \
        -o "$target.part" "$url" 2>/dev/null; then
    got=1
  elif command -v wget >/dev/null 2>&1 && wget -q -T 20 -O "$target.part" "$url" 2>/dev/null; then
    got=1
  fi

  if [[ $got -eq 1 ]]; then
    if verify "$target.part" "$exp_sha" "$exp_bytes"; then
      mv "$target.part" "$target"
      continue
    fi
    echo "  mirror bytes did not match the pin; discarding and trying the transport bridge"
    rm -f "$target.part"
  else
    echo "  official mirror unreachable from this host (expected in egress-restricted sandboxes)"
  fi

  ensure_bridge
  src="$BRIDGE_DIR/data/bridge/$bridge_name"
  if [[ ! -f "$src" ]]; then
    parts=()
    for ((p = 0; p < n_parts; p++)); do
      printf -v pp '%s%03d' "$bridge_name.part-" "$p"
      part="$BRIDGE_DIR/data/bridge/$pp"
      [[ -f "$part" ]] || { echo "FATAL: bridge part missing: $part" >&2; status=1; break; }
      parts+=("$part")
    done
    [[ ${#parts[@]} -gt 0 ]] || continue
    cat "${parts[@]}" > "$target.part"
  else
    cp "$src" "$target.part"
  fi
  if verify "$target.part" "$exp_sha" "$exp_bytes"; then
    mv "$target.part" "$target"
  else
    echo "FATAL: could not obtain a hash-verified $name" >&2
    rm -f "$target.part"
    status=1
  fi
done

if [[ $WITH_AUX -eq 1 ]]; then
  n_aux=$("$PY" -c "import json;print(len(json.load(open('$PINS'))['auxiliary']))")
  for ((i = 0; i < n_aux; i++)); do
    name=$("$PY" -c "import json,sys;print(json.load(open('$PINS'))['auxiliary'][$i]['canonical_name'])")
    path=$("$PY" -c "import json,sys;print(json.load(open('$PINS'))['auxiliary'][$i]['bridge_path'])")
    exp_sha=$("$PY" -c "import json,sys;print(json.load(open('$PINS'))['auxiliary'][$i]['sha256'])")
    exp_bytes=$("$PY" -c "import json,sys;print(json.load(open('$PINS'))['auxiliary'][$i]['bytes'])")
    target="$DEST/$name"
    if [[ -f "$target" ]] && verify "$target" "$exp_sha" "$exp_bytes" >/dev/null 2>&1; then
      echo "[$name] already present and verified"; continue
    fi
    echo "[$name] auxiliary channel, bridge only (not published on the competition data tab)"
    ensure_bridge
    mkdir -p "$(dirname "$target")"
    if [[ -f "$BRIDGE_DIR/$path" ]]; then
      cp "$BRIDGE_DIR/$path" "$target"
    else
      # some versions store the aux payload as a single part file
      alt="$BRIDGE_DIR/${path}.part-000"
      if [[ -f "$alt" ]]; then cp "$alt" "$target"; else
        echo "FATAL: auxiliary payload not found in bridge ($path)" >&2; status=1; continue; fi
    fi
    if ! verify "$target" "$exp_sha" "$exp_bytes"; then
      rm -f "$target"; status=1
    fi
  done
fi

if [[ $bridge_ok -eq 1 ]]; then
  rm -rf "$BRIDGE_DIR"
fi

echo
if [[ $status -eq 0 ]]; then
  echo "All requested rasters are in $DEST and hash-verified against config/data_pins.json"
  echo "Next:  $PY scripts/prepare_data.py"
else
  echo "One or more rasters could not be obtained. Nothing unverified was left in $DEST." >&2
fi
exit $status
