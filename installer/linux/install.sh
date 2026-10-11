#!/usr/bin/env bash
# Voxprint AI Movie Dubber - Linux installer (per user, no root needed).
#
#   ./install.sh                  install into ~/.local (program, shared Python 3.14 runtime, launcher, menu entry, .vxdub type)
#   ./install.sh --prefix DIR     put the program in DIR (default ~/.local/share/voxprint-movie-dubber)
#   ./install.sh --uninstall      remove the program and drop this app's shared-resource references
#   ./install.sh --skip-gpu-check CI ONLY: do not check the NVIDIA driver and GPU. GitHub runners have no GPU.
#   ./install.sh --skip-models    CI ONLY: do not download the AI models. A user install always downloads them.
#
# Requirements: x86_64 Linux (a distribution released in 2025 or later), an NVIDIA GeForce RTX 40-series GPU or newer
# (compute capability 8.9+) and an NVIDIA driver from the 600 branch or newer, about 25 GB of free disk space and an
# internet connection. The installer puts Python 3.14, PyTorch 2.11.0+cu130, torchaudio, torchcodec, the CTranslate2
# CUDA 12 libraries, ffmpeg and the default-pipeline models into ~/.local/share/voxprint/shared before the first launch.
# An interrupted download is continued from a .partial file.
set -euo pipefail

APP_ID="voxprint-movie-dubber"
REF_APP="movie-dubber"
MIME_TYPE="application/vnd.voxprint.dub+zip"
MIME_ICON="application-vnd.voxprint.dub+zip"
MIN_DRIVER=600
MIN_CC="8.9"
RUNTIME_VERSION="py3.14-torch2.11-cu130"
FFMPEG_VERSION="n8.1"
FFMPEG_ARCHIVE="ffmpeg-n8.1-latest-linux64-lgpl-8.1.tar.xz"
FFMPEG_BASE="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
BIN_DIR="$HOME/.local/bin"
PREFIX="$DATA_HOME/$APP_ID"
VX_HOME="$DATA_HOME/voxprint"
SHARED="$VX_HOME/shared"
RT="$SHARED/runtimes/$RUNTIME_VERSION"
ENV_DIR="$RT/env"
PY="$ENV_DIR/bin/python"
FF_DIR="$SHARED/ffmpeg/$FFMPEG_VERSION"
SKIP_GPU=0
SKIP_MODELS=0
UNINSTALL=0

say() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

usage() { sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-gpu-check) SKIP_GPU=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    --uninstall) UNINSTALL=1 ;;
    --prefix) [ $# -ge 2 ] || die "--prefix needs a folder"; PREFIX="$2"; shift ;;
    --prefix=*) PREFIX="${1#--prefix=}" ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown option: $1" ;;
  esac
  shift
done

refresh_desktop() {
  command -v update-mime-database >/dev/null 2>&1 && update-mime-database "$DATA_HOME/mime" >/dev/null 2>&1 || true
  command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$DATA_HOME/applications" >/dev/null 2>&1 || true
  command -v gtk-update-icon-cache >/dev/null 2>&1 && gtk-update-icon-cache -q -t "$DATA_HOME/icons/hicolor" >/dev/null 2>&1 || true
  return 0
}

add_ref() {
  python3 "$PREFIX/app/dubber/infra/shared_manifest.py" add --home "$VX_HOME" --app "$REF_APP" "$@"
}

download_resumable() {
  url="$1"
  dest="$2"
  partial="${dest}.partial"
  mkdir -p "$(dirname "$dest")"
  if [ -f "$partial" ]; then
    say "Resuming download ($(wc -c < "$partial") bytes already saved)"
  else
    say "Downloading $(basename "$dest")"
  fi
  curl -fL --retry 5 --retry-all-errors -C - -o "$partial" "$url"
  mv -f "$partial" "$dest"
}

install_ffmpeg() {
  mkdir -p "$FF_DIR"
  if [ -x "$FF_DIR/ffmpeg" ] && "$FF_DIR/ffmpeg" -version >/dev/null 2>&1; then
    say "[4/5] Reusing the shared ffmpeg"
    return 0
  fi
  archive="$FF_DIR/$FFMPEG_ARCHIVE"
  say "[4/5] Downloading ffmpeg (LGPL build)"
  download_resumable "$FFMPEG_BASE/$FFMPEG_ARCHIVE" "$archive"
  sums="$(mktemp)"
  download_resumable "$FFMPEG_BASE/checksums.sha256" "$sums"
  want="$(awk -v name="$FFMPEG_ARCHIVE" '$0 ~ name { print $1; exit }' "$sums")"
  have="$(sha256sum "$archive" | awk '{ print $1 }')"
  rm -f "$sums"
  if [ -z "$want" ] || [ "$want" != "$have" ]; then
    rm -f "$archive"
    die "ffmpeg download is corrupt (SHA-256 mismatch)"
  fi
  tmp="$(mktemp -d)"
  tar -xJf "$archive" -C "$tmp"
  find "$tmp" -type f \( -name ffmpeg -o -name ffprobe -o -name LICENSE.txt \) -exec cp {} "$FF_DIR/" \;
  if [ -f "$FF_DIR/LICENSE.txt" ]; then
    mv -f "$FF_DIR/LICENSE.txt" "$FF_DIR/FFMPEG-LICENSE.txt"
  fi
  chmod 755 "$FF_DIR/ffmpeg" "$FF_DIR/ffprobe"
  rm -rf "$tmp" "$archive"
  "$FF_DIR/ffmpeg" -version >/dev/null 2>&1 || die "ffmpeg was extracted but does not run"
}

if [ "$UNINSTALL" = 1 ]; then
  say "Removing Voxprint AI Movie Dubber from $PREFIX"
  if [ -f "$PREFIX/app/dubber/infra/shared_manifest.py" ]; then
    python3 "$PREFIX/app/dubber/infra/shared_manifest.py" release --home "$VX_HOME" --app "$REF_APP" || true
  fi
  rm -rf "$PREFIX"
  rm -f "$BIN_DIR/$APP_ID" "$DATA_HOME/applications/$APP_ID.desktop" "$DATA_HOME/mime/packages/$APP_ID.xml" \
        "$DATA_HOME/icons/hicolor/256x256/apps/$APP_ID.png" "$DATA_HOME/icons/hicolor/256x256/mimetypes/$MIME_ICON.png"
  refresh_desktop
  say "Done. Shared components this program was the last user of were removed."
  exit 0
fi

[ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ] || die "this package is for x86_64 Linux"
[ -f "$HERE/main.py" ] && [ -f "$HERE/requirements.txt" ] || die "run install.sh from the unpacked package folder"

# 1. GPU and driver
if [ "$SKIP_GPU" = 1 ]; then
  say "[1/5] Skipping the NVIDIA driver and GPU check (--skip-gpu-check, CI only)"
else
  command -v nvidia-smi >/dev/null 2>&1 || die "no NVIDIA driver found (nvidia-smi is missing). Install an NVIDIA driver from the $MIN_DRIVER branch or newer. NVIDIA driver downloads: https://www.nvidia.com/Download/index.aspx"
  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')"
  [ -n "$driver" ] || die "nvidia-smi did not report a driver version. NVIDIA driver downloads: https://www.nvidia.com/Download/index.aspx"
  major="${driver%%.*}"
  [ "$major" -ge "$MIN_DRIVER" ] 2>/dev/null || die "NVIDIA driver $driver is too old: the $MIN_DRIVER branch or newer is required. NVIDIA driver downloads: https://www.nvidia.com/Download/index.aspx"
  ok_gpu=""
  while IFS=, read -r name cc; do
    name="$(echo "$name" | sed 's/^ *//;s/ *$//')"; cc="$(echo "$cc" | tr -d ' ')"
    if [ -n "$cc" ] && [ "$cc" != "[N/A]" ] && awk -v a="$cc" -v b="$MIN_CC" 'BEGIN{exit !(a+0 >= b+0)}'; then
      ok_gpu="$name (compute capability $cc)"; break
    fi
  done < <(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null || true)
  [ -n "$ok_gpu" ] || die "no supported GPU: an NVIDIA GeForce RTX 40-series or newer (compute capability $MIN_CC+) is required. NVIDIA driver downloads: https://www.nvidia.com/Download/index.aspx"
  say "[1/5] GPU: $ok_gpu, driver $driver"
fi

# 2. uv
UV="$(command -v uv || true)"
if [ -z "$UV" ]; then
  say "[1/5] Downloading uv (package manager)"
  mkdir -p "$PREFIX/uv"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$PREFIX/uv" UV_NO_MODIFY_PATH=1 sh >/dev/null
  UV="$PREFIX/uv/uv"
fi
[ -x "$UV" ] || die "uv is not available"

# 3. program files
say "[2/5] Copying the program to $PREFIX/app"
mkdir -p "$PREFIX"
rm -rf "$PREFIX/app.new"
mkdir -p "$PREFIX/app.new"
for item in main.py dubber assets BUILD.json build_info.json requirements.txt runtime-constraints.txt LICENSE NOTICE README.md; do
  [ -e "$HERE/$item" ] && cp -a "$HERE/$item" "$PREFIX/app.new/"
done
rm -rf "$PREFIX/app"
mv "$PREFIX/app.new" "$PREFIX/app"

# 4. Shared Python 3.14 environment. A previous per-app environment is moved once and not left in place.
if [ -d "$PREFIX/env" ] && [ ! -x "$PY" ]; then
  say "[2/5] Moving the previous environment into $RT"
  mkdir -p "$(dirname "$RT")"
  rm -rf "$RT"
  mkdir -p "$RT"
  mv "$PREFIX/env" "$ENV_DIR"
fi
if [ -d "$VX_HOME/models" ] && [ ! -e "$SHARED/models" ]; then
  mkdir -p "$SHARED"
  mv "$VX_HOME/models" "$SHARED/models"
fi
export UV_PYTHON_INSTALL_DIR="$RT/python"
export UV_PYTHON_PREFERENCE=only-managed
export PYTHONUTF8=1
say "[2/5] Creating the Python 3.14 environment"
"$UV" venv "$ENV_DIR" --python 3.14 --allow-existing --quiet
read -r TV TA TC < <("$PY" - "$PREFIX/app/dubber/infra/runtime_lock.json" <<'PYEOF'
import json, sys
lock = json.load(open(sys.argv[1], encoding="utf-8"))
def base(dist):
    for w in lock.get("wheels", []):
        if w.get("dist") == dist:
            return str(w["version"]).split("+")[0]
    raise SystemExit(f"runtime lock has no {dist} wheel")
print(lock["torch_version"], base("torchaudio"), base("torchcodec"))
PYEOF
)
[ -n "${TC:-}" ] || die "cannot read the PyTorch versions from runtime_lock.json"
say "[2/5] Installing PyTorch $TV (CUDA 13.0 build), torchaudio $TA and torchcodec $TC"
"$UV" pip install --python "$PY" --quiet "torch==$TV" "torchaudio==$TA" "torchcodec==$TC" --torch-backend=cu130
PINS="$(mktemp)"
trap 'rm -f "$PINS"' EXIT
{ printf 'torch==%s+cu130\ntorchaudio==%s+cu130\ntorchcodec==%s+cu130\n' "$TV" "$TA" "$TC"; cat "$PREFIX/app/runtime-constraints.txt"; } > "$PINS"
say "[3/5] Installing the other packages and the CUDA 12 libraries"
"$UV" pip install --python "$PY" --quiet -r "$PREFIX/app/requirements.txt" -c "$PINS"
"$UV" pip install --python "$PY" --quiet "nvidia-cublas-cu12==12.9.2.10" "nvidia-cudnn-cu12==9.27.0.42"
"$PY" -c "import torch, transformers, PySide6, faster_whisper, ctranslate2; print('torch', torch.__version__, '- CUDA available:', torch.cuda.is_available())"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RT/.install-complete"
"$PY" - "$RT/runtime-key.json" "$TV" <<'PYEOF'
import json, sys
path, tv = sys.argv[1], sys.argv[2]
with open(path, "w", encoding="utf-8") as handle:
    json.dump({"python": "3.14", "torch": tv + "+cu130", "flavor": "cu130", "created_by": "movie-dubber"}, handle, indent=1)
    handle.write("\n")
PYEOF

# 5. ffmpeg, models, manifest
install_ffmpeg
mkdir -p "$SHARED/models"
if [ "$SKIP_MODELS" = 1 ]; then
  say "[5/5] Skipping the model download (--skip-models, CI only)"
else
  say "[5/5] Downloading the AI models the default pipeline needs"
  "$PY" "$PREFIX/app/main.py" --fetch-models
fi
ff_sha="$(sha256sum "$FF_DIR/ffmpeg" | awk '{ print $1 }')"
ff_size="$(wc -c < "$FF_DIR/ffmpeg" | tr -d ' ')"
add_ref --id runtime --version "$RUNTIME_VERSION" --path "$RT"
add_ref --id torch --version "${TV}+cu130" --path "$RT"
add_ref --id ctranslate2 --version cu12 --path "$RT"
add_ref --id ffmpeg --version "$FFMPEG_VERSION" --path "$FF_DIR" --sha256 "$ff_sha" --size "$ff_size"
add_ref --id models --version store --path "$SHARED/models"

# 6. launcher, menu entry, icons and the .vxdub file type
say "Adding the launcher $BIN_DIR/$APP_ID"
mkdir -p "$BIN_DIR"
cat > "$BIN_DIR/$APP_ID" <<LAUNCH
#!/bin/sh
# Voxprint AI Movie Dubber launcher (written by install.sh)
exec "$PY" "$PREFIX/app/main.py" "\$@"
LAUNCH
chmod 755 "$BIN_DIR/$APP_ID"

mkdir -p "$DATA_HOME/icons/hicolor/256x256/apps" "$DATA_HOME/icons/hicolor/256x256/mimetypes" \
         "$DATA_HOME/applications" "$DATA_HOME/mime/packages"
cp "$HERE/linux/voxprint-dubber-256.png" "$DATA_HOME/icons/hicolor/256x256/apps/$APP_ID.png"
cp "$HERE/linux/vxdub-256.png" "$DATA_HOME/icons/hicolor/256x256/mimetypes/$MIME_ICON.png"

cat > "$DATA_HOME/applications/$APP_ID.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=Voxprint AI Movie Dubber
Comment=Dub films into another language with local AI models
Exec=$BIN_DIR/$APP_ID %f
Icon=$APP_ID
Terminal=false
Categories=AudioVideo;Video;
MimeType=$MIME_TYPE;
StartupNotify=true
DESK

cat > "$DATA_HOME/mime/packages/$APP_ID.xml" <<MIME
<?xml version="1.0" encoding="UTF-8"?>
<mime-info xmlns="http://www.freedesktop.org/standards/shared-mime-info">
  <mime-type type="$MIME_TYPE">
    <comment>Voxprint dubbing project</comment>
    <sub-class-of type="application/zip"/>
    <icon name="$MIME_ICON"/>
    <glob pattern="*.vxdub" weight="60"/>
  </mime-type>
</mime-info>
MIME

refresh_desktop
command -v xdg-mime >/dev/null 2>&1 && xdg-mime default "$APP_ID.desktop" "$MIME_TYPE" >/dev/null 2>&1 || true

say "Installed. Start it from the application menu or run: $APP_ID"
case ":$PATH:" in *":$BIN_DIR:"*) ;; *) say "Note: $BIN_DIR is not on PATH; add it or run $BIN_DIR/$APP_ID" ;; esac
