#!/usr/bin/env bash
# Voxprint AI Movie Dubber - Linux installer (per user, no root needed).
#
#   ./install.sh                  install into ~/.local (program, Python 3.14 environment, launcher, menu entry, .vxdub type)
#   ./install.sh --prefix DIR     put the program and its environment in DIR (default ~/.local/share/voxprint-movie-dubber)
#   ./install.sh --uninstall      remove the program, the launcher, the menu entry and the .vxdub type (models are kept)
#   ./install.sh --skip-gpu-check CI ONLY: do not check the NVIDIA driver and GPU. GitHub runners have no GPU.
#                                 The program itself still refuses to dub on an unsupported GPU.
#
# Requirements: x86_64 Linux (a distribution released in 2025 or later), an NVIDIA GeForce RTX 40-series GPU or newer
# (compute capability 8.9+) and an NVIDIA driver from the 600 branch or newer, about 25 GB of free disk space and an
# internet connection. The installer creates a Python 3.14 environment with uv (downloaded when it is not installed)
# and installs PyTorch 2.11.0 (CUDA 13.0 build) and the packages from requirements.txt.
set -euo pipefail

APP_ID="voxprint-movie-dubber"
MIME_TYPE="application/vnd.voxprint.dub+zip"
MIME_ICON="application-vnd.voxprint.dub+zip"
MIN_DRIVER=600
MIN_CC="8.9"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
BIN_DIR="$HOME/.local/bin"
PREFIX="$DATA_HOME/$APP_ID"
SKIP_GPU=0
UNINSTALL=0

say() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

usage() { sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-gpu-check) SKIP_GPU=1 ;;
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

if [ "$UNINSTALL" = 1 ]; then
  say "Removing Voxprint AI Movie Dubber from $PREFIX"
  rm -rf "$PREFIX"
  rm -f "$BIN_DIR/$APP_ID" "$DATA_HOME/applications/$APP_ID.desktop" "$DATA_HOME/mime/packages/$APP_ID.xml" \
        "$DATA_HOME/icons/hicolor/256x256/apps/$APP_ID.png" "$DATA_HOME/icons/hicolor/256x256/mimetypes/$MIME_ICON.png"
  refresh_desktop
  say "Done. Downloaded AI models were kept."
  exit 0
fi

[ "$(uname -s)" = "Linux" ] && [ "$(uname -m)" = "x86_64" ] || die "this package is for x86_64 Linux"
[ -f "$HERE/main.py" ] && [ -f "$HERE/requirements.txt" ] || die "run install.sh from the unpacked package folder"

# 1. GPU and driver
if [ "$SKIP_GPU" = 1 ]; then
  say "Skipping the NVIDIA driver and GPU check (--skip-gpu-check, CI only)"
else
  command -v nvidia-smi >/dev/null 2>&1 || die "no NVIDIA driver found (nvidia-smi is missing). Install an NVIDIA driver from the $MIN_DRIVER branch or newer."
  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | tr -d ' ')"
  [ -n "$driver" ] || die "nvidia-smi did not report a driver version"
  major="${driver%%.*}"
  [ "$major" -ge "$MIN_DRIVER" ] 2>/dev/null || die "NVIDIA driver $driver is too old: the $MIN_DRIVER branch or newer is required"
  ok_gpu=""
  while IFS=, read -r name cc; do
    name="$(echo "$name" | sed 's/^ *//;s/ *$//')"; cc="$(echo "$cc" | tr -d ' ')"
    if [ -n "$cc" ] && [ "$cc" != "[N/A]" ] && awk -v a="$cc" -v b="$MIN_CC" 'BEGIN{exit !(a+0 >= b+0)}'; then
      ok_gpu="$name (compute capability $cc)"; break
    fi
  done < <(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null || true)
  [ -n "$ok_gpu" ] || die "no supported GPU: an NVIDIA GeForce RTX 40-series or newer (compute capability $MIN_CC+) is required"
  say "GPU: $ok_gpu, driver $driver"
fi

# 2. uv
UV="$(command -v uv || true)"
if [ -z "$UV" ]; then
  say "Downloading uv (package manager)"
  mkdir -p "$PREFIX/uv"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$PREFIX/uv" UV_NO_MODIFY_PATH=1 sh >/dev/null
  UV="$PREFIX/uv/uv"
fi
[ -x "$UV" ] || die "uv is not available"

# 3. program files
say "Copying the program to $PREFIX/app"
mkdir -p "$PREFIX"
rm -rf "$PREFIX/app.new"
mkdir -p "$PREFIX/app.new"
for item in main.py dubber assets BUILD.json build_info.json requirements.txt runtime-constraints.txt LICENSE NOTICE README.md; do
  [ -e "$HERE/$item" ] && cp -a "$HERE/$item" "$PREFIX/app.new/"
done
rm -rf "$PREFIX/app"
mv "$PREFIX/app.new" "$PREFIX/app"

# 4. Python 3.14 environment with PyTorch 2.11.0+cu130 and the requirements
ENV_DIR="$PREFIX/env"
say "Creating the Python 3.14 environment"
"$UV" venv "$ENV_DIR" --python 3.14 --allow-existing --quiet
PY="$ENV_DIR/bin/python"
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
say "Installing PyTorch $TV (CUDA 13.0 build), torchaudio $TA and torchcodec $TC"
"$UV" pip install --python "$PY" --quiet "torch==$TV" "torchaudio==$TA" "torchcodec==$TC" --torch-backend=cu130
PINS="$(mktemp)"
trap 'rm -f "$PINS"' EXIT
{ printf 'torch==%s+cu130\ntorchaudio==%s+cu130\ntorchcodec==%s+cu130\n' "$TV" "$TA" "$TC"; cat "$PREFIX/app/runtime-constraints.txt"; } > "$PINS"
say "Installing the other packages"
"$UV" pip install --python "$PY" --quiet -r "$PREFIX/app/requirements.txt" -c "$PINS"
"$PY" -c "import torch, transformers, PySide6, faster_whisper; print('torch', torch.__version__, '- CUDA available:', torch.cuda.is_available())"

# 5. launcher, menu entry, icons and the .vxdub file type
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
