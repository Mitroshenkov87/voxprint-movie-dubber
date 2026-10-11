#!/usr/bin/env bash
# VOXPrint AI Movie Dubber - Linux installer.
#
#   ./install.sh                  install for the system into /opt/voxprint-ai-suite/movie-dubber (uses sudo)
#   ./install.sh --prefix DIR     same layout under DIR, without root (CI and unprivileged runs)
#   ./install.sh --uninstall      remove the program and this app's shared-resource references
#   ./install.sh --prefix DIR --uninstall
#   ./install.sh --skip-gpu-check CI ONLY: do not check the NVIDIA driver and GPU. GitHub runners have no GPU.
#   ./install.sh --skip-models    CI ONLY: do not download the AI models. A user install always downloads them.
#
# A system install symlinks voxprint-movie-dubber and voxprint-dubber-cli into /usr/local/bin.
# With --prefix, those names are symlinked into DIR/bin. The menu entry Name is VOXPrint AI Movie Dubber.
# User data stays in ~/.local/share/voxprint. An older install under ~/.local/share/voxprint-movie-dubber
# is removed; that user data is kept.
#
# Requirements: x86_64 Linux (a distribution released in 2025 or later), an NVIDIA GeForce RTX 40-series GPU or newer
# (compute capability 8.9+) and an NVIDIA driver from the 600 branch or newer, about 25 GB of free disk space and an
# internet connection. The installer puts Python 3.14, PyTorch 2.11.0+cu130, torchaudio, torchcodec, the CTranslate2
# CUDA 12 libraries and ffmpeg into ~/.local/share/voxprint/shared, and the default-pipeline models into ~/.local/share/voxprint/models, before the first launch.
# An interrupted download is continued from a .partial file.
set -euo pipefail

APP_SLUG="movie-dubber"
GUI_BIN="voxprint-movie-dubber"
CLI_BIN="voxprint-dubber-cli"
DISPLAY_NAME="VOXPrint AI Movie Dubber"
DESKTOP_ID="voxprint-movie-dubber"
REF_APP="movie-dubber"
MIME_TYPE="application/vnd.voxprint.dub+zip"
MIME_ICON="application-vnd.voxprint.dub+zip"
OLD_PREFIX_NAME="voxprint-movie-dubber"
MIN_DRIVER=600
MIN_CC="8.9"
RUNTIME_VERSION="py3.14-torch2.11-cu130"
FFMPEG_VERSION="n8.1"
FFMPEG_ARCHIVE="ffmpeg-n8.1-latest-linux64-lgpl-8.1.tar.xz"
FFMPEG_BASE="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest"
DEFAULT_SUITE_ROOT="/opt/voxprint-ai-suite"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKIP_GPU=0
SKIP_MODELS=0
UNINSTALL=0
PREFIX_SET=0
PREFIX_DIR=""

say() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

usage() { sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-gpu-check) SKIP_GPU=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    --uninstall) UNINSTALL=1 ;;
    --prefix) [ $# -ge 2 ] || die "--prefix needs a folder"; PREFIX_SET=1; PREFIX_DIR="$2"; shift ;;
    --prefix=*) PREFIX_SET=1; PREFIX_DIR="${1#--prefix=}" ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown option: $1" ;;
  esac
  shift
done

if [ "$PREFIX_SET" = 1 ]; then
  [ -n "$PREFIX_DIR" ] || die "--prefix needs a folder"
  case "$PREFIX_DIR" in
    /*) ;;
    *) PREFIX_DIR="$(cd "$(dirname "$PREFIX_DIR")" && pwd)/$(basename "$PREFIX_DIR")" ;;
  esac
  SUITE_ROOT="$PREFIX_DIR"
  APP_DIR="$SUITE_ROOT/$APP_SLUG"
  BIN_DIR="$SUITE_ROOT/bin"
  SHARE_DIR="$SUITE_ROOT/share"
  SYSTEM=0
else
  SUITE_ROOT="$DEFAULT_SUITE_ROOT"
  APP_DIR="$SUITE_ROOT/$APP_SLUG"
  BIN_DIR="/usr/local/bin"
  SHARE_DIR="/usr/local/share"
  SYSTEM=1
fi

if [ -n "${VOXPRINT_INSTALL_HOME:-}" ]; then
  USER_HOME="$VOXPRINT_INSTALL_HOME"
elif [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != "root" ]; then
  USER_HOME="$(getent passwd "$SUDO_USER" | cut -d: -f6)"
  [ -n "$USER_HOME" ] || die "cannot find the home directory of $SUDO_USER"
else
  USER_HOME="$HOME"
fi
if [ -n "${VOXPRINT_INSTALL_XDG:-}" ]; then
  DATA_HOME="$VOXPRINT_INSTALL_XDG"
else
  DATA_HOME="$USER_HOME/.local/share"
fi
VX_HOME="$DATA_HOME/voxprint"
SHARED="$VX_HOME/shared"
RT="$SHARED/runtimes/$RUNTIME_VERSION"
ENV_DIR="$RT/env"
PY="$ENV_DIR/bin/python"
FF_DIR="$SHARED/ffmpeg/$FFMPEG_VERSION"
OLD_PREFIX="$DATA_HOME/$OLD_PREFIX_NAME"
TARGET_USER="${SUDO_USER:-}"

run_root() {
  if [ "$SYSTEM" != 1 ]; then
    "$@"
    return
  fi
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    command -v sudo >/dev/null 2>&1 || die "installing to $APP_DIR needs root. Run with sudo, or pass --prefix DIR."
    sudo "$@"
  fi
}

as_user() {
  if [ "$(id -u)" -eq 0 ] && [ -n "$TARGET_USER" ] && [ "$TARGET_USER" != "root" ]; then
    sudo -u "$TARGET_USER" -H -- env \
      HOME="$USER_HOME" \
      XDG_DATA_HOME="$DATA_HOME" \
      USER="$TARGET_USER" \
      LOGNAME="$TARGET_USER" \
      PATH="$PATH" \
      PYTHONUTF8=1 \
      UV_PYTHON_INSTALL_DIR="$RT/python" \
      UV_PYTHON_PREFERENCE=only-managed \
      "$@"
  else
    PYTHONUTF8=1 UV_PYTHON_INSTALL_DIR="$RT/python" UV_PYTHON_PREFERENCE=only-managed "$@"
  fi
}

refresh_desktop() {
  local root="$1"
  command -v update-mime-database >/dev/null 2>&1 && update-mime-database "$root/mime" >/dev/null 2>&1 || true
  command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$root/applications" >/dev/null 2>&1 || true
  command -v gtk-update-icon-cache >/dev/null 2>&1 && gtk-update-icon-cache -q -t "$root/icons/hicolor" >/dev/null 2>&1 || true
  return 0
}

remove_old_user_install() {
  # The previous layout lived in the home directory. Leave ~/.local/share/voxprint alone.
  # On uninstall, do not move an old environment back into the shared runtime.
  if [ "$UNINSTALL" != 1 ] && [ -d "$OLD_PREFIX/env" ] && [ ! -x "$PY" ]; then
    say "Moving the previous environment into $RT"
    mkdir -p "$(dirname "$RT")"
    rm -rf "$RT"
    mkdir -p "$RT"
    mv "$OLD_PREFIX/env" "$ENV_DIR"
  fi
  if [ -e "$OLD_PREFIX" ] || [ -e "$USER_HOME/.local/bin/$OLD_PREFIX_NAME" ] \
     || [ -e "$DATA_HOME/applications/$OLD_PREFIX_NAME.desktop" ]; then
    say "Removing the previous per-user install (user data in $VX_HOME is kept)"
  fi
  rm -rf "$OLD_PREFIX"
  rm -f "$USER_HOME/.local/bin/$OLD_PREFIX_NAME" \
        "$USER_HOME/.local/bin/$GUI_BIN" \
        "$DATA_HOME/applications/$OLD_PREFIX_NAME.desktop" \
        "$DATA_HOME/applications/$DESKTOP_ID.desktop" \
        "$DATA_HOME/mime/packages/$OLD_PREFIX_NAME.xml" \
        "$DATA_HOME/mime/packages/$DESKTOP_ID.xml" \
        "$DATA_HOME/icons/hicolor/256x256/apps/$OLD_PREFIX_NAME.png" \
        "$DATA_HOME/icons/hicolor/256x256/apps/$DESKTOP_ID.png" \
        "$DATA_HOME/icons/hicolor/256x256/mimetypes/$MIME_ICON.png"
  refresh_desktop "$DATA_HOME"
}

add_ref() {
  as_user "$PY" "$APP_DIR/dubber/infra/components.py" add --home "$VX_HOME" --app "$REF_APP" "$@"
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

write_launcher() {
  local name="$1"
  local path="$APP_DIR/$name"
  local body
  body="$(cat <<LAUNCH
#!/bin/sh
# $DISPLAY_NAME launcher (written by install.sh). Same entry point as main.py.
APP="$APP_DIR"
DATA="\${XDG_DATA_HOME:-\$HOME/.local/share}"
PY="\$DATA/voxprint/shared/runtimes/$RUNTIME_VERSION/env/bin/python"
if [ ! -x "\$PY" ]; then
  echo "$DISPLAY_NAME could not find the shared Python runtime: \$PY" >&2
  exit 1
fi
exec "\$PY" "\$APP/main.py" "\$@"
LAUNCH
)"
  if [ "$SYSTEM" = 1 ]; then
    printf '%s\n' "$body" | run_root tee "$path" >/dev/null
    run_root chmod 755 "$path"
    run_root ln -sfn "$path" "$BIN_DIR/$name"
  else
    printf '%s\n' "$body" > "$path"
    chmod 755 "$path"
    ln -sfn "$path" "$BIN_DIR/$name"
  fi
}

install_program_files() {
  local stage item
  stage="$(mktemp -d)"
  for item in main.py dubber assets BUILD.json build_info.json requirements.txt runtime-constraints.txt LICENSE NOTICE README.md; do
    [ -e "$HERE/$item" ] && cp -a "$HERE/$item" "$stage/"
  done
  if [ "$SYSTEM" = 1 ]; then
    run_root mkdir -p "$APP_DIR"
    run_root cp -a "$stage/." "$APP_DIR/"
  else
    # uv may already live in APP_DIR/uv. Keep it across the swap.
    rm -rf "$APP_DIR.new"
    mkdir -p "$APP_DIR.new"
    cp -a "$stage/." "$APP_DIR.new/"
    if [ -d "$APP_DIR/uv" ]; then
      cp -a "$APP_DIR/uv" "$APP_DIR.new/uv"
    fi
    rm -rf "$APP_DIR"
    mv "$APP_DIR.new" "$APP_DIR"
  fi
  rm -rf "$stage"
}

release_shared() {
  local script="$1"
  local release_log pending_line rest path
  release_log="$(mktemp)"
  if [ -x "$PY" ]; then
    as_user "$PY" "$script" release --home "$VX_HOME" --app "$REF_APP" >"$release_log" 2>&1 || true
  elif [ -f "$script" ]; then
    python3 "$script" release --home "$VX_HOME" --app "$REF_APP" >"$release_log" 2>&1 || true
  fi
  cat "$release_log" || true
  pending_line="$(awk -F'pending: ' '/^pending: / { print $2; exit }' "$release_log" || true)"
  rm -f "$release_log"
  if [ -n "${pending_line:-}" ] && [ "$pending_line" != "none" ]; then
    rest="$pending_line"
    while [ -n "$rest" ]; do
      path="${rest%%, *}"
      case "$path" in
        "$SHARED"/*) rm -rf "$path" ;;
      esac
      if [ "$path" = "$rest" ]; then
        break
      fi
      rest="${rest#*, }"
    done
  fi
}

if [ "$UNINSTALL" = 1 ]; then
  say "Removing $DISPLAY_NAME from $APP_DIR"
  if [ -f "$APP_DIR/dubber/infra/components.py" ]; then
    release_shared "$APP_DIR/dubber/infra/components.py"
  elif [ -f "$OLD_PREFIX/app/dubber/infra/components.py" ]; then
    release_shared "$OLD_PREFIX/app/dubber/infra/components.py"
  fi
  if [ "$SYSTEM" = 1 ]; then
    run_root rm -f "$BIN_DIR/$GUI_BIN" "$BIN_DIR/$CLI_BIN"
    run_root rm -rf "$APP_DIR"
    run_root rm -f "$SHARE_DIR/applications/$DESKTOP_ID.desktop" \
      "$SHARE_DIR/mime/packages/$DESKTOP_ID.xml" \
      "$SHARE_DIR/icons/hicolor/256x256/apps/$DESKTOP_ID.png" \
      "$SHARE_DIR/icons/hicolor/256x256/mimetypes/$MIME_ICON.png"
    run_root rmdir "$SUITE_ROOT" 2>/dev/null || true
  else
    rm -f "$BIN_DIR/$GUI_BIN" "$BIN_DIR/$CLI_BIN"
    rm -rf "$APP_DIR"
    rm -f "$SHARE_DIR/applications/$DESKTOP_ID.desktop" \
      "$SHARE_DIR/mime/packages/$DESKTOP_ID.xml" \
      "$SHARE_DIR/icons/hicolor/256x256/apps/$DESKTOP_ID.png" \
      "$SHARE_DIR/icons/hicolor/256x256/mimetypes/$MIME_ICON.png"
    rmdir "$BIN_DIR" "$SHARE_DIR/applications" "$SHARE_DIR/mime/packages" \
      "$SHARE_DIR/icons/hicolor/256x256/apps" "$SHARE_DIR/icons/hicolor/256x256/mimetypes" \
      "$SHARE_DIR/icons/hicolor/256x256" "$SHARE_DIR/icons/hicolor" "$SHARE_DIR/icons" \
      "$SHARE_DIR/mime" "$SHARE_DIR" "$SUITE_ROOT" 2>/dev/null || true
  fi
  remove_old_user_install
  refresh_desktop "$SHARE_DIR"
  say "Done. Shared components this program was the last user of were removed. User data in $VX_HOME was kept."
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

remove_old_user_install

# 2. uv
UV="$(command -v uv || true)"
if [ -z "$UV" ]; then
  say "[1/5] Downloading uv (package manager)"
  if [ "$SYSTEM" = 1 ]; then
    run_root mkdir -p "$APP_DIR/uv"
    curl -LsSf https://astral.sh/uv/install.sh | run_root env UV_INSTALL_DIR="$APP_DIR/uv" UV_NO_MODIFY_PATH=1 sh >/dev/null
  else
    mkdir -p "$APP_DIR/uv"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$APP_DIR/uv" UV_NO_MODIFY_PATH=1 sh >/dev/null
  fi
  UV="$APP_DIR/uv/uv"
fi
[ -x "$UV" ] || die "uv is not available"

# 3. program files
say "[2/5] Copying the program to $APP_DIR"
install_program_files

# 4. Shared Python 3.14 environment.
if [ -d "$SHARED/models" ] && [ ! -e "$VX_HOME/models" ]; then
  mv "$SHARED/models" "$VX_HOME/models"
fi
moved_runtime=0
for old in "$VX_HOME"/runtime-*; do
  [ -d "$old" ] || continue
  base="$(basename "$old")"
  case "$base" in
    runtime-[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f])
      if [ "$moved_runtime" = 0 ] && [ ! -f "$RT/runtime-key.json" ] && [ -f "$old/runtime-key.json" ] \
         && grep -q '"flavor"[[:space:]]*:[[:space:]]*"cu130"' "$old/runtime-key.json"; then
        mkdir -p "$(dirname "$RT")"
        rm -rf "$RT"
        mv "$old" "$RT"
        moved_runtime=1
        say "Moved the old runtime folder $base into $RT"
      else
        rm -rf "$old"
        say "Removed the old runtime folder $base"
      fi
      ;;
  esac
done
say "[2/5] Creating the Python 3.14 environment"
as_user "$UV" venv "$ENV_DIR" --python 3.14 --allow-existing --quiet
read -r TV TA TC < <("$PY" - "$APP_DIR/dubber/infra/runtime_lock.json" <<'PYEOF'
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
as_user "$UV" pip install --python "$PY" --quiet "torch==$TV" "torchaudio==$TA" "torchcodec==$TC" --torch-backend=cu130
PINS="$(mktemp)"
trap 'rm -f "$PINS"' EXIT
{ printf 'torch==%s+cu130\ntorchaudio==%s+cu130\ntorchcodec==%s+cu130\n' "$TV" "$TA" "$TC"; cat "$APP_DIR/runtime-constraints.txt"; } > "$PINS"
say "[3/5] Installing the other packages and the CUDA 12 libraries"
WHEEL_DIR="$(mktemp -d)"
"$PY" "$HERE/fetch_common_wheel.py" --dest "$WHEEL_DIR"
chmod 644 "$PINS"
chmod -R a+rX "$WHEEL_DIR"
as_user "$UV" pip install --python "$PY" --quiet --find-links "$WHEEL_DIR" -r "$APP_DIR/requirements.txt" -c "$PINS"
rm -rf "$WHEEL_DIR"
as_user "$UV" pip install --python "$PY" --quiet "nvidia-cublas-cu12==12.9.2.10" "nvidia-cudnn-cu12==9.27.0.42"
if "$PY" -c "from voxprint_common.runtime_manifest import cpu_onnxruntime_conflicts; raise SystemExit(0 if cpu_onnxruntime_conflicts() else 1)"; then
  say "Replacing the CPU onnxruntime package with onnxruntime-gpu 1.30.0"
  as_user "$UV" pip uninstall --python "$PY" --quiet onnxruntime
  as_user "$UV" pip install --python "$PY" --quiet "onnxruntime-gpu==1.30.0"
fi
"$PY" -c "import voxprint_common; from voxprint_common.runtime_manifest import cpu_onnxruntime_conflicts; assert voxprint_common.__version__ == '0.1.1', voxprint_common.__version__; raise SystemExit('The CPU package onnxruntime is installed. The shared runtime must not contain it.' if cpu_onnxruntime_conflicts() else 0)"
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
mkdir -p "$VX_HOME/models"
if [ "$SKIP_MODELS" = 1 ]; then
  say "[5/5] Skipping the model download (--skip-models, CI only)"
else
  say "[5/5] Downloading the AI models the default pipeline needs"
  as_user "$PY" "$APP_DIR/main.py" --fetch-models
fi
ff_sha="$(sha256sum "$FF_DIR/ffmpeg" | awk '{ print $1 }')"
add_ref --id python-3.14 --kind runtime --version "$RUNTIME_VERSION" --path "$RT"
add_ref --id torch-2.11.0-cu130 --kind wheels --version "${TV}+cu130" --path "$RT"
add_ref --id ctranslate2-cu12 --kind wheels --version cu12 --path "$RT"
add_ref --id ffmpeg-8 --kind tool --version "$FFMPEG_VERSION" --path "$FF_DIR" --sha256 "$ff_sha"
add_ref --id models --kind model --version store --path "$VX_HOME/models"

# 6. launchers, menu entry, icons and the .vxdub file type
say "Adding $GUI_BIN and $CLI_BIN in $BIN_DIR"
if [ "$SYSTEM" = 1 ]; then
  run_root mkdir -p "$BIN_DIR" \
    "$SHARE_DIR/icons/hicolor/256x256/apps" "$SHARE_DIR/icons/hicolor/256x256/mimetypes" \
    "$SHARE_DIR/applications" "$SHARE_DIR/mime/packages"
else
  mkdir -p "$BIN_DIR" \
    "$SHARE_DIR/icons/hicolor/256x256/apps" "$SHARE_DIR/icons/hicolor/256x256/mimetypes" \
    "$SHARE_DIR/applications" "$SHARE_DIR/mime/packages"
fi
write_launcher "$GUI_BIN"
write_launcher "$CLI_BIN"

if [ "$SYSTEM" = 1 ]; then
  run_root cp "$HERE/linux/voxprint-dubber-256.png" "$SHARE_DIR/icons/hicolor/256x256/apps/$DESKTOP_ID.png"
  run_root cp "$HERE/linux/vxdub-256.png" "$SHARE_DIR/icons/hicolor/256x256/mimetypes/$MIME_ICON.png"
else
  cp "$HERE/linux/voxprint-dubber-256.png" "$SHARE_DIR/icons/hicolor/256x256/apps/$DESKTOP_ID.png"
  cp "$HERE/linux/vxdub-256.png" "$SHARE_DIR/icons/hicolor/256x256/mimetypes/$MIME_ICON.png"
fi

desktop_body="$(cat <<DESK
[Desktop Entry]
Type=Application
Name=$DISPLAY_NAME
Comment=Dub films into another language with local AI models
Exec="$BIN_DIR/$GUI_BIN" %f
Icon=$DESKTOP_ID
Terminal=false
Categories=AudioVideo;Video;
MimeType=$MIME_TYPE;
StartupNotify=true
DESK
)"
mime_body="$(cat <<MIME
<?xml version="1.0" encoding="UTF-8"?>
<mime-info xmlns="http://www.freedesktop.org/standards/shared-mime-info">
  <mime-type type="$MIME_TYPE">
    <comment>VOXPrint dubbing project</comment>
    <sub-class-of type="application/zip"/>
    <icon name="$MIME_ICON"/>
    <glob pattern="*.vxdub" weight="60"/>
  </mime-type>
</mime-info>
MIME
)"
if [ "$SYSTEM" = 1 ]; then
  printf '%s\n' "$desktop_body" | run_root tee "$SHARE_DIR/applications/$DESKTOP_ID.desktop" >/dev/null
  printf '%s\n' "$mime_body" | run_root tee "$SHARE_DIR/mime/packages/$DESKTOP_ID.xml" >/dev/null
else
  printf '%s\n' "$desktop_body" > "$SHARE_DIR/applications/$DESKTOP_ID.desktop"
  printf '%s\n' "$mime_body" > "$SHARE_DIR/mime/packages/$DESKTOP_ID.xml"
fi

refresh_desktop "$SHARE_DIR"
command -v xdg-mime >/dev/null 2>&1 && xdg-mime default "$DESKTOP_ID.desktop" "$MIME_TYPE" >/dev/null 2>&1 || true

if [ "$(id -u)" -eq 0 ] && [ -n "$TARGET_USER" ] && [ "$TARGET_USER" != "root" ]; then
  chown -R "$TARGET_USER:$TARGET_USER" "$VX_HOME" || true
fi

say "Installed. Start it from the application menu or run: $GUI_BIN"
say "Command line: $CLI_BIN"
if [ "$SYSTEM" = 1 ]; then
  case ":$PATH:" in *":$BIN_DIR:"*) ;; *) say "Note: $BIN_DIR is not on PATH" ;; esac
else
  say "Launchers: $BIN_DIR/$GUI_BIN and $BIN_DIR/$CLI_BIN"
fi
