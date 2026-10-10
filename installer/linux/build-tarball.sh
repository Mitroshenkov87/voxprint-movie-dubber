#!/usr/bin/env bash
# Packs the Linux tarball: dist/VoxprintMovieDubber-Linux-x86_64.tar.gz and its .sha256.
# Run from the repository root. build_info.json (the CI build stamp) is included when it exists.
set -euo pipefail
name="VoxprintMovieDubber-Linux-x86_64"
stage="build/linux/$name"
rm -rf build/linux && mkdir -p "$stage/linux" dist
cp -a main.py dubber assets BUILD.json requirements.txt LICENSE NOTICE README.md "$stage/"
[ -f build_info.json ] && cp build_info.json "$stage/"
cp installer/runtime-constraints.txt "$stage/"
cp installer/linux/install.sh "$stage/"
cp installer/linux/voxprint-dubber-256.png installer/linux/vxdub-256.png "$stage/linux/"
find "$stage" -name __pycache__ -type d -prune -exec rm -rf {} +
find "$stage" -name '*.pyc' -delete
chmod 755 "$stage/install.sh"
tar -C build/linux --owner=0 --group=0 --numeric-owner -czf "dist/$name.tar.gz" "$name"
(cd dist && sha256sum "$name.tar.gz" > "$name.tar.gz.sha256")
ls -l "dist/$name.tar.gz"; cat "dist/$name.tar.gz.sha256"
