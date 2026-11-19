# Suite splash screen spec v1 (FROZEN 2026-10-11, owner: Suite Lead)

Approved by Aleksandr 2026-10-11. Implement as-is; changes need a new version of this file, not per-app tweaks.

Art: /workspace/suite-splash-approved/<app>.jpg (16:9). Copy into the repo as assets/splash.jpg; produce 2x via Enlarge for HiDPI.
Apps: audiobook-builder, movie-dubber, clarity, enlarge, canvas, plotweaver, model-forge. Lucent is exempt (instant open).

Style (for any regeneration): dark navy textured background; a mosaic of many small geometric shards cutting into each other; cobalt blue, warm yellow, crimson red and off-white in equal measure; no large flat colour areas; worn collage edges; the app motif is built from shards on the right; the left third stays empty for text.

Layout (window 960x540 logical, borderless, centred, rounded 12 px corners, drop shadow):
- Left third, 48 px padding: app name (suite UI font, semibold, 40 px, off-white), below it `X.Y.Z RC build N "Codename"` (16 px, 70% off-white) read from BUILD.json, and the stage text (14 px).
- Progress bar: bottom-left, width = left third minus padding, 4 px, rounded; track 20% off-white; fill in the app identity colour (PALETTE.md); indeterminate shimmer until the first real progress value.
- Timing: show within 150 ms of launch; minimum 600 ms on screen (no flash); fade out 200 ms when the main window is ready; never block longer than startup.
- One shared component (package voxprint-common, owner Suite Lead) used by every app; no app-private splash code.
