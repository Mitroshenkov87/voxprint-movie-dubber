# Voxprint AI Movie Dubber

Part of the Voxprint AI Media Suite, together with [Voxprint AI Audiobook Builder](https://github.com/Mitroshenkov87/voxprint-audiobook-builder).

Voxprint AI Movie Dubber is a Windows desktop application that adds an AI-dubbed audio track to a movie. It separates the dialogue from music and effects, recognises and translates the speech, re-voices it with a voice cloned from the film or taken from the Voxprint voice library and adds the result to the file as an extra audio track, leaving the video and the original audio untouched. You can listen to a one-minute preview first and start watching while the rest of the film is still being dubbed. Everything runs locally on your NVIDIA GPU.

This is the 1.0.0 release candidate. The build number and the codename (Bochan) come from `BUILD.json`.

## Install

**Requirements:** Windows 11 24H2 or newer (64-bit), an NVIDIA GeForce RTX 40-series graphics card or newer with an NVIDIA driver from the 600 branch or newer, about 25 GB of free disk space and an internet connection for the first model download. The setup and the program stop with a clear message on older hardware; there is no processor-only mode. A Linux package (distributions released in 2025 or later) is on the [roadmap](ROADMAP.md).

1. Download the installer from the [latest release](https://github.com/Mitroshenkov87/voxprint-movie-dubber/releases):
   - **Full installer** (offline, recommended): `VoxprintMovieDubber-Full-Setup.exe` together with every `VoxprintMovieDubber-Full-Setup-N.bin` part, all in the same folder. It carries Python and PyTorch, so setup needs no internet.
   - **Online installer**: the small `VoxprintMovieDubber-Setup.exe`, which downloads Python and PyTorch during setup.
2. Check the download against the SHA-256 in the release notes, run the `.exe` and follow the setup. Leave "Download the AI models now" ticked if you want the models (about 8 GB) right away; otherwise they are downloaded on first use.
3. Start **Voxprint AI Movie Dubber** from the Start menu (folder "Voxprint").

<!-- screenshot: setup wizard -->

## Use

1. Drag a movie or series episode into the window, or press **Choose a movie**.
2. Pick the dub language and press **Dub**. Everything else is preset: the program finds or makes the translation, picks the voices and fits every line into its time.
3. When it is done, the dubbed file sits next to the original as `<name>.dub-<language>.mkv`. It has the new audio track added; the original audio and subtitles are kept, and the video is not re-encoded.

<!-- screenshot: main window -->

You can listen to a one-minute preview first, start watching while the rest is still being dubbed, and review characters, voices and every line before dubbing. Step-by-step instructions are in the [User guide](docs/USER-GUIDE.md).

Other programs can call it without the window. `python main.py --dry-run --json` prints one JSON object and does not load or download models. The commands and exit codes are in the [command-line reference](docs/CLI.md).

## Roadmap

Next: meaning-based translation with a local language model, full command-line control and a Linux package. See [ROADMAP.md](ROADMAP.md).

**License:** Apache-2.0, see `LICENSE` and `NOTICE`. Third-party components keep their own licenses (`docs/THIRD_PARTY_NOTICES.md`).

## Philosophy

Voxprint apps are complete: the choices about models, settings and pipelines are already made, so you install the app and start working. Like software used to be, there is no account, no subscription and no online sign-in, and your files stay on your computer. Voxprint looks forward. It is built for current operating systems and recent NVIDIA GPUs, and it drops old platforms after two to three years instead of carrying them. Free and open source (Apache-2.0), with modern open formats by default. [Read more](PHILOSOPHY.md)

## ☕ Support the Project

If you find this project useful and would like to support its development, you can buy me a coffee!

- **Network:** TRON (TRC-20)
- **Accepted:** USDT or TRX
- **Address:** `TYveZBXaSpM4FEHa6iGrc7A6zuLfS4z3x6`

> ⚠️ **Important:** Please ensure you are sending funds **only** via the TRON (TRC-20) network. 
> Sending assets from other networks (such as Ethereum ERC-20, BSC BEP-20, etc.) to this address will result in **permanent loss of funds**.
