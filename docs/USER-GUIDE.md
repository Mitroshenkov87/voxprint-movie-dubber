# Voxprint AI Movie Dubber: User guide

This guide covers release candidate 1.0.0 RC, build 1000 "Chazak". For the requirements and the installers, see [Install](../README.md#install) in the README.

Screenshots are placeholders for now; they are added after the first test on real hardware.

## 1. First run

Start **Voxprint AI Movie Dubber** from the Start menu (folder "Voxprint"). The splash screen appears first, then the main window.

- If the PC has no NVIDIA RTX 40-series card or newer, or the driver is older than the 600 branch, the program says so and closes. There is no processor-only mode.
- The first dub downloads any AI models that are still missing (about 8 GB in total, one time) into the shared Voxprint models folder. Voxprint AI Audiobook Builder uses the same folder, so nothing is downloaded twice.
- The program checks how much video memory (VRAM) the card has and picks a matching model tier by itself: a 16 GB tier, or a 24 GB tier for cards with 24 GB or more. There is nothing to set.

<!-- screenshot: splash screen -->
<!-- screenshot: main window, empty -->

## 2. Open a video

Drag a movie or series episode into the window, or press **Choose a movie…**. MKV, MP4, AVI, MOV, WebM, M4V, TS and M2TS files are accepted.

The window shows the duration, the video format, the audio tracks and any subtitles in the file. **Original audio track** picks the track to translate from; automatic is the right choice in almost every case.

<!-- screenshot: film opened, file info -->

## 3. Languages and options

**Dub language** is the language of the new audio track (English, Russian or German).

Everything under **Options** is preset for the best result and remembered for the next film. Change only what you need:

| Option | What it does |
| --- | --- |
| Translation from | Subtitles in the file or next to it, a subtitle file you choose, subtitles found online (needs SubDL or OpenSubtitles keys in Settings), or none: the speech is then recognised and translated offline. |
| Different voices for different characters | Off: one voice reads every line (faster). On: the speakers are found automatically and each gets its own voice. |
| Profanity | **As in the original**, or **Soften** (no obscene words; softened lines are highlighted on the Lines tab). |
| Original voices under the dub | How loud the original dialogue stays under the dub (off to quiet). |
| Output file | **MKV** (recommended) or **MP4** (the video is copied; audio formats MP4 cannot hold are converted). |

Songs and musical inserts are kept in the original language.

<!-- screenshot: options panel -->

## 4. One-click dub

Press **Dub**. The progress bar shows the current step and the time left. The steps are: read the file, extract the audio, find subtitles, separate the voices from music and effects, recognise the speech, find the speakers, translate, prepare the voices, speak every line, mix, and write the new file.

A dub can be stopped at any time. Every step is kept in the project, so pressing **Dub** again continues where it stopped.

<!-- screenshot: dubbing in progress -->

## 5. Review before dubbing (optional)

Press **Prepare and review first…** instead of **Dub** to make the lines first and check them before any voice is spoken. A short "Worth a look" list points at the things that most often need a human eye.

### Characters and voices

With **Different voices for different characters** on, the **Characters and voices** tab shows one card per speaker with the amount of speech, a **Listen** button and the voice it will get.

- **Find speakers** runs the speaker search. The best search uses the pyannote model, which needs a free Hugging Face token (Settings > Hugging Face token); without it, a simpler search is used.
- Tick two or more cards and press **Merge selected** if one person was split into several speakers.
- **Key character**: key characters (usually one or two, about 20 % of the dialogue or more) get a voice like the actor's; the others get the closest voice from the library.
- **Actor likeness** (0-100 %, default 50 %): how much of the actor's own timbre an actor-like voice keeps. Lower values blend in more of the native library voice and carry less of the original accent.
- **Save to library** keeps a voice for later films (only if you have the right to use it).
- **Download free voices…** adds the free Voxprint voices to the shared voice library.

With one voice for the whole film, the tab has a single **Voice** choice instead.

<!-- screenshot: characters and voices, actor likeness slider -->

### Lines

The **Lines** tab lists every line with its time, speaker, original and translation.

- Double-click a translation to edit it. Shorter lines sound more natural.
- **Keep original** leaves a line untranslated.
- The **Fit** column shows lines that were too long for their time, sped up, moved into a pause, softened or kept in the original. **Only lines that are too long** filters the list.

<!-- screenshot: lines tab -->

## 6. Preview and watch

- **Preview fragment (1 min)** dubs one minute so you can hear the voices and the mix before the whole film.
- **Watch** opens the built-in player once the first five minutes are dubbed. You can start watching while the rest is still being dubbed; the player waits if it catches up with the dub.
- **Open in external player** opens the file in your usual player.

<!-- screenshot: built-in player -->

## 7. The result

The new file is written next to the original as `<name>.dub-<language>.mkv` (or `.mp4`). It contains the video (copied, not re-encoded), the new dubbed audio track, the original audio tracks and the subtitles. **Show the file** opens its folder.

<!-- screenshot: finished dub -->

## 8. Projects (.vxdub)

**Save** and **Save As…** write a `.vxdub` project file: the lines, characters, voices and the work done so far. Double-click a `.vxdub` file, or use **Open…**, to continue where you stopped. The video is not stored in the project; the project finds it by its path. If the video was moved, the program asks you to locate it and checks that it is the same file. The format is described in [formats/VXDUB.md](formats/VXDUB.md).

## 9. Settings

**Settings** holds the subtitle service keys, the Hugging Face token, the models folder, the engine choices and the device. Keys and the token stay in your user profile on this computer.

**VRAM tier (advanced)** is optional. Automatic picks the tier from the card; the override can only choose a tier the card supports. Models the card cannot run are greyed out in the lists.

The program pauses briefly between batches only when a long job keeps the card hot: full speed for the first 2.5 hours, then a pause while the 5-minute median GPU temperature is 83 °C or more, or while the card keeps throttling for more than a minute; work resumes at 75 °C. There is no setting for this.

## 10. Where things are

| What | Where |
| --- | --- |
| Program | `C:\Program Files\VoxprintMovieDubber` |
| AI models (shared with Audiobook Builder) | `%LOCALAPPDATA%\Voxprint\models` (can be changed in Settings) |
| Python runtime (shared) | `%LOCALAPPDATA%\Voxprint\runtime-<key>` |
| Projects and stage cache | `%LOCALAPPDATA%\VoxprintMovieDubber\projects` |
| Logs | `%LOCALAPPDATA%\VoxprintMovieDubber\logs` |
| Settings | `%LOCALAPPDATA%\VoxprintMovieDubber\state\settings.json` and the shared `%LOCALAPPDATA%\Voxprint\state\suite.json` |
| Diagnostic reports | saved to the Desktop, with a copy in `%LOCALAPPDATA%\VoxprintMovieDubber\reports` |

**Diagnostics** (the Start menu shortcut "Voxprint AI Movie Dubber - diagnostics", or the Diagnostics button) checks the card, memory and ffmpeg, loads every model and saves a report without personal data. See [DIAGNOSTICS.md](DIAGNOSTICS.md).

## 11. Uninstall

Use **Settings > Apps > Installed apps > Voxprint AI Movie Dubber > Uninstall**, or the uninstall shortcut in the Start menu folder "Voxprint".

- The uninstaller asks whether to delete your projects, logs and reports (default: keep).
- The shared models are deleted only when no other Voxprint program uses them and you answer Yes (default: keep).
- The shared Python runtime is removed when no other Voxprint program uses it.
- Voxprint AI Audiobook Builder is never touched.
