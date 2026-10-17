# Feature map

The shared audio screen is the main screen. Every control that used to live on the four-step window is still that control, opened from the overflow menu as the Review wizard. Expert mode is the side panel on the main screen. Nothing in this table was removed. A row marked **NEEDS DECISION** has no place on the new screen yet.

The wizard steps are Film, Characters, Lines, and Dub. The Dub step is the player, the mix, the one-minute preview, and Watch. Characters, Lines, and the player stay on Film until Prepare has finished, which is the same rule the old step buttons used.

## Main screen

| Old feature | New place |
| --- | --- |
| Choose a video (button, click, or drop; mkv, mp4, avi, mov, webm, m4v, ts, m2ts) | Main screen drop target. The Review wizard Film step has the same chooser. |
| Dub language: Russian, English, or German | Language chips on the main screen, the Expert language field, and the Film step combo. One language per film, as before. |
| Quality: Fast, Balanced, Best quality | Quality chips. Expert shows the concrete speech, separation, and batch values. |
| Voices: Keep actor voices, Neutral voices, or Custom | Voices chips. Custom opens a dialog titled Choose voices, on the shell theme. After Prepare, each row is a diarization speaker (Voice 1, Voice 2), a film excerpt, a library voice, and a preview of that library voice. Before Prepare it asks to prepare. The same multi-voice flag and per-speaker voice are what the Characters step and Expert use. Likeness stays in Expert. |
| Actor likeness, default 50% | Expert field shows 50%. The project still stores 0 to 1. Film step slider when one voice is used. Each character card has its own slider. |
| Audiobook narrator voice (Warm, Clear, Low) | Hidden. That row belongs to the audio surface and is not a film voice. Character voices stay on the Characters step and in Expert. |
| Different voices for different characters | Expert multi-voice field and the Film step checkbox. |
| Profanity: as in the original, or soften | Expert field and the Film step combo. Softened lines stay highlighted on the Lines step. |
| Original voices under the dub | Expert mix field, the Film step slider, and the Dub step slider. |
| Output file: MKV or MP4 | Expert field and the Film step combo. |
| Subtitles: automatic, none, a chosen file, or online when keys are set | Expert field is automatic or none. The Film step adds the file chooser and the status line. Keys are in Settings and in Expert. |
| Original audio track, including automatic | Film step, Original audio track. |
| Film details: duration, video format, audio tracks, embedded subtitles | Film step, under the file name. |
| One-click Dub, full pipeline through mix and mux, resume from finished stages | Dub on the main screen. It runs the same pipeline as the Film step Dub button. |
| Progress, current step, time left, cancel | Main screen progress and Cancel. The Film and Dub steps show the same job. |
| Prepare and review, then the "worth a look" notes | Film step, Prepare and review. Attention stays on that step and on the step buttons inside the wizard. |
| Show the finished file, open its folder | History Open and Show folder. Film step Show the file. Dub step Show the file. |
| Open, Save, and Save As for a .vxdub project | Overflow menu: Open…, Save, Save As…. The wizard File menu has the same three actions. A project file passed on the command line opens here. |
| Video moved: ask for the file and check it is the same one | Open project, same dialogs as before. |
| Video preview on the main screen | Video preview on main screen: shell video result region (Yitro), pending |

## Characters

| Old feature | New place |
| --- | --- |
| Character cards: name, speech time, line count, sample lines | Overflow menu, Characters. Review wizard, Characters step. |
| Voice for each character | Character card combo. |
| Listen | Character card Listen. |
| Key character | Character card checkbox. |
| Actor likeness slider on a card (0–100%) | Character card slider. |
| Merge selected cards | Characters step, Merge selected. |
| Find speakers | Characters step, Find speakers. |
| Rename a character | Character card name field. |
| Save an actor-like voice to the library, with the consent question | Character card, Save to library. |
| Download free voices | Characters step, Download free voices…. |
| One voice for the whole film: voice choice and likeness | Characters step, shown when multi-voice is off. The Film step has the same voice combo and slider. |

## Lines

| Old feature | New place |
| --- | --- |
| Line table: time, speaker, original, translation, fit, keep original | Overflow menu, Lines. Review wizard, Lines step. |
| Edit a translation (double-click) | Lines table, translation column. |
| Keep original | Lines table, keep column. |
| Fit flags: too long, sped up, shifted, softened, kept, AI | Lines table, Fit column. |
| Only lines that are too long | Lines step checkbox. |
| Reassign a line to another speaker | Lines table, speaker column, when multi-voice is on. |
| Softened lines highlighted | Lines table, translation cell. |
| Re-dub a line | Editing the translation clears that line's audio. The next Dub speaks it again and leaves finished lines as they are. |
| Editing locked while a job runs | Lines step, same rule as before. |

## Player, mix, and the job

| Old feature | New place |
| --- | --- |
| Built-in player: picture, dubbed sound, play, pause, stop, seek | Overflow menu, Player. Review wizard, Dub step. |
| Preview fragment (1 minute), including the poster frame | Dub step, Preview fragment (1 min). |
| Watch while the rest is still dubbing, and wait if playback catches up | Dub step, Watch. The Film step shows Watch once the first five minutes are ready. |
| Open in an external player | Dub step, Open in external player. |
| Mix level while watching the result | Dub step slider, kept in step with the Expert mix field and the Film slider. |
| Job log | Dub step log. |
| One running dub at a time, with cancel | Main screen Cancel, Film step Cancel, and Dub step Cancel. |
| MDL download queue | Overflow menu, Manage models (`voxprint-mdl --gui --app movie-dubber`). |
| A waiting list of several films | **NEEDS DECISION.** The old window ran one pipeline job. |

## Settings, models, and startup

| Old feature | New place |
| --- | --- |
| Settings: subtitle service keys, models folder, separation, diarization, TTS model, backend, device, VRAM tier | Overflow menu, Settings. The same fields are in the Expert panel. The wizard Settings button opens the same dialog. |
| OpenSubtitles user and password | Settings dialog. |
| Manage models | Overflow menu, Manage models. |
| Optional speaker models missing, and retry | A note on the main screen. The wizard shows the same banner and retry button. |
| Diagnostics, short run, report on the Desktop, open, show folder, copy | Overflow menu, Diagnostics. `--diagnose` still opens the window and starts the checks. |
| Check and repair shared runtime, ffmpeg, and models | Overflow menu, Check and repair. The same check still runs before the window opens. |
| Splash screen and the loading easter egg | Unchanged, before the main screen. |
| GPU and driver minimum, then close | Unchanged, before the main screen. |
| Version, build, and codename | Settings dialog and the wizard footer. |
| Interface language menu | Wizard gear menu. **NEEDS DECISION** for changing the main screen language without opening the window again. The main screen uses the language it was opened with. |
| Thermal pause on a long job | Unchanged. There was no control for it. |
| Songs and musical inserts kept in the original language | Unchanged pipeline behavior. There was no separate control. |

## Command line

The commands still run as `voxprint-movie-dubber-cli`. The rows below say where the same job can be started from the window.

| Old feature | New place |
| --- | --- |
| `version`, `--version` | Settings dialog version line. The command itself is unchanged. |
| `diagnose`, `--diagnose-cli` | Diagnostics dialog. |
| `--diagnose` | Opens the main screen and starts diagnostics. |
| `fetch-models`, `--fetch-models` | Manage models. |
| `run-project`, `--run-project` | Dub on the main screen and Dub on the Film step. |
| `info`, `project-info`, `--project-info` | Film step details. Line count is on the Lines step after Prepare. The command itself is unchanged. |
| `selftest`, `--selftest` | Check and repair. `--selftest` still checks, opens the window briefly, and exits. `--selftest --json` stays headless. |
| `--help`, `-h` | **NEEDS DECISION.** The command list stays on the command line. |
| `--json` | **NEEDS DECISION.** Headless output only. |
| `--dry-run` | **NEEDS DECISION.** Discovery JSON stays on the command line. |
| `--quick` | Diagnostics dialog, shorter run. |
| `--no-download` | Diagnostics dialog never downloads. |
| `--target` | Diagnostics use the language chosen on the main screen. |
| `--out` for a diagnostics report | **NEEDS DECISION.** The dialog writes the report to the Desktop and can open that folder. |
| `--skip`, `--tts-modes`, `--cpu-tts`, `--cpu-sep`, `--clip`, `--asr-repo` | **NEEDS DECISION.** Diagnostic flags with no window control. The command still accepts them. |
| `--tts-model` | Settings TTS model, which the diagnostics dialog reads. |
| `--models`, `--report` on fetch-models | **NEEDS DECISION.** Manage models is the window. The flags stay on the command. |
| `--languages`, `--source-lang`, `--target-lang`, `--stages` | Target language is the Language chips. Source stays automatic unless a project already has one. Stage selection stays on the command. **NEEDS DECISION** for picking a source language or a last stage in the window. |
| `--worker` | **NEEDS DECISION.** Internal worker process. |
| `--register-models-user`, `--unregister-models-user` | **NEEDS DECISION.** Installer and uninstaller. |
| `--register-runtime-user`, `--unregister-runtime-user` | **NEEDS DECISION.** Installer and uninstaller. |
| `--release-shared`, `--sync-suite-settings` | **NEEDS DECISION.** Installer and uninstaller. |
| `--out` for unregister or release-shared | **NEEDS DECISION.** Uninstaller. |
| Positional `.vxdub` file | Opens that project in the Review wizard. |

## Not a new preview pane

The main screen result area is still the audio player from `surface_for("movie-dubber")`: waveform bars and a chapters label. The film player with play, pause, and seek is the existing player on the Dub step.

Video preview on main screen: shell video result region (Yitro), pending. `ShellWindow._bind_video_result` looks for a widget named `videoResult` and does not draw one. The existing player stays on the Dub step until that region is in the pinned shell.
