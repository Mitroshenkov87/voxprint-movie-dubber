# Models

VOXPrint MDL downloads weights into the shared Voxprint model folder. Movie Dubber does not download them.
The app manifest is `mdl.json` at the program root. Each entry below is a model that manifest pins.
The licence in `dubber/models.py` is the licence on that model's card. A download is pinned to the revision
below, and MDL checks every file (size, then SHA-256) before the folder is used. Diarization and the speaker
embedding are optional: one voice does not need them. Qwen3-TTS is written out in full, with the same
repository and revision Audiobook Builder uses, because voxprint-common 0.2.4 cannot resolve a shared-component
id into files. See the TODO in `mdl.json`.

The pyannote diarization pipeline and the speaker embedding are ungated mirrors. File hashes are the `SHA256SUMS`
published with those revisions. The pipeline config loads embedding, segmentation and PLDA from the mirror folder.

## Registry

| Key | Repository | Revision | Licence | Checksums |
|---|---|---|---|---|
| `tts_1_7b` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | `fd4b254389122332181a7c3db7f27e918eec64e3` | Apache-2.0 | pinned in `mdl.json` (24 GB) |
| `tts_0_6b` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | `5d83992436eae1d760afd27aff78a71d676296fc` | Apache-2.0 | pinned in `mdl.json` |
| `asr` | `deepdml/faster-whisper-large-v3-turbo-ct2` | `4df90f75321148c3a29a9e2351b7ddf8f5b115a8` | MIT | pinned in `mdl.json` |
| `sep` | `JusperLee/TIGER-DnR` | `b7a59560bbca10febbcd46fb01600f868e587f57` | Apache-2.0 (weights), MIT (code) | pinned in `mdl.json` |
| `diar` | `Mitroshenkov87/voxprint-pyannote-speaker-diarization-community-1` | `9c851af727f1085a361777d13e32734f7f5ebb30` | CC-BY-4.0 | pinned in `mdl.json` |
| `embed` | `Mitroshenkov87/voxprint-pyannote-embedding` | `4655ca656e1d7b1d8fe8100b39b4fdbf3015c7b8` | MIT | pinned in `mdl.json` |
| `roformer` | `KimberleyJSN/melbandroformer` | `ac9b0614ab3cd7f77219e18ba494dfd93956c348` | MIT | pinned in `mdl.json` |
| `mt_en_ru` | `Helsinki-NLP/opus-mt-tc-big-en-zle` | `708be1d372fe4c358a352f404e6dc9ca0126ba48` | CC-BY-4.0 | pinned in `mdl.json` |
| `mt_ru_en` | `Helsinki-NLP/opus-mt-tc-big-zle-en` | `09a40f722d6d8b76aaad6fe51a06c914622a13d1` | CC-BY-4.0 | pinned in `mdl.json` |
| `mt_ru_de` | `Helsinki-NLP/opus-mt-tc-big-zle-de` | `b2e247f0c413ca6aa51a32f2f2be8666cf72405e` | CC-BY-4.0 | pinned in `mdl.json` |
| `mt_de_ru` | `Helsinki-NLP/opus-mt-tc-big-de-zle` | `d4db2a2cbaa6c2f1ea57d0ed40924d35767b05f9` | CC-BY-4.0 | pinned in `mdl.json` |
| `mt_en_de` | `Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-gmw` | `0a10a154b1d057720d03d8227bc59ea9633c590b` | Apache-2.0 | pinned in `mdl.json` |
| `mt_de_en` | `Helsinki-NLP/opus-mt-tc-bible-big-gmw-deu_eng_fra_por_spa` | `a6933ed14d06d91d797809258bdb2d52f5487a36` | Apache-2.0 | pinned in `mdl.json` |

The 2020 Helsinki-NLP `opus-mt-en-ru` and `opus-mt-de-en` cards are Apache-2.0. `opus-mt-ru-en` and `opus-mt-en-de` are
CC-BY-4.0. Those four checkpoints are not used. en↔ru and ru↔de use tc-big. en↔de uses tc-bible-big.

## Roles

Pipeline stages ask `dubber/model_resolver.py` for a role. The resolver finds the matching `mdl.json` entry
(model id, plus the VRAM tier for Qwen3-TTS), then the complete local folder. Stage code does not name a
repository, a revision or a path, and nothing is downloaded. A missing model raises `ModelUnavailable`: the CLI
exits with code 4 (`models`) and the window shows its Manage models notice.

| Role | `mdl.json` entry | Used by | Placeholder |
|---|---|---|---|
| `separation` | `sep` | separation stage (TIGER-DnR) | yes |
| `separation.best` | `roformer` | separation stage, `roformer` option | yes |
| `asr` | `asr` | recognition, reference-clip check | yes |
| `diarization` | `diar` | speaker stage (optional) | yes |
| `embedding` | `embed` | voice matching (optional) | yes |
| `translation.<src>-<tgt>` | `mt-<src>-<tgt>` | translation stage | no |
| `tts.1_7b` | `qwen3-tts`, tier `24gb` | speech stage | yes |
| `tts.0_6b` | `qwen3-tts`, tier `16gb` | speech stage | yes |

Entries marked `"placeholder": true` in `mdl.json` point at the current upstream weights until Model Forge publishes
the suite model for that role. Replacing one is a `mdl.json` change only.

Silero VAD is not in `mdl.json`: its weights ship inside the `silero-vad` and `faster-whisper` wheels, which the
runtime lock pins.

## Memory between stages

The model session unloads every resident model the next stage does not use and empties the CUDA cache before the
stage starts. Speech synthesis therefore starts with separation, recognition and diarization off the card. The TTS
batch is sized from the free VRAM measured before each batch, leaving 8 % of the card (between 1.5 and 2.5 GB)
unused; an out-of-memory error halves the batch. A VRAM tier picked by hand still caps the lines per call.

## Mel-Band RoFormer

`KimberleyJSN/melbandroformer` commit `ac9b0614ab3cd7f77219e18ba494dfd93956c348` (2026-04-22T12:30:27Z, "Update README.md")
sets the model-card licence to MIT. The previous card (`f45f9e3d8570a406c94ac34b29f49ce43fda3bc8`, 2025-06-17) was GPL-3.0.
The pinned revision is that MIT commit. `MelBandRoformer.ckpt` was uploaded earlier (`94a0e5de2622a4160198b158e6f8141296da887e`,
2024-08-06); the file at the MIT revision is the one that is downloaded and checksummed.

audio-separator looks up `vocals_mel_band_roformer.ckpt`. The store keeps the published filename and links that expected
name to `MelBandRoformer.ckpt`, so the package uses its yaml and does not download a second copy of the weights.

## Translation pairs

tc-big models, the same repositories as Voxprint AI Audiobook Builder:

| Pair | Model | Target token |
|---|---|---|
| en → ru | `Helsinki-NLP/opus-mt-tc-big-en-zle` | `>>rus<<` |
| ru → en | `Helsinki-NLP/opus-mt-tc-big-zle-en` | none |
| ru → de | `Helsinki-NLP/opus-mt-tc-big-zle-de` | none |
| de → ru | `Helsinki-NLP/opus-mt-tc-big-de-zle` | `>>rus<<` |
| en → de | `Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-gmw` | `>>deu<<` |
| de → en | `Helsinki-NLP/opus-mt-tc-bible-big-gmw-deu_eng_fra_por_spa` | `>>eng<<` |

en→de and de→en use the tc-bible-big models. The model cards list `>>deu<<` and `>>eng<<` among the valid target-language labels. The 2020 Opus-MT checkpoints are not used.

## CC BY 4.0 attributions

The Opus-MT tc-big models below are used unmodified under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

**Opus-MT tc-big** (en→ru, ru→en, ru→de, de→ru). Language Technology Research Group at the University of Helsinki (Helsinki-NLP).
Source: [https://huggingface.co/Helsinki-NLP](https://huggingface.co/Helsinki-NLP).
Licence: [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).
Tiedemann, Aulamo, Bakshandaeva, Boggia, Grönroos, Nieminen, Raganato, Scherrer, Vázquez, Virpioja: "Democratizing neural machine translation with OPUS-MT", Language Resources and Evaluation 58 (2024).

**pyannote speaker-diarization-community-1**. Hervé Bredin and the pyannote.audio authors.
Original: [https://huggingface.co/pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).
Mirror: [https://huggingface.co/Mitroshenkov87/voxprint-pyannote-speaker-diarization-community-1](https://huggingface.co/Mitroshenkov87/voxprint-pyannote-speaker-diarization-community-1) at `9c851af727f1085a361777d13e32734f7f5ebb30`.
Licence: [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).
No changes besides removing the access gate. Segmentation, the WeSpeaker embedding and PLDA are the subfolders of that mirror.

## MIT attribution

**pyannote/embedding**. Copyright (c) 2022 CNRS. pyannote.audio.
Original: [https://huggingface.co/pyannote/embedding](https://huggingface.co/pyannote/embedding).
Mirror: [https://huggingface.co/Mitroshenkov87/voxprint-pyannote-embedding](https://huggingface.co/Mitroshenkov87/voxprint-pyannote-embedding) at `4655ca656e1d7b1d8fe8100b39b4fdbf3015c7b8`.
Licence: MIT.
No changes besides removing the access gate.

## Apache-2.0 attributions

**Opus-MT tc-bible-big** (en→de, de→en). Language Technology Research Group at the University of Helsinki (Helsinki-NLP).
Source: [https://huggingface.co/Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-gmw](https://huggingface.co/Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-gmw) and [https://huggingface.co/Helsinki-NLP/opus-mt-tc-bible-big-gmw-deu_eng_fra_por_spa](https://huggingface.co/Helsinki-NLP/opus-mt-tc-bible-big-gmw-deu_eng_fra_por_spa).
Licence: [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0).
Both model cards list the licence as Apache-2.0. en→de requires the sentence-initial token `>>deu<<`. de→en requires `>>eng<<`.
