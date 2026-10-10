# Models

Movie Dubber downloads weights into the shared Voxprint model folder. Each entry below is the model the program
actually fetches. The licence in `dubber/models.py` and the licence in `dubber/infra/model_manifest.json` are the
licence on that model's Hugging Face card. A download is pinned to the revision below, and every file it fetches
is checked (size, then SHA-256) before the folder is used.

`pyannote/speaker-diarization-community-1` is gated and is not file-pinned here. Its licence is recorded on the
`diar` entry in the registry: CC-BY-4.0.

## Registry

| Key | Repository | Revision | Licence | Checksums |
|---|---|---|---|---|
| `tts_1_7b` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | `fd4b254389122332181a7c3db7f27e918eec64e3` | Apache-2.0 | pinned, verified on download |
| `tts_0_6b` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | `5d83992436eae1d760afd27aff78a71d676296fc` | Apache-2.0 | pinned, verified on download |
| `asr` | `deepdml/faster-whisper-large-v3-turbo-ct2` | `4df90f75321148c3a29a9e2351b7ddf8f5b115a8` | MIT | pinned, verified on download |
| `sep` | `JusperLee/TIGER-DnR` | `b7a59560bbca10febbcd46fb01600f868e587f57` | Apache-2.0 (weights), MIT (code) | pinned, verified on download |
| `diar` | `pyannote/speaker-diarization-community-1` | not file-pinned | CC-BY-4.0, gated | structural check only |
| `embed` | `pyannote/embedding` | `4db4899737a38b2d618bbd74350915aa10293cb2` | MIT, gated | pinned, verified on download |
| `roformer` | `KimberleyJSN/melbandroformer` | `ac9b0614ab3cd7f77219e18ba494dfd93956c348` | MIT | pinned, verified on download |
| `mt_en_ru` | `Helsinki-NLP/opus-mt-tc-big-en-zle` | `708be1d372fe4c358a352f404e6dc9ca0126ba48` | CC-BY-4.0 | pinned, verified on download |
| `mt_ru_en` | `Helsinki-NLP/opus-mt-tc-big-zle-en` | `09a40f722d6d8b76aaad6fe51a06c914622a13d1` | CC-BY-4.0 | pinned, verified on download |
| `mt_ru_de` | `Helsinki-NLP/opus-mt-tc-big-zle-de` | `b2e247f0c413ca6aa51a32f2f2be8666cf72405e` | CC-BY-4.0 | pinned, verified on download |
| `mt_de_ru` | `Helsinki-NLP/opus-mt-tc-big-de-zle` | `d4db2a2cbaa6c2f1ea57d0ed40924d35767b05f9` | CC-BY-4.0 | pinned, verified on download |
| `mt_en_de` | `Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-gmw` | `0a10a154b1d057720d03d8227bc59ea9633c590b` | Apache-2.0 | pinned, verified on download |
| `mt_de_en` | `Helsinki-NLP/opus-mt-tc-bible-big-gmw-deu_eng_fra_por_spa` | `a6933ed14d06d91d797809258bdb2d52f5487a36` | Apache-2.0 | pinned, verified on download |

The 2020 Helsinki-NLP `opus-mt-en-ru` and `opus-mt-de-en` cards are Apache-2.0. `opus-mt-ru-en` and `opus-mt-en-de` are
CC-BY-4.0. Those four checkpoints are not used. en↔ru and ru↔de use tc-big. en↔de uses tc-bible-big.

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

These components are used unmodified under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

**Opus-MT tc-big** (en→ru, ru→en, ru→de, de→ru). Language Technology Research Group at the University of Helsinki (Helsinki-NLP).
Source: [https://huggingface.co/Helsinki-NLP](https://huggingface.co/Helsinki-NLP).
Licence: [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).
Tiedemann, Aulamo, Bakshandaeva, Boggia, Grönroos, Nieminen, Raganato, Scherrer, Vázquez, Virpioja: "Democratizing neural machine translation with OPUS-MT", Language Resources and Evaluation 58 (2024).

## Apache-2.0 attributions

**Opus-MT tc-bible-big** (en→de, de→en). Language Technology Research Group at the University of Helsinki (Helsinki-NLP).
Source: [https://huggingface.co/Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-gmw](https://huggingface.co/Helsinki-NLP/opus-mt-tc-bible-big-deu_eng_fra_por_spa-gmw) and [https://huggingface.co/Helsinki-NLP/opus-mt-tc-bible-big-gmw-deu_eng_fra_por_spa](https://huggingface.co/Helsinki-NLP/opus-mt-tc-bible-big-gmw-deu_eng_fra_por_spa).
Licence: [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0).
Both model cards list the licence as Apache-2.0. en→de requires the sentence-initial token `>>deu<<`. de→en requires `>>eng<<`.

**pyannote speaker-diarization-community-1**. pyannote.
Source: [https://huggingface.co/pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).
Licence: [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).
Gated: the user accepts the model-card terms and supplies a Hugging Face token. `pyannote/embedding` is MIT, not CC BY.
