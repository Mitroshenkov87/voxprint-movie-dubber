# Roadmap

What we plan to add to Voxprint AI Movie Dubber. Plans can change; each release's notes list what actually shipped.

## Next version

- **Meaning-based translation:** an optional local language model translates each line with the scenes around it in mind, keeping the meaning and tone, adapting idioms and shortening long lines, instead of translating word for word.
- **Lines that fit the first time:** the program estimates how long a translated line will take to say before it voices it, so fewer lines need to be re-voiced and dubbing finishes sooner.
- **Full command-line control:** choosing the source, editing characters and voices, and making a preview will also work from the command line, so the whole job can be automated without opening the window.
- **Linux package:** an installable package for Linux distributions released in 2025 or later.
- **Tuning on real graphics cards:** speeds, batch sizes and memory use tuned from the first tests on real RTX hardware, so dubbing runs faster and more steadily on 16 GB and 24 GB cards.

## Later

- **Lip-sync-aware timing:** dubbed lines are timed to the actors' mouth movements where it helps, without changing the video.
- **Watch with almost no wait:** on cards with enough memory, you can start watching the dub shortly after pressing Dub, instead of waiting for a few minutes of dubbed film first.
- **Optional video re-encode:** if you want a smaller file, the video can be re-encoded to the open AV1 format on export; leaving the video untouched stays the default.

## Models and performance

- **Smart memory placement:** the program splits work between graphics memory, system memory and disk, so better models can run on 16 GB and 24 GB cards without a big speed loss:
  - **One part at a time:** a model's parts are loaded one after another and unloaded when done, instead of shrinking the model and losing quality.
  - **Small helpers off the graphics card:** speaker encoders, aligners and similar small parts stay in system memory or run on the processor, leaving graphics memory for the main models.
  - **Long context in system memory:** the translation model keeps the memory of earlier lines in system memory, so it can use more of the film's context.
  - **Bigger translation models:** for translation models built from many experts, the experts that are not in use wait in system memory, so a larger and better model fits on the card.
  - **Next step loads in advance:** while one step runs, the next step's model is read from disk into system memory, so the hand-over between steps is shorter.
