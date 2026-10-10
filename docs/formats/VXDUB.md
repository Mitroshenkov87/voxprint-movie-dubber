# `.vxdub` format version 1

A `.vxdub` file is a Voxprint dubbing project: the recognised lines, the translation, the voice
assignments, and the pipeline stage cache keys for one film. It does not contain the film.
The video stays where the user put it and is linked by path and a fast fingerprint.

The container follows the same rule as ODF and EPUB. The media type is
`application/vnd.voxprint.dub+zip`.

## Container

A `.vxdub` file is a ZIP archive.

1. The first member is named `mimetype`. It is stored (compression method 0), it has an empty
   extra field, and its bytes are exactly `application/vnd.voxprint.dub+zip` with no newline.
2. The JSON members follow, uncompressed or deflated:

   | Member | Role |
   | --- | --- |
   | `manifest.json` | Format version, app identity, timestamps, video reference, languages, settings |
   | `transcript.json` | Recognised lines: text, timings, speakers |
   | `translation.json` | Target-language text for those lines |
   | `voices.json` | Character to voice mapping, likeness, reference clips |
   | `job.json` | Pipeline stage states and cache keys, plus where each dubbed line was placed |

3. Optional voice-reference clips live under `assets/`. A clip is a file that already sits inside
   the local project folder (for example `voices/S1.wav` is stored as `assets/voices/S1.wav`).
4. Any other member is kept and written back on the next save. The film itself is never added,
   neither as a member nor as an asset.

JSON is UTF-8. Objects keep keys this version does not understand: a re-save writes them back.
A reader refuses `schema` greater than 1.

## `manifest.json`

```json
{
  "schema": 1,
  "app": {"id": "movie-dubber", "version": "1.0.0-rc", "build": 999},
  "created": "2026-10-10T09:14:00Z",
  "modified": "2026-10-10T09:14:00Z",
  "source": {
    "path": "/films/clip.mkv",
    "relative": "../films/clip.mkv",
    "size": 123456,
    "mtime": "2026-10-09T18:00:00Z",
    "fingerprint": "<sha256 hex>",
    "sha256": "<optional full sha256 hex>"
  },
  "source_lang": "en",
  "target_lang": "ru",
  "settings": {}
}
```

`schema` is the integer `1`. `app.id` is `movie-dubber`. `app.version` and `app.build` are the
program version and build that wrote the file. `created` and `modified` are ISO 8601 UTC with a
`Z` suffix and whole seconds. `created` is kept across saves; `modified` is the time of the save.

`source_lang` and `target_lang` repeat the languages stored in `settings` so a reader can see them
without interpreting the rest of the settings object. `settings` is the project's settings
dictionary (volumes, voice mode, subtitle choice, and the rest).

### Video reference

`source.path` is the absolute path at save time. `source.relative` is that file relative to the
directory that contains the `.vxdub`, with `/` separators, or `""` when the path cannot be made
relative. `source.size` is the byte size. `source.mtime` is the modification time in the same UTC
form as `created`.

`source.fingerprint` is the hex SHA-256 of these bytes, in order:

1. The first 16 MiB, or the whole file when it is shorter.
2. The last 16 MiB. When the file is at most 16 MiB this is the whole file again. When it is
   longer, the reader seeks to `size - 16 MiB`, so a file shorter than 32 MiB overlaps the head.
3. The file size as an unsigned 64-bit big-endian integer.

`source.sha256`, when present, is the hex SHA-256 of the entire file. Writers omit it unless a
full hash was requested or a previous save already stored one for the same fingerprint. Hashing
a whole movie on every save is not required.

Opening the file resolves the video in this order:

1. `source.path`, if that file exists, has the same size, and the fingerprint matches.
2. The `.vxdub` directory joined with `source.relative`, under the same checks.
3. Otherwise the program asks the user to locate the file and accepts it only when the size and
   fingerprint match. A different file is rejected and the project is not opened.

### Newer schemas

`schema` above 1 is an error. The message names the file's version and the version this program
opens, and tells the user to update the program. A save must not overwrite a newer file.

## `transcript.json`

Recognised lines. Each object has `id`, `start`, `end` (seconds), `text`, `speaker`, and the
other non-translation fields of a line (`kind`, `source`, `tag`, `keep_original`). Line order is
the order of this array.

## `translation.json`

`target_lang` and `lines`. Each line has the same `id` as the transcript plus `translation`,
`spoken`, `softened`, `review`, and `edited`.

## `voices.json`

`actor_weight` is the project default likeness (0 to 1). `single_voice` is the one-voice
assignment (`kind`, `id`, `actor_weight`). `characters` is one object per speaker:

- `id`, `name`
- `voice`: `kind` (`clone`, `library`, `actor`, `auto`) and library `id`
- `actor_weight` (number or null when the character uses the project default)
- `key` (true, false, or null)
- `seconds`, `ref_text`, `ref_audio` (path relative to the local project folder)
- `reference`: the member under `assets/` that holds the reference clip, when one was saved

`single_ref`, when present, is the reference for the single-voice mode (`audio`, `text`,
`reference`).

## `job.json`

`stages` is the pipeline stage map. A finished stage has `done: true` and `inputs`, the cache key
the runner compares on the next launch. `lines` holds the placement of each dubbed line: `id`,
`audio` (path of the synthesised take, relative to the project folder), `audio_s`, `place_start`,
`stretch`, and `fit`.

The audio takes, stems, and mix are not in the file. They stay in the local project folder, which
is chosen from the video's absolute path. Opening a `.vxdub` whose video path still maps to that
folder resumes the dub from `stages`. Opening it after the video has moved creates a new folder,
restores the script and the voice clips, and clears `done` so stages whose outputs are absent run
again. The `.vxdub` itself still records the cache keys from the save.

## Unknown keys and members

A reader keeps every JSON key it does not use, including keys inside `app`, `source`, `settings`,
line objects, and stage objects. On the next save those keys are written back. Members other than
`mimetype` and the five JSON names are copied back unchanged. Members and assets whose file name
is the video's file name are dropped so a re-save cannot start embedding the film.

Names under `assets/` are restored into the project folder only when the relative path does not
escape that folder.

## Writing

The file is written to a temporary name in the destination directory (`.<name>.<token>.tmp`) and
published with `os.replace`. A failed replace leaves the previous file in place and removes the
temporary file.

## File type

| | |
| --- | --- |
| Extension | `.vxdub` |
| Media type | `application/vnd.voxprint.dub+zip` |
| Windows ProgID | `Voxprint.MovieDubber.Project` |
| Windows description | Voxprint dubbing project |
| Open command | the program, with the project path as its first argument |

The document icon is `installer/vxdub.ico`, the application mark on a page. The Windows installer
registers the extension and the ProgID and deletes both on uninstall.
