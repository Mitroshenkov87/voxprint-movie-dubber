# Voxprint AI Movie Dubber — каркас (skeleton) v0.1

Windows-программа (PySide6), которая добавляет в фильм дополнительную дорожку с дубляжом.
**Это каркас:** окно, выбор файла и языка, заглушки шагов дубляжа и главное — **встроенная диагностика**:
она измеряет, как все модели конвейера ведут себя на вашем ноутбуке (RTX), и сохраняет **текстовый отчёт на Рабочий стол**.

Стек как у Voxprint Audiobook Builder: Python 3.11, PySide6 (тёмное «стеклянное» окно, Acrylic на Windows 11),
PyTorch + CUDA через `uv --torch-backend=auto`, Qwen3-TTS, Opus-MT. Отличие: TTS идёт через `faster-qwen3-tts` (CUDA Graphs)
и transformers 5.x — поэтому у программы **своё окружение**, не смешивайте с Audiobook Builder.

## Что измеряет диагностика
GPU / VRAM / RAM / драйвер / CUDA, питание ноутбука; наличие ffmpeg; доступность FlashAttention и CUDA Graphs (и работают ли они реально);
время загрузки каждой модели (VAD, разделение TIGER-DnR, ASR faster-whisper, диаризация pyannote, Opus-MT, Qwen3-TTS);
скорость TTS (RTF) на коротких фразах в 4 режимах: обычный SDPA / +FlashAttention-2 / CUDA Graphs / Graphs+FA2;
время каждого этапа на встроенном коротком клипе (`assets/test_clip`, 18 с, синтетический); ошибки с трейсбэками.
Каждая проверка в отдельном процессе и под защитой: сбой или краш одной не останавливает отчёт.
Отчёт: `Рабочий стол\Voxprint-MovieDubber-Diagnostics-<дата>.txt` (английский, удобно вставлять в чат; токены и имя пользователя скрываются).

## Быстрый старт (Windows 10/11 x64, NVIDIA)
1. Распакуйте `Voxprint-MovieDubber-<версия>-portable.zip` (или склонируйте репозиторий).
2. Один раз запустите **`setup.bat`** — скачает uv, Python 3.11, PyTorch (CUDA под ваш драйвер) и пакеты (~4 ГБ, нужен интернет).
3. Запустите **`diagnose.bat`** (или `run.bat` → кнопка «Запустить диагностику»). Первый запуск скачает модели (~6.5 ГБ; можно запретить галочкой).
   Для диаризации нужен токен Hugging Face и принятые условия модели `pyannote/speaker-diarization-community-1` — без него шаг будет пропущен.
4. Пришлите файл отчёта с Рабочего стола.
Если окно не открывается: `diagnose-console.bat`. Ключи: `--quick`, `--no-download`, `--out <файл>`.

## Разработка и сборка
```
python -m venv .venv && .venv\Scripts\activate         (Linux: source .venv/bin/activate)
pip install uv && uv pip install torch torchaudio --torch-backend=auto && uv pip install -r requirements-dev.txt
python -m pytest            # 59 тестов, GPU не нужен
python main.py              # окно;   python main.py --diagnose-cli   # диагностика в консоли
build.bat                   # тесты + portable zip (dist\);   build.bat exe  # + PyInstaller и установщик Inno Setup
```
Подробнее (установщик, CI, что не проверено) — `docs/BUILDING.md`.

## Статус
Проверено на Linux без GPU (CPU): весь конвейер диагностики, отчёт, GUI-smoke, 59 тестов, реальный запуск Qwen3-TTS 0.6B на CPU.
**Не проверено:** любые GPU-пути (CUDA Graphs, FlashAttention, VRAM-метрики), Windows-скрипты `*.bat`, PyInstaller/Inno Setup, Acrylic.
Лицензия: «все права защищены» (временно, см. `LICENSE`); открытая лицензия не выбрана. Стороннее — `THIRD_PARTY_NOTICES.md`.
