# Сборка и установщик (Voxprint AI Movie Dubber)

Всё ниже подготовлено на Linux и **не проверено на Windows** (кроме `tools/make_portable_zip.py` и тестов).

## Вариант A — portable zip (рекомендуется для первой диагностики)
`python tools/make_portable_zip.py` (на любой ОС) → `dist/Voxprint-MovieDubber-<ver>-portable.zip` (~1.5 МБ: исходники + bat-файлы + тестовый клип).
Пользователь распаковывает и запускает `setup.bat` (uv → Python 3.11 → torch CUDA → пакеты), затем `diagnose.bat`.
Плюсы: нет заморозки PyTorch/CUDA в exe (самое хрупкое место), модели и torch ставятся под драйвер пользователя. Установка ничего не меняет в системе.

## Вариант B — установщик (как у Audiobook Builder: PyInstaller --onedir + Inno Setup)
Требуется Windows x64, Python 3.11, при желании Inno Setup 6.
`build.bat exe` → тесты → `dist\VoxprintMovieDubber\` → `installer\Output\VoxprintMovieDubber-Setup.exe`
(`installer/VoxprintMovieDubber.iss`: установка на пользователя без админ-прав, ярлык «Диагностика», модели не включены).
Рабочие процессы запускаются как `VoxprintMovieDubber.exe --worker NAME args.json` — один exe в двух ролях; это нужно проверить на Windows.
Ожидаемые риски: размер (PyTorch+CUDA ≈ 3–5 ГБ — как у Audiobook Builder нужен «тонкий»/онлайн-вариант, см. его `installer/build_thin.bat`), скрытые импорты faster-qwen3-tts/transformers 5.

## Вариант C — GitHub Actions
`.github/workflows/build-windows.yml` (черновик, **не опубликован**): windows-latest, тесты на CPU-torch, диагностика на CPU (отчёт в artifacts),
portable zip, по флагу — PyInstaller + Inno Setup. GPU на раннерах нет, поэтому GPU-замеры возможны только на ноутбуке.
Чтобы использовать: создать приватный репозиторий (решение владельца), положить код, запустить workflow вручную.

## Открытые решения
1. Репозиторий: создать ли приватный репозиторий (имя, например `voxprint-movie-dubber`)? Пока всё только локально.
2. Где собирать установщик: на ноутбуке (`build.bat`) или в Actions приватного репозитория (бесплатный лимит минут).
3. Лицензия проекта (у Audiobook Builder — Apache-2.0; здесь не выбрана) и подписание exe.
4. FlashAttention на Windows: официальных колёс `flash-attn` нет; диагностика покажет SKIP, если пакет не установлен.
