"""Hardware gate: RTX 40 / Ada (compute capability 8.9) and NVIDIA driver branch 600, or the program does not install or start."""
from __future__ import annotations

from dubber.diag.report import Status
from dubber.infra.gpu_policy import (
    INSTALLER_LEAD,
    INSTALLER_LEAD_RU,
    SKIP_ENV,
    evaluate_nvidia_smi_query,
    failure_summary,
    found_text,
    gate_skipped,
    installer_message,
    startup_block,
)


def test_rtx_4090_laptop_8_9_passes():
    text = "NVIDIA GeForce RTX 4090 Laptop GPU, 8.9, 617.42\n"
    result = evaluate_nvidia_smi_query(text)
    gpu = result.gpus[0]
    assert result.ok
    assert gpu.name == "NVIDIA GeForce RTX 4090 Laptop GPU"
    assert gpu.compute == (8, 9)
    assert gpu.driver == "617.42" and gpu.driver_branch == 617
    assert "8.9" in result.summary and "617" in result.summary


def test_rtx_3080_8_6_fails():
    result = evaluate_nvidia_smi_query("NVIDIA GeForce RTX 3080, 8.6, 617.42\n")
    assert not result.ok
    assert result.gpus[0].compute == (8, 6)
    assert result.summary == failure_summary(result.gpus)
    assert "NVIDIA GeForce RTX 3080, compute capability 8.6, driver branch 617: below the minimum." in result.summary
    assert "compute capability 8.9 or higher" in result.summary
    assert "driver branch 600 or newer" in result.summary


def test_no_gpu_fails():
    for text in ("", "   \n", "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver\n"):
        result = evaluate_nvidia_smi_query(text)
        assert not result.ok and result.gpus == ()
        assert result.summary.startswith("No NVIDIA GPU found.")


def test_driver_560_fails():
    result = evaluate_nvidia_smi_query("NVIDIA GeForce RTX 4090 Laptop GPU, 8.9, 560.94\n")
    assert not result.ok
    assert result.gpus[0].compute == (8, 9) and result.gpus[0].driver_branch == 560
    assert "driver branch 560: below the minimum." in result.summary


def test_driver_branch_600_is_the_minimum_and_one_ada_gpu_is_enough():
    assert evaluate_nvidia_smi_query("NVIDIA GeForce RTX 4090 Laptop GPU, 8.9, 600.00\n").ok
    assert not evaluate_nvidia_smi_query("NVIDIA GeForce RTX 4090 Laptop GPU, 8.9, 599.99\n").ok
    mixed = "NVIDIA GeForce RTX 3080, 8.6, 617.42\nNVIDIA GeForce RTX 4090 Laptop GPU, 8.9, 617.42\n"
    result = evaluate_nvidia_smi_query(mixed)
    assert result.ok and result.qualifying[0].name.endswith("4090 Laptop GPU")


def test_installer_message_names_the_gpu_we_found():
    missing = evaluate_nvidia_smi_query("")
    old = evaluate_nvidia_smi_query("NVIDIA GeForce RTX 3080, 8.6, 560.70\n")
    assert found_text(missing.gpus) == "No NVIDIA GPU was found."
    assert found_text(old.gpus) == "NVIDIA GeForce RTX 3080, compute capability 8.6, driver 560.70 (branch 560)."
    text = installer_message(found_text(old.gpus))
    assert text.startswith(INSTALLER_LEAD)
    assert "A processor-only copy is not offered." in text
    assert "What we found: NVIDIA GeForce RTX 3080, compute capability 8.6, driver 560.70 (branch 560)." in text
    assert "Install a supported graphics card and driver, then run setup again." in text
    assert "вычислительная способность 8.9" in INSTALLER_LEAD_RU


def test_startup_blocks_without_cuda_or_below_8_9_and_ci_can_skip(monkeypatch):
    blocked = startup_block(False, None, "")
    assert blocked is not None and blocked.code == "no_cuda"
    low = startup_block(True, (8, 6), "NVIDIA GeForce RTX 3080")
    assert low is not None and low.code == "low_compute" and low.capability == "8.6" and low.name.startswith("NVIDIA")
    assert startup_block(True, (8, 9), "NVIDIA GeForce RTX 4090 Laptop GPU") is None
    assert startup_block(True, (9, 0), "NVIDIA GeForce RTX 5090") is None
    monkeypatch.delenv(SKIP_ENV, raising=False)
    assert not gate_skipped()
    monkeypatch.setenv(SKIP_ENV, "1")
    assert gate_skipped()


def test_window_gate_shows_the_localized_dialog(monkeypatch):
    import main as app_main
    from dubber import i18n

    monkeypatch.delenv(SKIP_ENV, raising=False)
    shown = {}

    class _Box:
        @staticmethod
        def critical(_parent, title, body):
            shown["title"] = title
            shown["body"] = body

    monkeypatch.setattr(app_main, "probe_torch", lambda: (False, None, ""))
    assert app_main._refuse_unsupported_gpu(_Box) is True
    assert shown["title"] == "This PC cannot run Voxprint AI Movie Dubber"
    assert "PyTorch does not see a usable NVIDIA GPU on this PC." in shown["body"]
    assert "compute capability 8.9 or higher" in shown["body"]
    assert "The program will close." in shown["body"]

    monkeypatch.setattr(app_main, "probe_torch", lambda: (True, (8, 6), "NVIDIA GeForce RTX 3080"))
    assert app_main._refuse_unsupported_gpu(_Box) is True
    assert "NVIDIA GeForce RTX 3080" in shown["body"] and "8.6" in shown["body"]

    monkeypatch.setattr(app_main, "probe_torch", lambda: (True, (8, 9), "NVIDIA GeForce RTX 4090 Laptop GPU"))
    assert app_main._refuse_unsupported_gpu(_Box) is False

    i18n.set_language("ru", save=False)
    monkeypatch.setattr(app_main, "probe_torch", lambda: (False, None, ""))
    assert app_main._refuse_unsupported_gpu(_Box) is True
    assert shown["title"] == "Этот компьютер не может запустить Voxprint AI Movie Dubber"
    assert "PyTorch не видит подходящую видеокарту NVIDIA на этом компьютере." in shown["body"]
    assert "8.9" in shown["body"] and "600" in shown["body"]

    monkeypatch.setenv(SKIP_ENV, "1")
    assert app_main._refuse_unsupported_gpu(_Box) is False


def test_startup_dialog_is_localized():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "dubber" / "locales"
    en = json.loads((root / "en.json").read_text(encoding="utf-8"))
    ru = json.loads((root / "ru.json").read_text(encoding="utf-8"))
    assert en["gpu.gate_title"] == "This PC cannot run Voxprint AI Movie Dubber"
    assert en["gpu.gate_no_cuda"] == "PyTorch does not see a usable NVIDIA GPU on this PC."
    assert "{name}" in en["gpu.gate_low_compute"] and "{capability}" in en["gpu.gate_low_compute"]
    assert "8.9" in en["gpu.gate_body"] and "{detail}" in en["gpu.gate_body"]
    assert ru["gpu.gate_title"] == "Этот компьютер не может запустить Voxprint AI Movie Dubber"
    assert "8.9" in ru["gpu.gate_body"] and "600" in ru["gpu.gate_body"]
    assert ru["gpu.gate_no_cuda"]
    assert "{name}" in ru["gpu.gate_low_compute"]


def test_installer_script_gates_on_compute_and_driver_and_keeps_cpu_for_ci_only():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    ps1 = (root / "installer" / "install-runtime.ps1").read_text(encoding="utf-8")
    iss = (root / "installer" / "VoxprintMovieDubber.iss").read_text(encoding="utf-8")
    assert "--query-gpu=name,compute_cap,driver_version" in ps1
    assert INSTALLER_LEAD in ps1 and INSTALLER_LEAD_RU in ps1
    assert '-Backend cpu' in ps1 or '$Backend -eq "cpu"' in ps1
    assert "exit 2" in ps1
    assert "-ge 560" not in ps1 and "-ge 570" not in ps1
    assert "cu126" not in ps1 and "cu128" not in ps1
    assert "-ge 580" in ps1 and "@(13, 0)" in ps1
    assert "HardwareRequired" in iss or "HardwareFile" in iss
    assert INSTALLER_LEAD in iss


def test_lock_drops_cu126_and_auto_never_selects_cpu():
    from dubber.infra import runtime

    lock = runtime.load_lock()
    assert "cu126" not in lock["flavors"] and "cu128" not in lock["flavors"]
    assert lock["python"] == "3.14" and lock["torch_version"] == "2.11.0"
    assert "cpu" in lock["flavors"] and "cu130" in lock["flavors"]
    assert not any(w.get("flavor") in ("cu126", "cu128") for w in lock["wheels"])
    assert any(w["dist"] == "torch" and w["version"] == "2.11.0+cu130" and "cp314" in w["file"] for w in lock["wheels"])
    assert any(w["dist"] == "torchaudio" and w["version"] == "2.11.0+cu130" for w in lock["wheels"])
    assert any(w["dist"] == "torchcodec" and w["version"] == "0.17.0+cu130" for w in lock["wheels"])
    assert any(w["dist"] == "nvidia-cublas-cu12" for w in lock["wheels"])
    assert any(w["dist"] == "nvidia-cudnn-cu12" for w in lock["wheels"])
    assert runtime.choose_flavor(lock, (12, 8)) is None
    assert runtime.choose_flavor(lock, (13, 0)) == "cu130"
    assert runtime.choose_flavor(lock, (13, 4)) == "cu130"
    assert runtime.choose_flavor(lock, (12, 6)) is None
    assert runtime.choose_flavor(lock, None) is None


def test_diagnostics_report_name_compute_and_driver_branch_and_fail_below_minimum(monkeypatch):
    from dubber.diag import checks_system as cs

    def payload(name, cap, driver, branch):
        return {
            "smi": "nvidia-smi",
            "cuda_driver_api": "13.0",
            "gpus": [{
                "name": name, "driver": driver, "compute_cap": cap, "driver_branch": branch,
                "vram_total_gb": 16.0, "vram_used_gb": 0.4, "power_limit_w": "150", "power_max_w": "175",
                "sm_max_mhz": "2000", "mem_max_mhz": "8000", "pcie": "gen4 x16", "temp_c": "40", "util": "0",
                "persistence": "Disabled", "display_active": "Enabled",
            }],
        }

    monkeypatch.setattr(cs, "query_nvidia_smi", lambda: payload(
        "NVIDIA GeForce RTX 4090 Laptop GPU", "8.9", "617.42", 617))
    ok = cs.check_gpu_smi()
    assert ok.status == Status.OK
    assert ok.metrics["compute_cap"] == "8.9" and ok.metrics["driver_branch"] == 617
    assert "compute capability 8.9" in ok.summary and "branch 617" in ok.summary

    monkeypatch.setattr(cs, "query_nvidia_smi", lambda: payload("NVIDIA GeForce RTX 3080", "8.6", "617.42", 617))
    low = cs.check_gpu_smi()
    assert low.status == Status.FAIL
    assert "8.6" in low.summary and "below the minimum" in low.summary

    monkeypatch.setattr(cs, "query_nvidia_smi", lambda: payload(
        "NVIDIA GeForce RTX 4090 Laptop GPU", "8.9", "560.94", 560))
    old = cs.check_gpu_smi()
    assert old.status == Status.FAIL and "560" in old.summary

    monkeypatch.setattr(cs, "query_nvidia_smi", lambda: {})
    missing = cs.check_gpu_smi()
    assert missing.status == Status.FAIL and missing.summary.startswith("No NVIDIA GPU found.")


def test_settings_ui_has_no_cpu_device(monkeypatch):
    import pytest

    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from dubber import settings
    from dubber.infra import suite
    from dubber.ui.dialogs import SettingsDialog

    QApplication.instance() or QApplication([])
    monkeypatch.delenv("VOXPRINT_SKIP_GPU_GATE", raising=False)
    assert settings.device() == "auto"
    suite.set_value("gpu", "cpu")
    assert settings.device() == "auto"
    dialog = SettingsDialog()
    choices = [dialog.cmb_device.itemData(i) for i in range(dialog.cmb_device.count())]
    assert choices == ["auto", "cuda"]
    dialog.cmb_device.setCurrentIndex(dialog.cmb_device.findData("cuda"))
    dialog.accept()
    assert settings.device() == "cuda" and suite.gpu() == "cuda:0"
    dialog.close()


def test_cuda12_dll_dirs_come_from_the_nvidia_wheels(tmp_path, monkeypatch):
    import os

    from dubber.infra import cuda_dlls

    site = tmp_path / "site"
    cublas = site / "nvidia" / "cublas" / "bin"
    cudnn = site / "nvidia" / "cudnn" / "bin"
    cublas.mkdir(parents=True)
    cudnn.mkdir(parents=True)
    (cublas / "cublas64_12.dll").write_bytes(b"cublas")
    (cudnn / "cudnn64_9.dll").write_bytes(b"cudnn")
    monkeypatch.setattr(cuda_dlls.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(cuda_dlls.sys, "path", [str(site)])
    dirs = cuda_dlls.candidate_dirs()
    assert any(d.parent.name == "cublas" for d in dirs)
    assert any(d.parent.name == "cudnn" for d in dirs)
    assert cuda_dlls.has_cublas12(dirs)

    added: list[str] = []
    monkeypatch.setattr(cuda_dlls.os, "add_dll_directory", lambda path: added.append(path), raising=False)
    monkeypatch.setattr(cuda_dlls.sys, "platform", "win32")
    monkeypatch.setattr(cuda_dlls, "candidate_dirs", lambda: [cublas, cudnn])
    old_path = os.environ.get("PATH", "")
    cuda_dlls._done.clear()
    try:
        registered = cuda_dlls.expose()
        assert registered == [str(cublas), str(cudnn)]
        assert added == [str(cublas), str(cudnn)]
    finally:
        os.environ["PATH"] = old_path
        cuda_dlls._done.clear()


def test_ctranslate2_cuda_check_fails_when_no_device_is_visible(monkeypatch):
    from dubber.diag import checks_system as cs
    from dubber.infra import cuda_dlls

    ok, text = cuda_dlls.cuda_device_summary(1)
    assert ok and text == "ctranslate2 sees 1 CUDA device."
    missing, missing_text = cuda_dlls.cuda_device_summary(None)
    assert not missing and missing_text.startswith("ctranslate2 is not installed")
    bad, bad_text = cuda_dlls.cuda_device_summary(0)
    assert not bad and "sees no CUDA device" in bad_text

    monkeypatch.setattr(cuda_dlls, "expose", lambda: [r"C:\site\nvidia\cublas\bin"])
    monkeypatch.setattr(cuda_dlls, "has_cublas12", lambda _dirs: True)
    monkeypatch.setattr(cuda_dlls, "cuda_device_count", lambda: 0)
    failed = cs.check_ctranslate2()
    assert failed.status == Status.FAIL and "no CUDA device" in failed.summary

    monkeypatch.setattr(cuda_dlls, "cuda_device_count", lambda: 1)
    passed = cs.check_ctranslate2()
    assert passed.status == Status.OK and "1 CUDA device" in passed.summary
