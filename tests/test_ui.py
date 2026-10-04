import pytest

pytest.importorskip("PySide6")

from dubber.ui import theme


def _lum(hex_color):
    h = hex_color.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def ratio(a, b):
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


@pytest.mark.parametrize("fg,bg", theme.CONTRAST_PAIRS)
def test_wcag_aa_contrast(fg, bg):
    assert ratio(fg, bg) >= 4.5, (fg, bg, ratio(fg, bg))


def test_style_sheet_formats_cleanly():
    css = theme.build_style(True)
    assert "{{" not in css and "QPushButton" in css


def test_window_builds_and_has_the_main_controls(qtbot=None):
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from dubber.ui.window import MainWindow
    w = MainWindow()
    for name in ("btn_file", "cmb_target", "btn_diag", "btn_start", "btn_open", "btn_copy"):
        assert hasattr(w, name), name
    assert w.target_lang() in ("ru", "en", "de", "fr", "es")
    opt = w.build_options()
    assert opt.target_lang == w.target_lang()
    w.set_language("ru")
    assert w.btn_diag.text()
    w.close()
