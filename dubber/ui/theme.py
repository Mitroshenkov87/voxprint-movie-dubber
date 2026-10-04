"""Dark 'glass' theme shared with Voxprint AI Audiobook Builder (same palette, same stylesheet rules, trimmed to what this window uses).

Rules (kept from the audiobook program): every text colour is opaque and passes WCAG AA (>= 4.5:1) on its panel; panels/buttons/inputs are
almost solid; only the window background is translucent in Acrylic mode (~87 % opaque tint), so a white desktop still gives a dark window.
On Windows 11 22H2+ the window gets a real Acrylic backdrop (``platform_win.apply_backdrop``); elsewhere the solid fallback colour.
"""
from __future__ import annotations

TEXT = "#f2f2f5"
TEXT_MUTED = "#c4c4d0"
TEXT_FAINT = "#a8a8b8"
TEXT_DISABLED = "#8e8e9e"
TEXT_ON_ACCENT = "#0b1220"
ACCENT = "#60a5fa"
ACCENT_HOVER = "#7db9ff"
ACCENT_SOFT = "#93c5fd"
ACCENT_STRONG = "#2563eb"
ROOT_PLAIN = "#17171c"
ROOT_GLASS = "rgba(16,16,22,222)"
CARD_PLAIN = "#202029"
CARD_GLASS = "rgba(34,34,44,238)"
CONTROL_BG = "#2c2c38"
CONTROL_HOVER = "#383846"
CONTROL_BORDER = "#4b4b5c"
AMBER = "#fcd34d"
NOTE_BG = "#2a2418"
NOTE_BORDER = "#6b5720"
GREEN = "#86efac"
RED = "#fca5a5"

STYLE_TEMPLATE = """
* {{{{ font-family: "Segoe UI Variable Text", "Segoe UI", "Ubuntu", "Noto Sans", "Cantarell", "DejaVu Sans", sans-serif; font-size: 14px; color: {text}; }}}}
QWidget#root {{{{ background: {{root_bg}}; }}}}
QLabel#title {{{{ font-size: 26px; font-weight: 600; }}}}
QLabel#subtitle {{{{ color: {muted}; }}}}
QFrame#card {{{{ background: {{card_bg}}; border: 1px solid {border}; border-radius: 12px; }}}}
QFrame#card[primary="true"] {{{{ border: 2px solid {accent}; }}}}
QLabel#sectiontitle {{{{ font-size: 16px; font-weight: 600; background: transparent; }}}}
QLabel#fileLabel, QLabel#status {{{{ color: {muted}; background: transparent; }}}}
QLabel#hint {{{{ color: {faint}; font-size: 12px; background: transparent; }}}}
QLabel#warn {{{{ color: {amber}; font-size: 12px; background: transparent; }}}}
QLabel#ok {{{{ color: {green}; background: transparent; }}}}
QLabel#footer {{{{ color: {faint}; font-size: 12px; }}}}
QLabel#chip {{{{ color: {faint}; padding: 3px 8px; border-radius: 10px; font-size: 12px; background: #2a2a35; }}}}
QLabel#chip[state="ready"] {{{{ color: {on_accent}; background: {soft}; font-weight: 600; }}}}
QLabel#chip[state="stub"] {{{{ color: {faint}; background: #2a2a35; }}}}
QPushButton {{{{ background: {control}; border: 1px solid {border}; border-radius: 8px; padding: 8px 16px; }}}}
QPushButton:hover {{{{ background: {hover}; }}}}
QPushButton:pressed {{{{ background: #23232d; }}}}
QPushButton:disabled {{{{ color: {disabled}; background: #1f1f28; border-color: #34343f; }}}}
QPushButton#primary {{{{ background: {accent}; border: 1px solid {soft}; font-size: 17px; font-weight: 600; padding: 14px 20px; color: {on_accent}; }}}}
QPushButton#primary:hover {{{{ background: {accent_hover}; }}}}
QPushButton#primary:disabled {{{{ background: #2a3a55; color: #9db0cc; border-color: #3a4d6e; }}}}
QPushButton#gear {{{{ padding: 6px 12px; font-size: 18px; }}}}
QProgressBar {{{{ background: {control}; border: 1px solid {border}; border-radius: 8px; height: 16px; text-align: center; }}}}
QProgressBar::chunk {{{{ background: {strong}; border-radius: 7px; }}}}
QComboBox {{{{ background: {control}; border: 1px solid {border}; border-radius: 8px; padding: 6px 12px; min-width: 130px; }}}}
QComboBox QAbstractItemView {{{{ background: #23232b; border: 1px solid {border}; selection-background-color: {strong}; }}}}
QLineEdit {{{{ background: {control}; border: 1px solid {border}; border-radius: 8px; padding: 6px 12px; selection-background-color: {strong}; }}}}
QPlainTextEdit {{{{ background: #1b1b23; border: 1px solid {border}; border-radius: 8px; padding: 6px; font-family: "Cascadia Mono", "Consolas", "DejaVu Sans Mono", monospace; font-size: 12px; }}}}
QCheckBox {{{{ spacing: 8px; background: transparent; }}}}
QCheckBox::indicator {{{{ width: 16px; height: 16px; border: 1px solid {accent}; border-radius: 4px; background: {control}; }}}}
QCheckBox::indicator:checked {{{{ background: {strong}; border-color: {soft}; }}}}
QScrollArea {{{{ background: transparent; border: none; }}}}
QWidget#content {{{{ background: transparent; }}}}
QScrollBar:vertical {{{{ background: transparent; width: 10px; margin: 2px; }}}}
QScrollBar::handle:vertical {{{{ background: #5a5a6c; border-radius: 4px; min-height: 30px; }}}}
QScrollBar::add-line, QScrollBar::sub-line {{{{ width: 0; height: 0; }}}}
QMenu {{{{ background: #23232b; border: 1px solid {border}; }}}}
QMenu::item:selected {{{{ background: {strong}; }}}}
""".format(text=TEXT, muted=TEXT_MUTED, faint=TEXT_FAINT, disabled=TEXT_DISABLED, on_accent=TEXT_ON_ACCENT, accent=ACCENT,
           accent_hover=ACCENT_HOVER, soft=ACCENT_SOFT, strong=ACCENT_STRONG, control=CONTROL_BG, hover=CONTROL_HOVER,
           border=CONTROL_BORDER, amber=AMBER, green=GREEN)

#: (foreground, background) pairs that must keep >= 4.5:1 (checked by tests/test_ui.py).
CONTRAST_PAIRS = [(TEXT, ROOT_PLAIN), (TEXT, CARD_PLAIN), (TEXT, CONTROL_BG), (TEXT, CONTROL_HOVER), (TEXT_MUTED, ROOT_PLAIN),
                  (TEXT_MUTED, CARD_PLAIN), (TEXT_FAINT, ROOT_PLAIN), (TEXT_FAINT, CARD_PLAIN), (TEXT_ON_ACCENT, ACCENT),
                  (TEXT_ON_ACCENT, ACCENT_HOVER), (TEXT_ON_ACCENT, ACCENT_SOFT), (TEXT, ACCENT_STRONG), (GREEN, CARD_PLAIN),
                  (AMBER, CARD_PLAIN), (RED, CARD_PLAIN), (TEXT_FAINT, "#2a2a35"), ("#ffffff", ACCENT_STRONG)]


def build_style(glass: bool) -> str:
    """Stylesheet: ``glass`` = Acrylic backdrop behind a translucent window, else the solid fallback."""
    if glass:
        return STYLE_TEMPLATE.format(root_bg=ROOT_GLASS, card_bg=CARD_GLASS)
    return STYLE_TEMPLATE.format(root_bg=ROOT_PLAIN, card_bg=CARD_PLAIN)
