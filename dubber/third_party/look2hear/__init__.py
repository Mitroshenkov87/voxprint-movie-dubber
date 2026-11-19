"""Vendored subset of TIGER (https://github.com/JusperLee/TIGER, MIT licence, (c) Kai Li) - only what TIGER-DnR inference needs.

Trimmed on purpose: the upstream ``look2hear.layers`` package imports librosa / torch_complex / distutils (training-only code);
here ``layers`` exposes only ``activations`` and ``normalizations``.  The model weights (JusperLee/TIGER-DnR) are Apache-2.0.
"""
