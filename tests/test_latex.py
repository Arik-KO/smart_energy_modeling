"""Tests for the LaTeX table builder."""

import math

from efs.evaluation.latex import escape, fmt, fmt_mean_std, latex_table


def test_number_formatting():
    assert fmt(3.14159, 2) == "3.14"
    assert fmt(float("nan")) == "--"
    assert fmt_mean_std(28.46, 3.81, 1) == "28.5 $\\pm$ 3.8"
    # A single run has no spread: show only the mean.
    assert fmt_mean_std(47.9, math.nan, 1) == "47.9"


def test_escape_special_characters():
    assert escape("dwt_lstm 5% & $") == r"dwt\_lstm 5\% \& \$"


def test_table_structure():
    text = latex_table("Caption", "tab:x", "lc", [["A", "B"]], [["1", "2"], ["MIDRULE"], ["3", "4"]])
    assert text.startswith("\\begin{table*}") and text.rstrip().endswith("\\end{table*}")
    assert "\\label{tab:x}" in text
    assert "1 & 2 \\\\" in text and "3 & 4 \\\\" in text
    assert text.count("\\hline") == 4          # top, below header, midrule, bottom
    booktabs = latex_table("C", "tab:y", "l", [["A"]], [["1"]], style="booktabs", wide=False)
    assert "\\toprule" in booktabs and booktabs.startswith("\\begin{table}")
