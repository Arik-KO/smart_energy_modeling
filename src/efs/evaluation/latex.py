"""Build LaTeX tables as plain strings (no extra dependencies)."""

from __future__ import annotations

import math

LATEX_SPECIAL = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_"}


def escape(text: str) -> str:
    """Escape characters that LaTeX would otherwise interpret."""
    result = str(text)
    for character, replacement in LATEX_SPECIAL.items():
        result = result.replace(character, replacement)
    return result


def fmt(value: float, digits: int = 2) -> str:
    """A number with fixed decimals; a dash for missing values."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    return f"{value:.{digits}f}"


def fmt_mean_std(mean: float, std: float, digits: int = 2) -> str:
    """'28.50 $\\pm$ 3.80', or just the mean when there is no spread (one run)."""
    if std is None or (isinstance(std, float) and math.isnan(std)):
        return fmt(mean, digits)
    return f"{fmt(mean, digits)} $\\pm$ {fmt(std, digits)}"


def latex_table(caption: str, label: str, column_format: str, header_rows: list[list[str]],
                body_rows: list[list[str]], style: str = "hline", notes: str | None = None,
                font_size: str | None = r"\small", wide: bool = True) -> str:
    """A complete table environment.

    style="hline" uses plain \\hline rules (no extra package); style="booktabs"
    uses \\toprule/\\midrule/\\bottomrule and needs \\usepackage{booktabs}.
    wide=True writes table* (full page width in two-column layouts).
    """
    if style == "booktabs":
        top, middle, bottom = r"\toprule", r"\midrule", r"\bottomrule"
    else:
        top, middle, bottom = r"\hline", r"\hline", r"\hline"

    # table* spans both columns of a two-column journal layout.
    environment = "table*" if wide else "table"
    lines = [f"\\begin{{{environment}}}[htbp]", r"\centering"]
    if font_size:
        lines.append(font_size)
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    # Scale the table down to the text width if it is wider (never up).
    # Needs \\usepackage{graphicx}, which any paper with figures already loads.
    lines.append(r"\resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{column_format}}}")
    lines.append(top)
    for row in header_rows:
        lines.append(" & ".join(row) + r" \\")
    lines.append(middle)
    for row in body_rows:
        if row == ["MIDRULE"]:
            lines.append(middle)
        else:
            lines.append(" & ".join(row) + r" \\")
    lines.append(bottom)
    lines.append(r"\end{tabular}}")
    if notes:
        lines.append(r"\par\smallskip")
        lines.append(f"{{\\footnotesize {notes}}}")
    lines.append(f"\\end{{{environment}}}")
    return "\n".join(lines) + "\n"
