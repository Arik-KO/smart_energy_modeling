"""Tests for the paper figure style."""

import numpy as np

from efs.evaluation.plotting import finalize_axes, new_figure, nice_ticks, save_figure

PLOT_CFG = {"dpi": 300, "figsize": [12, 6], "tall_figsize": [12, 8], "font_size": 10,
            "font_weight": "bold", "formats": ["png"]}


def test_ticks_always_enclose_the_data():
    rng = np.random.default_rng(0)
    for _ in range(200):
        low = rng.uniform(-50, 50)
        values = low + rng.uniform(0.01, 100) * rng.random(50)
        ticks = nice_ticks(values)
        assert ticks[0] <= values.min() and ticks[-1] >= values.max()


def test_figure_is_saved_at_300_dpi_without_title(tmp_path):
    fig, ax = new_figure(PLOT_CFG)
    hours = np.arange(720)
    actual = 1 + np.sin(hours / 24 * 2 * np.pi)
    ax.plot(hours, actual, label="Observation")
    ax.plot(hours, actual * 1.1, label="DWT-TNN")
    finalize_axes(ax, hours, [actual, actual * 1.1], "Hour", "Energy demand (kWh)", PLOT_CFG)
    assert ax.get_title() == ""
    assert ax.get_ylim()[1] >= (actual * 1.1).max()
    saved = save_figure(fig, tmp_path / "demo", PLOT_CFG)

    from PIL import Image
    with Image.open(saved[0]) as image:
        dpi = image.info["dpi"]
    assert round(dpi[0]) == 300
