"""Simple investment economics for the battery: NPV, payback, break-even cost."""

from __future__ import annotations


def annuity_factor(discount_rate: float, years: int) -> float:
    """Present value of receiving 1 per year for ``years`` years.

    Example: at 5% for 10 years, 1/1.05 + 1/1.05^2 + ... + 1/1.05^10 = 7.72.
    """
    if discount_rate == 0:
        return float(years)
    return (1.0 - (1.0 + discount_rate) ** (-years)) / discount_rate


def net_present_value(capital_cost: float, annual_saving: float, annual_om: float,
                      discount_rate: float, years: int) -> float:
    """NPV = -capital + (saving - O&M) x annuity factor."""
    return -capital_cost + (annual_saving - annual_om) * annuity_factor(discount_rate, years)


def simple_payback_years(capital_cost: float, annual_saving: float, annual_om: float) -> float:
    """Years of net savings needed to repay the capital (inf if never)."""
    net = annual_saving - annual_om
    return capital_cost / net if net > 0 else float("inf")


def break_even_cost_per_kwh(annual_saving: float, annual_om: float, capacity_kwh: float,
                            discount_rate: float, years: int) -> float:
    """The installed cost per kWh at which the battery would exactly pay for itself."""
    value = (annual_saving - annual_om) * annuity_factor(discount_rate, years)
    return value / capacity_kwh if capacity_kwh > 0 else float("nan")
