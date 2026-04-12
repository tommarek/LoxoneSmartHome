"""Price unit conversion utilities.

Single source of truth for EUR/MWh to CZK/kWh conversion.
All prices should be converted at storage time using these functions.
"""

from typing import Dict, Tuple


def eur_mwh_to_czk_kwh(price_eur_mwh: float, eur_czk_rate: float) -> float:
    """Convert EUR/MWh to CZK/kWh.

    Args:
        price_eur_mwh: Price in EUR per MWh (from OTE API)
        eur_czk_rate: EUR to CZK exchange rate (e.g., 25.0)

    Returns:
        Price in CZK per kWh
    """
    return price_eur_mwh * eur_czk_rate / 1000.0


def convert_price_dict(
    prices: Dict[Tuple[str, str], float],
    eur_czk_rate: float,
) -> Dict[Tuple[str, str], float]:
    """Convert entire price dict from EUR/MWh to CZK/kWh.

    Args:
        prices: Dict mapping (start_time, end_time) to price in EUR/MWh
        eur_czk_rate: EUR to CZK exchange rate

    Returns:
        Same dict structure with prices in CZK/kWh
    """
    return {key: eur_mwh_to_czk_kwh(p, eur_czk_rate) for key, p in prices.items()}
