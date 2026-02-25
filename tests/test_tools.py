from src.tools.stock_tools import (
    get_realtime_quote,
    get_stock_info,
    get_technical_indicators,
    search_stock_by_name,
)


def test_search_stock_by_name_returns_results() -> None:
    data = search_stock_by_name("茅台")
    assert data["total"] >= 1
    assert any(row["code"] == "600519" for row in data["results"])


def test_get_stock_info_market_mismatch() -> None:
    data = get_stock_info("600519", "hk_share")
    assert data["error"] is True
    assert data["code"] == "MARKET_MISMATCH"


def test_quote_and_indicators_have_expected_fields() -> None:
    quote = get_realtime_quote("00700", "hk_share")
    assert quote["code"] == "00700"
    assert "price" in quote

    indicators = get_technical_indicators("00700", "hk_share")
    assert indicators["code"] == "00700"
    assert "indicators" in indicators
