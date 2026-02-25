from __future__ import annotations

import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional

from src.models.entities import StockTool
from src.tools.catalog import STOCKS

try:
    import akshare as ak
except Exception:  # pragma: no cover - handled by runtime error path
    ak = None  # type: ignore


def _error(code: str, message: str, suggestion: str) -> Dict[str, Any]:
    return {
        "error": True,
        "code": code,
        "message": message,
        "suggestion": suggestion,
    }


def _ensure_akshare() -> Optional[Dict[str, Any]]:
    if ak is None:
        return _error(
            code="DATA_UNAVAILABLE",
            message="akshare 未安装或导入失败",
            suggestion="请安装并验证: pip install akshare",
        )
    return None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if text in {"", "None", "nan", "NaN", "-"}:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _to_int(value: Any) -> Optional[int]:
    num = _to_float(value)
    if num is None:
        return None
    return int(num)


def _exchange_of_a_share(code: str) -> str:
    if code.startswith(("6", "9", "5")):
        return "SSE"
    if code.startswith(("0", "2", "3")):
        return "SZSE"
    if code.startswith(("4", "8")):
        return "BSE"
    return "CN"


def _normalize_code(stock_code: str, market: str) -> Optional[str]:
    raw = str(stock_code or "").strip()
    digits = "".join(ch for ch in raw if ch.isdigit())

    if market == "a_share":
        if len(digits) == 6:
            return digits
        return None

    if market == "hk_share":
        if not digits:
            return None
        if len(digits) > 5:
            return None
        return digits.zfill(5)

    return None


def _is_market_mismatch(stock_code: str, market: str) -> bool:
    raw = str(stock_code or "").strip()
    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        return False
    if market == "a_share" and len(digits) <= 5:
        return True
    if market == "hk_share" and len(digits) == 6:
        return True
    return False


def _xq_symbol(code: str, market: str) -> str:
    if market == "hk_share":
        return code

    exchange = _exchange_of_a_share(code)
    if exchange == "SSE":
        return f"SH{code}"
    if exchange == "SZSE":
        return f"SZ{code}"
    if exchange == "BSE":
        return f"BJ{code}"
    return f"SH{code}"


def _a_daily_symbol(code: str) -> str:
    exchange = _exchange_of_a_share(code)
    if exchange == "SSE":
        return f"sh{code}"
    if exchange == "SZSE":
        return f"sz{code}"
    if exchange == "BSE":
        return f"bj{code}"
    return f"sh{code}"


def _call_with_retry(func: Any, *args: Any, **kwargs: Any) -> Any:
    last_exc: Optional[Exception] = None
    for i in range(3):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # pragma: no cover - depends on external APIs
            last_exc = exc
            if i < 2:
                time.sleep(0.5 * (i + 1))
    assert last_exc is not None
    raise last_exc


def _map_api_exception(exc: Exception) -> Dict[str, Any]:
    text = str(exc)
    if "429" in text or "rate" in text.lower() or "limit" in text.lower():
        return _error(
            code="API_RATE_LIMIT",
            message=f"数据源触发限流: {text}",
            suggestion="请稍后重试",
        )

    return _error(
        code="DATA_UNAVAILABLE",
        message=f"数据源暂不可用: {text}",
        suggestion="请稍后重试，或切换其他股票代码验证",
    )


@lru_cache(maxsize=1)
def _a_code_name_rows() -> List[Dict[str, str]]:
    assert ak is not None
    df = _call_with_retry(ak.stock_info_a_code_name)
    rows: List[Dict[str, str]] = []
    for _, row in df.iterrows():
        code = str(row.get("code", "")).strip()
        name = str(row.get("name", "")).strip()
        if len(code) == 6 and name:
            rows.append({"code": code, "name": name})
    return rows


@lru_cache(maxsize=1)
def _hk_name_rows() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []

    if ak is not None:
        try:
            ah = _call_with_retry(ak.stock_zh_ah_name)
            for _, row in ah.iterrows():
                code = str(row.get("代码", "")).strip().zfill(5)
                name = str(row.get("名称", "")).strip()
                if code and name:
                    rows.append({"code": code, "name": name})
        except Exception:
            pass

    for stock in STOCKS:
        if stock.market == "hk_share":
            rows.append({"code": stock.code.zfill(5), "name": stock.name})

    dedup: Dict[str, Dict[str, str]] = {}
    for item in rows:
        key = item["code"]
        if key not in dedup:
            dedup[key] = item
    return list(dedup.values())


def _local_stock_by_code(code: str, market: str) -> Optional[Dict[str, Any]]:
    for stock in STOCKS:
        if stock.code == code and stock.market == market:
            return {
                "code": stock.code,
                "name": stock.name,
                "full_name": stock.full_name,
                "industry": stock.industry,
                "market_cap_billion_cny": stock.market_cap_billion_cny,
                "listed_date": stock.listed_date,
                "exchange": stock.exchange,
                "market": stock.market,
            }
    return None


def _spot_xq_map(code: str, market: str) -> Dict[str, Any]:
    assert ak is not None
    symbol = _xq_symbol(code, market)
    df = _call_with_retry(ak.stock_individual_spot_xq, symbol=symbol)
    if df is None or df.empty:
        return {}

    out: Dict[str, Any] = {}
    for _, row in df.iterrows():
        key = str(row.get("item", "")).strip()
        if key:
            out[key] = row.get("value")
    return out


def _build_quote_from_xq(code: str, market: str) -> Dict[str, Any]:
    data = _spot_xq_map(code, market)
    if not data:
        return _error(
            code="STOCK_NOT_FOUND",
            message=f"未获取到股票 {code} 的行情数据",
            suggestion="请确认代码和市场类型，或稍后重试",
        )

    currency = str(data.get("货币", "CNY")).upper()
    fx_hkd_to_cny = 0.92

    amount = _to_float(data.get("成交额"))
    if amount is not None and currency == "HKD":
        amount = amount * fx_hkd_to_cny

    ts_raw = data.get("时间")
    if ts_raw:
        timestamp = str(ts_raw)
    else:
        timestamp = datetime.now(timezone.utc).isoformat()

    return {
        "code": code,
        "name": str(data.get("名称", code)),
        "price": _to_float(data.get("现价")),
        "change": _to_float(data.get("涨跌")),
        "change_pct": _to_float(data.get("涨幅")),
        "volume": _to_int(data.get("成交量")),
        "amount_million_cny": round((amount or 0.0) / 1_000_000, 2) if amount is not None else None,
        "high": _to_float(data.get("最高")),
        "low": _to_float(data.get("最低")),
        "open": _to_float(data.get("今开")),
        "prev_close": _to_float(data.get("昨收")),
        "timestamp": timestamp,
    }


def search_stock_by_name(name: str, market: str = "all") -> Dict[str, Any]:
    if not name or not isinstance(name, str):
        return _error(
            code="INVALID_PARAMETER",
            message="name 不能为空",
            suggestion="请传入有效股票名称，例如 '茅台' 或 '腾讯'",
        )

    allowed_markets = {"a_share", "hk_share", "all"}
    if market not in allowed_markets:
        return _error(
            code="INVALID_PARAMETER",
            message=f"market 参数非法: {market}",
            suggestion="market 仅支持 a_share/hk_share/all",
        )

    q = name.strip().lower()
    results: List[Dict[str, Any]] = []

    if market in {"a_share", "all"}:
        check = _ensure_akshare()
        a_rows: List[Dict[str, str]] = []
        if check is None:
            try:
                a_rows = _a_code_name_rows()
            except Exception:
                a_rows = []

        for row in a_rows:
            code = row["code"]
            nm = row["name"]
            if q in code.lower() or q in nm.lower():
                results.append(
                    {
                        "code": code,
                        "name": nm,
                        "market": "a_share",
                        "exchange": _exchange_of_a_share(code),
                    }
                )

    if market in {"hk_share", "all"}:
        hk_rows = _hk_name_rows()
        for row in hk_rows:
            code = row["code"]
            nm = row["name"]
            if q in code.lower() or q in nm.lower():
                results.append(
                    {
                        "code": code,
                        "name": nm,
                        "market": "hk_share",
                        "exchange": "HKEX",
                    }
                )

    # 按代码去重，保留首次命中
    dedup: Dict[tuple[str, str], Dict[str, Any]] = {}
    for item in results:
        key = (item["market"], item["code"])
        if key not in dedup:
            dedup[key] = item

    final = list(dedup.values())[:30]
    return {"results": final, "total": len(final)}


def get_stock_info(stock_code: str, market: str) -> Dict[str, Any]:
    if market not in {"a_share", "hk_share"}:
        return _error(
            code="INVALID_PARAMETER",
            message=f"不支持的市场: {market}",
            suggestion="market 仅支持 a_share 或 hk_share",
        )

    code = _normalize_code(stock_code, market)
    if not code:
        if _is_market_mismatch(stock_code, market):
            return _error(
                code="MARKET_MISMATCH",
                message=f"股票代码 {stock_code} 与市场 {market} 不匹配",
                suggestion="请检查 stock_code 与 market 是否匹配",
            )
        return _error(
            code="INVALID_PARAMETER",
            message=f"股票代码格式不正确: {stock_code}",
            suggestion="A 股为 6 位代码，港股为 5 位代码（可省略前导 0）",
        )

    check = _ensure_akshare()
    if check is not None:
        local = _local_stock_by_code(code, market)
        if local:
            return local
        return check

    try:
        quote_map = _spot_xq_map(code, market)
    except Exception as exc:
        quote_map = {}
        quote_exc = exc
    else:
        quote_exc = None

    if market == "a_share":
        try:
            profile = _call_with_retry(ak.stock_profile_cninfo, symbol=code)
            p = profile.iloc[0].to_dict() if profile is not None and not profile.empty else {}
        except Exception:
            p = {}

        if not p and quote_exc is not None:
            local = _local_stock_by_code(code, market)
            if local:
                return local
            return _map_api_exception(quote_exc)

        market_cap_raw = _to_float(quote_map.get("流通值")) or _to_float(quote_map.get("资产净值/总市值"))
        market_cap_billion = round((market_cap_raw or 0.0) / 1_000_000_000, 2) if market_cap_raw is not None else None

        listed_date = p.get("上市日期")
        listed_date = str(listed_date).split(" ")[0] if listed_date not in (None, "nan") else None

        return {
            "code": code,
            "name": str(p.get("A股简称") or quote_map.get("名称") or code),
            "full_name": str(p.get("公司名称") or p.get("A股简称") or quote_map.get("名称") or code),
            "industry": str(p.get("所属行业") or "未知"),
            "market_cap_billion_cny": market_cap_billion,
            "listed_date": listed_date,
            "exchange": _exchange_of_a_share(code),
            "market": "a_share",
        }

    # 港股
    try:
        sec = _call_with_retry(ak.stock_hk_security_profile_em, symbol=code)
        sec_map = sec.iloc[0].to_dict() if sec is not None and not sec.empty else {}
    except Exception:
        sec_map = {}

    try:
        company = _call_with_retry(ak.stock_hk_company_profile_em, symbol=code)
        company_map = company.iloc[0].to_dict() if company is not None and not company.empty else {}
    except Exception:
        company_map = {}

    if not sec_map and not company_map and quote_exc is not None:
        local = _local_stock_by_code(code, market)
        if local:
            return local
        return _map_api_exception(quote_exc)

    market_cap_raw = _to_float(quote_map.get("流通值")) or _to_float(quote_map.get("资产净值/总市值"))
    currency = str(quote_map.get("货币", "HKD")).upper()
    if market_cap_raw is not None and currency == "HKD":
        market_cap_raw = market_cap_raw * 0.92
    market_cap_billion = round((market_cap_raw or 0.0) / 1_000_000_000, 2) if market_cap_raw is not None else None

    listed_date = sec_map.get("上市日期")
    listed_date = str(listed_date).split(" ")[0] if listed_date not in (None, "nan") else None

    return {
        "code": code,
        "name": str(sec_map.get("证券简称") or quote_map.get("名称") or company_map.get("公司名称") or code),
        "full_name": str(company_map.get("公司名称") or company_map.get("英文名称") or sec_map.get("证券简称") or code),
        "industry": str(company_map.get("所属行业") or "未知"),
        "market_cap_billion_cny": market_cap_billion,
        "listed_date": listed_date,
        "exchange": "HKEX",
        "market": "hk_share",
    }


def get_realtime_quote(stock_code: str, market: str) -> Dict[str, Any]:
    if market not in {"a_share", "hk_share"}:
        return _error(
            code="INVALID_PARAMETER",
            message=f"不支持的市场: {market}",
            suggestion="market 仅支持 a_share 或 hk_share",
        )

    code = _normalize_code(stock_code, market)
    if not code:
        if _is_market_mismatch(stock_code, market):
            return _error(
                code="MARKET_MISMATCH",
                message=f"股票代码 {stock_code} 与市场 {market} 不匹配",
                suggestion="请检查 stock_code 与 market 是否匹配",
            )
        return _error(
            code="INVALID_PARAMETER",
            message=f"股票代码格式不正确: {stock_code}",
            suggestion="A 股为 6 位代码，港股为 5 位代码（可省略前导 0）",
        )

    check = _ensure_akshare()
    if check is not None:
        return check

    try:
        quote = _build_quote_from_xq(code, market)
        if quote.get("error"):
            return quote
        quote["market"] = market
        return quote
    except Exception as exc:
        return _map_api_exception(exc)


def _load_history(stock_code: str, market: str, period: str):
    assert ak is not None

    if market == "a_share":
        symbol = _a_daily_symbol(stock_code)
        df = _call_with_retry(ak.stock_zh_a_daily, symbol=symbol, adjust="qfq")
    else:
        df = _call_with_retry(ak.stock_hk_daily, symbol=stock_code, adjust="qfq")

    if df is None or df.empty:
        return df

    import pandas as pd

    df = df.copy()
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date")

    if period == "weekly":
        df = (
            df.set_index("date")
            .resample("W-FRI")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna(subset=["close"])
            .reset_index()
        )
    elif period == "monthly":
        df = (
            df.set_index("date")
            .resample("ME")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna(subset=["close"])
            .reset_index()
        )

    return df


def get_technical_indicators(
    stock_code: str,
    market: str,
    indicators: Optional[List[str]] = None,
    period: str = "daily",
) -> Dict[str, Any]:
    if market not in {"a_share", "hk_share"}:
        return _error(
            code="INVALID_PARAMETER",
            message=f"不支持的市场: {market}",
            suggestion="market 仅支持 a_share 或 hk_share",
        )

    code = _normalize_code(stock_code, market)
    if not code:
        if _is_market_mismatch(stock_code, market):
            return _error(
                code="MARKET_MISMATCH",
                message=f"股票代码 {stock_code} 与市场 {market} 不匹配",
                suggestion="请检查 stock_code 与 market 是否匹配",
            )
        return _error(
            code="INVALID_PARAMETER",
            message=f"股票代码格式不正确: {stock_code}",
            suggestion="A 股为 6 位代码，港股为 5 位代码（可省略前导 0）",
        )

    allowed_indicators = {"MA5", "MA10", "MA20", "MA60", "MACD", "RSI14", "KDJ"}
    requested = indicators or ["MA5", "MA20", "MACD", "RSI14"]
    for ind in requested:
        if ind not in allowed_indicators:
            return _error(
                code="INVALID_PARAMETER",
                message=f"不支持的指标: {ind}",
                suggestion="请使用 MA5/MA10/MA20/MA60/MACD/RSI14/KDJ",
            )

    if period not in {"daily", "weekly", "monthly"}:
        return _error(
            code="INVALID_PARAMETER",
            message=f"不支持的周期: {period}",
            suggestion="period 仅支持 daily/weekly/monthly",
        )

    check = _ensure_akshare()
    if check is not None:
        return check

    try:
        df = _load_history(code, market, period)
    except Exception as exc:
        return _map_api_exception(exc)

    if df is None or df.empty:
        return _error(
            code="DATA_UNAVAILABLE",
            message=f"未获取到 {code} 的历史数据",
            suggestion="请稍后重试",
        )

    close = df["close"]
    high = df["high"]
    low = df["low"]

    result: Dict[str, Any] = {}

    for ind in requested:
        if ind == "MA5":
            result[ind] = round(float(close.rolling(5).mean().iloc[-1]), 4)
        elif ind == "MA10":
            result[ind] = round(float(close.rolling(10).mean().iloc[-1]), 4)
        elif ind == "MA20":
            result[ind] = round(float(close.rolling(20).mean().iloc[-1]), 4)
        elif ind == "MA60":
            result[ind] = round(float(close.rolling(60).mean().iloc[-1]), 4)
        elif ind == "MACD":
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            dif = ema12 - ema26
            dea = dif.ewm(span=9, adjust=False).mean()
            hist = (dif - dea) * 2
            result[ind] = {
                "dif": round(float(dif.iloc[-1]), 4),
                "dea": round(float(dea.iloc[-1]), 4),
                "histogram": round(float(hist.iloc[-1]), 4),
            }
        elif ind == "RSI14":
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta.where(delta < 0, 0.0))
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss.replace(0, 1e-12)
            rsi = 100 - (100 / (1 + rs))
            result[ind] = round(float(rsi.iloc[-1]), 4)
        elif ind == "KDJ":
            low_n = low.rolling(9).min()
            high_n = high.rolling(9).max()
            rsv = (close - low_n) / (high_n - low_n).replace(0, 1e-12) * 100
            k = rsv.ewm(alpha=1 / 3, adjust=False).mean()
            d = k.ewm(alpha=1 / 3, adjust=False).mean()
            j = 3 * k - 2 * d
            result[ind] = {
                "k": round(float(k.iloc[-1]), 4),
                "d": round(float(d.iloc[-1]), 4),
                "j": round(float(j.iloc[-1]), 4),
            }

    as_of = str(df["date"].iloc[-1])
    as_of = as_of.split(" ")[0]
    return {
        "code": code,
        "period": period,
        "as_of": as_of,
        "indicators": result,
    }


def get_tool_definitions() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "search_stock_by_name",
                "description": "通过股票名称搜索股票代码。支持 A 股和港股。名称可以是公司简称或全称，会返回所有匹配的结果。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "股票名称，如 茅台、腾讯、平安"},
                        "market": {
                            "type": "string",
                            "enum": ["a_share", "hk_share", "all"],
                            "description": "市场类型，默认 all",
                            "default": "all",
                        },
                    },
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_stock_info",
                "description": "获取股票基本信息，包括公司全称、所属行业、总市值、上市日期等。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "stock_code": {"type": "string", "description": "股票代码"},
                        "market": {
                            "type": "string",
                            "enum": ["a_share", "hk_share"],
                            "description": "市场类型",
                        },
                    },
                    "required": ["stock_code", "market"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_realtime_quote",
                "description": "获取股票实时行情，包括当前价格、涨跌幅、成交量、成交额等。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "stock_code": {"type": "string", "description": "股票代码"},
                        "market": {
                            "type": "string",
                            "enum": ["a_share", "hk_share"],
                            "description": "市场类型",
                        },
                    },
                    "required": ["stock_code", "market"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_technical_indicators",
                "description": "获取股票技术指标，包括 MA、MACD、RSI、KDJ。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "stock_code": {"type": "string", "description": "股票代码"},
                        "market": {
                            "type": "string",
                            "enum": ["a_share", "hk_share"],
                            "description": "市场类型",
                        },
                        "indicators": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["MA5", "MA10", "MA20", "MA60", "MACD", "RSI14", "KDJ"],
                            },
                            "default": ["MA5", "MA20", "MACD", "RSI14"],
                        },
                        "period": {
                            "type": "string",
                            "enum": ["daily", "weekly", "monthly"],
                            "default": "daily",
                        },
                    },
                    "required": ["stock_code", "market"],
                },
            },
        },
    ]


def get_stock_tool_registry() -> Dict[str, StockTool]:
    defs = get_tool_definitions()
    tools = [
        StockTool(
            name="search_stock_by_name",
            description="通过股票名称模糊搜索股票代码",
            parameters=defs[0]["function"]["parameters"],
            supports_a_share=True,
            supports_hk_share=True,
            handler=search_stock_by_name,
        ),
        StockTool(
            name="get_stock_info",
            description="获取股票基础信息",
            parameters=defs[1]["function"]["parameters"],
            supports_a_share=True,
            supports_hk_share=True,
            handler=get_stock_info,
        ),
        StockTool(
            name="get_realtime_quote",
            description="获取实时行情",
            parameters=defs[2]["function"]["parameters"],
            supports_a_share=True,
            supports_hk_share=True,
            handler=get_realtime_quote,
        ),
        StockTool(
            name="get_technical_indicators",
            description="获取技术指标",
            parameters=defs[3]["function"]["parameters"],
            supports_a_share=True,
            supports_hk_share=True,
            handler=get_technical_indicators,
        ),
    ]
    return {t.name: t for t in tools}
