from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class StockMeta:
    code: str
    name: str
    full_name: str
    industry: str
    market: str
    exchange: str
    listed_date: str
    market_cap_billion_cny: float

STOCK_NAME_CODE = {
    "贵州茅台": "600519",
    "平安银行": "000001"
}

STOCKS: List[StockMeta] = [
    StockMeta(
        code="600519",
        name="贵州茅台",
        full_name="贵州茅台酒股份有限公司",
        industry="白酒",
        market="a_share",
        exchange="SSE",
        listed_date="2001-08-27",
        market_cap_billion_cny=2100.5,
    ),
    StockMeta(
        code="000001",
        name="平安银行",
        full_name="平安银行股份有限公司",
        industry="银行",
        market="a_share",
        exchange="SZSE",
        listed_date="1991-04-03",
        market_cap_billion_cny=180.2,
    ),
    StockMeta(
        code="601318",
        name="中国平安",
        full_name="中国平安保险(集团)股份有限公司",
        industry="保险",
        market="a_share",
        exchange="SSE",
        listed_date="2007-03-01",
        market_cap_billion_cny=900.7,
    ),
    StockMeta(
        code="00700",
        name="腾讯控股",
        full_name="Tencent Holdings Limited",
        industry="互联网",
        market="hk_share",
        exchange="HKEX",
        listed_date="2004-06-16",
        market_cap_billion_cny=2400.0,
    ),
    StockMeta(
        code="02318",
        name="中国平安",
        full_name="Ping An Insurance (Group) Company of China, Ltd.",
        industry="保险",
        market="hk_share",
        exchange="HKEX",
        listed_date="2004-06-24",
        market_cap_billion_cny=890.0,
    ),
    StockMeta(
        code="09988",
        name="阿里巴巴-W",
        full_name="Alibaba Group Holding Limited",
        industry="互联网",
        market="hk_share",
        exchange="HKEX",
        listed_date="2019-11-26",
        market_cap_billion_cny=1500.0,
    ),
]
