# 接口契约：股票分析工具集

**版本**: 1.0.0
**格式**: OpenAI Function Calling JSON Schema
**更新日期**: 2026-02-26

---

## 工具 1：search_stock_by_name

**描述**: 通过股票名称（模糊匹配）搜索对应的股票代码，同时支持 A 股和港股。

```json
{
  "name": "search_stock_by_name",
  "description": "通过股票名称搜索股票代码。支持 A 股和港股。名称可以是公司简称或全称，会返回所有匹配的结果。",
  "parameters": {
    "type": "object",
    "properties": {
      "name": {
        "type": "string",
        "description": "股票名称，如「茅台」「腾讯」「平安」"
      },
      "market": {
        "type": "string",
        "enum": ["a_share", "hk_share", "all"],
        "description": "市场类型。a_share=A 股，hk_share=港股，all=同时搜索（默认）",
        "default": "all"
      }
    },
    "required": ["name"]
  }
}
```

**返回格式**:
```json
{
  "results": [
    {
      "code": "600519",
      "name": "贵州茅台",
      "market": "a_share",
      "exchange": "SSE"
    },
    {
      "code": "00700",
      "name": "腾讯控股",
      "market": "hk_share",
      "exchange": "HKEX"
    }
  ],
  "total": 2
}
```

---

## 工具 2：get_stock_info

**描述**: 获取股票的基本信息，包括公司名称、所属行业、总市值等。

```json
{
  "name": "get_stock_info",
  "description": "获取股票基本信息，包括公司全称、所属行业、总市值、上市日期等。",
  "parameters": {
    "type": "object",
    "properties": {
      "stock_code": {
        "type": "string",
        "description": "股票代码。A 股使用 6 位数字代码（如 600519），港股使用 5 位数字代码（如 00700）"
      },
      "market": {
        "type": "string",
        "enum": ["a_share", "hk_share"],
        "description": "市场类型，必须与 stock_code 对应"
      }
    },
    "required": ["stock_code", "market"]
  }
}
```

**返回格式**:
```json
{
  "code": "600519",
  "name": "贵州茅台",
  "full_name": "贵州茅台酒股份有限公司",
  "industry": "白酒",
  "market_cap_billion_cny": 2100.5,
  "listed_date": "2001-08-27",
  "exchange": "SSE",
  "market": "a_share"
}
```

---

## 工具 3：get_realtime_quote

**描述**: 获取股票实时行情数据。

```json
{
  "name": "get_realtime_quote",
  "description": "获取股票实时行情，包括当前价格、涨跌幅、成交量、成交额等。",
  "parameters": {
    "type": "object",
    "properties": {
      "stock_code": {
        "type": "string",
        "description": "股票代码"
      },
      "market": {
        "type": "string",
        "enum": ["a_share", "hk_share"],
        "description": "市场类型"
      }
    },
    "required": ["stock_code", "market"]
  }
}
```

**返回格式**:
```json
{
  "code": "600519",
  "name": "贵州茅台",
  "price": 1680.00,
  "change": 12.50,
  "change_pct": 0.75,
  "volume": 3245678,
  "amount_million_cny": 54532.1,
  "high": 1695.00,
  "low": 1662.00,
  "open": 1668.00,
  "prev_close": 1667.50,
  "timestamp": "2026-02-26T10:30:00+08:00"
}
```

---

## 工具 4：get_technical_indicators

**描述**: 获取股票技术分析指标。

```json
{
  "name": "get_technical_indicators",
  "description": "获取股票技术指标，包括 MA（均线）、MACD、RSI、KDJ 等常用技术分析指标。",
  "parameters": {
    "type": "object",
    "properties": {
      "stock_code": {
        "type": "string",
        "description": "股票代码"
      },
      "market": {
        "type": "string",
        "enum": ["a_share", "hk_share"],
        "description": "市场类型"
      },
      "indicators": {
        "type": "array",
        "items": {
          "type": "string",
          "enum": ["MA5", "MA10", "MA20", "MA60", "MACD", "RSI14", "KDJ"]
        },
        "description": "需要计算的技术指标列表",
        "default": ["MA5", "MA20", "MACD", "RSI14"]
      },
      "period": {
        "type": "string",
        "enum": ["daily", "weekly", "monthly"],
        "description": "数据周期，默认 daily（日线）",
        "default": "daily"
      }
    },
    "required": ["stock_code", "market"]
  }
}
```

**返回格式**:
```json
{
  "code": "600519",
  "period": "daily",
  "as_of": "2026-02-26",
  "indicators": {
    "MA5": 1672.30,
    "MA20": 1645.80,
    "MACD": {
      "dif": 8.25,
      "dea": 6.10,
      "histogram": 4.30
    },
    "RSI14": 62.5,
    "KDJ": {
      "k": 68.2,
      "d": 64.5,
      "j": 75.6
    }
  }
}
```

---

## 错误响应格式

所有工具在发生错误时返回统一格式：

```json
{
  "error": true,
  "code": "STOCK_NOT_FOUND",
  "message": "未找到股票代码 99999，请确认代码和市场类型是否匹配",
  "suggestion": "请尝试使用 search_stock_by_name 先搜索股票代码"
}
```

**错误码清单**:

| 错误码 | 含义 |
|--------|------|
| `STOCK_NOT_FOUND` | 股票代码不存在 |
| `MARKET_MISMATCH` | 股票代码与市场类型不匹配（如 A 股代码传入 hk_share） |
| `API_RATE_LIMIT` | 数据源接口限流 |
| `DATA_UNAVAILABLE` | 数据暂时不可用（如非交易时间） |
| `INVALID_PARAMETER` | 参数格式错误 |
