# Core Logic Checklist

## Confirmed invariants / behaviors

- [x] Chat 请求必须有非空 `query`，并受长度约束（4000）
  - [src/web/app.py](/Users/tbxsx/Code/VibeRL/src/web/app.py:77)
- [x] `model` 字段长度受限（120）
  - [src/web/app.py](/Users/tbxsx/Code/VibeRL/src/web/app.py:84)
- [x] 历史列表 `limit` 非法值不会导致 500
  - [src/web/app.py](/Users/tbxsx/Code/VibeRL/src/web/app.py:53)
- [x] 服务内部错误对外脱敏，详细 traceback 仅落历史
  - [src/web/app.py](/Users/tbxsx/Code/VibeRL/src/web/app.py:103)
- [x] 历史文件损坏自动备份并自愈
  - [src/web/history_store.py](/Users/tbxsx/Code/VibeRL/src/web/history_store.py:23)
- [x] 历史记录超上限时按时间顺序裁剪
  - [src/web/history_store.py](/Users/tbxsx/Code/VibeRL/src/web/history_store.py:43)
- [x] 前端展示用户/agent数据使用 `textContent`（避免 HTML 注入）
  - [src/web/static/app.js](/Users/tbxsx/Code/VibeRL/src/web/static/app.js:47)
  - [src/web/static/app.js](/Users/tbxsx/Code/VibeRL/src/web/static/app.js:141)

## Failure modes

- 外部行情源不可用：`DATA_UNAVAILABLE`，可能导致 `/api/chat` 500（业务层异常）
  - [src/tools/stock_tools.py](/Users/tbxsx/Code/VibeRL/src/tools/stock_tools.py:136)
- 首次 A 股代码拉取慢：`stock_info_a_code_name` 冷启动延迟
  - [src/tools/stock_tools.py](/Users/tbxsx/Code/VibeRL/src/tools/stock_tools.py:153)
- 单请求内串行执行 chat + multi-agent，长尾时延明显
  - [src/web/service.py](/Users/tbxsx/Code/VibeRL/src/web/service.py:25)

## Performance hotspots (to measure)

- `search_stock_by_name` 冷缓存首次耗时
- `get_technical_indicators` 频繁拉全量历史数据后的 rolling 计算
- `/api/chat` 中双路径串行执行总耗时（chat + multi-agent）

## Suggested next improvements

1. 将历史存储从 JSON 文件迁移到 SQLite（支持并发与查询）
2. 为 `/api/chat` 添加超时控制与结果缓存
3. 对 `stock_tools` 增加可配置数据源优先级与熔断
4. 为前端增加请求取消和流式结果展示

## Inferred / runtime verification needed
- 压测下是否出现文件锁竞争，需要真实并发压测确认。
