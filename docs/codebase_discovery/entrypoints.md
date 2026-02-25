# Entrypoints

## Confirmed by code

### HTTP Entrypoints
- `/` -> `index()` in [src/web/app.py](/Users/tbxsx/Code/VibeRL/src/web/app.py:43)
  - Input: browser GET
  - Output: `index.html`
- `/api/health` -> `health()` in [src/web/app.py](/Users/tbxsx/Code/VibeRL/src/web/app.py:47)
  - Input: GET
  - Output: `{ok,time}`
- `/api/history` -> `history_list()` in [src/web/app.py](/Users/tbxsx/Code/VibeRL/src/web/app.py:51)
  - Input: GET + `limit`
  - Output: history summaries
- `/api/history/<record_id>` -> `history_detail()` in [src/web/app.py](/Users/tbxsx/Code/VibeRL/src/web/app.py:61)
  - Input: GET + record id
  - Output: full record
- `/api/chat` -> `chat()` in [src/web/app.py](/Users/tbxsx/Code/VibeRL/src/web/app.py:68)
  - Input: POST JSON `{query, model}`
  - Output: chat + multi-agent result record

### CLI Entrypoints
- `python src/web/app.py` -> `main()` in [src/web/app.py](/Users/tbxsx/Code/VibeRL/src/web/app.py:131)
- `python src/agent/run.py` -> `main()` in [src/agent/run.py](/Users/tbxsx/Code/VibeRL/src/agent/run.py:31)
- `python src/demo/multi_agent_demo.py` -> `main()` in [src/demo/multi_agent_demo.py](/Users/tbxsx/Code/VibeRL/src/demo/multi_agent_demo.py:317)
- `python src/debugger/proxy.py` -> `main()` in [src/debugger/proxy.py](/Users/tbxsx/Code/VibeRL/src/debugger/proxy.py:219)
- Data/eval CLIs:
  - [src/data/generate_sft.py](/Users/tbxsx/Code/VibeRL/src/data/generate_sft.py:32)
  - [src/data/validate.py](/Users/tbxsx/Code/VibeRL/src/data/validate.py:49)
  - [src/data/to_parquet.py](/Users/tbxsx/Code/VibeRL/src/data/to_parquet.py:14)
  - [src/eval/evaluate.py](/Users/tbxsx/Code/VibeRL/src/eval/evaluate.py:21)
  - [src/eval/compare.py](/Users/tbxsx/Code/VibeRL/src/eval/compare.py:15)
  - [src/eval/report.py](/Users/tbxsx/Code/VibeRL/src/eval/report.py:8)

## Routing summary
- Web 请求通过 `create_app()` 注入 `AgentQueryService` + `HistoryStore`，进入 `service.execute()` 后分叉为：
  - 单 Agent: `StockAnalysisAgent.run()`
  - 多 Agent: `MultiAgentOrchestrator.run()`
- 所有结果最终写入 `HistoryStore.add()`。

## Inferred / runtime verification needed
- 多 worker（gunicorn/uwsgi）部署下，JSON 文件存储是否仍满足并发一致性，需要运行时验证。
