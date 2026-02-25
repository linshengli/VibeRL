# Web 应用完整指南（多轮 + 流式 + Agentic + ReAct）

本指南对应以下实现：
- 后端：`src/web/app.py`
- 前端：`src/web/templates/index.html` + `src/web/static/app.css` + `src/web/static/app.js`
- 服务编排：`src/web/service.py`
- Agentic 编排器：`src/agentic/orchestrator.py`
- 历史存储：`src/web/history_store.py`

## 1. 你现在得到的能力

1. **多轮对话**
- 以 `conversation_id` 管理会话。
- 每次消息会写入一个 turn（`turn_index` 递增）。

2. **流式传输**
- `POST /api/chat/stream` 返回 `text/event-stream`。
- 前端实时显示：`delta`（文本增量）+ `agent_event`（agent 工作事件）。

3. **Agentic Agent**
- 使用动态决策循环（非固定流水线）
- 按状态选择下一步动作：`search / quote / indicators / info / reasoning / risk / reflection`

4. **ReAct / 推理 / 反思**
- `chat` 结果返回 `react_trace`、`reasoning_summary`
- `multi_agent` 返回每个标的的 `reasoning`、`reflection` 和全局 `reflection`

5. **规则 / 大模型切换**
- 前端输入区可点击切换：
  - `Rule-based`
  - `大模型`（`deepseek-chat`）
- `model` 会传到 chat 和 multi-agent 编排层；大模型不可用时会自动回退到规则推理。

## 2. 启动

```bash
cd /Users/tbxsx/Code/VibeRL
source .venv/bin/activate
pip install -r requirements.txt
python src/web/app.py --host 0.0.0.0 --port 5000 --model rule-based
```

浏览器打开：
- [http://127.0.0.1:5000](http://127.0.0.1:5000)

## 3. API

### 3.1 会话 API

- `GET /api/conversations?limit=100`
- `GET /api/conversations/<conversation_id>`

### 3.2 非流式聊天

`POST /api/chat`

请求：

```json
{
  "query": "比较茅台和腾讯",
  "model": "rule-based",
  "conversation_id": "可选，不传则自动新建"
}
```

响应（简化）：

```json
{
  "id": "turn_id",
  "conversation_id": "...",
  "turn_index": 2,
  "status": "ok",
  "chat": {"final_answer": "..."},
  "multi_agent": {"report": "...", "worklog": []},
  "warnings": []
}
```

### 3.3 流式聊天

`POST /api/chat/stream`

返回事件：
- `meta`: 会话元信息
- `agent_event`: agent 工作事件
- `delta`: 文本增量
- `record`: 最终完整 turn 记录
- `error`: 失败
- `done`: 结束

## 4. 输入限制与容错

- `query` 最长 4000 字符
- `model` 最长 120 字符
- 历史文件损坏自动备份 + 自愈
- `multi-agent` 失败时自动降级，优先保留 chat 结果

## 5. 测试

```bash
pytest -q tests/test_web_api.py tests/test_web_service.py tests/test_history_store.py tests/test_multi_agent_reporter.py tests/test_agentic_orchestrator.py
```

覆盖：
- 会话/历史 API
- 流式 endpoint
- 多轮上下文拼接
- multi-agent 降级
- ReAct 轨迹与推理摘要输出
- agentic 反思链路
- 历史文件损坏自愈
