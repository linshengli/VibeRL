# VibeRL 完整运行指南

本文基于当前仓库可执行代码（真实股票工具已接入 `akshare`）整理，覆盖从环境准备到单 Agent、评估、数据管线、多 Agent Demo 的完整流程。

SFT/RL 训练专项说明见：`docs/SFT_RL_TRAINING_GUIDE.md`。

## 1. 环境准备

在项目根目录执行：

```bash
cd /Users/tbxsx/Code/VibeRL
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

可选（若你要直接用 DeepSeek/OpenAI API 而不是 `rule-based`）：

```bash
export DEEPSEEK_API_KEY="你的key"
# 或
export OPENAI_API_KEY="你的key"
```

## 2. 快速健康检查

```bash
cd /Users/tbxsx/Code/VibeRL
pytest -q
```

预期：全部通过（当前仓库基线为 `20 passed`）。

## 3. 运行单 Agent（真实行情 + 技术指标）

### 3.1 最简模式（本地 rule-based 决策，真实工具数据）

```bash
python src/agent/run.py \
  --query "帮我查一下茅台和腾讯的行情与技术指标" \
  --model rule-based
```

### 3.2 使用在线模型 function-calling

```bash
python src/agent/run.py \
  --query "帮我查一下茅台最近的技术指标" \
  --model deepseek-chat
```

如果你启用了调试代理（第 7 节），可加：

```bash
--debug-proxy http://localhost:8080/v1
```

## 4. 运行评估

```bash
python src/eval/evaluate.py \
  --model rule-based \
  --test-cases data/test_cases.json \
  --output results/baseline_eval.json
```

模型对比：

```bash
python src/eval/compare.py \
  --models rule-based rule-based rule-based \
  --labels "基线" "SFT" "GRPO" \
  --test-cases data/test_cases.json \
  --output results/final_comparison.json

python src/eval/report.py \
  --input results/final_comparison.json \
  --output results/final_report.md
```

## 5. 运行 SFT 数据管线

```bash
python src/data/generate_sft.py \
  --output-dir data/sft \
  --num-samples 1000 \
  --model deepseek-chat \
  --market-dist '{"a_share": 0.4, "hk_share": 0.3, "mixed": 0.3}' \
  --seed 42

python src/data/validate.py --data-dir data/sft

python src/data/to_parquet.py \
  --input data/sft/samples.jsonl \
  --output-dir data/sft/parquet \
  --split 0.9
```

输出：
- `data/sft/parquet/train.parquet`
- `data/sft/parquet/val.parquet`

## 6. 运行多 Agent Demo（推荐）

```bash
python src/demo/multi_agent_demo.py \
  --query "请比较茅台和腾讯，给我行情、技术指标、风险建议" \
  --model rule-based \
  --json-output results/multi_agent_demo.json
```

产物：
- 终端打印多 Agent 汇总报告
- `results/multi_agent_demo.json`（结构化结果）

## 7. 启动调试代理（记录所有 LLM 请求到 SQLite）

```bash
python src/debugger/proxy.py \
  --port 8080 \
  --ui-port 8081 \
  --db-path debug/debug_records.db
```

查询记录：

```bash
curl 'http://localhost:8081/records?limit=20&offset=0'
curl 'http://localhost:8081/records/<record_id>'
```

## 8. 常见问题

1. `DATA_UNAVAILABLE` / 网络报错
- 原因：行情上游源临时不可用或被限流。
- 建议：稍后重试；必要时切换网络环境。

2. 首次 `search_stock_by_name` 比较慢
- 原因：A 股代码表首次会从数据源拉取并缓存。
- 建议：首次等待完成，后续调用会快很多。

3. 使用在线模型时报 Key 错误
- 检查 `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` 是否正确导出。



## 9. Web 三栏应用

完整说明见：`docs/WEB_APP_GUIDE.md`（现已支持**多轮会话 + 流式传输 + agentic 动态编排 + ReAct/推理/反思 + 规则/大模型切换 + RAG 上传检索 + Debugger 前端查看**）。

快速启动：

```bash
python src/web/app.py --host 0.0.0.0 --port 5000 --model rule-based
```

页面中可直接点击切换：
- `Rule-based`
- `大模型`（`deepseek-chat`，需配置可用 key）

## 10. RAG 资料库（Telegram / PDF / PPT）

Web 页面左侧可直接上传资料到 RAG：
- `文档(PDF/PPT/TXT/JSON)`
- `Telegram 导出`
- `第三方聊天导出`
- `WhatsApp 导出`
- `Discord 导出`
- `Slack 导出`

通过 API 上传示例：

```bash
curl -X POST http://127.0.0.1:5000/api/rag/upload \
  -F import_type=document \
  -F file=@/absolute/path/your_report.pdf
```

对话时启用 RAG：

```bash
curl -X POST http://127.0.0.1:5000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"总结我上传资料里的风险点","model":"rule-based","use_rag":true}'
```

更多见：`docs/RAG_GUIDE.md`

## 11. Debugger 记录前端查看

若使用了 `debugger/proxy.py` 作为 LLM 上游代理，web 会自动读取同一数据库并显示记录。

```bash
python src/web/app.py --host 0.0.0.0 --port 5000 --debug-db-path debug/debug_records.db
```
