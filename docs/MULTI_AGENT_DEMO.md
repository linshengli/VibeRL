# 多 Agent Demo 说明

示例脚本：`src/demo/multi_agent_demo.py`

## Agent 角色

- `PlannerAgent`：从用户问题中拆解标的和任务。
- `ResearchAgent`：调用 `StockAnalysisAgent` 执行 ReAct + 工具调用，拉取行情和技术指标。
- `FundamentalAgent`：补充公司基础信息（行业、市值、上市日期）。
- `RiskAgent`：根据涨跌幅、RSI、MACD 计算风险等级并给出建议。
- `ReporterAgent`：汇总所有子 Agent 的结果输出最终报告。

## 执行链路

`Planner -> Research -> Fundamental -> Risk -> Reporter`

## 运行命令

```bash
python src/demo/multi_agent_demo.py \
  --query "请比较茅台和腾讯，给我行情、技术指标、风险建议" \
  --model rule-based \
  --json-output results/multi_agent_demo.json
```

## 输出格式

- 控制台：人类可读汇总报告。
- JSON：每个标的都会包含：
  - `symbol`
  - `quote`
  - `indicators`
  - `info`
  - `risk`
  - `final_answer`
  - `trajectory_id`

