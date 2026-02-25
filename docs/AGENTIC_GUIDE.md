# Agentic Agent 说明

实现文件：`src/agentic/orchestrator.py`

## 与固定流程的区别

旧方案：固定 `Planner -> Research -> Fundamental -> Risk -> Reporter`。

新方案：动态决策循环（state-driven），每个标的按当前状态选择下一步动作：
- `search`
- `quote`
- `indicators`
- `info`
- `reasoning`
- `risk`
- `reflection`

当某些信息已满足、或查询并不需要某些信息时，会跳过对应步骤。

## 入口

Web 请求通过：
- `src/web/service.py` -> `_run_multi()` -> `AgenticOrchestrator.run()`
- 支持按 `model` 切换 `rule-based` / 大模型推理（不可用自动回退）

## 输出字段

- 每个标的：
  - `reasoning`
  - `reflection`
  - `action_trace`
- 全局：
  - `reflection`（聚合复盘）

## 流式事件

`AgenticOrchestrator.run(..., event_callback=...)` 会在关键节点发事件：
- 规划
- 每一步 agent 决策与结果
- 汇总完成

前端右侧 “Multi-Agent 工作信息” 会实时展示这些事件。
