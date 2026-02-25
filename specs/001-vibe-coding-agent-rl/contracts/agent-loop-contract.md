# 接口契约：ReAct Agent 循环

**版本**: 1.0.0
**更新日期**: 2026-02-26

---

## Agent 主入口接口

```python
class StockAnalysisAgent:
    def run(
        self,
        user_query: str,
        max_steps: int = 10,
        system_prompt: Optional[str] = None,
    ) -> AgentTrajectory:
        """
        执行完整的 ReAct Agent 循环。

        参数:
            user_query: 用户的自然语言查询
            max_steps: 最大工具调用轮数（默认 10）
            system_prompt: 自定义系统提示（不传则使用默认）

        返回:
            AgentTrajectory: 包含完整对话历史和最终结果

        异常:
            MaxStepsExceeded: 超过最大步数仍未完成
            ToolExecutionError: 工具调用发生不可恢复错误
        """
```

---

## 数据生成接口

```python
class SFTDataGenerator:
    def generate_batch(
        self,
        prompts: List[str],
        market_distribution: Dict[str, float] = {
            "a_share": 0.4,
            "hk_share": 0.3,
            "mixed": 0.3,
        },
        batch_size: int = 10,
    ) -> List[SFTSample]:
        """
        批量生成 SFT 训练数据。

        参数:
            prompts: 用户问题列表
            market_distribution: A 股/港股/混合的比例分布
            batch_size: 每批并发请求数

        返回:
            List[SFTSample]: 生成的样本列表
        """

    def save_to_parquet(
        self,
        samples: List[SFTSample],
        output_path: str,
        split: float = 0.9,
    ) -> Tuple[str, str]:
        """
        保存为 verl 兼容的 parquet 格式。

        参数:
            samples: 样本列表
            output_path: 输出目录
            split: 训练集比例（默认 0.9，剩余为验证集）

        返回:
            Tuple[train_path, val_path]: 训练集和验证集路径
        """
```

---

## 奖励函数接口

```python
class RewardComputer:
    def compute(
        self,
        trajectory: AgentTrajectory,
        expected_tools: Optional[List[str]] = None,
        use_llm_judge: bool = False,
    ) -> RewardSignal:
        """
        计算单条轨迹的奖励。

        参数:
            trajectory: Agent 执行轨迹
            expected_tools: 预期工具调用序列（用于规则匹配）
            use_llm_judge: 是否使用 LLM 对最终输出质量打分

        返回:
            RewardSignal: 包含各维度分数和综合奖励
        """
```

---

## 评估接口

```python
class AgentEvaluator:
    def evaluate(
        self,
        model_or_agent: Union[str, StockAnalysisAgent],
        test_cases: List[dict],
        metrics: List[str] = ["tool_accuracy", "output_quality", "amb_accuracy"],
    ) -> dict:
        """
        在测试集上运行评估并生成报告。

        参数:
            model_or_agent: 模型路径或已初始化的 Agent
            test_cases: 测试用例列表，每条含 query、expected_tools、ground_truth
            metrics: 需要计算的指标

        返回:
            dict: 各指标分数和汇总报告
        """
```

---

## 调试代理接口

```
POST http://localhost:8080/proxy
Content-Type: application/json

转发至实际 LLM API，同时记录到 SQLite。
与 OpenAI API 格式完全兼容，客户端只需修改 base_url。

GET http://localhost:8081/records?limit=20&offset=0
返回最近的调试记录列表。

GET http://localhost:8081/records/{record_id}
返回单条调试记录详情。
```
