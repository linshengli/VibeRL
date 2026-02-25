# 快速开始：Vibe Coding Agent RL 后训练

**目标读者**: 熟悉 Python 和深度学习基础，首次复现本项目的工程师
**预计时间**: 从零到完整运行约 15-23 天（见里程碑表）
**更新日期**: 2026-02-26

---

## 前置条件

- [ ] GPU 服务器就绪（推荐 AutoDL / Lambda / vast.ai，至少 4× L4 或 2× A100）
- [ ] Python 3.10+ 已安装
- [ ] CUDA 12.1+ 已配置
- [ ] Git 已安装
- [ ] DeepSeek API Key（用于数据生成）：https://platform.deepseek.com/

---

## Step 1：克隆仓库并配置环境

```bash
# 1. 克隆仓库
git clone <repo_url>
cd VibeRL

# 2. 创建 conda 环境（推荐，避免依赖冲突）
conda create -n viberl python=3.11 -y
conda activate viberl

# 3. 安装 PyTorch（CUDA 12.1 版本）
# 参考：https://pytorch.org/get-started/locally/
pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. 安装核心依赖
pip install verl          # RL 后训练框架：https://github.com/volcengine/verl
pip install vllm          # 推理引擎：https://docs.vllm.ai
pip install wandb         # 实验追踪：https://wandb.ai/
pip install transformers datasets accelerate peft
pip install akshare       # A 股数据：https://akshare.akfamily.xyz/
pip install openai        # OpenAI 兼容 API 客户端
pip install flask         # 调试 Web UI

# 5. 验证安装
python -c "import torch; print(torch.cuda.is_available())"  # 应输出 True
python -c "import verl; print('verl OK')"
python -c "import vllm; print('vllm OK')"
```

---

## Step 2：下载基础模型

```bash
# 方法一：使用 Hugging Face Hub（需要网络访问）
# 参考：https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='models/Qwen2.5-7B-Instruct')
"

# 方法二：使用 ModelScope（国内网络推荐）
pip install modelscope
python -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='models/Qwen2.5-7B-Instruct')
"
```

---

## Step 3：配置 API Keys

```bash
# 复制配置模板
cp config/config.example.yaml config/config.yaml

# 编辑配置文件，填入以下信息：
# - deepseek_api_key: 用于数据生成
# - wandb_api_key: 用于实验追踪（可选）
# - model_path: 本地模型路径
vim config/config.yaml
```

---

## Step 4：运行 Agent 原型（验证基线）

```bash
# 4.1 启动调试代理（可选，用于记录 API 请求）
python src/debugger/proxy.py --port 8080 &

# 4.2 运行单次 Agent 查询
python src/agent/run.py \
    --query "帮我查一下茅台的实时行情和最近的 MACD 指标" \
    --model deepseek-chat \
    --debug-proxy http://localhost:8080

# 预期输出：
# [Thought] 用户询问茅台的行情和技术指标，茅台是 A 股贵州茅台，代码 600519...
# [Action] search_stock_by_name(name="茅台", market="a_share")
# [Observation] {"results": [{"code": "600519", "name": "贵州茅台", ...}]}
# [Action] get_realtime_quote(stock_code="600519", market="a_share")
# ...
# [Final Answer] 贵州茅台（600519）当前价格 1680.00 元，...

# 4.3 运行基线评估（50 个测试用例）
python src/eval/evaluate.py \
    --model deepseek-chat \
    --test-cases data/test_cases.json \
    --output results/baseline_eval.json
```

---

## Step 5：生成 SFT 训练数据

```bash
# 5.1 生成 1000 条训练数据（约需 1-2 小时，使用 deepseek-chat API）
python src/data/generate_sft.py \
    --output-dir data/sft \
    --num-samples 1000 \
    --model deepseek-chat \
    --market-dist '{"a_share": 0.4, "hk_share": 0.3, "mixed": 0.3}' \
    --seed 42

# 5.2 验证数据格式
python src/data/validate.py --data-dir data/sft

# 5.3 转换为 verl parquet 格式
python src/data/to_parquet.py \
    --input data/sft/samples.jsonl \
    --output-dir data/sft/parquet \
    --split 0.9

# 预期输出：
# data/sft/parquet/train.parquet (900 条)
# data/sft/parquet/val.parquet (100 条)
```

---

## Step 6：运行 SFT 训练

```bash
# 6.1 启动 SFT 训练（4 卡）
torchrun --nproc_per_node=4 \
    -m verl.trainer.fsdp_sft_trainer \
    --config configs/sft_config.yaml

# 关键配置参数（见 configs/sft_config.yaml）：
# model_path: models/Qwen2.5-7B-Instruct
# train_files: data/sft/parquet/train.parquet
# val_files: data/sft/parquet/val.parquet
# num_train_epochs: 3
# gradient_checkpointing: true  # 必须开启！
# fsdp_offload: true             # 显存不足时开启

# 6.2 监控训练（新终端）
# 在 wandb dashboard 查看 loss 曲线：https://wandb.ai/

# 6.3 评估 SFT 效果
python src/eval/evaluate.py \
    --model checkpoints/sft/final \
    --test-cases data/test_cases.json \
    --output results/sft_eval.json

# 成功标准：工具调用准确率 ≥ 75%
```

---

## Step 7：运行 GRPO RL 训练

```bash
# 7.1 启动 GRPO 训练（需要 Ray + vLLM）
# 参考：https://verl.readthedocs.io/en/latest/algo/grpo.html
python -m verl.trainer.main_grpo \
    --config configs/grpo_config.yaml

# 关键配置参数（见 configs/grpo_config.yaml）：
# actor.model_path: checkpoints/sft/final
# algorithm.grpo.group_size: 8
# algorithm.grpo.kl_coef: 0.01
# trainer.learning_rate: 5e-7  # RL 阶段学习率要低！

# ⚠️ 重要提醒：
# - actor MUST 不开启 param_offload（否则训练速度下降 3-5 倍）
# - ref model 可以开启 param_offload（节省显存）

# 7.2 监控关键指标
# 若在 wandb 中发现 frac_reward_zero_std > 0.3，说明需要切换 NGRPO

# 7.3 若需切换 NGRPO
python -m verl.trainer.main_grpo \
    --config configs/grpo_config.yaml \
    --use_ngrpo true  # 启用虚拟满分样本
```

---

## Step 8：最终评估与对比

```bash
# 运行三方对比评估
python src/eval/compare.py \
    --models deepseek-chat checkpoints/sft/final checkpoints/grpo/final \
    --labels "基线(deepseek-chat)" "SFT(7B)" "GRPO(7B)" \
    --test-cases data/test_cases.json \
    --output results/final_comparison.json

# 生成评估报告
python src/eval/report.py \
    --input results/final_comparison.json \
    --output results/final_report.md
```

---

## 验证检查点

完成以下所有验证后，即为成功复现：

| 检查点 | 验证命令 | 成功标准 |
|--------|----------|----------|
| 环境就绪 | `python -c "import torch; assert torch.cuda.is_available()"` | 无报错 |
| Agent 原型 | 运行 Step 4 | 工具调用顺序合理，输出有意义 |
| 数据生成 | 运行 Step 5 | 1000 条数据，格式验证通过 |
| SFT 完成 | 运行 Step 6 | 工具准确率 ≥ 75% |
| RL 完成 | 运行 Step 7 | 工具准确率 ≥ 85% |
| A/港股区分 | 运行 Step 8 | 混淆场景准确率 ≥ 90% |

---

## 常见问题

**Q: OOM（显存不足）**
A: 依次尝试：开启 `gradient_checkpointing`，开启 `fsdp_offload`，减小 `per_device_batch_size`。

**Q: reward_std ≈ 0，模型不学习**
A: 切换 NGRPO（Step 7 中添加 `--use_ngrpo true`），详见 NGRPO 论文：https://arxiv.org/abs/2509.18851

**Q: SFT 后工具调用能力提升但推理链质量下降**
A: 这是梯度冲突问题，参考 DART 论文（https://arxiv.org/abs/2602.00994）使用 LoRA 解耦训练。

**Q: akshare 接口限流**
A: 在工具调用之间增加随机延迟（0.5-1 秒），或使用付费数据源替代。
