from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class StockTool:
    name: str
    description: str
    parameters: Dict[str, Any]
    supports_a_share: bool
    supports_hk_share: bool
    handler: Callable[..., Dict[str, Any]]


@dataclass
class FunctionCall:
    name: str
    arguments: str


@dataclass
class ToolCall:
    id: str
    type: str
    function: FunctionCall


@dataclass
class AgentMessage:
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


@dataclass
class AgentTrajectory:
    trajectory_id: str
    prompt: str
    messages: List[AgentMessage]
    tool_calls: List[ToolCall]
    final_output: str
    num_turns: int
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SFTSample:
    sample_id: str
    messages: List[Dict[str, Any]]
    market_type: str
    difficulty: str
    has_cot: bool
    source_model: str
    generated_at: str


@dataclass
class RewardSignal:
    trajectory_id: str
    tool_correctness: float
    output_quality: float
    efficiency_penalty: float
    final_reward: float
    expected_tools: List[str] = field(default_factory=list)
    actual_tools: List[str] = field(default_factory=list)
    tool_match_detail: List[bool] = field(default_factory=list)


@dataclass
class DebugRecord:
    record_id: str
    timestamp: str
    request_id: Optional[str]
    request_body: str
    response_body: str
    tool_calls: Optional[str]
    duration_ms: int
    status_code: int
    error: Optional[str]


@dataclass
class TrainingConfig:
    model_path: str
    output_dir: str
    fsdp_param_offload: bool = False
    fsdp_optimizer_offload: bool = True
    num_train_epochs: int = 3
    batch_size: int = 32
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    gradient_checkpointing: bool = True
    group_size: int = 8
    kl_coef: float = 0.01
    clip_ratio: float = 0.2
    temperature: float = 0.7
    use_ngrpo: bool = False
    ngrpo_threshold: float = 0.3
    wandb_project: str = "vibe-rl"
    wandb_run_name: str = "default"
    seed: int = 42
