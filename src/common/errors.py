class AgentError(Exception):
    """Base error for agent runtime."""


class MaxStepsExceeded(AgentError):
    """Raised when the agent fails to finish within max steps."""


class ToolExecutionError(AgentError):
    """Raised when a tool call cannot be recovered."""
