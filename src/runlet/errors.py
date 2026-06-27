class RunletError(Exception):
    code = "runlet_error"


class ModelError(RunletError):
    code = "model_error"


class ToolError(RunletError):
    code = "tool_error"


class ContextOverflowError(RunletError):
    code = "context_overflow"


class HookError(RunletError):
    code = "hook_error"


class PolicyStop(RunletError):
    code = "policy_stop"


class StateError(RunletError):
    code = "state_error"


class CancellationError(RunletError):
    code = "cancelled"


class InternalRuntimeError(RunletError):
    code = "internal_runtime_error"
