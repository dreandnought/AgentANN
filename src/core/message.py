from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Literal


class MessageType(str, Enum):
    FORWARD = "FORWARD"
    BACKWARD = "BACKWARD"
    WEIGHT_UPDATE = "WEIGHT_UPDATE"
    ERROR = "ERROR"
    PING = "PING"
    PONG = "PONG"
    RESET = "RESET"
    FORWARD_ACK = "FORWARD_ACK"


class AgentState(str, Enum):
    IDLE = "IDLE"
    WAITING_FORWARD_INPUT = "WAITING_FORWARD_INPUT"
    COMPUTING_FORWARD = "COMPUTING_FORWARD"
    FORWARD_SENT = "FORWARD_SENT"
    WAITING_BACKWARD_INPUT = "WAITING_BACKWARD_INPUT"
    COMPUTING_BACKWARD = "COMPUTING_BACKWARD"
    UPDATED = "UPDATED"
    ERROR = "ERROR"


@dataclass(slots=True)
class AgentMessage:
    type: MessageType
    sender_id: str
    target_id: str
    step_id: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value
        return d


WebEventType = Literal[
    "AGENT_STATE_CHANGE",
    "GLOBAL_METRICS",
    "GLOBAL_WEIGHTS",
    "LOG_LINE",
    "ERROR",
    "LLM_INTERACTION",
    "WEIGHT_LOADED",
]


@dataclass(slots=True)
class WebEvent:
    type: WebEventType
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "data": self.data}
