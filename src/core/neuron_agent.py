from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Awaitable

import numpy as np

from .message import AgentMessage, AgentState, MessageType, WebEvent


def _relu(x: float) -> float:
    return float(max(0.0, x))


def _relu_deriv(x: float) -> float:
    return 1.0 if x > 0.0 else 0.0


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


@dataclass(frozen=True)
class NeuronSpec:
    agent_id: str
    layer: str
    input_dim: int
    input_sources: list[str]
    output_targets: list[str]
    learning_rate: float
    activation: str = "relu"


class NeuronAgent:
    def __init__(
        self,
        spec: NeuronSpec,
        inbox: "asyncio.Queue[AgentMessage]",
        send: Callable[[AgentMessage], Awaitable[None]],
        publish_web_event: Callable[[WebEvent], Awaitable[None]],
        logs_dir: Path,
        seed: int,
        llm_client: Any | None = None,
        llm_semaphore: asyncio.Semaphore | None = None,
        llm_enabled: bool = False,
        llm_system_prompt: str = "",
        llm_timeout_seconds: float = 20.0,
        llm_max_retries: int = 3,
        llm_backoff_seconds: float = 1.0,
    ):
        self.spec = spec
        self.inbox = inbox
        self._send = send
        self._publish_web_event = publish_web_event
        self._state = AgentState.IDLE
        self._last_error: str | None = None

        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0.0, 0.02, size=(spec.input_dim,)).astype(np.float64)
        self.bias = float(rng.normal(0.0, 0.02))

        self._cache_step_id: str | None = None
        self._cache_input: np.ndarray | None = None
        self._cache_z: float | None = None
        self._cache_a: float | None = None

        self._forward_buffer: dict[str, dict[str, float]] = {}
        self._backward_buffer: dict[str, dict[str, float]] = {}

        self._logs_dir = logs_dir
        self._llm_client = llm_client
        self._llm_semaphore = llm_semaphore
        self._llm_enabled = llm_enabled
        self._llm_system_prompt = llm_system_prompt
        self._llm_timeout_seconds = llm_timeout_seconds
        self._llm_max_retries = llm_max_retries
        self._llm_backoff_seconds = llm_backoff_seconds

        logs_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(f"agent.{spec.agent_id}")
        self._logger.setLevel(logging.INFO)
        handler = logging.FileHandler(logs_dir / f"{spec.agent_id}.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s\t%(message)s"))
        self._logger.handlers = []
        self._logger.addHandler(handler)
        self._logger.propagate = False

    def _write_dead_letter(self, payload: dict[str, Any]) -> None:
        p = self._logs_dir / "dead_letter.log"
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    async def _llm_json(self, req: dict[str, Any]) -> dict[str, Any]:
        if not self._llm_enabled or self._llm_client is None:
            raise RuntimeError("LLM not enabled")

        last_err: Exception | None = None
        for attempt in range(self._llm_max_retries):
            await self._publish_web_event(
                WebEvent(
                    type="LLM_INTERACTION",
                    data={
                        "agent_id": self.spec.agent_id,
                        "step_id": self._cache_step_id,
                        "op": req.get("op", "unknown"),
                        "phase": "request",
                        "attempt": attempt + 1,
                        "request": req,
                    },
                )
            )
            try:
                sem = self._llm_semaphore
                if sem is None:
                    if req.get("op") == "forward":
                        await self._set_state(AgentState.COMPUTING_FORWARD, {"step_id": self._cache_step_id, "brain": "llm"})
                    elif req.get("op") == "backward":
                        await self._set_state(AgentState.COMPUTING_BACKWARD, {"step_id": self._cache_step_id, "brain": "llm"})
                    raw = await asyncio.wait_for(
                        self._llm_client.chat(self._llm_system_prompt, json.dumps(req, ensure_ascii=False)),
                        timeout=self._llm_timeout_seconds,
                    )
                else:
                    async with sem:
                        if req.get("op") == "forward":
                            await self._set_state(AgentState.COMPUTING_FORWARD, {"step_id": self._cache_step_id, "brain": "llm"})
                        elif req.get("op") == "backward":
                            await self._set_state(AgentState.COMPUTING_BACKWARD, {"step_id": self._cache_step_id, "brain": "llm"})
                        raw = await asyncio.wait_for(
                            self._llm_client.chat(self._llm_system_prompt, json.dumps(req, ensure_ascii=False)),
                            timeout=self._llm_timeout_seconds,
                        )
                resp = json.loads(raw)

                await self._publish_web_event(
                    WebEvent(
                        type="LLM_INTERACTION",
                        data={
                            "agent_id": self.spec.agent_id,
                            "step_id": self._cache_step_id,
                            "op": req.get("op", "unknown"),
                            "phase": "response",
                            "attempt": attempt + 1,
                            "response": resp,
                        },
                    ),
                )

                return resp
            except Exception as e:
                last_err = e
                await self._publish_web_event(
                    WebEvent(
                        type="LLM_INTERACTION",
                        data={
                            "agent_id": self.spec.agent_id,
                            "step_id": self._cache_step_id,
                            "op": req.get("op", "unknown"),
                            "phase": "error",
                            "attempt": attempt + 1,
                            "error": f"{type(e).__name__}: {e}",
                        },
                    )
                )
                await asyncio.sleep(self._llm_backoff_seconds * (2**attempt))

        self._write_dead_letter(
            {
                "event": "LLM_FAILURE",
                "agent_id": self.spec.agent_id,
                "step_id": self._cache_step_id,
                "request": req,
                "error": f"{type(last_err).__name__}: {last_err}",
            }
        )
        raise RuntimeError(f"LLM call failed: {type(last_err).__name__}: {last_err}") from last_err

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def last_error(self) -> str | None:
        return self._last_error

    async def _set_state(self, state: AgentState, detail: dict[str, Any] | None = None) -> None:
        self._state = state
        payload: dict[str, Any] = {"agent_id": self.spec.agent_id, "state": state.value}
        if detail:
            payload.update(detail)
        await self._publish_web_event(WebEvent(type="AGENT_STATE_CHANGE", data=payload))

    async def run(self) -> None:
        await self._set_state(AgentState.WAITING_FORWARD_INPUT)
        while True:
            msg = await self.inbox.get()
            if msg.type == MessageType.PING:
                await self._send(
                    AgentMessage(
                        type=MessageType.PONG,
                        sender_id=self.spec.agent_id,
                        target_id=msg.sender_id,
                        step_id=msg.step_id,
                        payload={},
                    )
                )
                continue

            if msg.type == MessageType.RESET:
                self._forward_buffer.clear()
                self._backward_buffer.clear()
                self._cache_step_id = None
                self._cache_input = None
                self._cache_z = None
                self._cache_a = None
                self._last_error = None
                await self._set_state(AgentState.WAITING_FORWARD_INPUT)
                continue

            try:
                if msg.type == MessageType.FORWARD:
                    await self._handle_forward(msg)
                elif msg.type == MessageType.BACKWARD:
                    await self._handle_backward(msg)
                elif msg.type == MessageType.ERROR:
                    self._last_error = str(msg.payload.get("error") or "unknown")
                    self._logger.info(json.dumps({"event": "ERROR", "payload": msg.payload}, ensure_ascii=False))
                    await self._set_state(AgentState.ERROR, {"error": self._last_error})
                elif msg.type == MessageType.FORWARD_ACK:
                    pass  # Silently ignore ACKs from output targets
                else:
                    self._logger.info(json.dumps({"event": "UNKNOWN_MESSAGE", "type": msg.type}, ensure_ascii=False))
            except Exception as e:
                self._last_error = f"{type(e).__name__}: {e}"
                self._logger.info(json.dumps({"event": "EXCEPTION", "error": self._last_error}, ensure_ascii=False))
                await self._set_state(AgentState.ERROR, {"error": self._last_error})
                await self._send(
                    AgentMessage(
                        type=MessageType.ERROR,
                        sender_id=self.spec.agent_id,
                        target_id="coordinator",
                        step_id=msg.step_id,
                        payload={"error": self._last_error},
                    )
                )

    async def _handle_forward(self, msg: AgentMessage) -> None:
        await self._set_state(AgentState.WAITING_FORWARD_INPUT, {"step_id": msg.step_id})
        
        if "coordinator" in self.spec.input_sources:
            # First layer receives a full vector 'x' from coordinator
            x = np.asarray(msg.payload["x"], dtype=np.float64)
        else:
            # Subsequent layers receive scalar 'a' from multiple upstream agents
            buf = self._forward_buffer.setdefault(msg.step_id, {})
            buf[msg.sender_id] = float(msg.payload["a"])
            if len(buf) < len(self.spec.input_sources):
                return
            x = np.array([buf[src] for src in self.spec.input_sources], dtype=np.float64)
            self._forward_buffer.pop(msg.step_id, None)

        self._cache_step_id = msg.step_id
        self._cache_input = x

        if self._llm_enabled:
            resp = await self._llm_json(
                {
                    "op": "forward",
                    "agent_id": self.spec.agent_id,
                    "activation": self.spec.activation,
                    "weights": self.weights.tolist(),
                    "bias": self.bias,
                    "x": x.tolist(),
                }
            )
            z = float(resp["z"])
            a = float(resp["a"])
        else:
            await self._set_state(AgentState.COMPUTING_FORWARD, {"step_id": msg.step_id, "brain": "cpu"})
            z = float(np.dot(self.weights, x) + self.bias)
            if self.spec.activation == "relu":
                a = _relu(z)
            elif self.spec.activation == "sigmoid":
                a = _sigmoid(z)
            else:
                a = z  # linear
        self._cache_z = z
        self._cache_a = a
        self._logger.info(json.dumps({"event": "FORWARD", "step_id": msg.step_id, "z": z, "a": a}, ensure_ascii=False))
        
        
        for target in self.spec.output_targets:
            await self._send(
                AgentMessage(
                    type=MessageType.FORWARD,
                    sender_id=self.spec.agent_id,
                    target_id=target,
                    step_id=msg.step_id,
                    payload={"a": a},
                )
            )

        for source in self.spec.input_sources:
            await self._send(
                AgentMessage(
                    type=MessageType.FORWARD_ACK,
                    sender_id=self.spec.agent_id,
                    target_id=source,
                    step_id=msg.step_id,
                    payload={"a": a},
                )
            )

        await self._set_state(AgentState.FORWARD_SENT, {"step_id": msg.step_id})
        await self._set_state(AgentState.WAITING_BACKWARD_INPUT, {"step_id": msg.step_id})

    async def _handle_backward(self, msg: AgentMessage) -> None:
        if self._cache_step_id != msg.step_id:
            return

        await self._set_state(AgentState.WAITING_BACKWARD_INPUT, {"step_id": msg.step_id})
        
        if "coordinator" in self.spec.output_targets:
            # Last layer receives a single gradient scalar from coordinator
            total_grad = float(msg.payload["grad"])
        else:
            # Other layers receive gradients from multiple downstream agents
            buf = self._backward_buffer.setdefault(msg.step_id, {})
            buf[msg.sender_id] = float(msg.payload["grad"])
            if len(buf) < len(self.spec.output_targets):
                return
            total_grad = float(sum(buf.values()))
            self._backward_buffer.pop(msg.step_id, None)

        if self._cache_input is None or self._cache_z is None or self._cache_a is None:
            raise RuntimeError("Missing forward cache")

        x = self._cache_input
        
        if self._llm_enabled:
            resp = await self._llm_json(
                {
                    "op": "backward",
                    "agent_id": self.spec.agent_id,
                    "activation": self.spec.activation,
                    "weights": self.weights.tolist(),
                    "bias": self.bias,
                    "x": x.tolist(),
                    "z": float(self._cache_z),
                    "a": float(self._cache_a),
                    "total_grad": float(total_grad),
                    "needs_grad_to_upstream": ("coordinator" not in self.spec.input_sources),
                }
            )
            delta = float(resp["delta"])
            d_w = np.asarray(resp["d_w"], dtype=np.float64)
            d_b = float(resp["d_b"])
            grad_to_upstream = resp.get("grad_to_upstream")
        else:
            await self._set_state(AgentState.COMPUTING_BACKWARD, {"step_id": msg.step_id, "brain": "cpu"})
            # Compute local delta based on activation function
            if self.spec.activation == "relu":
                delta = total_grad * _relu_deriv(float(self._cache_z))
            elif self.spec.activation == "sigmoid":
                a = self._cache_a
                delta = total_grad * (a * (1.0 - a))
            else:
                delta = total_grad  # linear

            d_w = delta * x
            d_b = delta
            grad_to_upstream = (delta * self.weights).tolist() if "coordinator" not in self.spec.input_sources else None

        # Send backward to upstream (if we are not the first layer)
        if "coordinator" not in self.spec.input_sources:
            for idx, target in enumerate(self.spec.input_sources):
                if grad_to_upstream is None:
                    raise RuntimeError("Missing upstream gradients")
                await self._send(
                    AgentMessage(
                        type=MessageType.BACKWARD,
                        sender_id=self.spec.agent_id,
                        target_id=target,
                        step_id=msg.step_id,
                        payload={"grad": float(grad_to_upstream[idx])},
                    )
                )

        # Update weights
        self.weights = self.weights - self.spec.learning_rate * d_w
        self.bias = float(self.bias - self.spec.learning_rate * d_b)

        self._logger.info(
            json.dumps(
                {
                    "event": "BACKWARD",
                    "step_id": msg.step_id,
                    "total_grad": total_grad,
                    "delta": float(delta),
                    "dw_norm": float(np.linalg.norm(d_w)),
                    "db": float(d_b),
                },
                ensure_ascii=False,
            )
        )

        await self._send(
            AgentMessage(
                type=MessageType.WEIGHT_UPDATE,
                sender_id=self.spec.agent_id,
                target_id="coordinator",
                step_id=msg.step_id,
                payload={"weights": self.weights.tolist(), "bias": self.bias},
            )
        )
        await self._set_state(AgentState.UPDATED, {"step_id": msg.step_id})
        await self._set_state(AgentState.WAITING_FORWARD_INPUT)
