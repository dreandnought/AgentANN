from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import numpy as np
import os

from .llm_client import LLMConfig, build_llm_client
from .message import AgentMessage, MessageType, WebEvent
from .neuron_agent import NeuronAgent, NeuronSpec


@dataclass(frozen=True)
class NetworkConfig:
    input_dim: int
    hidden_agents: list[str]
    output_agents: list[str]
    hidden_activation: str
    output_activation: str
    learning_rate: float

    @staticmethod
    def load(path: str | Path) -> "NetworkConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        net = raw["network"]
        hp = raw.get("hyperparameters") or {}
        return NetworkConfig(
            input_dim=int(net["input_dim"]),
            hidden_agents=list(net["hidden_layer"]["agents"]),
            output_agents=list(net["output_layer"]["agents"]),
            hidden_activation=net["hidden_layer"].get("activation", "relu"),
            output_activation=net["output_layer"].get("activation", "sigmoid"),
            learning_rate=float(hp.get("learning_rate") or 0.01),
        )


class Coordinator:
    def __init__(
        self,
        network_config_path: Path,
        global_weights_path: Path,
        logs_dir: Path,
        publish_web_event: Callable[[WebEvent], Awaitable[None]],
        llm_config_path: Path | None = None,
    ):
        self._cfg = NetworkConfig.load(network_config_path)
        self._global_weights_path = global_weights_path
        self._logs_dir = logs_dir
        self._publish_web_event = publish_web_event

        resolved_llm_path = llm_config_path or (network_config_path.parent / "llm_config.json")
        if resolved_llm_path.exists():
            self._llm_cfg = LLMConfig.load(resolved_llm_path)
            self._llm_client = build_llm_client(self._llm_cfg)
            self._llm_semaphore = asyncio.Semaphore(max(1, int(self._llm_cfg.max_concurrent_requests)))
            prompts_path = Path(self._llm_cfg.system_prompts_path)
            self._llm_system_prompt = prompts_path.read_text(encoding="utf-8") if prompts_path.exists() else ""
        else:
            self._llm_cfg = None
            self._llm_client = None
            self._llm_semaphore = None
            self._llm_system_prompt = ""

        self._inbox: "asyncio.Queue[AgentMessage]" = asyncio.Queue()
        self._agent_inboxes: dict[str, "asyncio.Queue[AgentMessage]"] = {
            agent_id: asyncio.Queue() for agent_id in (self._cfg.hidden_agents + self._cfg.output_agents)
        }
        self._agents: dict[str, NeuronAgent] = {}
        self._agent_specs: dict[str, NeuronSpec] = {}
        self._agent_tasks: dict[str, asyncio.Task[None]] = {}

        self._running = asyncio.Event()
        self._shutdown = asyncio.Event()
        self._loop_task: asyncio.Task[None] | None = None
        self._control_lock = asyncio.Lock()

        self._step_counter = 0
        self._global_weights: dict[str, Any] = {}

    async def _send(self, msg: AgentMessage) -> None:
        if msg.target_id == "coordinator":
            await self._inbox.put(msg)
            return
        q = self._agent_inboxes.get(msg.target_id)
        if not q:
            await self._inbox.put(
                AgentMessage(
                    type=MessageType.ERROR,
                    sender_id="coordinator",
                    target_id="coordinator",
                    step_id=msg.step_id,
                    payload={"error": f"unknown target: {msg.target_id}", "original": msg.to_dict()},
                )
            )
            return
        await q.put(msg)

    @property
    def config(self) -> NetworkConfig:
        return self._cfg

    def is_running(self) -> bool:
        return self._running.is_set()

    async def start(self) -> None:
        async with self._control_lock:
            if self._loop_task is None:
                if not self._agents:
                    self._init_agents()
                self._loop_task = asyncio.create_task(self._run_loop())
            self._running.set()

    async def pause(self) -> None:
        async with self._control_lock:
            self._running.clear()

    async def reset(self) -> None:
        async with self._control_lock:
            self._running.clear()
            await self._broadcast(MessageType.RESET, payload={})
            self._step_counter = 0
            self._global_weights = {}
            self._write_global_weights()
            await self._publish_web_event(WebEvent(type="GLOBAL_WEIGHTS", data={"weights": self._global_weights}))

    async def shutdown(self) -> None:
        async with self._control_lock:
            self._running.clear()
            self._shutdown.set()
            if self._loop_task:
                await asyncio.wait([self._loop_task], timeout=2.0)
            for t in self._agent_tasks.values():
                t.cancel()

    async def snapshot(self) -> dict[str, Any]:
        llm_has_key = False
        llm_env = None
        if self._llm_cfg and self._llm_cfg.enabled:
            llm_env = self._llm_cfg.api_key_env
            llm_has_key = bool(os.getenv(self._llm_cfg.api_key_env))
        return {
            "running": self.is_running(),
            "step": self._step_counter,
            "llm_enabled": bool(self._llm_cfg and self._llm_cfg.enabled),
            "llm_has_key": llm_has_key,
            "llm_api_key_env": llm_env,
            "agents": {
                aid: {"state": self._agents[aid].state.value, "last_error": self._agents[aid].last_error}
                for aid in self._agents
            },
            "global_weights_keys": list(self._global_weights.keys()),
        }

    async def load_weights(self, weights_data: dict[str, Any]) -> None:
        async with self._control_lock:
            if not self._agents:
                self._init_agents()
            
            # Reset all agents to clear any previous ERROR states before loading
            await self._broadcast(MessageType.RESET, payload={})

            raw_weights = weights_data
            missing = set(self._agents.keys()) - set(raw_weights.keys())
            if missing:
                raise ValueError(f"Missing weights for agents: {sorted(missing)}")
            for aid, agent in self._agents.items():
                if aid in raw_weights:
                    w = raw_weights[aid]["weights"]
                    b = raw_weights[aid]["bias"]
                    if not isinstance(w, list) or len(w) != agent.spec.input_dim:
                        raise ValueError(
                            f"Invalid weights shape for {aid}: expected len={agent.spec.input_dim}, got len={len(w) if isinstance(w, list) else 'non-list'}"
                        )
                    agent.weights = np.array(w, dtype=np.float64)
                    agent.bias = float(b)
                    await self._publish_web_event(
                        WebEvent(
                            type="WEIGHT_LOADED",
                            data={
                                "agent_id": aid,
                                "weights": w,
                                "bias": float(b),
                            }
                        )
                    )
                    await asyncio.sleep(0.05)
            
            self._global_weights = raw_weights
            self._write_global_weights()

    async def inference_single(self, x: list[float]) -> dict[str, Any]:
        """Perform a single forward pass without triggering backward update"""
        async with self._control_lock:
            if not self._agents:
                self._init_agents()
        
        step_id = f"infer-{int(time.time() * 1000)}"

        timeout_budget = 5.0
        if self._llm_cfg and self._llm_cfg.enabled:
            retries = int(self._llm_cfg.retry_strategy.max_retries)
            backoff = float(self._llm_cfg.retry_strategy.backoff_seconds)
            llm_timeout = float(self._llm_cfg.request_timeout_seconds)
            backoff_total = backoff * (2**retries - 1)
            timeout_budget = max(timeout_budget, retries * llm_timeout + backoff_total + 1.0)
        
        for hid in self._cfg.hidden_agents:
            await self._agent_inboxes[hid].put(
                AgentMessage(
                    type=MessageType.FORWARD,
                    sender_id="coordinator",
                    target_id=hid,
                    step_id=step_id,
                    payload={"x": x},
                )
            )

            # Wait for FORWARD_ACK from this hidden agent before proceeding to the next
            temp_msgs = []
            deadline = time.time() + timeout_budget
            while time.time() < deadline:
                timeout = max(0.0, deadline - time.time())
                try:
                    msg = await asyncio.wait_for(self._inbox.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    for m in temp_msgs: await self._inbox.put(m)
                    raise TimeoutError(f"Timeout waiting for ACK from {hid}")
                
                if msg.type == MessageType.FORWARD_ACK and msg.sender_id == hid and msg.step_id == step_id:
                    break
                elif msg.type == MessageType.ERROR and msg.step_id == step_id:
                    for m in temp_msgs: await self._inbox.put(m)
                    raise RuntimeError(str(msg.payload.get("error") or "agent_error"))
                else:
                    temp_msgs.append(msg)
            
            for m in temp_msgs:
                await self._inbox.put(m)

        outputs: dict[str, float] = {}
        deadline = time.time() + timeout_budget
        while time.time() < deadline and len(outputs) < len(self._cfg.output_agents):
            timeout = max(0.0, deadline - time.time())
            try:
                msg = await asyncio.wait_for(self._inbox.get(), timeout=timeout)
                if msg.type == MessageType.ERROR and msg.step_id == step_id:
                    raise RuntimeError(str(msg.payload.get("error") or "agent_error"))
                if msg.type == MessageType.FORWARD and msg.step_id == step_id:
                    outputs[msg.sender_id] = float(msg.payload["a"])
            except asyncio.TimeoutError:
                break
        
        if len(outputs) < len(self._cfg.output_agents):
            raise TimeoutError(f"Inference timeout, received {len(outputs)}/{len(self._cfg.output_agents)} outputs")

        out_vec = np.array([outputs.get(aid, 0.0) for aid in self._cfg.output_agents], dtype=np.float64)
        pred_class = int(np.argmax(out_vec))
        confidence = float(np.max(out_vec))

        res = {
            "prediction": pred_class,
            "confidence": confidence,
            "probabilities": out_vec.tolist(),
            "raw_outputs": outputs
        }

        await self._publish_web_event(
            WebEvent(
                type="INFERENCE_RESULT",
                data=res
            )
        )

        return res

    def _init_agents(self) -> None:
        def _seed(agent_id: str) -> int:
            return abs(hash(agent_id)) % (2**31)

        llm_enabled = bool(self._llm_cfg and self._llm_cfg.enabled)
        llm_timeout_seconds = float(self._llm_cfg.request_timeout_seconds) if self._llm_cfg else 20.0
        llm_max_retries = int(self._llm_cfg.retry_strategy.max_retries) if self._llm_cfg else 3
        llm_backoff_seconds = float(self._llm_cfg.retry_strategy.backoff_seconds) if self._llm_cfg else 1.0

        for aid in self._cfg.hidden_agents:
            spec = NeuronSpec(
                agent_id=aid,
                layer="hidden",
                input_dim=self._cfg.input_dim,
                input_sources=["coordinator"],
                output_targets=self._cfg.output_agents,
                learning_rate=self._cfg.learning_rate,
                activation=self._cfg.hidden_activation,
            )
            self._agent_specs[aid] = spec
            agent = NeuronAgent(
                spec=spec,
                inbox=self._agent_inboxes[aid],
                send=self._send,
                publish_web_event=self._publish_web_event,
                logs_dir=self._logs_dir,
                seed=_seed(aid),
                llm_client=self._llm_client,
                llm_semaphore=self._llm_semaphore,
                llm_enabled=llm_enabled,
                llm_system_prompt=self._llm_system_prompt,
                llm_timeout_seconds=llm_timeout_seconds,
                llm_max_retries=llm_max_retries,
                llm_backoff_seconds=llm_backoff_seconds,
            )
            self._agents[aid] = agent
            self._agent_tasks[aid] = asyncio.create_task(agent.run())

        for aid in self._cfg.output_agents:
            spec = NeuronSpec(
                agent_id=aid,
                layer="output",
                input_dim=len(self._cfg.hidden_agents),
                input_sources=self._cfg.hidden_agents,
                output_targets=["coordinator"],
                learning_rate=self._cfg.learning_rate,
                activation=self._cfg.output_activation,
            )
            self._agent_specs[aid] = spec
            agent = NeuronAgent(
                spec=spec,
                inbox=self._agent_inboxes[aid],
                send=self._send,
                publish_web_event=self._publish_web_event,
                logs_dir=self._logs_dir,
                seed=_seed(aid),
                llm_client=self._llm_client,
                llm_semaphore=self._llm_semaphore,
                llm_enabled=llm_enabled,
                llm_system_prompt=self._llm_system_prompt,
                llm_timeout_seconds=llm_timeout_seconds,
                llm_max_retries=llm_max_retries,
                llm_backoff_seconds=llm_backoff_seconds,
            )
            self._agents[aid] = agent
            self._agent_tasks[aid] = asyncio.create_task(agent.run())

    async def _broadcast(self, msg_type: MessageType, payload: dict[str, Any]) -> None:
        step_id = f"control-{int(time.time() * 1000)}"
        for aid, q in self._agent_inboxes.items():
            await q.put(
                AgentMessage(
                    type=msg_type,
                    sender_id="coordinator",
                    target_id=aid,
                    step_id=step_id,
                    payload=payload,
                )
            )

    def _write_dead_letter(self, payload: dict[str, Any]) -> None:
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        p = self._logs_dir / "dead_letter.log"
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    async def _health_check(self, timeout_seconds: float = 1.0) -> list[str]:
        if not self._agents:
            return []

        step_id = f"health-{int(time.time() * 1000)}"
        for aid in self._agents.keys():
            await self._agent_inboxes[aid].put(
                AgentMessage(
                    type=MessageType.PING,
                    sender_id="coordinator",
                    target_id=aid,
                    step_id=step_id,
                    payload={},
                )
            )

        got: set[str] = set()
        buffered: list[AgentMessage] = []
        deadline = time.time() + timeout_seconds
        while time.time() < deadline and len(got) < len(self._agents):
            timeout = max(0.0, deadline - time.time())
            try:
                msg = await asyncio.wait_for(self._inbox.get(), timeout=timeout)
            except asyncio.TimeoutError:
                break

            if msg.type == MessageType.PONG and msg.step_id == step_id:
                got.add(msg.sender_id)
            else:
                buffered.append(msg)

        for msg in buffered:
            await self._inbox.put(msg)

        missing = sorted(set(self._agents.keys()) - got)
        return missing

    async def _restart_agent(self, agent_id: str, reason: str) -> None:
        t = self._agent_tasks.get(agent_id)
        if t:
            t.cancel()

        spec = self._agent_specs.get(agent_id)
        if not spec:
            return

        llm_enabled = bool(self._llm_cfg and self._llm_cfg.enabled)
        llm_timeout_seconds = float(self._llm_cfg.request_timeout_seconds) if self._llm_cfg else 20.0
        llm_max_retries = int(self._llm_cfg.retry_strategy.max_retries) if self._llm_cfg else 3
        llm_backoff_seconds = float(self._llm_cfg.retry_strategy.backoff_seconds) if self._llm_cfg else 1.0

        seed = abs(hash(agent_id)) % (2**31)
        agent = NeuronAgent(
            spec=spec,
            inbox=self._agent_inboxes[agent_id],
            send=self._send,
            publish_web_event=self._publish_web_event,
            logs_dir=self._logs_dir,
            seed=seed,
            llm_client=self._llm_client,
            llm_semaphore=self._llm_semaphore,
            llm_enabled=llm_enabled,
            llm_system_prompt=self._llm_system_prompt,
            llm_timeout_seconds=llm_timeout_seconds,
            llm_max_retries=llm_max_retries,
            llm_backoff_seconds=llm_backoff_seconds,
        )

        if agent_id in self._global_weights:
            agent.weights = np.array(self._global_weights[agent_id]["weights"], dtype=np.float64)
            agent.bias = float(self._global_weights[agent_id]["bias"])

        self._agents[agent_id] = agent
        self._agent_tasks[agent_id] = asyncio.create_task(agent.run())
        self._write_dead_letter({"event": "AGENT_RESTART", "agent_id": agent_id, "reason": reason})

    async def _health_check_and_restart(self, reason: str) -> None:
        missing = await self._health_check(timeout_seconds=1.0)
        for aid in missing:
            await self._restart_agent(aid, reason=reason)

    async def _run_loop(self) -> None:
        rng = np.random.default_rng(42)
        while not self._shutdown.is_set():
            await self._running.wait()
            await self._health_check_and_restart(reason="pre_step")
            self._step_counter += 1
            step_id = f"step-{self._step_counter}"

            x = rng.random(self._cfg.input_dim, dtype=np.float64)
            label = int(rng.integers(0, len(self._cfg.output_agents)))
            y = np.zeros(len(self._cfg.output_agents), dtype=np.float64)
            y[label] = 1.0

            for hid in self._cfg.hidden_agents:
                await self._agent_inboxes[hid].put(
                    AgentMessage(
                        type=MessageType.FORWARD,
                        sender_id="coordinator",
                        target_id=hid,
                        step_id=step_id,
                        payload={"x": x.tolist()},
                    )
                )

            outputs: dict[str, float] = {}
            weight_updates_expected = len(self._cfg.hidden_agents) + len(self._cfg.output_agents)
            weight_updates_seen = 0
            local_losses: dict[str, float] = {}

            deadline = time.time() + 5.0
            while time.time() < deadline and len(outputs) < len(self._cfg.output_agents):
                timeout = max(0.0, deadline - time.time())
                try:
                    msg = await asyncio.wait_for(self._inbox.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    break

                if msg.type == MessageType.FORWARD and msg.step_id == step_id:
                    outputs[msg.sender_id] = float(msg.payload["a"])
                    continue

                if msg.type == MessageType.ERROR:
                    await self._publish_web_event(WebEvent(type="ERROR", data={"step_id": msg.step_id, **msg.payload}))
                    await self._health_check_and_restart(reason="forward_error")
                    self._write_dead_letter({"event": "FORWARD_ERROR", "step_id": msg.step_id, **msg.payload})
                    await self.reset()
                    break

            if len(outputs) < len(self._cfg.output_agents):
                await self._publish_web_event(
                    WebEvent(
                        type="ERROR",
                        data={"step_id": step_id, "error": "forward_timeout", "received": len(outputs)},
                    )
                )
                await self._health_check_and_restart(reason="forward_timeout")
                self._write_dead_letter({"event": "FORWARD_TIMEOUT", "step_id": step_id, "received": len(outputs)})
                await self.reset()
                continue

            out_vec = np.array([outputs.get(aid, 0.0) for aid in self._cfg.output_agents], dtype=np.float64)
            diff = out_vec - y
            loss = float(0.5 * np.sum(diff * diff))
            pred = int(np.argmax(out_vec))
            acc = 1.0 if pred == label else 0.0

            grads = diff
            for i, aid in enumerate(self._cfg.output_agents):
                await self._agent_inboxes[aid].put(
                    AgentMessage(
                        type=MessageType.BACKWARD,
                        sender_id="coordinator",
                        target_id=aid,
                        step_id=step_id,
                        payload={"grad": float(grads[i])},
                    )
                )

            deadline = time.time() + 5.0
            while time.time() < deadline and weight_updates_seen < weight_updates_expected:
                timeout = max(0.0, deadline - time.time())
                try:
                    msg = await asyncio.wait_for(self._inbox.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    break

                if msg.type == MessageType.WEIGHT_UPDATE and msg.step_id == step_id:
                    self._global_weights[msg.sender_id] = {
                        "weights": msg.payload.get("weights"),
                        "bias": msg.payload.get("bias"),
                    }
                    if "loss" in msg.payload:
                        local_losses[msg.sender_id] = float(msg.payload["loss"])
                    self._write_global_weights()
                    weight_updates_seen += 1
                    continue

                if msg.type == MessageType.ERROR:
                    await self._publish_web_event(WebEvent(type="ERROR", data={"step_id": msg.step_id, **msg.payload}))
                    await self._health_check_and_restart(reason="weight_update_error")
                    self._write_dead_letter({"event": "WEIGHT_UPDATE_ERROR", "step_id": msg.step_id, **msg.payload})
                    await self.reset()
                    break

            if weight_updates_seen < weight_updates_expected:
                await self._publish_web_event(
                    WebEvent(
                        type="ERROR",
                        data={
                            "step_id": step_id,
                            "error": "weight_update_timeout",
                            "received": weight_updates_seen,
                            "expected": weight_updates_expected,
                        },
                    )
                )
                await self._health_check_and_restart(reason="weight_update_timeout")
                self._write_dead_letter(
                    {
                        "event": "WEIGHT_UPDATE_TIMEOUT",
                        "step_id": step_id,
                        "received": weight_updates_seen,
                        "expected": weight_updates_expected,
                    }
                )
                await self.reset()
                continue

            await self._publish_web_event(
                WebEvent(
                    type="GLOBAL_METRICS",
                    data={
                        "step_id": step_id,
                        "sample_id": self._step_counter,
                        "label": label,
                        "pred": pred,
                        "current_loss": loss,
                        "accuracy": acc,
                    },
                )
            )

            await self._publish_web_event(
                WebEvent(type="GLOBAL_WEIGHTS", data={"step_id": step_id, "weights": self._global_weights})
            )

    def _write_global_weights(self) -> None:
        tmp = self._global_weights_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._global_weights, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._global_weights_path)
