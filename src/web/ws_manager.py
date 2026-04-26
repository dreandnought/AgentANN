from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from starlette.websockets import WebSocket


@dataclass
class WebSocketManager:
    _connections: set[WebSocket] = field(default_factory=set)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    async def broadcast_json(self, payload: dict[str, Any]) -> None:
        async with self._lock:
            conns = list(self._connections)

        for ws in conns:
            try:
                await ws.send_json(payload)
            except Exception:
                await self.disconnect(ws)
