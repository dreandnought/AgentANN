from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

from src.core.coordinator import Coordinator
from src.core.message import WebEvent
from src.web.ws_manager import WebSocketManager


def create_app() -> Starlette:
    base_dir = Path(__file__).resolve().parents[2]
    static_dir = Path(__file__).resolve().parent / "static"
    config_dir = base_dir / "config"

    ws_manager = WebSocketManager()

    async def publish_web_event(evt: WebEvent) -> None:
        await ws_manager.broadcast_json(evt.to_dict())

    coordinator = Coordinator(
        network_config_path=config_dir / "network_config.json",
        global_weights_path=base_dir / "global_weights.json",
        logs_dir=base_dir / "logs",
        publish_web_event=publish_web_event,
    )

    async def index(request: Any) -> FileResponse:
        _ = request
        return FileResponse(str(static_dir / "index.html"))

    async def get_state(request: Any) -> JSONResponse:
        _ = request
        return JSONResponse(await coordinator.snapshot())

    async def control_start(request: Any) -> JSONResponse:
        _ = request
        await coordinator.start()
        return JSONResponse({"ok": True})

    async def control_pause(request: Any) -> JSONResponse:
        _ = request
        await coordinator.pause()
        return JSONResponse({"ok": True})

    async def control_reset(request: Any) -> JSONResponse:
        _ = request
        await coordinator.reset()
        return JSONResponse({"ok": True})

    async def load_weights(request: Any) -> JSONResponse:
        form = await request.form()
        if "file" not in form:
            return JSONResponse({"ok": False, "error": "missing_file"}, status_code=400)
        
        file = form["file"]
        content = await file.read()
        try:
            import json
            weights_data = json.loads(content)
            await coordinator.load_weights(weights_data)
            return JSONResponse({"ok": True})
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    from starlette.background import BackgroundTasks

    async def infer(request: Any) -> JSONResponse:
        try:
            data = await request.json()
            features = data.get("features")
            if not features or len(features) != 4:
                return JSONResponse({"ok": False, "error": "requires exactly 4 features"}, status_code=400)
            
            x = [float(f) for f in features]
            async def run_inference_with_error_handling(features_x: list[float]) -> None:
                try:
                    await coordinator.inference_single(features_x)
                except Exception as ex:
                    await publish_web_event(WebEvent(type="ERROR", data={"error": f"Inference failed: {str(ex)}"}))

            tasks = BackgroundTasks()
            tasks.add_task(run_inference_with_error_handling, x)
            return JSONResponse({"ok": True, "message": "started"}, background=tasks)
        except Exception as e:
            return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    async def websocket_endpoint(websocket: WebSocket) -> None:
        await ws_manager.connect(websocket)
        try:
            await websocket.send_json({"type": "STATE_SNAPSHOT", "data": await coordinator.snapshot()})
            while True:
                msg = await websocket.receive_json()
                action = msg.get("action")
                if action == "START_TRAINING":
                    await coordinator.start()
                elif action == "PAUSE_TRAINING":
                    await coordinator.pause()
                elif action == "RESET":
                    await coordinator.reset()
                elif action == "PING":
                    await websocket.send_json({"type": "PONG", "data": {}})
        except WebSocketDisconnect:
            await ws_manager.disconnect(websocket)
        except Exception:
            await ws_manager.disconnect(websocket)

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: Starlette):
        yield
        await coordinator.shutdown()

    routes = [
        Route("/", index, methods=["GET"]),
        Route("/api/state", get_state, methods=["GET"]),
        Route("/api/weights/load", load_weights, methods=["POST"]),
        Route("/api/infer", infer, methods=["POST"]),
        Route("/api/control/start", control_start, methods=["POST"]),
        Route("/api/control/pause", control_pause, methods=["POST"]),
        Route("/api/control/reset", control_reset, methods=["POST"]),
        WebSocketRoute("/ws", websocket_endpoint),
    ]

    app = Starlette(routes=routes, lifespan=lifespan)
    app.state.ws_manager = ws_manager
    app.state.coordinator = coordinator
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    return app


app = create_app()
