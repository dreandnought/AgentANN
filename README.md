# Multi-Agent BP Network (LLM Agent Cluster)

## 目录
- `config/llm_config.json`: 大模型 API 接入配置（API Key 通过环境变量提供）
- `config/network_config.json`: 网络结构配置（H0-H7 / O0-O2）
- `src/web/main.py`: FastAPI 后端入口（HTTP + WebSocket + 静态前端）
- `src/web/static/`: 实时监控页面
- `global_weights.json`: 实时权重（由 Coordinator 统一写入）
- `logs/`: 每个 Agent 的独立日志

## 本地启动

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
export LLM_API_KEY="..."
uvicorn src.web.main:app --host 0.0.0.0 --port 8000
```

浏览器打开 `http://localhost:8000/`。
