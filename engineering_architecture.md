# 工程实现方案：大模型接入与后端架构

## 一、 系统技术栈选型
1. **核心计算与并发**: Python 3.10+, `asyncio`
2. **Web 后端**: `Starlette` (提供静态页面和 WebSocket 实时双向通信；在常规 Python 版本环境下可无缝替换为 `FastAPI`)
3. **LLM 接口客户端**: `openai` (Python官方库，支持自定义 BaseURL 和 API Key)
4. **日志与持久化**: 原生的 `logging` 模块 + `json` 文件读写

## 二、 大模型 API 接入配置 (`llm_config.json`)
为了让系统灵活支持不同的模型（如 GPT-4, Claude-3, 或者本地部署的 Llama-3/Qwen），我们需要一个独立的配置文件 `llm_config.json`。

```json
{
  "api_provider": "openai",
  "base_url": "https://api.openai.com/v1",
  "api_key_env": "LLM_API_KEY",
  "model_name": "gpt-4o-mini",
  "temperature": 0.0,
  "max_tokens": 1024,
  "system_prompts_path": "agent_prompts.md",
  "retry_strategy": {
    "max_retries": 3,
    "backoff_seconds": 1.0
  }
}
```

## 三、 Agent 集群引擎设计
在 Web 服务启动时，我们会触发一个后台的 `asyncio.create_task(run_agent_cluster())`：

1. **LLM Wrapper 封装**:
   创建一个异步的 `call_llm(prompt, agent_id)` 函数，封装对大模型的请求，处理 JSON 解析（强制要求 LLM 输出包含梯度和更新后权重的 JSON 格式）。
2. **NeuronAgent 改造**:
   原先的代码硬编码数学计算，现在改为：将收到的输入 $X$ 组装成 Prompt，发送给 `call_llm`，等待 LLM 吐出计算结果（激活值 $a$ 或梯度 $dW, db$），然后再将其通过 Queue 发给下游。
3. **状态监控上报**:
   在每个 Agent 内部加入钩子：每当状态改变（等待中、计算中、已发送、出错）时，不仅写入 `logs/H3.log`，同时通过全局的 `WebSocket Manager` 广播一条状态消息给前端。

## 四、 核心目录结构
```text
/workspace
├── config/
│   ├── network_config.json      # 网络拓扑
│   └── llm_config.json          # 大模型API配置
├── docs/
│   ├── system_design.md
│   ├── agent_prompts.md
│   └── error_handling.md
├── src/
│   ├── core/
│   │   ├── message.py           # 内部消息协议
│   │   ├── neuron_agent.py      # LLM驱动的Agent类
│   │   ├── coordinator.py       # 协调器
│   │   └── llm_client.py        # 大模型请求封装
│   ├── web/
│   │   ├── main.py              # Starlette 入口
│   │   ├── ws_manager.py        # WebSocket 管理器
│   │   └── static/              # 前端页面
│   │       ├── index.html
│   │       └── app.js
├── data/                        # Iris 数据存放
├── logs/                        # 独立日志
└── global_weights.json          # 实时权重
```
