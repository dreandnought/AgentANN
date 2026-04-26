# 前端监控页面设计方案

为了实时展示这 20 个 Agent (神经元) 模拟 BP 网络的过程，前端页面需要像一个**“作战指挥中心”**。

## 一、 核心界面布局 (UI Layout)

1. **顶栏 (Header)**：
   - 当前训练状态（Epoch, 样本序号）。
   - 当前全局 Loss 和 Accuracy 曲线图。
   - “开始训练”、“暂停”、“重置” 控制按钮。

2. **左侧面板 - 网络拓扑与状态 (Topology Map)**：
   - 使用 `Echarts` 或 `D3.js` 绘制一个 2 层的网络图（输入节点1个 -> 隐藏层10个 -> 输出层10个）。
   - **状态颜色指示**：
     - 灰色 (Gray)：空闲/挂起等待中 (Pending/Waiting)。
     - 蓝色闪烁 (Blue Pulse)：前向传播计算中 (Forwarding)。
     - 绿色闪烁 (Green Pulse)：前向消息已发送，等待反向梯度 (Forward Done)。
     - 橙色闪烁 (Orange Pulse)：反向传播计算中 (Backwarding)。
     - 红色 (Red)：发生错误/崩溃/超时 (Error/Dead)。

3. **右侧面板 - Agent 实时探针 (Agent Inspector)**：
   - 当用户在左侧拓扑图中点击某个神经元（例如 `H3`）时，右侧会展示该 Agent 的实时内部细节：
     - **身份**：隐藏层 Agent H3。
     - **当前任务**：等待 O0-O9 的梯度...
     - **当前参数**：权重 $W$ 的均值/方差、偏置 $b$ 的值。
     - **最近一次交互**：从 LLM 获得的原始 JSON 响应摘要。
     - **实时终端 (Terminal)**：通过 WebSocket 实时 Tail 读取该 Agent 的专属日志 `logs/H3.log`。

## 二、 WebSocket 通信协议

前端与 FastAPI 后端建立一个持久的 `ws://localhost:8000/ws` 连接。

**后端向前端推送的消息格式 (Server -> Client)：**
```json
{
  "type": "AGENT_STATE_CHANGE",
  "data": {
    "agent_id": "H3",
    "state": "COMPUTING_FORWARD",
    "timestamp": 1713800000
  }
}
```
```json
{
  "type": "GLOBAL_METRICS",
  "data": {
    "sample_id": 42,
    "current_loss": 0.45,
    "accuracy": 0.85
  }
}
```

**前端向后端发送的指令 (Client -> Server)：**
```json
{
  "action": "START_TRAINING",
  "params": {
    "batch_size": 1
  }
}
```

## 三、 技术栈
- HTML5 / CSS3 (原生或 TailwindCSS 保证轻量美观)
- Vanilla JavaScript / 现代浏览器 Fetch API
- `Echarts` (用于绘制神经元拓扑关系图和 Loss 曲线图)
- WebSocket (双向实时通信)