# 🌟 Multi-Agent 模拟神经网络：让大模型化身“神经元”的探索之旅

## 📖 项目简介
本项目是一个极具实验性和前瞻性的分布式 AI 系统。我们打破了传统深度学习框架中基于矩阵运算的黑盒模式，**将神经网络（BP）中的每一个“神经元”具象化为一个独立的大语言模型（LLM）Agent**。
在这个系统中，特征提取、前向传播、甚至反向传播的梯度计算，均由大语言模型通过理解提示词并执行数学推理来完成。这不仅为神经网络带来了极强的可解释性，也探索了 LLM 在充当确定性微观计算单元时的潜力。

---

## 🛠️ 核心技术细节

### 1. 提示词工程 (Prompt Engineering)：构建绝对确定性的数学计算器
要让大模型完美承担神经元的数学计算任务，必须克服其“幻觉”和“输出格式不稳定”的弱点。我们采用了**严格的 JSON-in / JSON-out 协议**：

*   **消除 Markdown 干扰**：明确要求大模型不使用 ` ```json ` 等任何 Markdown 代码块包裹，直接输出纯净的 JSON 字符串。这极大降低了后端的解析错误率（避免了 `SyntaxError: Unexpected token` 等问题）。
*   **精准的上下文注入**：系统会动态生成包含神经元内部状态的 JSON 作为 Prompt，包括：
    *   `op`: 当前操作（`forward` 或 `backward`）
    *   `activation`: 激活函数类型（如 `relu`, `sigmoid`）
    *   `weights` & `bias`: 当前神经元的精准浮点数权重与偏置。
    *   `x`: 上游传来的输入向量。
*   **计算目标具象化**：在 Prompt 中明确定义计算公式，例如 $z = w \cdot x + b$ 和 $a = \sigma(z)$，让模型按部就班地返回带有 `"z"` 和 `"a"` 键的 JSON 结果。

### 2. Harness 工程 (Agent 调度与控制)：驾驭不可控的 LLM
在一个网络中调度十几个甚至几十个 Agent 共同协作，必须建立强大的 Harness（安全带/脚手架）机制：

*   **异步状态机**：每个 NeuronAgent 维护着一套严格的状态机（如 `IDLE`, `COMPUTING_FORWARD`, `WAITING_BACKWARD_INPUT` 等）。前端 ECharts 能够根据这些状态实时渲染节点颜色的变化（灰色->黄色->蓝色->绿色）。
*   **并发控制与同步屏障 (Barrier)**：为了应对公有云大模型 API 的并发速率限制（Rate Limits），我们引入了 `FORWARD_ACK` 机制。Coordinator（协调器）在分发任务时，强制等待当前隐藏层 Agent 返回 ACK 后，再触发下一个 Agent，将并发调用优雅地降级为**严格串行**，彻底解决了并发超限问题。
*   **深度思考超时适配**：针对现代推理模型（如具备 Deep Thinking 能力的模型），系统配置了动态可调的超时阈值（如 `request_timeout_seconds: 60.0`），并在底层使用 `asyncio.wait_for` 进行精准拦截，辅以指数退避的重试机制，保证网络的健壮性。
*   **可视化探针 (Probe)**：构建了前端 Agent 探针页面，点击任意神经元即可实时查阅其当前权重、最新发往 LLM 的 JSON Request 以及 LLM 返回的 JSON Response，实现 100% 透明的白盒调试。

### 3. 网络间参数传递 (Parameter Passing)：基于 Actor 模型的异步通信
传统神经网络通过张量（Tensor）在内存中流动，而本项目采用**异步消息传递（Message Passing）**架构：

*   **信箱机制 (Inbox)**：每个神经元被封装为一个独立的 `asyncio.Task`，拥有专属的 `asyncio.Queue` 作为 Inbox。
*   **前向传播组装 (Forward Buffer)**：
    *   **隐藏层**：直接接收 Coordinator 广播的全量特征向量 $x$。
    *   **输出层**：采用 `_forward_buffer` 机制，在收到隐藏层发来的每一个标量 $a$ 时进行缓存；当收齐所有上游神经元的输出后，自动组装成当前层的输入向量 $x$，随后触发自身的 LLM 运算。
*   **非阻塞式结果聚合**：引入 `BackgroundTasks`，使得长时间挂起的推理任务在后台执行。Coordinator 收集到所有输出节点的 $a$ 值后，执行 `softmax/argmax` 计算最高概率分类，并通过 WebSocket (`INFERENCE_RESULT` 事件) 主动将最终结果推送至前端网页，彻底解决传统 HTTP 超时（504 Gateway Timeout）的痛点。

---

## 🚀 总结与展望
本项目成功证明了 **“LLM as a Compute Node”** 的可行性。通过极简的 Prompt 设计、基于 ACK 屏障的并发 Harness 调度以及纯异步的 Actor 通信模型，我们构建出了一个运行稳定、高度可视化的“大模型神经网络”。未来，这种架构可以进一步扩展为由专一领域专家（Expert Agents）组成的宏观推理网络，在复杂逻辑决策任务中发挥更大的威力！