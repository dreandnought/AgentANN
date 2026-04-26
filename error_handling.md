# 多智能体BP网络容错与异常检测机制

在分布式多智能体系统中，任何一个节点的崩溃或消息丢失都会导致整个网络陷入死锁。为此，本系统采用以下 4 重机制来保证训练的鲁棒性：

## 1. 异步等待超时 (Timeout Mechanism)
所有接收消息的操作必须使用 `asyncio.wait_for` 设定超时时间（默认 5.0 秒）。
- **Agent 层面**：例如输出层 Agent 在等待隐藏层 10 个激活值时，若超时，将抛弃已收集的部分数据，向 Coordinator 上报 `TIMEOUT_ERROR`，并清空当前状态，准备接收下一个样本。
- **Coordinator 层面**：若在规定时间内未收齐输出层的预测结果或参数更新确认，Coordinator 将中断当前 Batch，下发 `RESET` 指令给所有 Agent，跳过当前样本。

## 2. 异常捕获与广播 (Error Propagation)
所有 Agent 的计算逻辑（前向加权和、反向梯度计算）必须被 `try...except Exception` 包裹。
- 一旦发生异常（如数值溢出 NaN、维度不匹配等），Agent 必须：
  1. 记录详细的 traceback 到自身的独立日志。
  2. 构造一条 `msg_type="ERROR"` 的消息，发送给 Coordinator 以及所有正在等待它的下游 Agent。
- 下游 Agent 收到 `ERROR` 消息后，立刻终止当前的等待状态，清空缓存，防止被拖入死锁。

## 2.5 LLM 调用超时与重试约束 (LLM Timeout & Retry)
当启用换脑（LLM 驱动）时，Agent 的前向/反向计算会通过 LLM 客户端执行，系统将对 LLM 调用施加强约束：
- **请求超时**：每次 LLM 调用都有独立超时（`config/llm_config.json` 的 `request_timeout_seconds`）。
- **最大重试次数**：失败后按指数退避重试（`retry_strategy.max_retries` / `retry_strategy.backoff_seconds`）。
- **失败上报**：超过最大重试次数仍失败则 Agent 向 Coordinator 发送 `ERROR`，并写入死信日志以便审计。

## 3. 健康检查与热重启 (Health Check & Restart)
Coordinator 充当系统的“守护进程”。
- 每个 epoch 开始前，Coordinator 会广播 `PING` 消息。
- 未能在 1 秒内回复 `PONG` 的 Agent 会被判定为“已死亡”。
- Coordinator 会取消（cancel）死亡 Agent 的旧协程任务，重新实例化该 Agent，并从 `global_weights.json` 中读取其崩溃前的权重进行状态恢复（热启动）。

## 4. 死信记录 (Dead-letter Logging)
- 专门维护一个 `logs/dead_letter.log`。
- 所有引发异常的原始输入数据、消息体以及完整的报错堆栈，都将被写入此日志，方便开发者在系统自动恢复后进行事后审计和 Bug 修复。
