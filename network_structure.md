# 网络拓扑与配置约定 (Network Structure)

## 为什么不需要 4 个 Agent 维护输入层？
在标准的前馈神经网络中，输入层本身不包含任何可学习的参数（无权重，无偏置）。如果为输入层分配 4 个 Agent，它们将退化为纯粹的“信使”，导致不必要的通信开销和调度复杂性。

因此，**参数的维护责任属于接收方**。输入层由 Coordinator 直接代理打包发送，总共只需要 **11 个 Agent** (8隐藏层 + 3输出层) 协同即可。

## 全局配置文件结构

系统启动前，所有 Agent 和协调器都会读取 `network_config.json` 文件，以此明确自身所属层级、输入维度、以及消息的上下游路由目标。

配置结构如下：

```json
{
  "network": {
    "input_dim": 4,
    "hidden_layer": {
      "layer_name": "Hidden",
      "agents": ["H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7"],
      "activation": "relu",
      "input_source": "coordinator",
      "output_targets": ["O0", "O1", "O2"]
    },
    "output_layer": {
      "layer_name": "Output",
      "agents": ["O0", "O1", "O2"],
      "activation": "sigmoid",
      "input_source": ["H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7"],
      "output_targets": "coordinator"
    }
  },
  "hyperparameters": {
    "learning_rate": 0.01
  }
}
```

## 预训练权重文件结构（weights.json）

你提供的外部预训练权重文件用于覆盖各 Agent 的 `weights` 与 `bias`。文件结构为一个以 `agent_id` 为 key 的 JSON 对象：

```json
{
  "H0": { "weights": [0.0, 0.0, 0.0, 0.0], "bias": 0.0 },
  "H1": { "weights": [0.0, 0.0, 0.0, 0.0], "bias": 0.0 },
  "H2": { "weights": [0.0, 0.0, 0.0, 0.0], "bias": 0.0 },
  "H3": { "weights": [0.0, 0.0, 0.0, 0.0], "bias": 0.0 },
  "H4": { "weights": [0.0, 0.0, 0.0, 0.0], "bias": 0.0 },
  "H5": { "weights": [0.0, 0.0, 0.0, 0.0], "bias": 0.0 },
  "H6": { "weights": [0.0, 0.0, 0.0, 0.0], "bias": 0.0 },
  "H7": { "weights": [0.0, 0.0, 0.0, 0.0], "bias": 0.0 },

  "O0": { "weights": [0.0, 0.0, "... 共8个 ..."], "bias": 0.0 },
  "O1": { "weights": [0.0, 0.0, "... 共8个 ..."], "bias": 0.0 },
  "O2": { "weights": [0.0, 0.0, "... 共8个 ..."], "bias": 0.0 }
}
```

约定：
- Hidden 层（H0~H7）：`weights` 长度必须为 4。
- Output 层（O0~O2）：`weights` 长度必须为 8（对应 H0~H7 的 8 个输入激活）。
- `bias` 为标量（number）。
