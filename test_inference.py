import asyncio
import json
from pathlib import Path
import numpy as np

from src.core.coordinator import Coordinator
from src.core.message import WebEvent

async def dummy_publish(evt: WebEvent) -> None:
    pass

async def main():
    base_dir = Path("/workspace")
    config_dir = base_dir / "config"
    
    # Initialize coordinator
    coordinator = Coordinator(
        network_config_path=config_dir / "network_config.json",
        global_weights_path=base_dir / "global_weights.json",
        logs_dir=base_dir / "logs",
        publish_web_event=dummy_publish,
    )

    # 1. Load the pretrained weights
    print("[1] Loading pretrained weights...")
    with open(base_dir / "weights.json", "r") as f:
        weights_data = json.load(f)
    await coordinator.load_weights(weights_data)
    print("    Weights loaded successfully into 11 Agents.")

    # 2. Provide test features (Iris sepal_length, sepal_width, petal_length, petal_width)
    x_input = [5.1, 3.5, 1.4, 0.2]
    print(f"[2] Provided test features: {x_input}")

    # 3. Perform distributed inference
    print("[3] Broadcasting input to Multi-Agent Network for inference...")
    try:
        result = await coordinator.inference_single(x_input)
        print("\n=== Inference Result ===")
        print(f"    Predicted Class: {result['prediction']}")
        print(f"    Confidence:      {result['confidence'] * 100:.2f}%")
        print("\nOutput Activations (O0~O2):")
        for i, p in enumerate(result['probabilities']):
            print(f"  Class {i}: {p * 100:>6.2f}%")
    except Exception as e:
        print(f"Error during inference: {e}")
    finally:
        await coordinator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
