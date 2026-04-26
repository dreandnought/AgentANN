import torch
import json
import numpy as np

# Load weights
with open("weights.json", "r") as f:
    w = json.load(f)

x = np.array([5.1, 3.5, 1.4, 0.2], dtype=np.float32)

# H0-H7
h_out = []
for i in range(8):
    wi = np.array(w[f"H{i}"]["weights"])
    bi = w[f"H{i}"]["bias"]
    z = np.dot(wi, x) + bi
    a = max(0.0, z)
    h_out.append(a)

h_out = np.array(h_out, dtype=np.float32)

# O0-O2
o_out = []
for i in range(3):
    wi = np.array(w[f"O{i}"]["weights"])
    bi = w[f"O{i}"]["bias"]
    z = np.dot(wi, h_out) + bi
    a = 1.0 / (1.0 + np.exp(-z))
    o_out.append(a)

print("Manual inference:", o_out)
