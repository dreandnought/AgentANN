import json
import numpy as np

with open("weights.json", "r") as f:
    w = json.load(f)

x = np.array([5.1, 3.5, 1.4, 0.2], dtype=np.float32)

h_out = []
for i in range(8):
    wi = np.array(w[f"H{i}"]["weights"])
    bi = w[f"H{i}"]["bias"]
    z = np.dot(wi, x) + bi
    a = max(0.0, float(z))
    h_out.append(a)

print("x =", json.dumps(h_out))
print("O0 weights =", json.dumps(w["O0"]["weights"]))
print("O0 bias =", w["O0"]["bias"])
