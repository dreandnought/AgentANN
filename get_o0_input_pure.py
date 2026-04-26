import json

with open("weights.json", "r") as f:
    w = json.load(f)

x = [5.1, 3.5, 1.4, 0.2]

h_out = []
for i in range(8):
    wi = w[f"H{i}"]["weights"]
    bi = w[f"H{i}"]["bias"]
    z = sum(a*b for a, b in zip(wi, x)) + bi
    a = max(0.0, z)
    h_out.append(a)

print("O0 input x:", json.dumps(h_out))
print("O0 weights:", json.dumps(w["O0"]["weights"]))
print("O0 bias:", w["O0"]["bias"])
