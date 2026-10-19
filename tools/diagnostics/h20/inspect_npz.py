import json, numpy as np, sys
p = sys.argv[1]
z = np.load(p)
print("npz keys:", list(z.keys()))
for k in z.keys():
    a = z[k]
    print(k, getattr(a, "shape", None), getattr(a, "dtype", type(a)))
mp = p.replace("arrays.npz", "metrics.json")
m = json.load(open(mp))
print("metrics keys:", list(m.keys()))
