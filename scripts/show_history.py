import json
import sys

h = json.load(open("models/runs/gpt-cpu-tiny/history.json", encoding="utf-8"))
print("eval points:", len(h))
print("step  train   val     lr")
for r in h[::4] + [h[-1]]:
    print(f"{r['step']:>4} {r['train_loss']:.4f} {r['val_loss']:.4f} {r['lr']:.2e}")
best = min(h, key=lambda r: r["val_loss"])
print(f"best val: {best['val_loss']:.4f} at step {best['step']}")
print(f"elapsed: {h[-1]['elapsed_s'] / 60:.1f} min")
