CSTR graph encoder for PPO reproduction.

The checkpoint is the epoch-3 frozen encoder used by the CSTR graph PPO
results. It is kept locally so graph PPO reproduction scripts use a stable
reference encoder checkpoint.

Files:
- `best_dynamics_encoder.pt`: frozen graph encoder checkpoint.
- `metrics.jsonl`: encoder pretraining metrics.
