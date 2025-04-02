# Reward Configuration

This project supports reward-weighted SDS loss to guide 3D generation using external reward models. The following configuration fields can be added to your YAML config file to control this behavior:

```yaml
reward_strategy: "min-max"
reward_model: "image-reward"
n_noises: 7
reward_duration: 1.0
```

## 🔹 `reward_strategy`

Controls **how** the reward signal is used to guide training.

**Supported values:**

- `"none"` – Standard SDS training without reward guidance.
- `"min-max"` – Moves towards the best and away from the worst candidate (based on reward).
- `"best"` – Moves only towards the best candidate(s).
- `"weighted"` - Uses a softmax-weighted combination of all candidates based on their reward.

> We found "min-max" to perform best in practice, but feel free to experiment or add your own strategies.
To keep things simple, strategy-specific hyperparameters (e.g., number of candidates or weights) are fixed and can be modified in [`reward_strategy.py`](guidance/reward_strategy.py).

---

## 🔹 `reward_model`

Specifies **which model** is used to compute the reward score from rendered images.

**Supported values:**

- `"image-reward"` – Uses ImageReward to score image-text alignment.
- `"aesthetic"` – Uses an Aesthetic Score Predictor to rate image quality.
- `"none"` - No reward model is used, should be used with `reward_strategy: "none"`.

> You can add custom reward models by extending the [`RewardModel`](guidance/reward_model.py) class.

---

## 🔹 `n_noises`

Number of noise samples generated per training step.
These samples are used to produce multiple candidate outputs, from which the reward signal is computed and used to guide optimization.

> A higher value can lead to more stable reward estimation but increases computation cost.

---

## 🔹 `reward_duration`

A float value between `0.0` and `1.0` that determines **how long** the reward model is used during training (as a portion of total steps).

- `0.0` → Reward is never used (equivalent to `reward_strategy: "none"`).
- `1.0` → Reward is used throughout the entire training process.

> Example: `reward_duration: 0.5` means reward-guided training is used for the first half of training, and standard SDS is used for the second half.

---
