# Reward Configuration

This project supports reward-weighted SDS loss to guide 2D generation using external reward models. The following configuration arguments can be used to control this behavior:

```yaml
reward_strategy: "min-max"
reward_model: "image-reward"
n_noises: 7
```

## 🔹 `reward_strategy`

Controls **how** the reward signal is used to guide training.

**Supported values:**

- `"none"` – Standard SDS training without reward guidance.
- `"min-max"` – Moves towards the best and away from the worst candidate (based on reward).
- `"best"` – Moves only towards the best candidate(s).
- `"weighted"` - Uses a softmax-weighted combination of all candidates based on their reward.

> We found "min-max" to perform best in practice, but feel free to experiment or add your own strategies.
To keep things simple, strategy-specific hyperparameters (e.g., number of candidates or weights) are fixed and can be modified in [`reward_strategy.py`](reward_strategy.py).

---

## 🔹 `reward_model`

Specifies **which model** is used to compute the reward score from rendered images.

**Supported values:**

- `"image-reward"` – Uses ImageReward to score image-text alignment.
- `"aesthetic"` – Uses an Aesthetic Score Predictor to rate image quality.
- `"none"` - No reward model is used, should be used with `reward_strategy: "none"`.

> You can add custom reward models by extending the [`RewardModel`](reward_model.py) class.

---

## 🔹 `n_noises`

Number of noise samples generated per training step.
These samples are used to produce multiple candidate outputs, from which the reward signal is computed and used to guide optimization.

> A higher value can lead to more stable reward estimation but increases computation cost.

---
