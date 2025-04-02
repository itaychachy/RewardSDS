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

- `"none"` – standard SDS training without reward guidance.
- `"min-max"` – moves towards the best and away from the worst candidates (reward based).
- `"best"` – moves only towards the best candidate(s) (reward based).
- `"weighted"` - uses a softmax-weighted combination of all candidates (reward based).

> We found "min-max" to perform best in practice, but feel free to experiment or add your own strategies.
To keep things simple, strategy-specific hyperparameters (e.g., number of candidates or weights) are fixed and can be modified in [`reward_strategy.py`](reward_strategy.py).

---

## 🔹 `reward_model`

Specifies **which model** is used to compute the reward score from rendered images.

**Supported values:**

- `"image-reward"` – uses [ImageReward](https://github.com/THUDM/ImageReward) to score image-text alignment.
- `"aesthetic"` – uses [Aesthetic Score Predictor](https://github.com/LAION-AI/aesthetic-predictor) to score image quality.
- `"none"` - no reward model is used, should be used with `reward_strategy: "none"`.

> You can add custom reward models by extending the [`RewardModel`](reward_model.py) class.

---

## 🔹 `n_noises`

Number of noise samples to consider per training step. These samples are used to produce multiple candidate outputs, from which the reward signal is computed and used to guide optimization.

> A higher value can lead to more stable reward estimation but increases computation cost.

---
