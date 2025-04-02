# RewardSDS: Aligning Score Distillation via Reward-Weighted Sampling 
<p>
    🌐 <a href="https://itaychachy.github.io/reward-sds/" target="_blank">Project</a> | 📃 <a href="https://arxiv.org/abs/2503.09601" target="_blank">Paper</a>
</p>

<img src="assets/teaser.gif" width="500" alt="Teaser gif">

___

**TL;DR:**  
Introducing **RewardSDS**, a novel approach that weights noise samples based on alignment scores from a reward model, producing a weighted SDS loss that prioritizes gradients from noise samples that yield aligned high-reward output.
___

# Experimenting with RewardSDS

### Prerequisites
 This project has been tested with `Python 3.8`, `CUDA 11.8`, and `GPU L40S`.

## 3D Experiments
We provide our code for text-based NeRF optimization (with MVDream as the 2D prior) as an extension for Threestudio. To use it, please install [threestudio](https://github.com/threestudio-project/threestudio) first and then follow the following steps:

### Extension Installation
```bash
# Update path according to your threestudio installation
cp -r threestudio-reward-sds ../threestudio/custom/
cd ../threestudio/custom/threestudio-reward-sds

# First install xformers (https://github.com/facebookresearch/xformers#installing-xformers)
# cuda 11.8 version
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118
# cuda 12.1 version
# pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121

# Then install other dependencies
pip install -r requirements.txt
```

### Run 3D optimization
In the `threestudio` repo:
```bash
python launch.py --config custom/threestudio-reward-sds/configs/reward-mvdream-sd21.yaml --train --gpu 0 system.prompt_processor.prompt="A penguin with a brown bag in the snow"
```
Checkout the [README](threestudio-reward-sds/README.md) for configuration details. In addition, other optimization details (shading, resume from checkpoints, etc.) can be found in the [MVDream](https://github.com/DSaurus/threestudio-mvdream/tree/main) repo.

## 2D Experiments
We offer a simpler installation than Threestudio with minimal dependencies if you want to run experiments in 2D.

### Installation
```bash
# Create a new conda environment
conda create --name reward-sds -y python=3.8
conda activate reward-sds
pip install --upgrade pip

# Install dependencies
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
cd 2D_experiments
pip install -r requirements.txt
```

### Run 2D optimization
In the `2D_experiments` directory:
```bash
python generate.py --prompt "A white car and a red sheep"
```

See [`generate.py`](2D_experiments/generate.py) for more options, including but not limited to:
* `--reward_strategy`, `reward_model`, `n_noises` - reward related fields, details in [README](README.md).
* `--prompt` - text prompt for the generated image.
* `--mode` - choose between SDS-like loss functions [SDS](https://dreamfusion3d.github.io),  [VSD](https://ml.cs.tsinghua.edu.cn/prolificdreamer/), [sds-bridge](https://sds-bridge.github.io/).

## Evaluation
You may find the evaluation scripts useful for reproducing the results in the paper. They are available in the [`evaluation`](evaluation) directory (follow the TODOs in each script).

## Acknowledgements
This project is based on the following repositories:
* [Threestudio](https://github.com/threestudio-project/threestudio)
* [MVDream](https://github.com/DSaurus/threestudio-mvdream)
* [SDS-Bridge](https://github.com/davidmcall/SDS-Bridge?tab=readme-ov-file)
* [ImageReward](https://github.com/THUDM/ImageReward)

### Citation
Found RewardSDS useful? Please consider citing our work:
```bibtex
@misc{chachy2025rewardsdsaligningscoredistillation,
      title={RewardSDS: Aligning Score Distillation via Reward-Weighted Sampling}, 
      author={Itay Chachy and Guy Yariv and Sagie Benaim},
      year={2025},
      eprint={2503.09601},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.09601}, 
}
```