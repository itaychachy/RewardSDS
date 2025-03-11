# RewardSDS: Aligning Score Distillation via Reward-Weighted Sampling 
<p>
    🌐 <a href="https://itaychachy.github.io/reward-sds/" target="_blank">Project</a> | 📃 <a href="https://arxiv.org/abs/2503.09601" target="_blank">Paper</a>
</p>

**The code will be published soon...**

<img src="assets/teaser.gif" width="500" alt="Teaser gif">

___

> **Abtstract** \
> Score Distillation Sampling (SDS) has emerged as an effective technique for leveraging 2D diffusion priors for tasks such as text-to-3D generation. While powerful, SDS struggles with achieving fine-grained alignment to user intent. To overcome this, we introduce **RewardSDS**, a novel approach that weights noise samples based on alignment scores from a reward model, producing a weighted SDS loss. This loss prioritizes gradients from noise samples that yield aligned high-reward output. Our approach is broadly applicable and can extend SDS-based methods. In particular, we demonstrate its applicability to Variational Score Distillation (VSD) by introducing RewardVSD. We evaluate RewardSDS and RewardVSD on text-to-image, 2D editing, and text-to-3D generation tasks, showing significant improvements over SDS and VSD on a diverse set of metrics measuring generation quality and alignment to desired reward models, enabling state-of-the-art performance.
___


### Citation
If you find RewardSDS helpful, please consider citing:
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