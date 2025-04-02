from abc import ABC, abstractmethod
import torch.nn.functional as F
from os.path import expanduser
import os
from urllib.request import urlretrieve
import torch
from torchvision.transforms.functional import to_pil_image
from PIL import Image


class RewardModel(ABC):
    @abstractmethod
    def compute_scores(self, images, prompt):
        pass


class ImageReward(RewardModel):
    def __init__(self, device=torch.device("cuda"), model="ImageReward-v1.0"):
        import TensorImageReward as ImageReward
        self.reward = ImageReward.load(model)
        self.device = device
        self.reward.requires_grad_(False)

    def compute_scores(self, images, prompt):
        images = F.interpolate(images, (224, 224), mode='bilinear', align_corners=False).to(torch.float32)
        return self.reward.inference_rank(prompt, images)[1]


class AestheticReward(RewardModel):
    def __init__(
        self,
        device=torch.device("cuda"),
        clip_model="openai/clip-vit-large-patch14",
        aesthetic_model="vit_l_14",
    ):
        from transformers import CLIPProcessor, CLIPModel
        self.device = device
        self.clip_model = CLIPModel.from_pretrained(clip_model)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model)
        self.clip_model.to(device)
        self.clip_model.eval()

        # Load the aesthetic model.
        self.aesthetic_model = get_aesthetic_model(aesthetic_model)
        self.aesthetic_model.to(device)

    def compute_scores(self, images, prompt):
        # Convert each tensor image to a PIL image.
        pil_images = [img if isinstance(img, Image.Image) else to_pil_image(img.cpu()) for img in images]

        # Process the list of PIL images.
        inputs = self.clip_processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract image features using the CLIP model.
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)

        # Normalize the image features.
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Predict aesthetic scores.
        with torch.no_grad():
            scores = self.aesthetic_model(image_features)

        # Ensure the output is a 1D tensor (one score per image).
        # If aesthetic_model returns a shape like (n, 1), you might want to squeeze the last dimension.
        if scores.ndim > 1 and scores.size(1) == 1:
            scores = scores.squeeze(1)

        return scores


def get_aesthetic_model(clip_model="vit_l_14"):
    home = expanduser("~")
    cache_folder = os.path.join(home, ".cache", "emb_reader")
    path_to_model = os.path.join(cache_folder, "sa_0_4_" + clip_model + "_linear.pth")
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"
            + clip_model
            + "_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)
    import torch.nn as nn
    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError("Unsupported clip_model: " + clip_model)
    s = torch.load(path_to_model, map_location="cpu")
    m.load_state_dict(s)
    m.eval()
    return m
