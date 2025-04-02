import os
import json
import argparse
from PIL import Image
import numpy as np
import torch
from collections import defaultdict
from transformers import CLIPProcessor, CLIPModel
from torchvision.transforms.functional import to_pil_image
from os.path import expanduser
from urllib.request import urlretrieve

def get_aesthetic_model(clip_model="vit_l_14"):
    """Load the aesthetic model."""
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

def load_prompts(json_path):
    """Load prompts from the given JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    # TODO: Expects a list, modify the data structure as needed based on the JSON file format.
    return data

def load_image(image_path):
    """Load an image from disk and convert it to RGB."""
    np_im = np.array(Image.open(image_path).convert("RGB"))
    return Image.fromarray(np_im.astype(np.uint8))

def compute_aesthetic_score(image, clip_model, clip_processor, aesthetic_model, device):
    """
    Compute the Aesthetic Score of an image.
    This function processes the image with the CLIPProcessor and extracts image features using the CLIPModel.
    The aesthetic_model then predicts a score based on these features.
    Note: The text prompt is ignored in aesthetic scoring.
    """
    # Ensure the image is a PIL Image
    pil_image = image if isinstance(image, Image.Image) else to_pil_image(image)
    inputs = clip_processor(images=pil_image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    with torch.no_grad():
        score = aesthetic_model(image_features)
    return score.item()

def main():
    parser = argparse.ArgumentParser(
        description="Load generated images and prompts, and compute Aesthetic Scores."
    )
    # TODO: Modify the default values or set the required arguments based on your setup.
    parser.add_argument(
        "--data_path",
        type=str,
        default="data.json",
        help="Path to JSON file containing prompts."
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="images/",
        help="Directory containing generated images."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="aesthetic_scores.json",
        help="Output JSON file to save the Aesthetic scores."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on."
    )
    parser.add_argument(
        "--n_views",
        type=int,
        default=1,
        help="Number of views to evaluate for each prompt. Enables 3D evaluation, default value is 1 for 2D evaluation."
    )
    args = parser.parse_args()

    # Load the CLIP model and processor for feature extraction.
    model_id = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(model_id)
    clip_processor = CLIPProcessor.from_pretrained(model_id)
    clip_model.to(args.device)
    clip_model.eval()

    # Load the aesthetic model.
    aesthetic_model = get_aesthetic_model("vit_l_14")
    aesthetic_model.to(args.device)

    # Load prompts from the provided JSON.
    prompts = load_prompts(args.data_path)
    results = defaultdict(float)
    for prompt in prompts:
        for i in range(args.n_views):
            # TODO: Modify this based on the path conversion in the generation script.
            # Convert prompt to a safe filename as in the generation script.
            safe_name = prompt.replace(" ", "_").replace("/", "_").replace(":", "_").replace(",", "_")
            image_path = os.path.join(args.images_dir, safe_name + f'{i}.png')
            if not os.path.exists(image_path):
                print(f"Warning: Image not found for prompt:\n  '{prompt}'\n  Expected at: {image_path}")
                continue

            # Load image and compute Aesthetic score.
            image = load_image(image_path)
            score = compute_aesthetic_score(image, clip_model, clip_processor, aesthetic_model, args.device)
            results[prompt] += score
        results[prompt] /= args.n_views
        print(f"Prompt: {prompt}\nAesthetic Score: {results[prompt]}\n")

    # Save the results to a JSON file.
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Aesthetic scores saved to {args.output_path}")

    # Calculate and print the average Aesthetic score over all computed scores.
    if results:
        avg_score = sum(results.values()) / len(results)
        print(f"Average Aesthetic Score: {avg_score}")
    else:
        print("No Aesthetic scores computed.")

if __name__ == "__main__":
    main()
