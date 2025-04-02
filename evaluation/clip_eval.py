import os
import json
import argparse
from PIL import Image
import torch
import numpy as np
from collections import defaultdict
from transformers import CLIPProcessor, CLIPModel
from torchvision.transforms.functional import to_pil_image


def load_prompts(json_path):
    """Load prompts from the given JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    # TODO: Expects a list, modify the data structure as needed based on the JSON file format.
    return data


def load_image(image_path):
    """Load an image from disk and convert it to RGB."""
    np_im = np.array(Image.open(image_path).convert("RGB"))
    return Image.fromarray(np_im.astype("uint8"))

def compute_clip_similarity_score(image, text_features, clip_model, clip_processor, device):
    """
    Compute the CLIP similarity score between an image and precomputed text_features.
    The score is the cosine similarity between the image features and text features.
    """
    # Ensure the image is a PIL Image.
    pil_image = image if isinstance(image, Image.Image) else to_pil_image(image)
    image_inputs = clip_processor(images=pil_image, return_tensors="pt", padding=True)
    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}

    with torch.no_grad():
        image_features = clip_model.get_image_features(**image_inputs)
    # Normalize image features.
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    # Compute cosine similarity.
    similarity = (image_features * text_features).sum(dim=-1)
    return similarity.item()


def main():
    parser = argparse.ArgumentParser(
        description="Load generated images and prompts, and compute CLIP similarity scores."
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
        default="clip_similarity_scores.json",
        help="Output JSON file to save the CLIP similarity scores."
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

    # Load the CLIP model and processor.
    model_id = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(model_id)
    clip_processor = CLIPProcessor.from_pretrained(model_id)
    clip_model.to(args.device)
    clip_model.eval()

    # Load prompts from the provided JSON.
    prompts = load_prompts(args.data_path)
    results = defaultdict(float)
    for prompt in prompts:
        # Precompute the text features for the prompt.
        text_inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True)
        text_inputs = {k: v.to(args.device) for k, v in text_inputs.items()}
        with torch.no_grad():
            text_features = clip_model.get_text_features(**text_inputs)
        # Normalize text features.
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        for i in range(args.n_views):
            # TODO: Modify this based on the path conversion in the generation script.
            # Convert prompt to a safe filename as in the generation script.
            safe_name = prompt.replace(" ", "_").replace("/", "_").replace(":", "_").replace(",", "_")
            image_path = os.path.join(args.images_dir, safe_name + f'{i}.png')
            if not os.path.exists(image_path):
                print(f"Warning: Image not found for prompt:\n  '{prompt}'\n  Expected at: {image_path}")
                continue

            # Load image and compute CLIP similarity score.
            image = load_image(image_path)
            score = compute_clip_similarity_score(image, text_features, clip_model, clip_processor, args.device)
            results[prompt] += score

        results[prompt] /= args.n_views
        print(f"Prompt: {prompt}\nCLIP Score: {results[prompt]}\n")

    # Save the results to a JSON file.
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"CLIP similarity scores saved to {args.output_path}")

    # Calculate and print the overall average CLIP gsimilarity score.
    if results:
        overall_avg = sum(results.values()) / len(results)
        print(f"Overall Average CLIP Similarity Score: {overall_avg}")
    else:
        print("No CLIP similarity scores computed.")


if __name__ == "__main__":
    main()
