import os
import json
import argparse
from PIL import Image
import torch
import numpy as np
from collections import defaultdict
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

def compute_image_reward_score(ir_model, image, prompt):
    """
    Compute the ImageReward score between an image and text prompt.
    """
    # Ensure the image is a PIL Image.
    pil_image = image if isinstance(image, Image.Image) else to_pil_image(image)
    return ir_model.score(prompt, pil_image)


def main():
    parser = argparse.ArgumentParser(
        description="Load generated images and prompts, and compute ImageReward scores."
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
        default="image_reward_scores.json",
        help="Output JSON file to save the ImageReward scores."
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

    # Load the ImageReward model.
    import ImageReward as IR
    ir_model = IR.load("ImageReward-v1.0")
    ir_model.to(args.device)
    ir_model.eval()

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

            # Load image and compute ImageReward score.
            image = load_image(image_path)
            score = compute_image_reward_score(ir_model, image, prompt)
            results[prompt] += score

        results[prompt] /= args.n_views
        print(f"Prompt: {prompt}\nImageReward Score: {results[prompt]}\n")

    # Save the results to a JSON file.
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"nImageReward scores saved to {args.output_path}")

    # Calculate and print the overall average ImageReward score.
    if results:
        overall_avg = sum(results.values()) / len(results)
        print(f"Overall Average ImageReward Score: {overall_avg}")
    else:
        print("No ImageReward scores computed.")


if __name__ == "__main__":
    main()
