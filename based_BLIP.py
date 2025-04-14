import os
import torch
from transformers import AutoProcessor, BlipForQuestionAnswering
from datasets import load_dataset
from PIL import Image

# -------- 1. Set up BLIP model and processor --------
device = torch.device("mps" if torch.cuda.is_available() else "cpu")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
model.to(device)
model.eval()

# -------- 2. Load the TextVQA dataset from Huggingface --------
# here we load the "validation" split.
dataset = load_dataset("lmms-lab/textvqa", split="validation")
print(f"Loaded {len(dataset)} test samples.")

# -------- 3. Image handling function --------
def load_sample_image(sample):
    """
    Depending on how the dataset stores its image, try to:
      - Use an embedded image (if the dataset feature is of type Image)
      - Or load the image from a file path given in "img_fn".
    """
    if "image" in sample:
        # the dataset may store the image as an Image feature already decoded.
        return sample["image"]
    elif "img_fn" in sample:
        # otherwise, load from the provided file path.
        img_path = sample["img_fn"]
        if os.path.exists(img_path):
            return Image.open(img_path).convert("RGB")
        else:
            raise FileNotFoundError(f"Image file not found: {img_path}")
    else:
        raise KeyError("No image data found in sample.")

# -------- 4. Run inference and compute VQA accuracy --------
# The VQA accuracy metric: for each question:
# score = min(1, (# times predicted answer appears in ground truth answers) / 3)
total_score = 0.0
total_samples = 0

predictions = []

# Iterate over all test samples.
for idx, sample in enumerate(dataset):
    # retrieve question text and ground-truth answers.
    question_text = sample.get("question")
    gt_answers = sample.get("answers")
    
    if gt_answers is None:
        print(f"Sample {idx} has no ground truth answers; skipping.")
        continue

    # normalise ground truth answers: assume they are a list of strings.
    if isinstance(gt_answers, list):
        gt_answers = [ans.strip().lower() for ans in gt_answers if isinstance(ans, str)]
    elif isinstance(gt_answers, str):
        gt_answers = [gt_answers.strip().lower()]
    else:
        print(f"Sample {idx} has an unexpected answer format; skipping.")
        continue

    # load the image for this sample.
    try:
        image = load_sample_image(sample)
    except Exception as e:
        print(f"Error loading image for sample {idx}: {e}; skipping sample.")
        continue

    # process the image and the question.
    inputs = processor(images=image, text=question_text, return_tensors="pt").to(device)

    # disable gradients for speed and memory savings
    with torch.no_grad():
        outputs = model.generate(**inputs)
    # Decode the output (normalise by stripping extra spaces and lowercasing)
    pred_answer = processor.decode(outputs[0], skip_special_tokens=True).strip().lower()

    # Compute VQA score for this question.
    matching_count = sum(1 for ans in gt_answers if ans == pred_answer)
    score = min(1.0, matching_count / 3.0)
    total_score += score
    total_samples += 1

    predictions.append({
        "sample_index": idx,
        "question": question_text,
        "predicted_answer": pred_answer,
        "ground_truth_answers": gt_answers,
        "score": score,
    })

    if (idx + 1) % 50 == 0:
        print(f"Processed {idx+1}/{len(dataset)} samples; Current average accuracy: {total_score/total_samples:.4f}")

# -------- 5. Final VQA Accuracy --------
if total_samples > 0:
    final_accuracy = total_score / total_samples
    print(f"\nFinal VQA Accuracy on the TextVQA Test Set: {final_accuracy:.4f}")
else:
    print("No samples were processed; please check the dataset format.")