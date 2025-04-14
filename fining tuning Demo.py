import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
from transformers import AutoProcessor, BlipForQuestionAnswering
from datasets import load_dataset
import pytesseract

###############################################
# Function to extract text from an image using OCR
###############################################
def extract_text_from_image(image):
    """
    Uses pytesseract to extract text from a PIL Image.
    Returns the recognized text (as a string).
    """
    ocr_text = pytesseract.image_to_string(image)
    return ocr_text.strip()

###############################################
# 1. Define the Fine-Tuning Model Wrapper
###############################################
class BLIPVQAFineTuner(nn.Module):
    """
    This module wraps the BLIP VQA model and injects a small MLP on top of the vision encoder.
    Its forward() method is used during training.
    Its generate() method performs greedy decoding using the modified vision features.
    The MLP output is summed with the original vision features,
    and the combined features are passed to the text decoder.
    """
    def __init__(self, base_model, mlp_hidden_dim=256):
        super().__init__()
        self.base_model = base_model
        try:
            vision_hidden_size = self.base_model.config.vision_hidden_size
        except AttributeError:
            vision_hidden_size = 768
        self.mlp = nn.Sequential(
            nn.Linear(vision_hidden_size, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, vision_hidden_size)
        )
    
    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        # Get image features from the vision encoder.
        vision_outputs = self.base_model.vision_model(pixel_values=pixel_values)
        original_feats = vision_outputs[0]  # [B, vis_seq_len, hidden_size]
        new_feats = self.mlp(original_feats)
        combined_feats = original_feats + new_feats
        # Pass combined features to the text decoder for training.
        text_outputs = self.base_model.text_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=combined_feats,
            labels=labels,
            return_dict=True
        )
        return text_outputs

    def generate(self, pixel_values, input_ids, attention_mask, max_length=15):
        """
        Greedy decoding for batch_size=1.
        At each step, we reinitialize the attention mask as all ones for the current generated sequence.
        We also stop if the sequence length reaches the model’s maximum position embeddings.
        """
        self.eval()
        with torch.no_grad():
            vision_outputs = self.base_model.vision_model(pixel_values=pixel_values)
            original_feats = vision_outputs[0]
            new_feats = self.mlp(original_feats)
            combined_feats = original_feats + new_feats

            generated = input_ids.clone()  # starting sequence (shape [1, seq_len])
            for _ in range(max_length):

                # Reinitialize attention mask as ones (matching the current sequence length).
                cur_mask = torch.ones_like(generated, dtype=attention_mask.dtype, device=generated.device)
                outputs = self.base_model.text_decoder(
                    input_ids=generated,
                    attention_mask=cur_mask,
                    encoder_hidden_states=combined_feats,
                    return_dict=True
                )
                logits = outputs.logits  # shape [1, current_seq_len, vocab_size]
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # shape [1, 1]
                generated = torch.cat([generated, next_token], dim=-1)
                if next_token.item() == processor.tokenizer.eos_token_id:
                    break
            return generated

###############################################
# 2. Load Base Model, Processor, and Set Up the Fine-Tuner
###############################################
# Load the pretrained BLIP VQA model and processor.
base_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
# Here, we override the default text max length to a smaller value (e.g., 128) so that our input is shorter.
processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base", truncation=True, max_length=128)
base_model.processor = processor

# Wrap the base model.
custom_model = BLIPVQAFineTuner(base_model, mlp_hidden_dim=256)

# Optionally, lower the image resolution to speed processing.
processor.image_processor.size = {"height": 224, "width": 224}

# Freeze all parameters of the base_model except for the additional MLP and the text_decoder.
# (Unfreezing text_decoder helps the model learn to generate correct answers.)
for name, param in base_model.named_parameters():
    if "text_decoder" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
# Ensure our added MLP is trainable.
for param in custom_model.mlp.parameters():
    param.requires_grad = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
custom_model.to(device)

###############################################
# 3. Data Preparation: Collate Functions & DataLoaders
###############################################
# For training: we tokenize questions using a max_length smaller than the model’s max_position_embeddings.
def collate_fn_train(samples):
    images, prompts, answers = [], [], []
    for s in samples:
        # Use the "image" field.
        if "image" in s and s["image"] is not None:
            img = s["image"]
        else:
            continue
        if "question" not in s or "answers" not in s or not s["answers"]:
            continue
        # Extract OCR text from the image.
        ocr_text = extract_text_from_image(img)
        # Create a prompt that concatenates the question and OCR output.
        # For example:
        #   "What is the brand of the camera?\nOCR: Dakota Digital\nAnswer:"
        prompt = s["question"].strip() + "\nOCR: " + ocr_text + "\nAnswer:"
        images.append(img)
        prompts.append(prompt)
        answers.append(s["answers"][0])
    if len(images) == 0 or not (len(images) == len(prompts) == len(answers)):
        return None
    model_inputs = processor(
        images=images,
        text=prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True
    )
    labels = processor(
        text=answers,
        return_tensors="pt",
        padding="max_length",
        truncation=True
    ).input_ids
    model_inputs["labels"] = labels
    return model_inputs

def collate_fn_eval(samples):
    images, prompts, answers = [], [], []
    for s in samples:
        if "image" in s and s["image"] is not None:
            img = s["image"]
        else:
            continue
        if "question" not in s or "answers" not in s or not s["answers"]:
            continue
        ocr_text = extract_text_from_image(img)
        prompt = s["question"].strip() + "\nOCR: " + ocr_text + "\nAnswer:"
        images.append(img)
        prompts.append(prompt)
        answers.append(s["answers"])  # Keep full list for evaluation.
    if len(images) == 0 or not (len(images) == len(prompts) == len(answers)):
        return None
    model_inputs = processor(
        images=images,
        text=prompts,
        return_tensors="pt",
        padding="max_length",
        truncation=True
    )
    return model_inputs, answers
# Load dataset (using a small subset for demonstration).
dataset = load_dataset("lmms-lab/textvqa")
train_dataset = dataset["train"].select(range(200))   # 20 training samples
val_dataset = dataset["validation"].select(range(20))  # 20 validation samples

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn_train)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn_eval)

###############################################
# 4. Minimal Training Loop
###############################################
optimizer = optim.AdamW(custom_model.mlp.parameters(), lr=1e-4)  # only fine-tuning the MLP
num_epochs = 2  # One epoch for demonstration

custom_model.train()
print("Starting training...\n")
for epoch in range(num_epochs):
    for step, batch in enumerate(train_loader):
        if batch is None or batch["pixel_values"].size(0) == 0:
            continue
        for key in batch:
            batch[key] = batch[key].to(device)
        optimizer.zero_grad()
        outputs = custom_model(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        loss = outputs.loss
        if loss is None:
            continue
        loss.backward()
        optimizer.step()
        if (step + 1) % 5 == 0:
            print(f"Epoch {epoch+1}, Step {step+1}, Loss: {loss.item():.4f}")
print("\nTraining complete!\n")

###############################################
# 5. Evaluation on Validation Set (VQA Accuracy)
###############################################
def vqa_accuracy(pred_answer, gt_answers):
    """
    Compute standard VQA accuracy:
       score = min(1, (# of matching ground-truth answers)/3)
    gt_answers is a list of answer strings.
    """
    pred_answer = pred_answer.strip().lower()
    match_count = sum(1 for ans in gt_answers if ans.strip().lower() == pred_answer)
    return min(1.0, match_count / 3.0)

custom_model.eval()
total_score = 0.0
total_samples = 0

print("Starting evaluation...\n")
for item in val_loader:
    if item is None:
        continue
    model_inputs, gt_answers = item
    for key in model_inputs:
        model_inputs[key] = model_inputs[key].to(device)
    with torch.no_grad():
        generated_ids = custom_model.generate(
            pixel_values=model_inputs["pixel_values"],
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            max_length=15
        )
    pred_text = processor.decode(generated_ids[0], skip_special_tokens=True)
    # Use the list of ground-truth answers (for this sample, e.g. 10 answers)
    sample_score = vqa_accuracy(pred_text, gt_answers[0])
    total_score += sample_score
    total_samples += 1
    print(f"Predicted: {pred_text}\nGround Truths: {gt_answers[0]}\nScore: {sample_score:.3f}\n")

if total_samples > 0:
    final_acc = total_score / total_samples
    print(f"Validation VQA Accuracy: {final_acc:.4f}")
else:
    print("No validation samples processed.")
