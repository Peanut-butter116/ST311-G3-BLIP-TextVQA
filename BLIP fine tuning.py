import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image

from transformers import AutoProcessor, BlipForQuestionAnswering
from datasets import load_dataset

# -------- 1. Define Custom Fine-Tuning Model Wrapper ---------

class BLIPTextVQAFineTuner(nn.Module):
    """
    This custom module wraps the BLIP VQA model (Salesforce/blip-vqa-base)
    and injects additional MLP layers after the vision encoder.
    
    It uses BLIP's available parameters (e.g. vision_hidden_size) as documented at:
    https://huggingface.co/docs/transformers/en/model_doc/blip

    The enhanced image features are then passed to the text decoder.
    """
    def __init__(self, blip_vqa_model, mlp_hidden_dim=512):
        super().__init__()
        self.blip_vqa = blip_vqa_model

        # Retrieve the vision hidden size from BLIP's config (as documented)
        try:
            vision_hidden_size = self.blip_vqa.config.vision_hidden_size
        except AttributeError:
            vision_hidden_size = 768  # fallback if not available

        # Define a simple MLP block: vision_hidden_size -> mlp_hidden_dim -> vision_hidden_size
        self.enhance_mlp = nn.Sequential(
            nn.Linear(vision_hidden_size, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, vision_hidden_size)
        )
    
    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, labels=None):
        """
        pixel_values: A tensor of shape (batch, 3, H, W) from the BLIP processor.
        input_ids, attention_mask: Tokenized question input.
        labels: Tokenized target answers (for computing loss during training).
        """
        # 1) Run the vision encoder to obtain image features.
        vision_outputs = self.blip_vqa.vision_model(pixel_values=pixel_values)
        # vision_outputs[0] has shape: [batch, num_patches, vision_hidden_size]
        vision_embeds = vision_outputs[0]

        # 2) Pass the features through our extra MLP layers.
        enhanced_embeds = self.enhance_mlp(vision_embeds)

        # 3) Feed the enhanced embeddings into the text decoder.
        text_outputs = self.blip_vqa.text_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=enhanced_embeds,
            labels=labels,
            return_dict=True
        )
        # text_outputs.logits: [batch, seq_len, vocab_size]
        # text_outputs.loss: Cross-entropy loss if labels are provided.
        logits = text_outputs.logits
        loss = text_outputs.loss

        return {"loss": loss, "logits": logits}


# -------- 2. Instantiate the Base BLIP Model & Processor --------

# Load the pretrained BLIP VQA model and its processor.
base_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")
# For convenience, we attach the processor to the base model.
base_model.processor = processor

# Wrap the base model in our custom fine-tuning module.
custom_model = BLIPTextVQAFineTuner(base_model, mlp_hidden_dim=512)
device = torch.device("mps" if torch.cuda.is_available() else "cpu")
custom_model.to(device)


# -------- 3. Load the TextVQA Dataset from Hugging Face --------

# The textvqa dataset provides train/validation splits.
dataset = load_dataset("lmms-lab/textvqa")
train_dataset = dataset["train"]
val_dataset   = dataset["validation"]

# Use a subset of the dataset for demonstration purposes
train_dataset = train_dataset.select(range(2000))
val_dataset = val_dataset.select(range(500))

print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")


# -------- 4. Prepare a Data Collation Function --------

def collate_fn(samples):
    images = []
    questions = []
    answers = []

    for s in samples:
        img = None  # Initialize to None at the start of each loop iteration.

        # 1) Load the image either from an 'image' field
        if "image" in s and isinstance(s["image"], Image.Image):
            img = s["image"]
        elif "img_fn" in s:
            img_path = s["img_fn"]
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Warning: Could not load {img_path}: {e}")
                # We can either skip or assign a placeholder image
                continue
        else:
            # If neither 'image' nor 'img_fn' is available, skip or handle the sample differently
            continue

        images.append(img)

        # 2) Gather question
        questions.append(s["question"])

        # 3) Gather one ground-truth answer (or a placeholder if none are found)
        if "answers" in s and isinstance(s["answers"], list) and len(s["answers"]) > 0:
            answers.append(s["answers"][0])
        else:
            answers.append("unknown")

    # 4) Preprocess with your BLIP processor
    model_inputs = processor(
        images=images,
        text=questions,
        return_tensors="pt",
        padding="max_length",
        truncation=True
    )

    # 5) Tokenise the answers for labels
    labels = processor(
        text=answers,
        return_tensors="pt",
        padding="max_length",
        truncation=True
    ).input_ids

    model_inputs["labels"] = labels

    return model_inputs


# Create DataLoaders for training and validation splits.
train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset,   batch_size=48, shuffle=False, collate_fn=collate_fn)


# -------- 5. Fine-Tuning Training Loop --------

optimizer = optim.AdamW(custom_model.parameters(), lr=1e-5)
num_epochs = 1
custom_model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    for step, batch in enumerate(train_loader):
        # Move all batch tensors to the device.
        for key in batch:
            batch[key] = batch[key].to(device)
        
        optimizer.zero_grad()
        outputs = custom_model(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        loss = outputs["loss"]
        if loss is None:
            continue
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if (step + 1) % 50 == 0:
            avg_loss = running_loss / 50
            print(f"Epoch {epoch+1} | Step {step+1} | Loss: {avg_loss:.4f}")
            running_loss = 0.0

print("Fine-tuning complete!")
torch.save(custom_model.state_dict(), "fine_tuned_blip_textvqa.pt")


# -------- 6. Evaluate on Validation Set using VQA Accuracy --------

custom_model.eval()

def vqa_accuracy(pred_answer, gt_answers):
    """
    Computes the standard VQA accuracy:
        score = min(1, (# of times pred_answer appears in gt_answers)/3)
    """
    pred_answer = pred_answer.strip().lower()
    match_count = sum(1 for ans in gt_answers if ans.strip().lower() == pred_answer)
    return min(1.0, match_count / 3.0)

total_score = 0.0
total_samples = 0

# Loop over validation samples one-by-one for simplicity.
for val_sample in val_dataset:
    # Load image.
    if "image" in val_sample and isinstance(val_sample["image"], Image.Image):
        img = val_sample["image"]
    elif "img_fn" in val_sample:
        img_path = val_sample["img_fn"]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
    else:
        continue
    
    question = val_sample["question"]
    gt_answers = val_sample["answers"]

    # Preprocess the single sample.
    encoding = processor(images=img, text=question, return_tensors="pt", padding=True, truncation=True)
    encoding = {k: v.to(device) for k, v in encoding.items()}
    
    # Use greedy generation for simplicity.
    with torch.no_grad():
        generated_ids = base_model.generate(
            pixel_values=encoding["pixel_values"],
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            max_length=15
        )
    pred_answer = processor.decode(generated_ids[0], skip_special_tokens=True)
    
    # Compute VQA accuracy for this sample.
    score = vqa_accuracy(pred_answer, gt_answers)
    total_score += score
    total_samples += 1

if total_samples > 0:
    final_accuracy = total_score / total_samples
    print(f"Validation VQA Accuracy: {final_accuracy:.4f}")
else:
    print("No validation samples were processed.")
