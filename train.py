import torch
from torch.utils.data import DataLoader, random_split
# CHANGE 1: Import RoBERTa classes
from transformers import RobertaTokenizerFast, RobertaForMaskedLM, get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import pandas as pd
from log_dataloader import HTTPLogsDataset 

# --- CONFIG ---
CSV_FILE = "./dataset/train_dataset.csv"
EPOCHS = 30
BATCH_SIZE = 8
LR = 5e-5
MAX_LENGTH = 256
# CHANGE 2: Use the DistilRoBERTa model ID
MODEL_PATH = "distilroberta-base"
SAVE_PATH = "distilroberta_http_mlm"
VAL_SPLIT = 0.1
PATIENCE = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- 1. SETUP TOKENIZER & MODEL ---
# CHANGE 3: Initialize RoBERTa Tokenizer (Fast is recommended)
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_PATH)
# CHANGE 4: Load into RobertaForMaskedLM (DistilRoBERTa uses this architecture)
model = RobertaForMaskedLM.from_pretrained(MODEL_PATH)
model.to(device)

# --- 2. DEFINE TOKENS ---
# We split tokens into two groups for the "Smart Masking" strategy

# Group A: IMMUTABLE STRUCTURE (Never mask these)
# These form the "grammar" of the request.
STRUCTURAL_WRAPPERS = [
    '<request_method>', '</request_method>',
    '<request_path>', '</request_path>',
    '<request_protocol>', '</request_protocol>',
    '<request_body>', '</request_body>'
]

# Group B: DATA TYPES (Allow masking these)
# We want the model to predict: "I expect an <INT> here."
DATA_TYPE_TOKENS = [
    '<INT>', '<FLOAT>', '<UUID>', '<HASH>',
    '<B64>', '<HEX>', '<TIME>', '<EMAIL>',
    '<IP>'
]

# Add ALL tokens to the tokenizer so it recognizes them as single units
ALL_SPECIAL_TOKENS = STRUCTURAL_WRAPPERS + DATA_TYPE_TOKENS
tokenizer.add_special_tokens({'additional_special_tokens': ALL_SPECIAL_TOKENS})
model.resize_token_embeddings(len(tokenizer))

# --- 3. COLLATE FN ---
def collate_fn(batch):
    return tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )

# --- 4. DATASET ---
# --- 4. DATASET ---
dataset = HTTPLogsDataset(CSV_FILE) # No max_length here

# Limit to 150k samples if requested
TOTAL_SAMPLES_TO_USE = 50000
if len(dataset) > TOTAL_SAMPLES_TO_USE:
    print(f"Limiting dataset from {len(dataset)} to {TOTAL_SAMPLES_TO_USE} samples.")
    dataset, _ = random_split(dataset, [TOTAL_SAMPLES_TO_USE, len(dataset) - TOTAL_SAMPLES_TO_USE])

train_size = int((1 - VAL_SPLIT) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# --- 5. OPTIMIZER ---
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_dataloader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# --- 6. SMART MASKING FUNCTION (Updated Logic) ---
def mask_tokens_for_mlm(input_ids, tokenizer, mask_prob=0.15):
    device = input_ids.device
    labels = input_ids.clone()

    probability_matrix = torch.full(labels.shape, mask_prob, device=device)

    # 1. Protect Standard Special Tokens (RoBERTa uses <s>, </s>, <pad>, <mask>)
    # Tokenizer properties handle the ID lookup automatically
    protected_token_ids = set([
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
        tokenizer.mask_token_id,
    ])

    # 2. Protect STRUCTURAL WRAPPERS only
    # We DO NOT protect DATA_TYPE_TOKENS (<INT>, <HEX>, etc.)
    # This forces the model to learn that "id=" should be followed by <INT>
    protected_token_ids.update(tokenizer.convert_tokens_to_ids(STRUCTURAL_WRAPPERS))

    # Apply protection mask
    special_tokens_mask = torch.zeros_like(labels, dtype=torch.bool, device=device)
    for i in range(labels.size(0)):
        for j in range(labels.size(1)):
            if input_ids[i, j].item() in protected_token_ids:
                special_tokens_mask[i, j] = True

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Ignore loss for unmasked tokens

    # 80% -> [MASK] (In RoBERTa, this is the <mask> token)
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=device)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.mask_token_id

    # 10% -> Random
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=device)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, device=device)
    input_ids[indices_random] = random_words[indices_random]

    # 10% -> Original

    return input_ids, labels

# --- 7. TRAINING & VALIDATION LOOPS (Unchanged) ---
def validate(model, dataloader):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            masked_input_ids, labels = mask_tokens_for_mlm(input_ids, tokenizer)
            
            outputs = model(
                input_ids=masked_input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            if not torch.isnan(outputs.loss):
                total_loss += outputs.loss.item()
                num_batches += 1
                
    model.train()
    return total_loss / num_batches if num_batches > 0 else float('inf')

best_val_loss = float('inf')
patience_counter = 0


torch.cuda.empty_cache()
model.train()
for epoch in range(EPOCHS):
    loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    total_loss = 0.0
    num_batches = 0

    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        masked_input_ids, labels = mask_tokens_for_mlm(input_ids, tokenizer)

        outputs = model(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss

        if torch.isnan(loss):
            continue

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1
        loop.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])

    avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_val_loss = validate(model, val_dataloader)

    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_path = os.path.join(SAVE_PATH, "best_model")
        os.makedirs(best_model_path, exist_ok=True)
        model.save_pretrained(best_model_path)
        tokenizer.save_pretrained(best_model_path)
        print(f"✓ New best model saved! Val Loss: {best_val_loss:.4f}")
    else:
        patience_counter += 1
        print(f"No improvement for {patience_counter} epoch(s)")
        if patience_counter >= PATIENCE:
            print("Early stopping triggered")
            break

print("Training Complete.")