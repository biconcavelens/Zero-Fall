
import torch
import json
import numpy as np
from transformers import DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer

MAX_LENGTH = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_sequence(log_data):
    """Build sequence - handles both flat and nested JSON structures"""
    body_bytes = log_data.get('body_bytes_sent', '') or log_data.get('body_bytes', '')
    method = log_data.get('method', '') or log_data.get('request.method', '')
    path = log_data.get('path', '') or log_data.get('request.path', '')
    protocol = log_data.get('protocol', '') or log_data.get('request.protocol', '')
    body = log_data.get('request_body', '') or log_data.get('body', '')
    
    seq = (
        f"[CLS] "
        f"<body_bytes> {body_bytes} </body_bytes> [SEP] "
        f"<request_method> {method} </request_method> [SEP] "
        f"<request_path> {path} </request_path> [SEP] "
        f"<request_protocol> {protocol} </request_protocol> [SEP] "
        f"<request_body> {body} </request_body> [SEP]"
    )
    return seq

def mask_tokens(input_ids, tokenizer, mask_prob=0.15):
    device = input_ids.device
    labels = input_ids.clone()
    probability_matrix = torch.full(labels.shape, mask_prob, device=device)
    special_tokens_mask = torch.zeros_like(labels, dtype=torch.bool, device=device)
    
    for i, ids in enumerate(input_ids):
        tokens = tokenizer.convert_ids_to_tokens(ids.cpu())
        for j, token in enumerate(tokens):
            if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
                special_tokens_mask[i, j] = True
    
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    
    if masked_indices.sum() == 0:
        rand_idx = torch.randint(1, labels.shape[1]-1, (1,), device=device)
        masked_indices[0, rand_idx] = True
    
    labels[~masked_indices] = -100
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=device)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.mask_token_id
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=device)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long, device=device)
    input_ids[indices_random] = random_words[indices_random]
    
    return input_ids, labels

def calculate_cost(log_text, tokenizer, model, num_runs=5):
    """Run multiple times and average to reduce variance from random masking"""
    errors = []
    
    model.eval()
    for i in range(num_runs):
        encoding = tokenizer(
            log_text,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='pt'
        ).to(device)
        
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        
        with torch.no_grad():
            masked_input, labels = mask_tokens(input_ids.clone(), tokenizer)
            outputs = model(
                input_ids=masked_input,
                attention_mask=attention_mask,
                labels=labels
            )
        
        # Handle both scalar and batched loss
        loss_val = outputs.loss
        if loss_val.ndim == 0:
            error = loss_val.item()
        else:
            error = loss_val.mean().item()
        
        errors.append(error)
    
    # Return average
    return np.mean(errors)

def main():
    model_path = "model.pth"
    config_path = "config.json"
    tokenizer_path = "./"
    input_file = "benign.json"
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"Loading configuration from {config_path}...")
    config = DistilBertConfig.from_json_file(config_path)

    print("Initializing model architecture...")
    model = DistilBertForMaskedLM(config)

    print(f"Loading weights from {model_path}...")
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"Error: {model_path} not found. Please run convert_to_pth.py first.")
        return
        
    model.to(device)
    model.eval()

    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)

    print(f"Reading input from {input_file}...")
    with open(input_file, 'r') as f:
        log_data = json.load(f)
    
    formatted_log = build_sequence(log_data)
    print(f"\nFormatted Log Sequence:\n{formatted_log}")
    
    print("\nCalculating reconstruction cost (average of 5 runs)...")
    cost = calculate_cost(formatted_log, tokenizer, model, num_runs=5)
    
    print(f"\nCalculated Cost (Reconstruction Loss): {cost:.4f}")

if __name__ == "__main__":
    main()
