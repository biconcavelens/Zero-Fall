
import torch
from transformers import DistilBertForMaskedLM
import os

def convert_safetensors_to_pth():
    model_path = "./" # Current directory containing model.safetensors and config.json
    output_model_file = "model.pth"

    print(f"Loading model from {model_path}...")
    try:
        # Load the model. Transformers automatically detects safetensors if available.
        model = DistilBertForMaskedLM.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Saving model state_dict to {output_model_file}...")
    torch.save(model.state_dict(), output_model_file)
    print(f"Successfully converted and saved to {output_model_file}")

if __name__ == "__main__":
    convert_safetensors_to_pth()
