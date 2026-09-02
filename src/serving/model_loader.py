"""
Model loading utilities for quantized LLMs.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Tuple

# Model configuration
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 4-bit quantization config (works on any platform)
QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
) if DEVICE == "cuda" else None

def load_model_and_tokenizer() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load Mistral-7B model and tokenizer from HuggingFace.
    
    On GPU (CUDA): Uses bitsandbytes 4-bit quantization for reduced memory.
    On CPU: Loads full precision (quantization not supported on CPU).
    Model downloaded and cached locally.
    
    Returns:
        Tuple of (model, tokenizer)
    
    Raises:
        RuntimeError if model fails to load
    """
    print(f"Loading model from {MODEL_ID}...")
    print(f"Using device: {DEVICE}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
        )
        
        # Load model with quantization on GPU, or full precision on CPU
        if DEVICE == "cuda":
            print("Loading with 4-bit quantization (bitsandbytes)...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                quantization_config=QUANTIZATION_CONFIG,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
        else:
            print("Loading on CPU (no quantization). This will be slow.")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                device_map="cpu",
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )
        
        print("✓ Model loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
) -> str:
    """
    Generate text using the loaded model.
    
    Args:
        model: The LLM model
        tokenizer: The tokenizer
        prompt: Input text prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.7 = slightly creative)
    
    Returns:
        Generated text
    """
    # Encode prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
            top_k=50,
            do_sample=True,
        )
    
    # Decode and clean
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


if __name__ == "__main__":
    # Quick test
    model, tokenizer = load_model_and_tokenizer()
    
    prompt = "Explain quantum computing in one sentence:"
    result = generate_text(model, tokenizer, prompt, max_tokens=50)
    
    print(f"\nPrompt: {prompt}")
    print(f"Response: {result}")
