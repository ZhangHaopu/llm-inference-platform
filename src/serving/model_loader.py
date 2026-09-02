"""
Model loading utilities for quantized LLMs.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple

# Model configuration
MODEL_ID = "TheBloke/Mistral-7B-Instruct-v0.1-GPTQ"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model_and_tokenizer() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load quantized Mistral-7B model and tokenizer from HuggingFace.
    
    Uses GPTQ quantization (4-bit) for reduced memory footprint.
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
            use_auth_token=False
        )
        
        # Load quantized model
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            trust_remote_code=True,
            use_auth_token=False,
            torch_dtype=torch.float16,  # Half precision for quantized models
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
