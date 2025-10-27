"""
Memory-efficient configuration for ICW experiments.
This file provides options based on your hardware.
"""

import torch

# Detect available hardware
HAS_CUDA = torch.cuda.is_available()
HAS_MPS = torch.backends.mps.is_available()  # Apple Silicon

print(f"CUDA available: {HAS_CUDA}")
print(f"MPS (Apple Silicon) available: {HAS_MPS}")

# Memory profiles for different models
MODEL_PROFILES = {
    "qwen-2.5-7b": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "full_memory": "13GB",
        "4bit_memory": "4-5GB",
        "8bit_memory": "7GB"
    },
    "llama-3.1-8b": {
        "name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "full_memory": "16GB",
        "4bit_memory": "5-6GB",
        "8bit_memory": "8GB"
    },
    "qwen-2.5-1.5b": {
        "name": "Qwen/Qwen2.5-1.5B-Instruct",
        "full_memory": "3GB",
        "4bit_memory": "1GB",
        "8bit_memory": "1.5GB"
    },
    "qwen-2.5-3b": {
        "name": "Qwen/Qwen2.5-3B-Instruct",
        "full_memory": "6GB",
        "4bit_memory": "2GB",
        "8bit_memory": "3GB"
    }
}

# Configuration options based on your hardware
def get_model_config(strategy="auto"):
    """
    Get model configuration based on available hardware.
    
    Strategies:
    - "auto": Automatically select based on available memory
    - "4bit": Use 4-bit quantization (best memory savings)
    - "8bit": Use 8-bit quantization (balance)
    - "full": Full precision (needs lots of memory)
    - "cpu": CPU-only mode (slow but works everywhere)
    - "small": Use smaller model (fast, less memory)
    """
    
    if strategy == "small":
        return {
            "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
            "quantization": None,
            "device_map": "auto",
            "description": "Small model (1.5B) - fast, low memory"
        }
    
    elif strategy == "4bit":
        from transformers import BitsAndBytesConfig
        return {
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "quantization": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            ),
            "device_map": "auto",
            "description": "4-bit quantization - ~4GB VRAM"
        }
    
    elif strategy == "8bit":
        from transformers import BitsAndBytesConfig
        return {
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "quantization": BitsAndBytesConfig(load_in_8bit=True),
            "device_map": "auto",
            "description": "8-bit quantization - ~7GB VRAM"
        }
    
    elif strategy == "cpu":
        return {
            "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
            "quantization": None,
            "device_map": "cpu",
            "description": "CPU-only mode - slow but works anywhere"
        }
    
    elif strategy == "full":
        return {
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "quantization": None,
            "device_map": "auto",
            "dtype": torch.float16,
            "description": "Full precision - needs 13GB+ VRAM"
        }
    
    else:  # auto
        if HAS_CUDA:
            # Check available GPU memory
            try:
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"Available GPU memory: {gpu_mem:.1f} GB")
                
                if gpu_mem > 12:
                    return get_model_config("4bit")
                elif gpu_mem > 8:
                    return get_model_config("8bit")
                else:
                    return get_model_config("small")
            except:
                return get_model_config("4bit")
        elif HAS_MPS:
            # Apple Silicon - use smaller model
            print("Apple Silicon detected - using smaller model")
            return get_model_config("small")
        else:
            # CPU only
            print("No GPU detected - using CPU mode")
            return get_model_config("cpu")

# Recommended configurations
RECOMMENDED_CONFIGS = {
    "MacBook (M1/M2/M3)": "small",  # 1.5B model works great
    "GPU with 8GB VRAM": "4bit",     # 4-bit quantization
    "GPU with 16GB VRAM": "8bit",    # 8-bit or full
    "GPU with 24GB+ VRAM": "full",   # Full precision
    "No GPU / CPU only": "cpu"       # CPU mode (slow)
}

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ICW Memory Configuration Helper")
    print("="*60)
    
    print("\nAvailable Model Profiles:")
    for key, profile in MODEL_PROFILES.items():
        print(f"\n{key}:")
        print(f"  Model: {profile['name']}")
        print(f"  Full precision: {profile['full_memory']}")
        print(f"  4-bit quantized: {profile['4bit_memory']}")
        print(f"  8-bit quantized: {profile['8bit_memory']}")
    
    print("\n" + "="*60)
    print("Recommended Configurations:")
    print("="*60)
    for hardware, strategy in RECOMMENDED_CONFIGS.items():
        print(f"{hardware}: strategy='{strategy}'")
    
    print("\n" + "="*60)
    print("Auto-detected configuration:")
    print("="*60)
    config = get_model_config("auto")
    for key, value in config.items():
        if key != "quantization":
            print(f"{key}: {value}")
