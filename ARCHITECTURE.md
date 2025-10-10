# Architecture Update: Model-Agnostic Design

## Overview

The codebase has been updated to use a modern, model-agnostic architecture based on the Hugging Face chat template system. This makes it trivial to switch between different models without changing any prompt formatting code.

## Key Changes

### 1. Removed Model-Specific Formatting

**Before (Phi-3 specific)**:
```python
def format_phi_prompt(instruction, query):
    return f"<|system|>{instruction}<|end|>\n<|user|>{query}<|end|>\n<|assistant|>"

prompt = format_phi_prompt("Be helpful", "What is AI?")
output = generator(prompt)[0]['generated_text']
response = output.split('<|assistant|>')[-1].strip()
```

**After (Model-agnostic)**:
```python
messages = [
    {"role": "system", "content": "Be helpful"},
    {"role": "user", "content": "What is AI?"}
]
response = generate_response(messages)
```

### 2. New `generate_response()` Function

```python
def generate_response(messages):
    """
    Generate response using tokenizer.apply_chat_template + model.generate + tokenizer.decode.
    This makes it easier to switch between models.
    """
    # Apply chat template - automatically formats for the specific model
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate
    outputs = model.generate(input_ids, **generation_config)
    
    # Decode only the new tokens (exclude the input)
    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response
```

### 3. Universal Message Format

All ICW methods now use the same message format:

```python
# Unicode ICW
messages = [
    {"role": "system", "content": "Insert zero-width spaces after words..."},
    {"role": "user", "content": query}
]

# Initials ICW  
messages = [
    {"role": "system", "content": "Start words with vowels..."},
    {"role": "user", "content": query}
]

# Lexical ICW
messages = [
    {"role": "system", "content": "Use green words..."},
    {"role": "user", "content": query}
]

# Acrostics ICW
messages = [
    {"role": "system", "content": "Start sentences with SECRET..."},
    {"role": "user", "content": query}
]
```

## Benefits

### 1. Easy Model Switching
Just change the `MODEL_NAME` - no code changes needed:

```python
# Switch from Qwen to Llama
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# That's it! Everything else works automatically
```

### 2. Support for Any Chat Model
Works with any model that has a chat template:
- Qwen (all versions)
- Llama (2, 3, 3.1, etc.)
- Phi (3, 4, etc.)
- Mistral
- Gemma
- And many more!

### 3. Automatic Prompt Formatting
Each model has different chat formats:
- **Qwen**: `<|im_start|>system\n...<|im_end|>`
- **Llama**: `<|begin_of_text|><|start_header_id|>system<|end_header_id|>`
- **Phi**: `<|system|>...<|end|>`

The `apply_chat_template()` method handles all of this automatically!

### 4. Cleaner Code
- No more model-specific formatting functions
- No manual string splitting to extract responses
- Consistent API across all watermarking methods

## How It Works

### Step-by-Step

1. **Define Messages**
   ```python
   messages = [
       {"role": "system", "content": "System instruction"},
       {"role": "user", "content": "User query"}
   ]
   ```

2. **Apply Chat Template**
   ```python
   input_ids = tokenizer.apply_chat_template(
       messages,
       add_generation_prompt=True,  # Adds the assistant prefix
       return_tensors="pt"
   )
   ```
   
   For Qwen, this produces:
   ```
   <|im_start|>system
   System instruction<|im_end|>
   <|im_start|>user
   User query<|im_end|>
   <|im_start|>assistant
   ```

3. **Generate**
   ```python
   outputs = model.generate(input_ids, **generation_config)
   ```

4. **Decode Only New Tokens**
   ```python
   response = tokenizer.decode(
       outputs[0][input_ids.shape[1]:],  # Skip the prompt
       skip_special_tokens=True
   )
   ```

### Configuration

```python
generation_config = {
    "max_new_tokens": 200,
    "do_sample": TEMPERATURE > 0,
    "temperature": TEMPERATURE if TEMPERATURE > 0 else None,
    "top_p": 0.9,
    "pad_token_id": tokenizer.pad_token_id,
}
```

## Model Updates

### Removed
- ❌ **Phi-3**: Does not have strong reasoning capabilities

### Kept/Added
- ✅ **Qwen 2.5**: Default - excellent reasoning, multilingual
- ✅ **Llama-3.1**: Strong reasoning, instruction following
- ✅ **Phi-4**: Enhanced reasoning (when available)

## Compatibility Notes

### Requirements
- `transformers >= 4.34.0` (for `apply_chat_template`)
- `torch >= 2.0.0`

### Trust Remote Code
Some models (like Qwen) require `trust_remote_code=True`:

```python
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, 
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
```

### Padding Token
Some models don't define a padding token by default:

```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

## Migration Guide

If you have custom ICW methods, update them as follows:

### Old Format
```python
def my_custom_watermark(query):
    instruction = "Your watermarking instruction"
    return format_phi_prompt(instruction, query)

for query in queries:
    prompt = my_custom_watermark(query)
    output = generator(prompt)[0]['generated_text']
    response = output.split('<|assistant|>')[-1].strip()
    results.append(response)
```

### New Format
```python
def my_custom_watermark(query):
    instruction = "Your watermarking instruction"
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": query}
    ]

for query in queries:
    messages = my_custom_watermark(query)
    response = generate_response(messages)
    results.append(response)
```

## Testing Different Models

### Quick Test Script

```python
# Test all models with the same prompt
models = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
]

messages = [
    {"role": "system", "content": "Be concise."},
    {"role": "user", "content": "What is AI?"}
]

for model_name in models:
    print(f"\n{'='*50}")
    print(f"Testing: {model_name}")
    print(f"{'='*50}")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Generate
    response = generate_response(messages)
    print(f"Response: {response}")
```

## Advantages for Research

1. **Reproducibility**: Same message format across all experiments
2. **Comparability**: Fair comparison between models (no formatting bias)
3. **Extensibility**: Easy to add new models or ICW methods
4. **Maintainability**: Single source of truth for generation logic

## Summary

The new architecture is:
- ✅ **Model-agnostic**: Works with any chat model
- ✅ **Simpler**: Less code, fewer bugs
- ✅ **Flexible**: Easy to add new models
- ✅ **Standard**: Uses Hugging Face best practices
- ✅ **Research-focused**: Models with reasoning capabilities

This design follows the advice from your mentor and makes the codebase much more flexible for future research!
