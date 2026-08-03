import os
import sys

def test_qwen():
    model_path = os.path.abspath("./models/qwen")
    
    if not os.path.isdir(model_path) or not os.listdir(model_path):
        print(f"Error: Model directory '{model_path}' is empty or does not exist.")
        print("Please run the download script first.")
        sys.exit(1)
        
    print("Loading model and tokenizer from local folder...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        # Load tokenizer and model from the local directory
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device detected: {device}")
        
        # Select optimal dtype (bfloat16 for modern GPUs, float16 for older GPUs, float32 for CPU)
        if device == "cuda":
            optimal_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            optimal_dtype = torch.float32
        print(f"Using dtype: {optimal_dtype}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=optimal_dtype,
            device_map="auto"
        )
        print("Model and tokenizer loaded successfully!")
        
        # Build the conversation payload using the model's recommended chat template
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, who are you?"}
        ]
        
        # Format the messages using the model's tokenizer chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Prepare inputs
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        print("\nGenerating response (this may take a few seconds)...")
        # Generate output
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        # Extract the new tokens
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        # Decode and print response
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print("\n--- Model Response ---")
        print(response)
        print("----------------------")
        
    except Exception as e:
        print(f"An error occurred during testing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_qwen()
