import os
import sys

def interactive_chat():
    model_path = os.path.abspath("./models/qwen")
    
    if not os.path.isdir(model_path) or not os.listdir(model_path):
        print(f"Error: Model directory '{model_path}' is empty or does not exist.")
        print("Please run the download script first.")
        sys.exit(1)
        
    print("Loading model and tokenizer from local folder...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        # Load tokenizer and model locally
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
        print("\n=== Interactive Chat Session ===")
        print("Type your message and press Enter. Type 'exit' or 'quit' to end the session.\n")
        
        # Initialize conversation history
        messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        
        while True:
            try:
                user_input = input("You: ")
                if user_input.strip().lower() in ["exit", "quit"]:
                    print("Exiting chat session. Goodbye!")
                    break
                
                if not user_input.strip():
                    continue
                
                # Append user message to the conversation history
                messages.append({"role": "user", "content": user_input})
                
                # Format using the model's chat template
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # Tokenize inputs
                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
                
                # Generate response
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                # Extract new tokens only (slice off the input prompt IDs)
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                
                # Decode response
                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                print(f"Qwen: {response}\n")
                
                # Append assistant response to history to maintain conversation context
                messages.append({"role": "assistant", "content": response})
                
            except KeyboardInterrupt:
                print("\nExiting chat session. Goodbye!")
                break
                
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    interactive_chat()
