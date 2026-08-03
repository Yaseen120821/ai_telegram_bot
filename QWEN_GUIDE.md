# Setup Guide: Qwen2.5-0.5B-Instruct Local Setup

This beginner-friendly guide walks you through downloading, saving, verifying, and running the Hugging Face model **Qwen2.5-0.5B-Instruct** locally inside your project folder (`./models/qwen/`) without relying on the default global Hugging Face cache.

---

## 1. Prerequisites and Installation

To download and run this model, you need Python (version 3.8+) and several packages. If you are using the project's existing virtual environment, these packages are already installed. For a fresh setup or other machines, install the dependencies using the following commands:

### Active Virtual Environment Command
On Windows (from the root of your project directory):
```powershell
# Create a virtual environment if you don't have one
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\Activate.ps1

# Install the latest stable version of Hugging Face Transformers and its dependencies
.venv\Scripts\pip install transformers torch accelerate huggingface_hub
```

### Dependency Freeze
After installing, save the dependencies to your `requirements.txt`:
```powershell
.venv\Scripts\pip freeze > requirements.txt
```

---

## 2. Model Download Script (`scripts/download_model.py`)

Create a script to download the model directly to your project directory. 

We use the `huggingface_hub` library's `snapshot_download` function. This approach fetches the files directly to your target directory without loading the weights into memory (RAM), which saves CPU/GPU resources. Setting `local_dir_use_symlinks=False` ensures that the actual model files are saved directly in your folder, making it fully portable.

### The Python Code:

```python
import os
import sys

def download_qwen():
    print("Checking dependencies...")
    try:
        import torch
        import transformers
        import huggingface_hub
        print(f"PyTorch version: {torch.__version__}")
        print(f"Transformers version: {transformers.__version__}")
        print(f"Hugging Face Hub version: {huggingface_hub.__version__}")
    except ImportError as e:
        print(f"Error importing dependencies: {e}")
        print("Please run: .venv\\Scripts\\pip install transformers torch accelerate huggingface_hub")
        sys.exit(1)
        
    # The identifier of the repository on Hugging Face
    repo_id = "Qwen/Qwen2.5-0.5B-Instruct"
    # Resolve the absolute path to save the model locally
    local_dir = os.path.abspath("./models/qwen")
    
    print(f"Starting download of '{repo_id}' directly to '{local_dir}'...")
    
    try:
        from huggingface_hub import snapshot_download
        
        # Download files and write them directly as actual files (not symlinks)
        huggingface_hub.snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # Critical: Writes physical files to make the model portable
            repo_type="model"
        )
        print("\nDownload completed successfully!")
        
        # Verify the download by listing all saved files and sizes
        print("\nDownloaded files verification list:")
        total_size_bytes = 0
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, local_dir)
                size_bytes = os.path.getsize(file_path)
                total_size_bytes += size_bytes
                size_mb = size_bytes / (1024 * 1024)
                print(f" - {rel_path} ({size_mb:.2f} MB)")
        
        total_size_gb = total_size_bytes / (1024 * 1024 * 1024)
        print(f"\nTotal Model Size on Disk: {total_size_gb:.2f} GB")
                
    except Exception as e:
        print(f"An error occurred during download: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_qwen()
```

### Explaining the Download Code Line-by-Line:
* **`import os`, `import sys`**: Core Python standard libraries used to manage paths, walk files, and manage execution exit codes.
* **`try: ... except ImportError`**: Verifies that the required libraries are available in the active environment.
* **`repo_id = "Qwen/Qwen2.5-0.5B-Instruct"`**: Tells Hugging Face Hub which repository to fetch.
* **`local_dir = os.path.abspath("./models/qwen")`**: Resolves `./models/qwen` to an absolute path for safety during local execution.
* **`from huggingface_hub import snapshot_download`**: Imports the Hugging Face utility used to fetch whole repositories.
* **`snapshot_download(...)`**:
  * **`repo_id`**: The target repository.
  * **`local_dir`**: The path where files are written.
  * **`local_dir_use_symlinks=False`**: Prevents Hugging Face from creating pointer symlinks back to its default cache. This results in standard, physical files inside your `./models/qwen` folder.
  * **`repo_type="model"`**: Explicitly specifies that we are downloading a model (as opposed to a dataset or space).
* **`os.walk(local_dir)`**: Recursively lists and computes sizes of all downloaded files to confirm successful completion.

---

## 3. Explaining Downloaded Model Files

Once downloaded, you will find several file types in `./models/qwen/`. Here is what each file does:

| File Name | Purpose | Description |
| :--- | :--- | :--- |
| **`config.json`** | Model Architecture | Contains high-level configuration parameters for the model (e.g., number of layers, attention heads, vocabulary size, activation functions). |
| **`generation_config.json`** | Generation Defaults | Defines default settings for generating text, such as maximum output length, temperature, top-p sampling, and stop tokens. |
| **`model.safetensors`** | Model Weights | Contains the actual weights (parameters) of the neural network. The `.safetensors` format is preferred over `.bin` (PyTorch pickle format) because it prevents execution of arbitrary code during loading and is faster to read. |
| **`tokenizer.json`** | Tokenization Engine | Represents the complete serialized tokenizer configuration, including the BPE (Byte Pair Encoding) vocabulary and formatting rules. |
| **`tokenizer_config.json`**| Tokenizer Settings | Stores user-level settings for the tokenizer, such as default padding strategies, truncation limits, and custom formatting properties. |
| **`vocab.json` / `merges.txt`** | Vocabulary & Merging | Maps tokens (sub-words) to unique numerical IDs and determines how characters should be merged together to build tokens. |
| **`special_tokens_map.json`** | Special Tokens | Maps roles (like start-of-sequence, end-of-sequence, system, user, and assistant) to specific tokens, allowing the model to distinguish where prompts end and responses begin. |

---

## 4. Model Testing Script (`tests/test_model.py`)

Create a script to load the model locally, format a conversation using the Chat Template, perform inference, and print the response.

### The Python Code:

```python
import os
import sys

def test_qwen():
    # Resolve the absolute path to your local model folder
    model_path = os.path.abspath("./models/qwen")
    
    # Check if the folder contains files
    if not os.path.isdir(model_path) or not os.listdir(model_path):
        print(f"Error: Model directory '{model_path}' is empty or does not exist.")
        print("Please run the download script first.")
        sys.exit(1)
        
    print("Loading model and tokenizer from local folder...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        # Load the tokenizer. local_files_only=True prevents any external network calls.
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        
        # Check if CUDA (GPU) is available, fallback to CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device detected: {device}")
        
        # Load the model weights and architecture
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" # Automatically places layers on target device (e.g., GPU/CPU)
        )
        print("Model and tokenizer loaded successfully!")
        
        # Define conversation context (Instruct models require structure)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, who are you?"}
        ]
        
        # Apply the model's chat template to format the conversation
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Convert text prompt to numerical tensor IDs and place on the model's device
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        print("\nGenerating response (this may take a few seconds)...")
        # Run inference
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        # Slice off the prompt IDs so we only print the new model response
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        # Decode response IDs back to readable text
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print("\n--- Model Response ---")
        print(response)
        print("----------------------")
        
    except Exception as e:
        print(f"An error occurred during testing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_qwen()
```

### Explaining the Test Code Line-by-Line:
* **`local_files_only=True`**: Tells the Transformers library to load components strictly from the folder specified. It will crash immediately if any files are missing instead of trying to download them from the internet.
* **`device = "cuda" if torch.cuda.is_available() else "cpu"`**: Automatically detects if a compatible NVIDIA GPU is available for faster text generation.
* **`torch_dtype`**: Loads weights in FP16 (half-precision) for GPUs to save memory, or FP32 (single-precision) for stable CPU execution.
* **`device_map="auto"`**: Automatically splits model layers across devices if you have multiple GPUs, or delegates memory mapping.
* **`tokenizer.apply_chat_template`**: Automatically structures the chat history using Qwen's specific formatting (ChatML syntax `<|im_start|>user...<|im_end|>`).
* **`model.generate(...)`**: Performs the forward pass, using parameters:
  * `max_new_tokens=512`: Limits response lengths.
  * `do_sample=True`, `temperature=0.7`, `top_p=0.9`: Configures creativity/randomness.
* **`tokenizer.batch_decode(..., skip_special_tokens=True)`**: Decodes the resulting token list into standard text while removing formatting tokens.

---

## 4.5. Manual Interactive Testing

To manually test the model by typing your own prompts and chatting with it in real-time, you can use the interactive script or run it directly in an interactive Python session.

### Method A: Use the Interactive Chat Script (`scripts/interactive_chat.py`)
We have created an interactive CLI script that loads the local model and allows you to chat with it in your command prompt. It retains the conversation history so that the model understands context in multi-turn conversations.

#### The Python Code (`scripts/interactive_chat.py`):
```python
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
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
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
                
                # Append user message
                messages.append({"role": "user", "content": user_input})
                
                # Format using template
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
                
                # Extract new tokens only
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                
                # Decode response
                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                print(f"Qwen: {response}\n")
                
                # Append assistant response to history
                messages.append({"role": "assistant", "content": response})
                
            except KeyboardInterrupt:
                print("\nExiting chat session. Goodbye!")
                break
                
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    interactive_chat()
```

#### Run the CLI Chat:
Run the following command from the root of your project directory:
```powershell
.venv\Scripts\python scripts/interactive_chat.py
```
This lets you talk to your model in real time directly from the shell!

---

### Method B: Line-by-Line Testing inside an Interactive Python Shell
If you want to quickly test specific prompt templates or parameters manually, you can open an interactive Python shell and run commands manually.

1. **Start Python interactive shell in your virtual environment**:
   ```powershell
   .venv\Scripts\python
   ```

2. **Paste the following commands line-by-line**:
   ```python
   # 1. Imports
   import os
   import torch
   from transformers import AutoTokenizer, AutoModelForCausalLM

   # 2. Setup Paths
   model_path = os.path.abspath("./models/qwen")

   # 3. Load Tokenizer and Model
   tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
   model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float32, device_map="auto")

   # 4. Input Prompt (Write whatever you want here)
   prompt = "Translate this phrase to French: 'Artificial Intelligence is the future.'"

   # 5. Format Chat Template
   messages = [
       {"role": "system", "content": "You are a helpful assistant."},
       {"role": "user", "content": prompt}
   ]
   formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

   # 6. Generate and Print Response
   inputs = tokenizer([formatted_text], return_tensors="pt").to(model.device)
   outputs = model.generate(**inputs, max_new_tokens=100)
   response_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
   print(tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0])
   ```

3. **Type `exit()`** to exit the Python shell.

---

## 5. Hugging Face Login for Gated Models

The `Qwen/Qwen2.5-0.5B-Instruct` model is open-access and does **not** require a login to download. However, if you work with gated models (e.g., Llama 3, Gemma 2), you must authenticate first:

1. **Get your Token**: Create a read-access token on Hugging Face: **Settings -> Access Tokens**.
2. **Login via CLI**:
   Run this in your terminal inside your virtual environment:
   ```powershell
   .venv\Scripts\huggingface-cli login
   ```
   Paste your token when prompted.
3. **Alternative (Environment Variable)**:
   Set the `HF_TOKEN` environment variable in PowerShell:
   ```powershell
   $env:HF_TOKEN="your_hugging_face_token_here"
   ```
4. **Alternative (Python inline)**:
   Pass the token directly in the Python download script:
   ```python
   snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False, token="your_token_here")
   ```

---

## 6. Downloading Methods Compared

| Method | How It Works | Best For | Pros & Cons |
| :--- | :--- | :--- | :--- |
| **Transformers (`from_pretrained`)** | Downloads components on-demand while initializing Python objects. | Fast prototyping | **Pros**: Fits standard workflow.<br>**Cons**: Loads full model weights into system RAM during download; downloads to the default HF cache directory. |
| **`huggingface_hub` (`snapshot_download`)** | Fetches the raw file repository over HTTP APIs directly to a directory. | **Production deployments, pipeline scripts (Recommended)** | **Pros**: Very fast; no system memory overhead; customizable file placement (no symlinks).<br>**Cons**: Requires the `huggingface_hub` library. |
| **Git LFS (`git clone`)** | Uses Git Large File Storage extension to track and download binary files. | Development and contribution | **Pros**: Native Git versioning support.<br>**Cons**: Very slow; requires installing Git LFS on Windows; creates bulky `.git` directory history. |

---

## 7. Storage and Management on Windows

### Where is the model stored by default?
If you run standard `from_pretrained` commands without specifying a directory, Hugging Face saves cache files on Windows in:
`C:\Users\<Your-Username>\.cache\huggingface\hub`

### How to move the model to another computer
Because we used `local_dir_use_symlinks=False` in `snapshot_download`, your `./models/qwen` directory contains actual physical files rather than links. To move it:
1. **Compress**: Compress the `models/qwen` folder into a single ZIP file (e.g., `qwen_model.zip`).
2. **Transfer**: Copy the ZIP file via USB, external drive, or local network to the destination computer.
3. **Extract**: Place the extracted files in a directory (e.g., `C:\SANA_AI\models\qwen`).
4. **Load**: Configure your Python script to point to the new path, and set `local_files_only=True`. No internet connection is needed on the destination computer.

---

## 8. Troubleshooting and Best Practices

### Common Errors and Fixes:
1. **`HTTPError / Connection Timeout`**:
   * *Cause*: Unstable network connection or Hugging Face blockages.
   * *Fix*: Set the Hugging Face mirror endpoint (especially if in regions with connectivity issues).
     ```powershell
     $env:HF_ENDPOINT="https://hf-mirror.com"
     ```
     Then re-run the download script.
2. **`OutOfMemoryError` (GPU CUDA OOM)**:
   * *Cause*: GPU does not have enough VRAM to run the model.
   * *Fix*: Fallback to CPU by setting `device = "cpu"` or load the model in 8-bit/4-bit quantization using `bitsandbytes`.
3. **`local_files_only` errors**:
   * *Cause*: Trying to load from a directory that is missing config files or has not finished downloading.
   * *Fix*: Re-run the download script to ensure all files are fetched.
4. **`Permission Denied`**:
   * *Cause*: Running the script in a directory where the user account lacks write permission.
   * *Fix*: Open PowerShell/Terminal as Administrator, or place the project folder in user-accessible paths like `D:\` or `C:\Users\<User>\Documents`.

### How to Update the Model:
If the model authors release an update to `Qwen/Qwen2.5-0.5B-Instruct` on Hugging Face, simply run the `scripts/download_model.py` script again. 
* The `snapshot_download` function verifies files using hash validation (SHA256).
* It will skip files that are already identical and only download changed or updated files, saving you time and bandwidth.
