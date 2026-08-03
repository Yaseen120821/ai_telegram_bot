import glob
import os
import sys

def patch_florence2_files():
    cache_dir = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")
    if not os.path.exists(cache_dir):
        print(f"Cache dir {cache_dir} does not exist yet.")
        return
    
    # 1. Patch configuration_florence2.py
    cfg_files = glob.glob(os.path.join(cache_dir, "**", "configuration_florence2.py"), recursive=True)
    for fpath in cfg_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()

            modified = False
            if "self.forced_bos_token_id is None" in content:
                content = content.replace(
                    "self.forced_bos_token_id is None",
                    'getattr(self, "forced_bos_token_id", None) is None'
                )
                modified = True
            
            if "class Florence2LanguageConfig(PretrainedConfig):" in content and "forced_bos_token_id = None" not in content:
                content = content.replace(
                    "class Florence2LanguageConfig(PretrainedConfig):",
                    "class Florence2LanguageConfig(PretrainedConfig):\n    forced_bos_token_id = None"
                )
                modified = True

            if modified:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"Patched config: {fpath}")
        except Exception as e:
            print(f"Error patching config {fpath}: {e}")

    # 2. Patch processing_florence2.py
    proc_files = glob.glob(os.path.join(cache_dir, "**", "processing_florence2.py"), recursive=True)
    for fpath in proc_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()

            modified = False
            if "tokenizer.additional_special_tokens +" in content:
                content = content.replace(
                    "tokenizer.additional_special_tokens +",
                    'list(getattr(tokenizer, "additional_special_tokens", [])) +'
                )
                modified = True

            if modified:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"Patched processor: {fpath}")
        except Exception as e:
            print(f"Error patching processor {fpath}: {e}")

    # 3. Patch modeling_florence2.py
    model_files = glob.glob(os.path.join(cache_dir, "**", "modeling_florence2.py"), recursive=True)
    for fpath in model_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()

            modified = False
            if "return self.language_model._supports_flash_attn_2" in content:
                content = content.replace(
                    "return self.language_model._supports_flash_attn_2",
                    'return getattr(self, "language_model", None) is not None and getattr(self.language_model, "_supports_flash_attn_2", False)'
                )
                modified = True

            if "return self.language_model._supports_sdpa" in content:
                content = content.replace(
                    "return self.language_model._supports_sdpa",
                    'return getattr(self, "language_model", None) is not None and getattr(self.language_model, "_supports_sdpa", False)'
                )
                modified = True

            # Patch Cache indexing compatibility: past_key_values[0][0].shape[2] -> past_key_values.get_seq_length()
            if "past_key_values[0][0].shape[2]" in content:
                content = content.replace(
                    "past_key_values[0][0].shape[2]",
                    "(past_key_values.get_seq_length() if hasattr(past_key_values, 'get_seq_length') else past_key_values[0][0].shape[2])"
                )
                modified = True

            if "past_key_values[idx]" in content:
                content = content.replace(
                    "past_key_values[idx]",
                    "(past_key_values[idx] if isinstance(past_key_values, (list, tuple)) else None)"
                )
                modified = True

            # Patch meta tensor linspace error: [x.item() for x in torch.linspace(...)]
            if "[x.item() for x in torch.linspace(" in content:
                content = content.replace(
                    "[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths)*2)]",
                    "torch.linspace(0, drop_path_rate, sum(depths)*2, device='cpu').tolist()"
                )
                modified = True

            if modified:
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"Patched model: {fpath}")
        except Exception as e:
            print(f"Error patching model {fpath}: {e}")

if __name__ == "__main__":
    patch_florence2_files()
