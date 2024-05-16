from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig

def download_model(model_name, save_directory):
    """Download model and tokenizer files and save them to a specified directory."""
    # Create configuration object
    config = LlamaConfig.from_pretrained(model_name)
    config.save_pretrained(save_directory)
    
    # Create tokenizer object
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_directory)
    
    # Create model object
    model = LlamaForCausalLM.from_pretrained(model_name)
    model.save_pretrained(save_directory)

    print(f"Model and tokenizer have been saved to {save_directory}")

# Usage
model_name = "google/gemma-7b-it"
save_directory = "./gemma7B"
print("Downloading")
download_model(model_name, save_directory)