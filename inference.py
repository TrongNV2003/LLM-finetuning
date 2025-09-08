import json
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.prompts import PROMPT_TEMPLATE

class Inference:
    def __init__(self, base_model_path: str, adapter_path: str):
        self.prompt_template = PROMPT_TEMPLATE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading tokenizer from {base_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading base model from {base_model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print(f"Loading LoRA adapter from {adapter_path}")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        
    def format_prompt(self, text: str) -> str:
        return self.prompt_template.format(text=text, label="")

    def inference(
        self, 
        text: str, 
        max_new_tokens: int = 50,
        temperature: float = 0.3,
        do_sample: bool = True
    ) -> str:
        """
        Infer the denoised address from the text input
        
        Args:
            text: The input address
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            The normalized address
        """
        prompt = self.format_prompt(text)
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.05
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        prompt_length = len(prompt)
        generated_part = generated_text[prompt_length:].strip()
        
        return generated_part

    def batch_inference(self, texts: list) -> list:
        results = []
        for text in texts:
            result = self.inference(text)
            results.append(result)
        return results

def main():
    base_model_path = "Qwen/Qwen3-0.6B"
    adapter_path = "./finetuning-checkpoints"
    
    try:
        infer = Inference(base_model_path, adapter_path)
        
        test_addresses = [
            "68, Xthuy, Cgiay, HN",
            "123, Ngtroi, Q1, HCM",
            "45, Lthanh, Hkiem, HN",
            "789, Bachmai, Hbt, HN",
            "56, Tramhung, Q3, HCM"
        ]
        
        print("\nTesting the model:")
        print("=" * 60)
        
        for address in test_addresses:
            result = infer.inference(address)
            print(f"Input:  {address}")
            print(f"Output: {result}")
            print("-" * 60)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
