import os
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

import json
import math
import hydra
import torch
from loguru import logger
from dotenv import load_dotenv
from omegaconf import DictConfig
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed,
)
from peft import PeftModel, PeftConfig

from src.utils import load_dataset

load_dotenv()


class CPTEvaluator:
    """Evaluate a CPT fine-tuned model.
    
    Computes perplexity on validation data and runs generation tests
    to qualitatively assess domain knowledge.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.data_args = cfg.dataset
        self.model_args = cfg.model
        self.model_path = cfg.training_arguments.output_dir

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            use_fast=self.model_args.use_fast_tokenizer,
            trust_remote_code=True,
            cache_dir=self.model_args.cache_dir,
            token=cfg.token,
            revision=self.model_args.revision,
            padding_side=self.model_args.padding_side,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = self._load_model(cfg)

        logger.info(f"Model loaded from: {self.model_path}")

    def _load_model(self, cfg: DictConfig):
        """Load model with PEFT adapters if available, otherwise load full model."""
        peft_config_path = os.path.join(self.model_path, "adapter_config.json")
        has_peft_adapter = os.path.exists(peft_config_path)

        if has_peft_adapter:
            logger.info("Loading model with PEFT adapters")

            peft_config = PeftConfig.from_pretrained(self.model_path)
            base_model_name = peft_config.base_model_name_or_path or self.model_args.model_name_or_path

            bnb_config = None
            if hasattr(self.model_args, 'qlora') and self.model_args.qlora:
                logger.info("Using QLoRA with 4-bit quantization for evaluation")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=self.compute_dtype
                )

            config = AutoConfig.from_pretrained(
                base_model_name,
                pad_token_id=self.tokenizer.pad_token_id,
                trust_remote_code=self.model_args.trust_remote_code,
                cache_dir=self.model_args.cache_dir,
                use_cache=self.model_args.use_cache,
                revision=self.model_args.revision,
                token=cfg.token,
            )
            config.deterministic_flash_attention = True

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                trust_remote_code=self.model_args.trust_remote_code,
                config=config,
                token=cfg.token,
                cache_dir=self.model_args.cache_dir,
                revision=self.model_args.revision,
                quantization_config=bnb_config,
                attn_implementation=self.model_args.attn_implementation,
                torch_dtype=self.compute_dtype,
                low_cpu_mem_usage=self.model_args.low_cpu_mem_usage,
            )

            model = PeftModel.from_pretrained(base_model, self.model_path)
            logger.info("PEFT adapters loaded successfully")

        else:
            logger.info("Loading full fine-tuned model (no PEFT adapters)")

            config = AutoConfig.from_pretrained(
                self.model_path,
                pad_token_id=self.tokenizer.pad_token_id,
                trust_remote_code=self.model_args.trust_remote_code,
                cache_dir=self.model_args.cache_dir,
                use_cache=self.model_args.use_cache,
                token=cfg.token,
            )
            config.deterministic_flash_attention = True

            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=self.model_args.trust_remote_code,
                config=config,
                token=cfg.token,
                cache_dir=self.model_args.cache_dir,
                attn_implementation=self.model_args.attn_implementation,
                torch_dtype=self.compute_dtype,
                low_cpu_mem_usage=self.model_args.low_cpu_mem_usage,
            )

        return model

    def evaluate_perplexity(self):
        """Compute perplexity on validation set.
        
        Perplexity = exp(average cross-entropy loss).
        Lower perplexity means the model better predicts the text.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        val_data_path = os.path.join(project_root, self.data_args.validation_file)
        val_data = load_dataset(val_data_path)
        max_length = getattr(self.model_args, 'model_max_length', 2048)

        logger.info(f"Evaluating perplexity on {len(val_data)} samples")

        total_loss = 0.0
        total_tokens = 0

        for idx, sample in enumerate(val_data):
            text = sample["text"]

            tokenized = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )
                loss = outputs.loss

            # Number of tokens in this sample (excluding padding)
            num_tokens = attention_mask.sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            if (idx + 1) % 100 == 0:
                running_ppl = math.exp(total_loss / total_tokens)
                logger.info(f"  [{idx + 1}/{len(val_data)}] Running perplexity: {running_ppl:.4f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)

        logger.info(f"Average loss: {avg_loss:.4f}")
        logger.info(f"Perplexity: {perplexity:.4f}")
        logger.info(f"Total tokens evaluated: {total_tokens:,}")

        return {"perplexity": perplexity, "avg_loss": avg_loss, "total_tokens": total_tokens}

    def generate_text(self, prompt: str, max_new_tokens: int = 256):
        """Generate text continuation from a prompt.
        
        Useful for qualitatively assessing domain knowledge after CPT.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        max_length = getattr(self.model_args, 'model_max_length', 2048)

        tokenized = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}

        with torch.no_grad():
            output = self.model.generate(
                **tokenized,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
            )

        input_length = tokenized["input_ids"].shape[1]
        generated_tokens = output[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return generated_text

    def full_evaluation(self, generation_prompts: list = None):
        """Run full evaluation: perplexity + optional generation tests.
        
        Args:
            generation_prompts: List of prompts to test text generation.
                If None, skips generation testing.
        """
        ppl_result = self.evaluate_perplexity()

        generation_results = []
        if generation_prompts:
            logger.info(f"Running generation tests with {len(generation_prompts)} prompts")
            for i, prompt in enumerate(generation_prompts):
                generated = self.generate_text(prompt)
                generation_results.append({
                    "prompt": prompt,
                    "generated": generated,
                })
                logger.info(f"  Prompt {i+1}: {prompt[:80]}...")
                logger.info(f"  Generated: {generated[:200]}...")

        results = {
            "model_path": self.model_path,
            "perplexity": ppl_result,
            "generation_results": generation_results,
        }

        results_file = "cpt_evaluation_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Evaluation results saved to {results_file}")

        # 4. Print report
        print(f"\n{'='*60}")
        print(f"{'CPT Evaluation Results':^60}")
        print(f"{'='*60}")
        print(f"Model: {self.model_path}")
        print(f"Perplexity:     {ppl_result['perplexity']:.4f}")
        print(f"Average Loss:   {ppl_result['avg_loss']:.4f}")
        print(f"Total Tokens:   {ppl_result['total_tokens']:,}")
        if generation_results:
            print(f"\nGeneration Tests: {len(generation_results)}")
            for i, gen in enumerate(generation_results):
                print(f"\n  [{i+1}] Prompt: {gen['prompt'][:80]}...")
                print(f"      Output: {gen['generated'][:200]}...")
        print(f"{'='*60}")

        return results


@hydra.main(version_base=None, config_path="../../conf", config_name="cpt")
def main(cfg: DictConfig):
    set_seed(cfg.seed, deterministic=True)

    logger.info(f"Starting CPT evaluation with model from: {cfg.training_arguments.output_dir}")

    evaluator = CPTEvaluator(cfg=cfg)

    generation_prompts = [
        # Optional: add domain-specific generation prompts for testing
    ]

    evaluator.full_evaluation(
        generation_prompts=generation_prompts if generation_prompts else None
    )


if __name__ == "__main__":
    main()
