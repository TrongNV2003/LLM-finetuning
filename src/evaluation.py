import os
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import json
import hydra
import torch
import asyncio
from loguru import logger
from dotenv import load_dotenv
from omegaconf import DictConfig
from concurrent.futures import ThreadPoolExecutor
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    set_seed,
)
from peft import PeftModel, PeftConfig

from src.utils import load_dataset
from src.metrics import EvaluateMetrics
from src.prompts import PROMPT_TEMPLATE

load_dotenv()


class LLMEvaluator:
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
        self.metrics = EvaluateMetrics()
        
        logger.info(f"Model loaded from: {self.model_path}")
        
    def _load_model(self, cfg: DictConfig):
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
            logger.info("✅ PEFT adapters loaded successfully")
            
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

    def format_prompt(self, text: str) -> str:
        """Format prompt exactly like in training"""
        prompt_str = PROMPT_TEMPLATE.format(text=text)
        messages = [
            {"role": "user", "content": prompt_str}
        ]
            
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

    async def evaluate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        
        test_data_path = os.path.join(parent_dir, self.data_args.test_file)
        test_data = load_dataset(test_data_path)

        max_concurrent_queries = self.cfg.evaluation_arguments.max_concurrent_queries
        
        logger.info(f"Processing {len(test_data)} samples with max {max_concurrent_queries} concurrent queries")
        
        semaphore = asyncio.Semaphore(max_concurrent_queries)
        
        async def predict(idx, sample):
            async with semaphore:
                return await self._predict(idx, sample, device)
        
        tasks = [predict(idx, sample) for idx, sample in enumerate(test_data)]
        
        query_results = await asyncio.gather(*tasks)
        
        predictions = []
        references = []
        inputs = []
        
        for idx, pred, ref, inp in sorted(query_results, key=lambda x: x[0]):
            predictions.append(pred)
            references.append(ref)
            inputs.append(inp)
        
        result = self.metrics.metrics_evaluate(predictions, references)
        self.metrics.print_evaluation_report(result, "Model Evaluation Results")

        detailed_results = []
        for i, (inp, pred, ref) in enumerate(zip(inputs, predictions, references)):
            detailed_results.append({
                'id': i,
                'input': inp,
                'prediction': pred,
                'reference': ref,
                'exact_match': pred.strip().lower() == ref.strip().lower()
            })
        
        results_file = "evaluation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'model_path': self.model_path,
                'metrics': {
                    'exact_match': result.exact_match,
                    'bleu_score': result.bleu_score,
                    'rouge_l': result.rouge_l,
                    'lexical_similarity': result.lexical_similarity,
                    'num_samples': result.num_samples
                },
                'detailed_results': detailed_results
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Evaluation results saved to {results_file}")
        
        return result

    async def _predict(self, idx, sample, device):
        """Process a single query asynchronously"""
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = loop.run_in_executor(
                executor, 
                self._run_single_inference, 
                idx, sample, device
            )
            return await future
    
    def _run_single_inference(self, idx, sample, device):
        text = sample['text']
        label = sample['label']
        
        prompt = self.format_prompt(text)
        eval_args = self.cfg.evaluation_arguments
        
        tokenized = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        
        try:
            with torch.no_grad():
                autocast_enabled = torch.cuda.is_available() and self.compute_dtype in (torch.float16, torch.bfloat16)
                with torch.autocast(device_type='cuda' if autocast_enabled else 'cpu', 
                                  dtype=self.compute_dtype, enabled=autocast_enabled):
                    output = self.model.generate(
                        **tokenized,
                        max_new_tokens=eval_args.max_new_tokens,
                        temperature=eval_args.temperature,
                        top_p=eval_args.top_p,
                        top_k=eval_args.top_k,
                        repetition_penalty=eval_args.repetition_penalty,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        do_sample=eval_args.do_sample,
                    )
            
            input_length = tokenized["input_ids"].shape[1]
            generated_tokens = output[0][input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            if generated_text.startswith("### Địa chỉ hoàn thiện:"):
                generated_text = generated_text.replace("### Địa chỉ hoàn thiện:", "").strip()
            elif generated_text.startswith("Địa chỉ hoàn thiện:"):
                generated_text = generated_text.replace("Địa chỉ hoàn thiện:", "").strip()

            logger.debug(f"Sample {idx}: Input='{text}', Output='{generated_text}'")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return idx, generated_text, label, text
            
        except RuntimeError as e:
            logger.error(f"Error processing sample {idx}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise e

    def evaluate_single(self, text: str) -> str:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        
        eval_args = self.cfg.evaluation_arguments
        prompt = self.format_prompt(text)
        
        tokenized = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        
        with torch.no_grad():
            autocast_enabled = torch.cuda.is_available() and self.compute_dtype in (torch.float16, torch.bfloat16)
            with torch.autocast(device_type='cuda' if autocast_enabled else 'cpu', 
                              dtype=self.compute_dtype, enabled=autocast_enabled):
                output = self.model.generate(
                    **tokenized,
                    max_new_tokens=eval_args.max_new_tokens,
                    temperature=eval_args.temperature,
                    top_p=eval_args.top_p,
                    top_k=eval_args.top_k,
                    repetition_penalty=eval_args.repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    do_sample=eval_args.do_sample,
                )
        
        input_length = tokenized["input_ids"].shape[1]
        generated_tokens = output[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        if generated_text.startswith("### Địa chỉ hoàn thiện:"):
            generated_text = generated_text.replace("### Địa chỉ hoàn thiện:", "").strip()
        elif generated_text.startswith("Địa chỉ hoàn thiện:"):
            generated_text = generated_text.replace("Địa chỉ hoàn thiện:", "").strip()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return generated_text


@hydra.main(version_base=None, config_path="conf", config_name="finetuning")
def main(cfg: DictConfig):
    set_seed(cfg.seed, deterministic=True)

    model_path = cfg.training_arguments.output_dir

    logger.info(f"Starting evaluation with model from: {model_path}")
    
    evaluator = LLMEvaluator(cfg=cfg)

    asyncio.run(evaluator.evaluate())

if __name__ == "__main__":
    main()