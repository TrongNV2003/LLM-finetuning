import os
import sys
import json
import torch
import argparse
from loguru import logger
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.metrics import EvaluateMetrics
from src.prompts import PROMPT_TEMPLATE


def load_trained_model(base_model_name: str, adapter_path: str):
    logger.info(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    logger.info(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    return model, tokenizer


def generate_prediction(model, tokenizer, text: str, max_new_tokens: int = 50) -> str:
    """Generate prediction for a single input"""

    prompt = PROMPT_TEMPLATE.format(text=text, label="")

    inputs = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.05
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    prompt_length = len(prompt)
    generated_part = generated_text[prompt_length:].strip()
    
    return generated_part


def evaluate_on_dataset(model, tokenizer, dataset_path: str, output_dir: str):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Evaluating on {len(data)} samples...")
    
    predictions = []
    references = []
    inputs = []
    
    for i, item in enumerate(data):
        if i % 10 == 0:
            logger.info(f"Processing {i}/{len(data)}...")
        
        input_text = item['text']
        reference = item['label']
        
        prediction = generate_prediction(model, tokenizer, input_text)
        
        predictions.append(prediction)
        references.append(reference)
        inputs.append(input_text)
    
    metrics_calculator = EvaluateMetrics()
    result = metrics_calculator.metrics_evaluate(predictions, references)
    
    metrics_calculator.print_evaluation_report(result, "Model Evaluation Results")
    
    os.makedirs(output_dir, exist_ok=True)
    
    detailed_results = []
    for i, (inp, pred, ref) in enumerate(zip(inputs, predictions, references)):
        detailed_results.append({
            'id': i,
            'input': inp,
            'prediction': pred,
            'reference': ref,
            'exact_match': pred.strip().lower() == ref.strip().lower()
        })
    
    results_file = os.path.join(output_dir, "evaluation_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': {
                'exact_match': result.exact_match,
                'bleu_score': result.bleu_score,
                'rouge_l': result.rouge_l,
                'semantic_similarity': result.semantic_similarity,
                'num_samples': result.num_samples
            },
            'detailed_results': detailed_results
        }, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    errors = [item for item in detailed_results if not item['exact_match']]
    error_analysis = {
        'total_samples': len(detailed_results),
        'correct_predictions': len(detailed_results) - len(errors),
        'incorrect_predictions': len(errors),
        'accuracy': (len(detailed_results) - len(errors)) / len(detailed_results),
        'error_examples': errors[:10]
    }
    
    error_file = os.path.join(output_dir, "error_analysis.json")
    with open(error_file, 'w', encoding='utf-8') as f:
        json.dump(error_analysis, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Error analysis saved to {error_file}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained address denoising model")
    parser.add_argument("--base_model", required=True, help="Base model name (e.g., microsoft/DialoGPT-medium)")
    parser.add_argument("--adapter_path", required=True, help="Path to LoRA adapter")
    parser.add_argument("--test_data", required=True, help="Path to test dataset JSON file")
    parser.add_argument("--output_dir", default="evaluation_results", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        model, tokenizer = load_trained_model(args.base_model, args.adapter_path)
        
        result = evaluate_on_dataset(model, tokenizer, args.test_data, args.output_dir)
        
        logger.info("Evaluation completed successfully!")
        logger.info(f"Overall score: {result.overall_score:.4f}")
        logger.info(f"Exact match: {result.exact_match:.4f}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
