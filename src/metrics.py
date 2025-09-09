import json
import numpy as np
from typing import List
from loguru import logger
from pyvi import ViTokenizer
from bert_score import score
from dataclasses import dataclass
from difflib import SequenceMatcher
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

@dataclass
class EvaluationResult:
    exact_match: float
    bleu_score: float
    rouge_l: float
    semantic_similarity: float
    num_samples: int


class EvaluateMetrics:
    def __init__(self):
        pass
    
    def _word_tokenize(self, text: str) -> List[str]:
        return ViTokenizer.tokenize(text.strip()).split()

    def exact_match(self, predictions: List[str], references: List[str]) -> float:
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
        
        matches = sum(1 for pred, ref in zip(predictions, references) 
                     if pred.strip().lower() == ref.strip().lower())
        return matches / len(predictions)
    
    def bleu_score(self, predictions: List[str], references: List[str]) -> float:
        smoothing = SmoothingFunction().method1
        scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = self._word_tokenize(pred.lower())
            ref_tokens = [self._word_tokenize(ref.lower())]

            score = sentence_bleu(ref_tokens, pred_tokens,
                                smoothing_function=smoothing)
            scores.append(score)
        
        return np.mean(scores)
    
    def rouge_l_score(self, predictions: List[str], references: List[str]) -> float:
        scores = []
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.lower().split()
            ref_tokens = ref.lower().split()
            
            if len(pred_tokens) == 0 or len(ref_tokens) == 0:
                scores.append(0.0)
                continue
            
            lcs_length = self._lcs_length(pred_tokens, ref_tokens)
            
            if len(pred_tokens) > 0 and len(ref_tokens) > 0:
                precision = lcs_length / len(pred_tokens)
                recall = lcs_length / len(ref_tokens)
                
                if precision + recall > 0:
                    f_score = 2 * precision * recall / (precision + recall)
                else:
                    f_score = 0.0
            else:
                f_score = 0.0
            
            scores.append(f_score)
        
        return np.mean(scores)
    
    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate longest common subsequence length"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def semantic_similarity_score(self, predictions: List[str], references: List[str]) -> float:
        """Calculate semantic similarity using string matching"""
        scores = []
        
        for pred, ref in zip(predictions, references):
            similarity = SequenceMatcher(None, pred.lower(), ref.lower()).ratio()
            scores.append(similarity)
        
        return np.mean(scores)

    def metrics_evaluate(self, predictions: List[str], references: List[str]) -> EvaluationResult:
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")
        
        exact_match = self.exact_match(predictions, references)
        bleu = self.bleu_score(predictions, references)
        rouge_l = self.rouge_l_score(predictions, references)
        semantic_sim = self.semantic_similarity_score(predictions, references)
        
        return EvaluationResult(
            exact_match=exact_match,
            bleu_score=bleu,
            rouge_l=rouge_l,
            semantic_similarity=semantic_sim,
            num_samples=len(predictions)
        )
    
    def print_evaluation_report(self, result: EvaluationResult, title: str = "Evaluation Results"):
        """Print a formatted evaluation report"""
        print(f"\n{'='*60}")
        print(f"{title:^60}")
        print(f"{'='*60}")
        print(f"Number of samples: {result.num_samples}")
        print(f"{'='*60}")
        
        print(f"Exact Match Accuracy:    {result.exact_match:.4f}")
        print(f"Semantic Similarity:     {result.semantic_similarity:.4f}")
        print(f"BLEU Score:              {result.bleu_score:.4f}")
        print(f"ROUGE-L Score:           {result.rouge_l:.4f}")

def compute_metrics_fn(tokenizer):
    """Create a compute_metrics function for use with Transformers Trainer"""
    metrics_calculator = EvaluateMetrics()
    
    def compute_metrics(eval_pred):
        try:
            predictions, labels = eval_pred
            
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            
            if predictions.ndim > 2:
                predictions = np.argmax(predictions, axis=-1)
            
            predictions = predictions.astype(np.int32)
            
            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            labels = labels.astype(np.int32)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            processed_preds = []
            processed_labels = []
            
            for pred, label in zip(decoded_preds, decoded_labels):
                pred_split = pred.split("### Địa chỉ hoàn thiện:")
                label_split = label.split("### Địa chỉ hoàn thiện:")
                
                if len(pred_split) > 1:
                    processed_preds.append(pred_split[-1].strip())
                else:
                    processed_preds.append(pred.strip())
                    
                if len(label_split) > 1:
                    processed_labels.append(label_split[-1].strip())
                else:
                    processed_labels.append(label.strip())
            
            result = metrics_calculator.metrics_evaluate(processed_preds, processed_labels)
            
            return {
                'exact_match': result.exact_match,
                'bleu_score': result.bleu_score,
                'rouge_l': result.rouge_l,
                'semantic_similarity': result.semantic_similarity,
            }
            
        except Exception as e:
            logger.error(f"Error computing metrics: {e}")
            logger.error(f"Predictions type: {type(predictions)}, shape: {getattr(predictions, 'shape', 'N/A')}")
            logger.error(f"Labels type: {type(labels)}, shape: {getattr(labels, 'shape', 'N/A')}")
            return {
                'exact_match': 0.0,
                'bleu_score': 0.0,
                'rouge_l': 0.0,
                'semantic_similarity': 0.0,
            }
    
    return compute_metrics


if __name__ == "__main__":
    predictions = [
        "68, Xuân Thủy, Cầu Giấy, Hà Nội",
        "123, Nguyễn Du, Q1, HCM",
        "45, Lê Thánh, Hoàn Kiếm, Hà Nội"
    ]
    
    references = [
        "68, Xuân Thủy, Cầu Giấy, Hà Nội",
        "123, Nguyễn Du, Quận 1, Hồ Chí Minh", 
        "45, Lê Thánh Tông, Hoàn Kiếm, Hà Nội"
    ]
    
    metrics = EvaluateMetrics()
    result = metrics.metrics_evaluate(predictions, references)
    metrics.print_evaluation_report(result, "Test Evaluation")
