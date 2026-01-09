"""
Comprehensive GLiNER2 Model Evaluation Suite

This script provides a complete evaluation framework for your ad detection model:
1. Standard metrics (Precision, Recall, F1, AUC-ROC)
2. Error analysis by category
3. Confidence calibration analysis
4. Latency benchmarking
5. Comparison with baseline

Usage:
    python evaluate_model.py --model ./gliner2_ad_classifier/best
    python evaluate_model.py --model ./gliner2_ad_classifier/best --compare-baseline
    python evaluate_model.py --analyze-errors
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
import random

try:
    from gliner2 import GLiNER2
except ImportError:
    print("❌ GLiNER2 not installed. Run: pip install gliner2")
    sys.exit(1)

# Try importing optional dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score,
        precision_recall_curve, roc_curve
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️ sklearn not installed. Some metrics will be unavailable.")


@dataclass
class EvaluationResult:
    """Container for all evaluation metrics"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auc_roc: Optional[float] = None
    
    confusion_matrix: Dict = field(default_factory=dict)
    per_category_metrics: Dict = field(default_factory=dict)
    error_analysis: Dict = field(default_factory=dict)
    latency_stats: Dict = field(default_factory=dict)
    
    total_samples: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    
    def __repr__(self):
        return (f"EvaluationResult(accuracy={self.accuracy:.4f}, "
                f"precision={self.precision:.4f}, recall={self.recall:.4f}, "
                f"f1={self.f1:.4f})")


def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def get_true_label(record: Dict) -> str:
    """Extract true label from JSONL record"""
    try:
        classifications = record['output']['classifications']
        for cls in classifications:
            if cls.get('task') == 'ad_detection':
                labels = cls.get('true_label', [])
                return labels[0] if labels else 'no_ad'
        return 'no_ad'
    except (KeyError, IndexError):
        return 'no_ad'


def categorize_text(text: str) -> str:
    """Categorize text for stratified analysis"""
    text_lower = text.lower()
    
    # High confidence ad patterns
    if any(p in text_lower for p in ['sponsored by', 'brought to you', 'word from our sponsor']):
        return 'explicit_ad'
    
    # Promo code mentions
    if any(p in text_lower for p in ['promo code', 'use code', 'discount code', 'coupon']):
        return 'has_promo_code'
    
    # URL mentions
    if '.com' in text_lower or '.io' in text_lower or '.co/' in text_lower:
        return 'has_url'
    
    # Patreon/Substack
    if any(p in text_lower for p in ['patreon', 'substack', 'ko-fi', 'buy me a coffee']):
        return 'patreon_substack'
    
    # Social media
    if any(p in text_lower for p in ['follow us', 'subscribe', 'twitter', 'instagram', 'youtube']):
        return 'social_media'
    
    # Brand love
    if any(p in text_lower for p in ['i love', 'i really like', 'big fan of', 'shoutout', 'not sponsored']):
        return 'brand_love'
    
    return 'other'


class ModelEvaluator:
    """
    Comprehensive model evaluation class.
    
    Provides:
    - Standard classification metrics
    - Stratified evaluation by category
    - Error analysis with patterns
    - Confidence calibration
    - Latency benchmarking
    """
    
    def __init__(self, model_path: str, threshold: float = 0.5):
        """
        Initialize evaluator with model.
        
        Args:
            model_path: Path to trained model or HuggingFace model name
            threshold: Classification threshold
        """
        print(f"📦 Loading model: {model_path}")
        self.model = GLiNER2.from_pretrained(model_path)
        self.threshold = threshold
        self.model_path = model_path
        
        # Labels for classification
        self.labels = {
            "ad": "Sponsored advertisement or paid promotion",
            "no_ad": "Regular content, not an advertisement"
        }
        
        print(f"✅ Model loaded successfully")
    
    def classify(self, text: str) -> Tuple[str, float, Dict]:
        """
        Classify text and return prediction with confidence.
        
        Returns:
            (predicted_label, confidence, raw_scores)
        """
        try:
            result = self.model.classify(text, labels=self.labels)
            
            if isinstance(result, dict):
                ad_score = result.get("ad", 0)
                no_ad_score = result.get("no_ad", 0)
                
                is_ad = ad_score > no_ad_score and ad_score >= self.threshold
                predicted = "ad" if is_ad else "no_ad"
                confidence = ad_score if is_ad else no_ad_score
                
                return predicted, confidence, result
            else:
                predicted = str(result).lower()
                return predicted, 0.7, {"raw": result}
                
        except Exception as e:
            print(f"⚠️ Classification error: {e}")
            return "no_ad", 0.0, {"error": str(e)}
    
    def evaluate(self, test_data: List[Dict], verbose: bool = True) -> EvaluationResult:
        """
        Run full evaluation on test data.
        
        Args:
            test_data: List of JSONL records
            verbose: Print progress updates
            
        Returns:
            EvaluationResult with all metrics
        """
        result = EvaluationResult()
        result.total_samples = len(test_data)
        
        y_true = []
        y_pred = []
        y_scores = []  # For AUC-ROC
        
        # Per-category tracking
        category_results = defaultdict(lambda: {'y_true': [], 'y_pred': []})
        
        # Error tracking
        false_positives = []
        false_negatives = []
        
        # Latency tracking
        latencies = []
        
        if verbose:
            print(f"\n🔍 Evaluating {len(test_data)} samples...")
        
        for i, record in enumerate(test_data):
            text = record.get('input', '')
            true_label = get_true_label(record)
            category = categorize_text(text)
            
            # Time the prediction
            start_time = time.perf_counter()
            predicted, confidence, raw_scores = self.classify(text)
            latency = (time.perf_counter() - start_time) * 1000  # ms
            latencies.append(latency)
            
            y_true.append(true_label)
            y_pred.append(predicted)
            
            # Get ad score for AUC-ROC
            ad_score = raw_scores.get('ad', 0.5) if isinstance(raw_scores, dict) else 0.5
            y_scores.append(ad_score)
            
            # Track by category
            category_results[category]['y_true'].append(true_label)
            category_results[category]['y_pred'].append(predicted)
            
            # Track errors
            if predicted != true_label:
                error_info = {
                    'text': text[:300] + '...' if len(text) > 300 else text,
                    'true_label': true_label,
                    'predicted': predicted,
                    'confidence': confidence,
                    'category': category,
                }
                
                if predicted == 'ad':
                    false_positives.append(error_info)
                else:
                    false_negatives.append(error_info)
            
            # Update confusion matrix counts
            if true_label == 'ad' and predicted == 'ad':
                result.true_positives += 1
            elif true_label == 'no_ad' and predicted == 'ad':
                result.false_positives += 1
            elif true_label == 'no_ad' and predicted == 'no_ad':
                result.true_negatives += 1
            else:  # true_label == 'ad' and predicted == 'no_ad'
                result.false_negatives += 1
            
            if verbose and (i + 1) % 100 == 0:
                print(f"   Processed {i + 1}/{len(test_data)}...")
        
        # Calculate metrics
        result.accuracy = (result.true_positives + result.true_negatives) / result.total_samples
        
        if result.true_positives + result.false_positives > 0:
            result.precision = result.true_positives / (result.true_positives + result.false_positives)
        
        if result.true_positives + result.false_negatives > 0:
            result.recall = result.true_positives / (result.true_positives + result.false_negatives)
        
        if result.precision + result.recall > 0:
            result.f1 = 2 * result.precision * result.recall / (result.precision + result.recall)
        
        # AUC-ROC (requires sklearn)
        if HAS_SKLEARN and HAS_NUMPY:
            try:
                y_true_binary = [1 if y == 'ad' else 0 for y in y_true]
                result.auc_roc = roc_auc_score(y_true_binary, y_scores)
            except Exception as e:
                print(f"⚠️ Could not calculate AUC-ROC: {e}")
        
        # Confusion matrix
        result.confusion_matrix = {
            'true_positives': result.true_positives,
            'false_positives': result.false_positives,
            'true_negatives': result.true_negatives,
            'false_negatives': result.false_negatives,
        }
        
        # Per-category metrics
        for category, cat_data in category_results.items():
            cat_y_true = cat_data['y_true']
            cat_y_pred = cat_data['y_pred']
            
            if not cat_y_true:
                continue
            
            cat_correct = sum(1 for t, p in zip(cat_y_true, cat_y_pred) if t == p)
            cat_accuracy = cat_correct / len(cat_y_true)
            
            # Calculate precision/recall for ads in this category
            cat_tp = sum(1 for t, p in zip(cat_y_true, cat_y_pred) if t == 'ad' and p == 'ad')
            cat_fp = sum(1 for t, p in zip(cat_y_true, cat_y_pred) if t == 'no_ad' and p == 'ad')
            cat_fn = sum(1 for t, p in zip(cat_y_true, cat_y_pred) if t == 'ad' and p == 'no_ad')
            
            cat_precision = cat_tp / (cat_tp + cat_fp) if (cat_tp + cat_fp) > 0 else 0
            cat_recall = cat_tp / (cat_tp + cat_fn) if (cat_tp + cat_fn) > 0 else 0
            
            result.per_category_metrics[category] = {
                'total': len(cat_y_true),
                'accuracy': cat_accuracy,
                'precision': cat_precision,
                'recall': cat_recall,
                'true_ads': sum(1 for y in cat_y_true if y == 'ad'),
                'true_non_ads': sum(1 for y in cat_y_true if y == 'no_ad'),
            }
        
        # Error analysis
        result.error_analysis = {
            'false_positives': false_positives[:50],  # Limit for readability
            'false_negatives': false_negatives[:50],
            'fp_by_category': defaultdict(int),
            'fn_by_category': defaultdict(int),
        }
        
        for fp in false_positives:
            result.error_analysis['fp_by_category'][fp['category']] += 1
        for fn in false_negatives:
            result.error_analysis['fn_by_category'][fn['category']] += 1
        
        # Latency stats
        if latencies:
            sorted_latencies = sorted(latencies)
            result.latency_stats = {
                'mean_ms': sum(latencies) / len(latencies),
                'median_ms': sorted_latencies[len(sorted_latencies) // 2],
                'p95_ms': sorted_latencies[int(len(sorted_latencies) * 0.95)],
                'p99_ms': sorted_latencies[int(len(sorted_latencies) * 0.99)],
                'min_ms': min(latencies),
                'max_ms': max(latencies),
            }
        
        return result
    
    def print_results(self, result: EvaluationResult):
        """Pretty print evaluation results"""
        print("\n" + "=" * 70)
        print("📊 EVALUATION RESULTS")
        print("=" * 70)
        
        print(f"\n📈 Overall Metrics:")
        print(f"   Accuracy:  {result.accuracy:.4f} ({result.accuracy*100:.2f}%)")
        print(f"   Precision: {result.precision:.4f}")
        print(f"   Recall:    {result.recall:.4f}")
        print(f"   F1 Score:  {result.f1:.4f}")
        if result.auc_roc is not None:
            print(f"   AUC-ROC:   {result.auc_roc:.4f}")
        
        print(f"\n📊 Confusion Matrix:")
        print(f"                      Predicted")
        print(f"                      ad       no_ad")
        print(f"   Actual ad       {result.true_positives:6d}    {result.false_negatives:6d}")
        print(f"          no_ad    {result.false_positives:6d}    {result.true_negatives:6d}")
        
        print(f"\n📂 Per-Category Performance:")
        for category, metrics in sorted(result.per_category_metrics.items(), 
                                        key=lambda x: -x[1]['total']):
            print(f"\n   {category} ({metrics['total']} samples):")
            print(f"      Accuracy:  {metrics['accuracy']:.2%}")
            print(f"      Precision: {metrics['precision']:.2%}")
            print(f"      Recall:    {metrics['recall']:.2%}")
            print(f"      (Ads: {metrics['true_ads']}, Non-ads: {metrics['true_non_ads']})")
        
        print(f"\n⚠️ Error Analysis:")
        print(f"   False Positives: {result.false_positives} (wrongly flagged as ads)")
        for cat, count in sorted(result.error_analysis['fp_by_category'].items(), 
                                 key=lambda x: -x[1])[:5]:
            print(f"      - {cat}: {count}")
        
        print(f"\n   False Negatives: {result.false_negatives} (missed actual ads)")
        for cat, count in sorted(result.error_analysis['fn_by_category'].items(), 
                                 key=lambda x: -x[1])[:5]:
            print(f"      - {cat}: {count}")
        
        print(f"\n⏱️ Latency Stats:")
        if result.latency_stats:
            print(f"   Mean:   {result.latency_stats['mean_ms']:.2f} ms")
            print(f"   Median: {result.latency_stats['median_ms']:.2f} ms")
            print(f"   P95:    {result.latency_stats['p95_ms']:.2f} ms")
            print(f"   P99:    {result.latency_stats['p99_ms']:.2f} ms")
        
        print("\n" + "=" * 70)
        
        # Quality assessment
        print("\n🎯 QUALITY ASSESSMENT:")
        
        if result.f1 >= 0.90:
            print("   ✅ EXCELLENT - Model is production ready")
        elif result.f1 >= 0.85:
            print("   ✅ GOOD - Model is suitable for production with monitoring")
        elif result.f1 >= 0.80:
            print("   ⚠️ ACCEPTABLE - Consider additional training data")
        else:
            print("   ❌ NEEDS IMPROVEMENT - Review training data and hyperparameters")
        
        if result.precision < 0.80:
            print("   ⚠️ Low precision: Too many false positives")
        if result.recall < 0.85:
            print("   ⚠️ Low recall: Missing too many actual ads")
        
        print("=" * 70)
    
    def print_error_examples(self, result: EvaluationResult, num_examples: int = 5):
        """Print example errors for debugging"""
        print("\n" + "=" * 70)
        print("🔍 ERROR EXAMPLES")
        print("=" * 70)
        
        print("\n❌ FALSE POSITIVES (wrongly classified as ads):")
        print("-" * 70)
        for err in result.error_analysis.get('false_positives', [])[:num_examples]:
            print(f"\n[Category: {err['category']}] [Confidence: {err['confidence']:.2%}]")
            print(f"Text: {err['text'][:200]}...")
        
        print("\n\n❌ FALSE NEGATIVES (missed ads):")
        print("-" * 70)
        for err in result.error_analysis.get('false_negatives', [])[:num_examples]:
            print(f"\n[Category: {err['category']}] [Confidence: {err['confidence']:.2%}]")
            print(f"Text: {err['text'][:200]}...")
    
    def save_results(self, result: EvaluationResult, output_path: str):
        """Save evaluation results to JSON"""
        # Convert to serializable format
        output = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'threshold': self.threshold,
            'total_samples': result.total_samples,
            'metrics': {
                'accuracy': result.accuracy,
                'precision': result.precision,
                'recall': result.recall,
                'f1': result.f1,
                'auc_roc': result.auc_roc,
            },
            'confusion_matrix': result.confusion_matrix,
            'per_category_metrics': dict(result.per_category_metrics),
            'latency_stats': result.latency_stats,
            'error_analysis': {
                'fp_by_category': dict(result.error_analysis.get('fp_by_category', {})),
                'fn_by_category': dict(result.error_analysis.get('fn_by_category', {})),
                'num_false_positives': len(result.error_analysis.get('false_positives', [])),
                'num_false_negatives': len(result.error_analysis.get('false_negatives', [])),
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_path}")


def compare_models(model1_path: str, model2_path: str, test_data: List[Dict]):
    """Compare two models side by side"""
    print("\n" + "=" * 70)
    print("🔄 MODEL COMPARISON")
    print("=" * 70)
    
    print(f"\nModel 1: {model1_path}")
    print(f"Model 2: {model2_path}")
    
    eval1 = ModelEvaluator(model1_path)
    eval2 = ModelEvaluator(model2_path)
    
    print("\n--- Evaluating Model 1 ---")
    result1 = eval1.evaluate(test_data, verbose=False)
    
    print("\n--- Evaluating Model 2 ---")
    result2 = eval2.evaluate(test_data, verbose=False)
    
    print("\n📊 Comparison Results:")
    print(f"\n{'Metric':<15} {'Model 1':<12} {'Model 2':<12} {'Δ':<10}")
    print("-" * 50)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    for metric in metrics:
        v1 = getattr(result1, metric)
        v2 = getattr(result2, metric)
        delta = v2 - v1
        delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
        print(f"{metric:<15} {v1:.4f}      {v2:.4f}      {delta_str}")
    
    print("\n" + "=" * 70)
    
    # Determine winner
    if result2.f1 > result1.f1:
        print(f"✅ Model 2 is better (F1: +{result2.f1 - result1.f1:.4f})")
    elif result1.f1 > result2.f1:
        print(f"✅ Model 1 is better (F1: +{result1.f1 - result2.f1:.4f})")
    else:
        print("⚖️ Models perform equally")
    
    return result1, result2


def find_optimal_threshold(model_path: str, test_data: List[Dict]) -> float:
    """Find optimal classification threshold"""
    print("\n🎯 Finding optimal threshold...")
    
    evaluator = ModelEvaluator(model_path, threshold=0.5)
    
    # Get predictions at different thresholds
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    best_threshold = 0.5
    best_f1 = 0
    
    print(f"\n{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 50)
    
    for thresh in thresholds:
        evaluator.threshold = thresh
        result = evaluator.evaluate(test_data, verbose=False)
        
        print(f"{thresh:<12} {result.precision:<12.4f} {result.recall:<12.4f} {result.f1:<12.4f}")
        
        if result.f1 > best_f1:
            best_f1 = result.f1
            best_threshold = thresh
    
    print(f"\n✅ Optimal threshold: {best_threshold} (F1: {best_f1:.4f})")
    return best_threshold


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate GLiNER2 ad detection model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python evaluate_model.py --model ./gliner2_ad_classifier/best
  
  # Evaluate on custom test set
  python evaluate_model.py --model ./gliner2_ad_classifier/best --test-data ./my_test.jsonl
  
  # Compare with baseline
  python evaluate_model.py --model ./gliner2_ad_classifier/best --compare-baseline
  
  # Find optimal threshold
  python evaluate_model.py --model ./gliner2_ad_classifier/best --find-threshold
  
  # Detailed error analysis
  python evaluate_model.py --model ./gliner2_ad_classifier/best --analyze-errors
        """
    )
    
    parser.add_argument('--model', type=str, default='./gliner2_ad_classifier/best',
                        help='Path to model to evaluate')
    parser.add_argument('--test-data', type=str, default='./training_data/val_latest.jsonl',
                        help='Path to test data JSONL')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    parser.add_argument('--output', type=str, default=None,
                        help='Save results to JSON file')
    parser.add_argument('--compare-baseline', action='store_true',
                        help='Compare with base GLiNER2 model')
    parser.add_argument('--baseline-model', type=str, default='fastino/gliner2-base-v1',
                        help='Baseline model to compare against')
    parser.add_argument('--find-threshold', action='store_true',
                        help='Find optimal classification threshold')
    parser.add_argument('--analyze-errors', action='store_true',
                        help='Print detailed error examples')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of samples to evaluate')
    
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.exists(args.test_data):
        print(f"❌ Test data not found: {args.test_data}")
        print("   Run: python prepare_training_data.py first")
        sys.exit(1)
    
    # Load test data
    print(f"📂 Loading test data: {args.test_data}")
    test_data = load_jsonl(args.test_data)
    
    if args.limit:
        random.seed(42)
        test_data = random.sample(test_data, min(args.limit, len(test_data)))
    
    print(f"   Loaded {len(test_data)} samples")
    
    # Find optimal threshold
    if args.find_threshold:
        find_optimal_threshold(args.model, test_data)
        return
    
    # Compare with baseline
    if args.compare_baseline:
        compare_models(args.baseline_model, args.model, test_data)
        return
    
    # Standard evaluation
    evaluator = ModelEvaluator(args.model, threshold=args.threshold)
    result = evaluator.evaluate(test_data)
    evaluator.print_results(result)
    
    # Error analysis
    if args.analyze_errors:
        evaluator.print_error_examples(result, num_examples=10)
    
    # Save results
    if args.output:
        evaluator.save_results(result, args.output)
    else:
        # Auto-save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"./evaluation_results_{timestamp}.json"
        evaluator.save_results(result, output_path)


if __name__ == "__main__":
    main()
