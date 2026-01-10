#!/usr/bin/env python3
"""
Inference script for the fine-tuned DeBERTa Ad Classifier.

This script loads a trained model and makes predictions on new text.

Usage:
    python predict.py --model_path deberta_ad_classifier/output/run_XXXXX/best_model --text "Your text here"
    python predict.py --model_path best_model --input_file texts.txt
    python predict.py --model_path best_model --interactive
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class AdDetector:
    """Ad Detection model using fine-tuned DeBERTa."""
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        """
        Initialize the Ad Detector.
        
        Args:
            model_path: Path to the fine-tuned model directory
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            max_length: Maximum sequence length for tokenization
        """
        self.model_path = model_path
        self.max_length = max_length
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Get label mapping
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
        
        logger.info(f"Model loaded successfully. Labels: {self.id2label}")
    
    def predict(
        self,
        text: Union[str, List[str]],
        return_probabilities: bool = True,
    ) -> Union[Dict, List[Dict]]:
        """
        Make predictions on input text.
        
        Args:
            text: Single text string or list of texts
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary or list of dictionaries with predictions
        """
        single_input = isinstance(text, str)
        if single_input:
            texts = [text]
        else:
            texts = text
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        # Process results
        results = []
        for i, (probs, logit) in enumerate(zip(probabilities, logits)):
            pred_id = torch.argmax(probs).item()
            pred_label = self.id2label[pred_id]
            confidence = probs[pred_id].item()
            
            result = {
                "text": texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i],
                "prediction": pred_label,
                "confidence": round(confidence, 4),
                "is_ad": pred_label == "ad",
            }
            
            if return_probabilities:
                result["probabilities"] = {
                    self.id2label[j]: round(p.item(), 4)
                    for j, p in enumerate(probs)
                }
            
            results.append(result)
        
        return results[0] if single_input else results
    
    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        return_probabilities: bool = True,
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        Make predictions on a large batch of texts.
        
        Args:
            texts: List of texts to classify
            batch_size: Number of texts per batch
            return_probabilities: Whether to return class probabilities
            show_progress: Whether to show progress
            
        Returns:
            List of prediction dictionaries
        """
        all_results = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = self.predict(batch, return_probabilities)
            all_results.extend(batch_results)
            
            if show_progress:
                batch_num = i // batch_size + 1
                logger.info(f"Processed batch {batch_num}/{total_batches}")
        
        return all_results


def interactive_mode(detector: AdDetector):
    """Run interactive mode for testing predictions."""
    print("\n" + "=" * 60)
    print("DeBERTa Ad Detector - Interactive Mode")
    print("=" * 60)
    print("Enter text to classify. Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            text = input("Enter text: ").strip()
            if text.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            if not text:
                print("Please enter some text.\n")
                continue
            
            result = detector.predict(text)
            
            print(f"\n{'='*40}")
            print(f"Prediction: {result['prediction'].upper()}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Probabilities:")
            for label, prob in result["probabilities"].items():
                print(f"  - {label}: {prob:.2%}")
            print(f"{'='*40}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="DeBERTa Ad Classifier Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model directory",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Text to classify",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="File with texts to classify (one per line or JSONL)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output file for predictions (JSON)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing multiple texts",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="Device to use for inference",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.text and not args.input_file and not args.interactive:
        parser.error("Please provide --text, --input_file, or --interactive")
    
    # Initialize detector
    detector = AdDetector(
        model_path=args.model_path,
        device=args.device,
    )
    
    # Interactive mode
    if args.interactive:
        interactive_mode(detector)
        return
    
    # Single text prediction
    if args.text:
        result = detector.predict(args.text)
        print(json.dumps(result, indent=2))
        return
    
    # File processing
    if args.input_file:
        input_path = Path(args.input_file)
        
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            sys.exit(1)
        
        # Load texts
        texts = []
        if input_path.suffix == ".jsonl":
            with open(input_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        texts.append(data.get("text") or data.get("input", ""))
        else:
            with open(input_path, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Loaded {len(texts)} texts from {input_path}")
        
        # Make predictions
        results = detector.predict_batch(texts, batch_size=args.batch_size)
        
        # Output
        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {args.output_file}")
        else:
            print(json.dumps(results, indent=2))
        
        # Summary
        ad_count = sum(1 for r in results if r["is_ad"])
        no_ad_count = len(results) - ad_count
        logger.info(f"\nSummary:")
        logger.info(f"  Total: {len(results)}")
        logger.info(f"  Ads: {ad_count} ({ad_count/len(results)*100:.1f}%)")
        logger.info(f"  No Ads: {no_ad_count} ({no_ad_count/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()
