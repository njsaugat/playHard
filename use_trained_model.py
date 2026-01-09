"""
Use Trained GLiNER2 Ad Classifier

This script demonstrates how to use your trained GLiNER2 model
for ad detection. It can replace the base model in ad_detector.py.
"""

import os
import json
from typing import Optional, Tuple, Dict, List

try:
    from gliner2 import GLiNER2
except ImportError:
    raise ImportError("GLiNER2 not installed. Run: pip install gliner2")


class TrainedAdClassifier:
    """
    Wrapper for the trained GLiNER2 ad classifier.
    Can be used as a drop-in replacement for ad detection.
    """
    
    def __init__(
        self, 
        model_path: str = "./gliner2_ad_classifier/best",
        threshold: float = 0.5
    ):
        """
        Initialize the trained classifier.
        
        Args:
            model_path: Path to the trained model directory
            threshold: Confidence threshold for ad classification
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at: {model_path}")
        
        print(f"📦 Loading trained model from: {model_path}")
        self.model = GLiNER2.from_pretrained(model_path)
        self.threshold = threshold
        
        # Classification labels with descriptions
        self.labels = {
            "ad": "Sponsored advertisement, promotion, or paid endorsement",
            "no_ad": "Regular content, not a sponsored advertisement"
        }
        
        print(f"✅ Model loaded successfully")
    
    def classify(self, text: str) -> Tuple[bool, float, str]:
        """
        Classify if text is an advertisement.
        
        Args:
            text: Transcript text to classify
            
        Returns:
            (is_ad: bool, confidence: float, label: str)
        """
        try:
            result = self.model.classify(text, labels=self.labels)
            
            if isinstance(result, dict):
                ad_score = result.get("ad", 0)
                no_ad_score = result.get("no_ad", 0)
                
                is_ad = ad_score > no_ad_score and ad_score >= self.threshold
                confidence = ad_score if is_ad else no_ad_score
                label = "ad" if is_ad else "no_ad"
                
                return is_ad, confidence, label
            else:
                # Fallback for string result
                is_ad = str(result).lower() == "ad"
                return is_ad, 0.7 if is_ad else 0.3, "ad" if is_ad else "no_ad"
                
        except Exception as e:
            print(f"⚠️ Classification error: {e}")
            return False, 0.0, "no_ad"
    
    def classify_batch(self, texts: List[str]) -> List[Tuple[bool, float, str]]:
        """
        Classify multiple texts.
        
        Args:
            texts: List of transcript texts
            
        Returns:
            List of (is_ad, confidence, label) tuples
        """
        results = []
        for text in texts:
            results.append(self.classify(text))
        return results
    
    def get_detailed_result(self, text: str) -> Dict:
        """
        Get detailed classification result with all scores.
        """
        try:
            result = self.model.classify(text, labels=self.labels)
            
            if isinstance(result, dict):
                ad_score = result.get("ad", 0)
                no_ad_score = result.get("no_ad", 0)
                is_ad = ad_score > no_ad_score and ad_score >= self.threshold
                
                return {
                    "text_preview": text[:200] + "..." if len(text) > 200 else text,
                    "is_ad": is_ad,
                    "scores": {
                        "ad": ad_score,
                        "no_ad": no_ad_score
                    },
                    "confidence": max(ad_score, no_ad_score),
                    "label": "ad" if is_ad else "no_ad",
                    "threshold": self.threshold
                }
            else:
                return {
                    "text_preview": text[:200] + "..." if len(text) > 200 else text,
                    "is_ad": str(result).lower() == "ad",
                    "label": str(result),
                    "confidence": 0.7
                }
                
        except Exception as e:
            return {
                "error": str(e),
                "is_ad": False,
                "confidence": 0.0
            }


def integrate_with_ad_detector(trained_model_path: str):
    """
    Example of how to integrate the trained model with ad_detector.py
    """
    print("""
================================================================================
HOW TO USE TRAINED MODEL WITH ad_detector.py
================================================================================

Option 1: Modify ad_detector.py to use trained model
------------------------------------------------------

In ad_detector.py, change the __init__ method:

    def __init__(self, model_name: str = None):
        # Use trained model if path provided, otherwise use base model
        if model_name and os.path.exists(model_name):
            self.model = GLiNER2.from_pretrained(model_name)
        else:
            self.model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")

Then use:
    detector = AdDetector(model_name="./gliner2_ad_classifier/best")


Option 2: Use TrainedAdClassifier directly
------------------------------------------

    from use_trained_model import TrainedAdClassifier
    
    classifier = TrainedAdClassifier("./gliner2_ad_classifier/best")
    
    is_ad, confidence, label = classifier.classify(transcript_text)
    
    if is_ad:
        print(f"AD detected with {confidence:.0%} confidence")
    else:
        print(f"Not an ad ({confidence:.0%} confidence)")


Option 3: Create a hybrid detector (recommended)
------------------------------------------------

    class HybridAdDetector:
        def __init__(self, trained_model_path: str = "./gliner2_ad_classifier/best"):
            from ad_detector import AdDetector
            from use_trained_model import TrainedAdClassifier
            
            # Load both models
            self.base_detector = AdDetector()  # For entity extraction, JSON, etc.
            self.classifier = TrainedAdClassifier(trained_model_path)
        
        def detect(self, text: str):
            # Use trained classifier for better accuracy
            is_ad, confidence, _ = self.classifier.classify(text)
            
            # Get structured data from base detector
            segment = self.base_detector.analyze_segment(text)
            
            # Override classification with trained model
            segment.is_ad_classification = is_ad
            segment.classification_confidence = confidence
            
            return segment

================================================================================
""")


def demo(model_path: str = "./gliner2_ad_classifier/best"):
    """Demo the trained classifier"""
    
    # Sample texts for testing
    sample_texts = [
        # Clear advertisement
        """
        This episode is brought to you by BetterHelp. If you're going through 
        a tough time, therapy can help. Visit betterhelp.com/podcast for 10% off 
        your first month. Use code PODCAST for an additional discount.
        """,
        
        # Brand love / shoutout (should be no_ad)
        """
        I've been using this amazing coffee maker from Ninja, and honestly, 
        I just love it. Nobody paid me to say this, I just really like it.
        """,
        
        # Regular content
        """
        So today we're going to be talking about the history of artificial 
        intelligence. It's fascinating how far we've come since the 1950s.
        """,
        
        # Subtle ad
        """
        Speaking of staying organized, I've been using Notion for all my 
        project management. Head to notion.com/creators to get started free.
        """
    ]
    
    print("=" * 70)
    print("🎯 Trained GLiNER2 Ad Classifier Demo")
    print("=" * 70)
    
    if not os.path.exists(model_path):
        print(f"\n❌ Trained model not found at: {model_path}")
        print("   Run train_gliner2_classifier.py first to train a model.")
        print("\n   Or use the base model for demo:")
        model_path = "fastino/gliner2-base-v1"
        print(f"   Using: {model_path}")
    
    try:
        classifier = TrainedAdClassifier(model_path)
    except FileNotFoundError:
        print(f"\n⚠️ Using base GLiNER2 model for demo...")
        classifier = TrainedAdClassifier.__new__(TrainedAdClassifier)
        classifier.model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
        classifier.threshold = 0.5
        classifier.labels = {
            "ad": "Sponsored advertisement",
            "no_ad": "Regular content"
        }
    
    print("\n" + "-" * 70)
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n📝 Sample {i}:")
        print(f"   {text.strip()[:100]}...")
        
        result = classifier.get_detailed_result(text.strip())
        
        if result.get("is_ad"):
            print(f"   ✅ AD DETECTED (confidence: {result.get('confidence', 0):.0%})")
        else:
            print(f"   ❌ Not an ad (confidence: {result.get('confidence', 0):.0%})")
        
        if "scores" in result:
            print(f"   Scores - ad: {result['scores']['ad']:.2f}, no_ad: {result['scores']['no_ad']:.2f}")
    
    print("\n" + "=" * 70)
    integrate_with_ad_detector(model_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Use trained GLiNER2 ad classifier')
    parser.add_argument('--model-path', type=str, default='./gliner2_ad_classifier/best',
                        help='Path to trained model')
    parser.add_argument('--text', type=str, default=None,
                        help='Text to classify')
    parser.add_argument('--demo', action='store_true', help='Run demo')
    
    args = parser.parse_args()
    
    if args.demo or args.text is None:
        demo(args.model_path)
    else:
        classifier = TrainedAdClassifier(args.model_path)
        result = classifier.get_detailed_result(args.text)
        print(json.dumps(result, indent=2))
