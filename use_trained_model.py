"""
Use Trained GLiNER2 Ad Classifier

This script demonstrates how to use your trained GLiNER2 model
for ad detection. It can replace the base model in ad_detector.py.

Model paths can be:
- "latest" - loads the most recent trained model
- A run ID like "run_20260110_143052" - loads specific checkpoint
- A full path to a model directory
"""

import os
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List

try:
    from gliner2 import GLiNER2
except ImportError:
    raise ImportError("GLiNER2 not installed. Run: pip install gliner2")


def resolve_model_path(
    model_path: str = "latest",
    base_dir: str = "./gliner2_ad_classifier"
) -> str:
    """
    Resolve a model path from various formats.
    
    Args:
        model_path: One of:
            - "latest" - use the most recent model
            - A run ID like "run_20260110_143052"
            - A full path to a model directory
        base_dir: Base directory where models are stored
    
    Returns:
        Resolved absolute path to the model directory
    """
    base_path = Path(base_dir)
    
    # If it's "latest", resolve through the symlink
    if model_path == "latest":
        latest_link = base_path / "latest" / "best"
        if latest_link.exists():
            return str(latest_link.resolve())
        
        # Fallback: find most recent run manually
        runs_dir = base_path / "runs"
        if runs_dir.exists():
            runs = sorted(
                [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            for run in runs:
                best_model = run / "best"
                if best_model.exists():
                    return str(best_model)
        
        # Legacy: check old location
        old_path = base_path / "best"
        if old_path.exists():
            return str(old_path)
        
        raise FileNotFoundError(f"No trained models found in: {base_dir}")
    
    # If it's a run ID
    if model_path.startswith("run_"):
        run_model = base_path / "runs" / model_path / "best"
        if run_model.exists():
            return str(run_model)
        raise FileNotFoundError(f"Model not found for run: {model_path}")
    
    # It's a direct path
    if os.path.exists(model_path):
        return model_path
    
    raise FileNotFoundError(f"Model not found at: {model_path}")


def list_available_models(base_dir: str = "./gliner2_ad_classifier") -> List[Dict]:
    """
    List all available trained model checkpoints.
    
    Returns:
        List of checkpoint info dicts
    """
    base_path = Path(base_dir)
    runs_dir = base_path / "runs"
    
    models = []
    
    if runs_dir.exists():
        for run_dir in sorted(runs_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if run_dir.is_dir() and run_dir.name.startswith("run_"):
                best_model = run_dir / "best"
                metadata_file = run_dir / "training_metadata.json"
                
                info = {
                    "run_id": run_dir.name,
                    "path": str(best_model) if best_model.exists() else None,
                    "has_model": best_model.exists(),
                }
                
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            info["metadata"] = json.load(f)
                    except Exception:
                        pass
                
                models.append(info)
    
    # Check legacy path
    legacy_path = base_path / "best"
    if legacy_path.exists():
        models.append({
            "run_id": "legacy",
            "path": str(legacy_path),
            "has_model": True,
        })
    
    return models


class TrainedAdClassifier:
    """
    Wrapper for the trained GLiNER2 ad classifier.
    Can be used as a drop-in replacement for ad detection.
    
    Supports loading models by:
    - "latest" - most recent checkpoint
    - Run ID (e.g., "run_20260110_143052")
    - Full path to model directory
    """
    
    def __init__(
        self, 
        model_path: str = "latest",
        threshold: float = 0.5,
        base_dir: str = "./gliner2_ad_classifier"
    ):
        """
        Initialize the trained classifier.
        
        Args:
            model_path: "latest", a run_id, or path to the trained model
            threshold: Confidence threshold for ad classification
            base_dir: Base directory where models are stored
        """
        # Resolve the model path
        resolved_path = resolve_model_path(model_path, base_dir)
        
        print(f"📦 Loading trained model from: {resolved_path}")
        self.model = GLiNER2.from_pretrained(resolved_path)
        self.model_path = resolved_path
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
            # GLiNER2 uses schema-based extraction
            schema = self.model.create_schema().classification(
                "ad_detection",
                ["ad", "no_ad"]
            )
            
            result = self.model.extract(text, schema, include_confidence=True)
            
            if isinstance(result, dict):
                ad_detection = result.get("ad_detection", {})
                
                # Handle both formats: {'label': 'ad', 'confidence': 0.9} or just 'ad'
                if isinstance(ad_detection, dict):
                    label = ad_detection.get("label", "no_ad")
                    confidence = ad_detection.get("confidence", 0.5)
                else:
                    label = str(ad_detection) if ad_detection else "no_ad"
                    confidence = 0.7
                
                is_ad = label == "ad" and confidence >= self.threshold
                
                return is_ad, confidence, "ad" if is_ad else "no_ad"
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
            # GLiNER2 uses schema-based extraction
            schema = self.model.create_schema().classification(
                "ad_detection",
                ["ad", "no_ad"]
            )
            
            result = self.model.extract(text, schema, include_confidence=True)
            
            if isinstance(result, dict):
                ad_detection = result.get("ad_detection", {})
                
                if isinstance(ad_detection, dict):
                    label = ad_detection.get("label", "no_ad")
                    confidence = ad_detection.get("confidence", 0.5)
                else:
                    label = str(ad_detection) if ad_detection else "no_ad"
                    confidence = 0.7
                
                is_ad = label == "ad" and confidence >= self.threshold
                
                return {
                    "text_preview": text[:200] + "..." if len(text) > 200 else text,
                    "is_ad": is_ad,
                    "scores": {
                        "ad": confidence if label == "ad" else 1 - confidence,
                        "no_ad": confidence if label == "no_ad" else 1 - confidence
                    },
                    "confidence": confidence,
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

MODEL CHECKPOINT SYSTEM:
------------------------
Models are now versioned with timestamps. Each training run creates a new 
checkpoint that's preserved:

    ./gliner2_ad_classifier/
        runs/
            run_20260110_143052/best/    <- first training
            run_20260110_160215/best/    <- second training
            run_20260111_091030/best/    <- third training
        latest -> runs/run_20260111_091030  (symlink to newest)

To list all checkpoints:
    python train_gliner2_classifier.py --list-checkpoints

To use a specific checkpoint:
    classifier = TrainedAdClassifier("run_20260110_143052")


Option 1: Use "latest" (recommended)
------------------------------------

    from use_trained_model import TrainedAdClassifier
    
    # Automatically loads the most recent model
    classifier = TrainedAdClassifier("latest")
    
    is_ad, confidence, label = classifier.classify(transcript_text)


Option 2: Use a specific checkpoint
-----------------------------------

    from use_trained_model import TrainedAdClassifier
    
    # Load a specific version (for reproducibility/rollback)
    classifier = TrainedAdClassifier("run_20260110_143052")
    
    is_ad, confidence, label = classifier.classify(transcript_text)


Option 3: List and compare models
---------------------------------

    from use_trained_model import list_available_models
    
    models = list_available_models()
    for m in models:
        print(f"{m['run_id']}: {m.get('metadata', {}).get('best_validation_loss', 'N/A')}")


Option 4: Create a hybrid detector
----------------------------------

    class HybridAdDetector:
        def __init__(self, model_version: str = "latest"):
            from ad_detector import AdDetector
            from use_trained_model import TrainedAdClassifier
            
            # Load both models
            self.base_detector = AdDetector()  # For entity extraction
            self.classifier = TrainedAdClassifier(model_version)
        
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


def demo(model_path: str = "latest"):
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
    
    # Show available models
    models = list_available_models()
    if models:
        print(f"\n📋 Available checkpoints: {len(models)}")
        for m in models[:3]:
            print(f"   - {m['run_id']}")
        if len(models) > 3:
            print(f"   ... and {len(models) - 3} more")
    
    try:
        classifier = TrainedAdClassifier(model_path)
        print(f"\n🎯 Using model: {classifier.model_path}")
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        print("   Run train_gliner2_classifier.py first to train a model.")
        print("\n⚠️ Using base GLiNER2 model for demo...")
        classifier = TrainedAdClassifier.__new__(TrainedAdClassifier)
        classifier.model = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
        classifier.model_path = "fastino/gliner2-base-v1 (base)"
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
    
    parser = argparse.ArgumentParser(
        description='Use trained GLiNER2 ad classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use the latest trained model
  python use_trained_model.py --demo
  
  # Use a specific checkpoint
  python use_trained_model.py --model run_20260110_143052 --demo
  
  # Classify specific text
  python use_trained_model.py --text "This episode is brought to you by..."
  
  # List all available models
  python use_trained_model.py --list-models
        """
    )
    parser.add_argument('--model', type=str, default='latest',
                        help='Model to use: "latest", a run_id, or full path')
    parser.add_argument('--text', type=str, default=None,
                        help='Text to classify')
    parser.add_argument('--demo', action='store_true', help='Run demo')
    parser.add_argument('--list-models', action='store_true', help='List available models')
    
    args = parser.parse_args()
    
    if args.list_models:
        print("=" * 70)
        print("📋 Available Model Checkpoints")
        print("=" * 70)
        models = list_available_models()
        if not models:
            print("\n   No trained models found.")
            print("   Run: python train_gliner2_classifier.py")
        else:
            for m in models:
                print(f"\n[{'✓' if m['has_model'] else '✗'}] {m['run_id']}")
                if m.get('path'):
                    print(f"    Path: {m['path']}")
                if m.get('metadata'):
                    meta = m['metadata']
                    if meta.get('timestamp'):
                        print(f"    Trained: {meta['timestamp']}")
                    if meta.get('best_validation_loss'):
                        print(f"    Best val loss: {meta['best_validation_loss']:.4f}")
    elif args.text:
        classifier = TrainedAdClassifier(args.model)
        result = classifier.get_detailed_result(args.text)
        print(json.dumps(result, indent=2))
    else:
        demo(args.model)