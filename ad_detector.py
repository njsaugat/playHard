"""
High-Confidence Ad Detection System
Built on GLiNER2 NER with multi-signal scoring
"""

from gliner2 import GLiNER2
import requests
import re
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

URL ="https://storage.googleapis.com/core-production-3c790.appspot.com/transcripts/RSS-6ff682d5-ae7d-44f0-88b8-b37b00575bbd-1762068399131.json"
class AdConfidence(Enum):
    """Confidence levels for ad detection"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CERTAIN = 4


@dataclass
class AdSegment:
    """Represents a detected ad segment with metadata"""
    start_time: float
    end_time: float
    text: str
    confidence: AdConfidence
    confidence_score: float  # 0.0 to 1.0
    signals: dict = field(default_factory=dict)
    entities: dict = field(default_factory=dict)

    def __repr__(self):
        return (f"AdSegment({self.start_time:.2f}s-{self.end_time:.2f}s, "
                f"confidence={self.confidence.name}, score={self.confidence_score:.2f})")


class AdDetector:
    """
    Multi-signal ad detection system using NER as foundation.
    
    Detection signals:
    1. NER entities (companies, products, brands)
    2. Ad-specific phrase patterns
    3. Call-to-action patterns (URLs, promo codes)
    4. Transition markers
    5. Entity density scoring
    """
    
    # Ad-specific phrases (weighted by specificity)
    AD_PHRASES = {
        # High confidence markers (weight: 0.9)
        "sponsored by": 0.9,
        "brought to you by": 0.9,
        "this episode is brought to you": 0.9,
        "this podcast is brought to you": 0.9,
        "word from our sponsor": 0.9,
        "message from our sponsor": 0.9,
        "thanks to our sponsor": 0.9,
        "partner of the show": 0.85,
        "sponsored content": 0.9,
        
        # Promo code patterns (weight: 0.85)
        "promo code": 0.85,
        "use code": 0.85,
        "discount code": 0.85,
        "coupon code": 0.85,
        "use my code": 0.85,
        "special code": 0.8,
        "exclusive code": 0.8,
        
        # Call to action (weight: 0.7-0.8)
        "sign up": 0.5,
        "free trial": 0.75,
        "free shipping": 0.75,
        "percent off": 0.7,
        "% off": 0.7,
        "limited time": 0.65,
        "special offer": 0.75,
        "exclusive offer": 0.75,
        "check them out": 0.6,
        "check it out": 0.4,
        "go to": 0.35,
        "visit": 0.3,
        "head over to": 0.5,
        "head to": 0.4,
        
        # Money-back / risk-free
        "money back guarantee": 0.7,
        "risk free": 0.65,
        "no risk": 0.6,
        "cancel anytime": 0.65,
        
        # Transition markers (weight: 0.5-0.6)
        "back to the show": 0.5,
        "now back to": 0.5,
        "let's get back": 0.4,
    }
    
    # URL pattern for detecting website mentions
    URL_PATTERN = re.compile(
        r'\b(?:https?://)?(?:www\.)?'
        r'([a-zA-Z0-9][-a-zA-Z0-9]*\.(com|co|io|org|net|app|ai|xyz|ly|me|tv|fm))'
        r'(?:/[^\s]*)?',
        re.IGNORECASE
    )
    
    # Promo code pattern (e.g., "use code PODCAST20")
    PROMO_CODE_PATTERN = re.compile(
        r'\b(?:code|promo|coupon)[:\s]+([A-Z0-9]{3,15})\b',
        re.IGNORECASE
    )
    
    # Discount pattern (e.g., "20% off", "save $50")
    DISCOUNT_PATTERN = re.compile(
        r'\b(?:(\d{1,2})\s*%\s*off|save\s*\$?\s*(\d+)|(\d{1,2})\s*percent\s*off)\b',
        re.IGNORECASE
    )
    
    def __init__(self, model_name: str = "fastino/gliner2-base-v1"):
        """Initialize the ad detector with GLiNER2 model"""
        self.extractor = GLiNER2.from_pretrained(model_name)
        self.entity_labels = ["company", "product", "brand", "organization"]
        
    def _extract_entities(self, text: str) -> dict:
        """Extract named entities using GLiNER2"""
        try:
            result = self.extractor.extract_entities(text, self.entity_labels)
            if isinstance(result, dict) and "entities" in result:
                return result["entities"]
        except Exception:
            pass
        return {}
    
    def _detect_ad_phrases(self, text: str) -> tuple[float, list[str]]:
        """
        Detect ad-specific phrases in text.
        Returns (max_weight, list of matched phrases)
        """
        text_lower = text.lower()
        matched = []
        max_weight = 0.0
        
        for phrase, weight in self.AD_PHRASES.items():
            if phrase in text_lower:
                matched.append(phrase)
                max_weight = max(max_weight, weight)
        
        return max_weight, matched
    
    def _detect_urls(self, text: str) -> list[str]:
        """Detect URLs/website mentions in text"""
        matches = self.URL_PATTERN.findall(text)
        # findall returns tuples due to groups, extract the domain
        return [m[0] if isinstance(m, tuple) else m for m in matches]
    
    def _detect_promo_codes(self, text: str) -> list[str]:
        """Detect promo/coupon codes"""
        return self.PROMO_CODE_PATTERN.findall(text)
    
    def _detect_discounts(self, text: str) -> list[str]:
        """Detect discount mentions"""
        matches = self.DISCOUNT_PATTERN.findall(text)
        discounts = []
        for m in matches:
            # Each match is a tuple of groups
            val = next((v for v in m if v), None)
            if val:
                discounts.append(val)
        return discounts
    
    def _calculate_entity_density(self, text: str, entities: dict) -> float:
        """
        Calculate entity density score.
        Higher density of commercial entities = more likely an ad.
        """
        word_count = len(text.split())
        if word_count < 5:
            return 0.0
        
        entity_count = sum(
            len(v) if isinstance(v, list) else 0 
            for v in entities.values()
        )
        
        # Normalize: 1 entity per 20 words is baseline
        # More than that increases score
        density = entity_count / (word_count / 20)
        return min(density, 1.0)  # Cap at 1.0
    
    def _compute_confidence_score(self, signals: dict) -> float:
        """
        Compute overall confidence score from all signals.
        Uses weighted combination with boosting for multiple signals.
        """
        score = 0.0
        
        # Base NER signal (companies + products present)
        has_company = bool(signals.get("companies"))
        has_product = bool(signals.get("products"))
        
        if has_company and has_product:
            score += 0.25  # Both present = decent base
        elif has_company or has_product:
            score += 0.1   # One present = weak signal alone
        
        # Ad phrase signal (strongest indicator)
        phrase_score = signals.get("phrase_score", 0.0)
        if phrase_score > 0:
            score += phrase_score * 0.5  # Up to 0.45 contribution
        
        # URL signal
        if signals.get("urls"):
            score += 0.2
            # Boost if URL + entity
            if has_company:
                score += 0.1
        
        # Promo code signal (very strong)
        if signals.get("promo_codes"):
            score += 0.35
        
        # Discount signal
        if signals.get("discounts"):
            score += 0.15
        
        # Entity density bonus
        density = signals.get("entity_density", 0.0)
        if density > 0.5:
            score += density * 0.1
        
        # Multi-signal boost: if 3+ signals present, boost confidence
        signal_count = sum([
            has_company or has_product,
            phrase_score > 0,
            bool(signals.get("urls")),
            bool(signals.get("promo_codes")),
            bool(signals.get("discounts")),
        ])
        
        if signal_count >= 4:
            score *= 1.2  # 20% boost for strong multi-signal
        elif signal_count >= 3:
            score *= 1.1  # 10% boost
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _score_to_confidence(self, score: float) -> AdConfidence:
        """Convert numeric score to confidence level"""
        if score >= 0.85:
            return AdConfidence.CERTAIN
        elif score >= 0.65:
            return AdConfidence.HIGH
        elif score >= 0.45:
            return AdConfidence.MEDIUM
        elif score >= 0.25:
            return AdConfidence.LOW
        return AdConfidence.NONE
    
    def analyze_segment(self, text: str, start_time: float = 0.0, 
                       end_time: float = 0.0) -> AdSegment:
        """
        Analyze a single transcript segment for ad content.
        
        Args:
            text: The transcript text
            start_time: Start timestamp in seconds
            end_time: End timestamp in seconds
            
        Returns:
            AdSegment with detection results
        """
        # Gather all signals
        signals = {}
        
        # 1. NER extraction
        entities = self._extract_entities(text)
        signals["companies"] = entities.get("company", []) + entities.get("organization", [])
        signals["products"] = entities.get("product", []) + entities.get("brand", [])
        
        # 2. Ad phrase detection
        phrase_score, matched_phrases = self._detect_ad_phrases(text)
        signals["phrase_score"] = phrase_score
        signals["matched_phrases"] = matched_phrases
        
        # 3. URL detection
        signals["urls"] = self._detect_urls(text)
        
        # 4. Promo code detection
        signals["promo_codes"] = self._detect_promo_codes(text)
        
        # 5. Discount detection
        signals["discounts"] = self._detect_discounts(text)
        
        # 6. Entity density
        signals["entity_density"] = self._calculate_entity_density(text, entities)
        
        # Compute final score
        confidence_score = self._compute_confidence_score(signals)
        confidence = self._score_to_confidence(confidence_score)
        
        return AdSegment(
            start_time=start_time,
            end_time=end_time,
            text=text,
            confidence=confidence,
            confidence_score=confidence_score,
            signals=signals,
            entities=entities,
        )
    
    def detect_ads(self, transcript_data: dict, 
                   min_confidence: AdConfidence = AdConfidence.MEDIUM,
                   merge_adjacent: bool = True,
                   merge_gap_seconds: float = 5.0) -> list[AdSegment]:
        """
        Detect ads in a full transcript.
        
        Args:
            transcript_data: Dict with "transcriptByWords" key containing segments
            min_confidence: Minimum confidence level to report
            merge_adjacent: Whether to merge adjacent ad segments
            merge_gap_seconds: Maximum gap between segments to merge
            
        Returns:
            List of detected AdSegments
        """
        segments = transcript_data.get("transcriptByWords", [])
        detected = []
        
        for segment in segments:
            if not isinstance(segment, dict):
                continue
                
            text = segment.get("words", "")
            if not text or not isinstance(text, str):
                continue
            
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            
            result = self.analyze_segment(text, start, end)
            
            if result.confidence.value >= min_confidence.value:
                detected.append(result)
        
        # Merge adjacent segments if requested
        if merge_adjacent and detected:
            detected = self._merge_adjacent_segments(detected, merge_gap_seconds)
        
        return detected
    
    def _merge_adjacent_segments(self, segments: list[AdSegment], 
                                  max_gap: float) -> list[AdSegment]:
        """Merge adjacent ad segments that are close together"""
        if not segments:
            return segments
        
        # Sort by start time
        sorted_segs = sorted(segments, key=lambda s: s.start_time)
        merged = [sorted_segs[0]]
        
        for seg in sorted_segs[1:]:
            last = merged[-1]
            
            # Check if segments should be merged
            gap = seg.start_time - last.end_time
            if gap <= max_gap:
                # Merge: extend the last segment
                merged[-1] = AdSegment(
                    start_time=last.start_time,
                    end_time=seg.end_time,
                    text=last.text + " " + seg.text,
                    confidence=max(last.confidence, seg.confidence, 
                                   key=lambda c: c.value),
                    confidence_score=max(last.confidence_score, seg.confidence_score),
                    signals=self._merge_signals(last.signals, seg.signals),
                    entities=self._merge_entities(last.entities, seg.entities),
                )
            else:
                merged.append(seg)
        
        return merged
    
    def _merge_signals(self, s1: dict, s2: dict) -> dict:
        """Merge signal dictionaries"""
        merged = {}
        for key in set(s1.keys()) | set(s2.keys()):
            v1, v2 = s1.get(key), s2.get(key)
            if isinstance(v1, list) and isinstance(v2, list):
                merged[key] = list(set(v1 + v2))
            elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                merged[key] = max(v1, v2)
            else:
                merged[key] = v1 or v2
        return merged
    
    def _merge_entities(self, e1: dict, e2: dict) -> dict:
        """Merge entity dictionaries"""
        merged = {}
        for key in set(e1.keys()) | set(e2.keys()):
            v1 = e1.get(key, [])
            v2 = e2.get(key, [])
            merged[key] = list(set(v1 + v2))
        return merged


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    # Initialize detector
    print("Loading ad detection model...")
    detector = AdDetector()
    
    # Fetch transcript
    url = URL
    print(f"Fetching transcript from: {url[:60]}...")
    
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    
    # Detect ads
    print("\nAnalyzing transcript for ads...\n")
    print("=" * 70)
    
    ads = detector.detect_ads(
        data,
        min_confidence=AdConfidence.MEDIUM,
        merge_adjacent=True,
        merge_gap_seconds=10.0
    )
    
    if not ads:
        print("No ads detected with medium or higher confidence.")
    else:
        print(f"Found {len(ads)} ad segment(s):\n")
        
        for i, ad in enumerate(ads, 1):
            print(f"{'─' * 70}")
            print(f"AD #{i}")
            print(f"{'─' * 70}")
            print(f"  Time:       {format_timestamp(ad.start_time)} → {format_timestamp(ad.end_time)}")
            print(f"  Confidence: {ad.confidence.name} ({ad.confidence_score:.0%})")
            
            # Show detected signals
            if ad.signals.get("matched_phrases"):
                print(f"  Phrases:    {', '.join(ad.signals['matched_phrases'][:5])}")
            if ad.signals.get("companies"):
                print(f"  Companies:  {', '.join(ad.signals['companies'][:5])}")
            if ad.signals.get("products"):
                print(f"  Products:   {', '.join(ad.signals['products'][:5])}")
            if ad.signals.get("urls"):
                print(f"  URLs:       {', '.join(ad.signals['urls'][:3])}")
            if ad.signals.get("promo_codes"):
                print(f"  Promo codes: {', '.join(ad.signals['promo_codes'])}")
            if ad.signals.get("discounts"):
                print(f"  Discounts:  {', '.join(ad.signals['discounts'])}% off")
            
            # Show text preview
            preview = ad.text[:200] + "..." if len(ad.text) > 200 else ad.text
            print(f"\n  Text preview:")
            print(f"  \"{preview}\"")
            print()
    
    print("=" * 70)
    print("Detection complete.")
    
    # Also show lower confidence detections for reference
    low_conf_ads = detector.detect_ads(
        data,
        min_confidence=AdConfidence.LOW,
        merge_adjacent=True,
    )
    
    low_only = [a for a in low_conf_ads if a.confidence == AdConfidence.LOW]
    if low_only:
        print(f"\n[Note: {len(low_only)} additional low-confidence segment(s) detected]")
