"""
High-Confidence Ad Detection System
Built on GLiNER2 with multi-signal scoring using:
- Entity extraction with descriptions
- JSON structure extraction
- Classification
- Relation extraction
- Combined schemas for single-pass efficiency
- Exclusion filters for non-ads (Patreon, social plugs, etc.)
- Brand love/shoutout detection
"""

from gliner2 import GLiNER2
import requests
import re
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum

URL = "https://storage.googleapis.com/core-production-3c790.appspot.com/transcripts/RSS-6ff682d5-ae7d-44f0-88b8-b37b00575bbd-1762068399131.json"


class AdConfidence(Enum):
    """Confidence levels for ad detection"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CERTAIN = 4


class ExclusionReason(Enum):
    """Reasons for excluding a segment from being an ad"""
    NONE = "none"
    PATREON_SUBSTACK = "patreon_substack"
    SOCIAL_MEDIA_PLUG = "social_media_plug"
    AD_FREE_SUBSCRIPTION = "ad_free_subscription"
    BRAND_LOVE_SHOUTOUT = "brand_love_shoutout"
    SELF_PROMOTION = "self_promotion"


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
    # New structured data from GLiNER2 JSON extraction
    structured_data: dict = field(default_factory=dict)
    relations: list = field(default_factory=list)
    is_ad_classification: Optional[bool] = None
    classification_confidence: float = 0.0
    # Exclusion tracking
    excluded: bool = False
    exclusion_reason: ExclusionReason = ExclusionReason.NONE
    is_brand_love: bool = False
    # Context expansion tracking
    context_expanded: bool = False
    original_start: Optional[float] = None
    original_end: Optional[float] = None

    def __repr__(self):
        status = f"EXCLUDED:{self.exclusion_reason.value}" if self.excluded else f"confidence={self.confidence.name}"
        return (f"AdSegment({self.start_time:.2f}s-{self.end_time:.2f}s, "
                f"{status}, score={self.confidence_score:.2f})")


class AdDetector:
    """
    Multi-signal ad detection system using GLiNER2's full capabilities:
    
    1. Entity extraction with descriptions - More accurate NER
    2. Classification - Direct ad/non-ad classification
    3. JSON structure extraction - Structured sponsor data in one pass
    4. Relation extraction - Sponsor relationships
    5. Combined schemas - All extraction in single inference
    6. Exclusion filters - Filter out Patreon, social plugs, etc.
    7. Brand love detection - Low confidence for shoutouts without CTAs
    """
    
    # =========================================================================
    # EXCLUSION PATTERNS - Things that are NEVER ads
    # =========================================================================
    
    # Patreon/Substack patterns (creator support, not sponsor ads)
    PATREON_SUBSTACK_PATTERNS = [
        r'\bpatreon\.com\b',
        r'\bpatreon\b',
        r'\bsubstack\.com\b',
        r'\bsubstack\b',
        r'\bsupport\s+(?:us|me|the\s+show)\s+on\s+patreon\b',
        r'\bbecome\s+a\s+patron\b',
        r'\bjoin\s+(?:our|my)\s+patreon\b',
        r'\bsubscribe\s+(?:to\s+)?(?:our|my)\s+substack\b',
        r'\bpatreon\s+supporters?\b',
        r'\bpatreon\s+members?\b',
        r'\bko-?fi\b',
        r'\bbuy\s+me\s+a\s+coffee\b',
    ]
    
    # Social media plug patterns (self-promotion, not sponsor ads)
    SOCIAL_MEDIA_PATTERNS = [
        r'\bfollow\s+(?:us|me)\s+on\s+(?:twitter|x|instagram|tiktok|facebook|youtube)\b',
        r'\b(?:twitter|x|instagram|tiktok|facebook)\.com/\w+\b',
        r'\bsubscribe\s+to\s+(?:our|my|the)\s+(?:youtube|channel)\b',
        r'\blike\s+and\s+subscribe\b',
        r'\bhit\s+(?:the\s+)?(?:subscribe|notification)\s+button\b',
        r'\b(?:our|my)\s+social\s+media\b',
        r'\bfind\s+(?:us|me)\s+on\s+social\b',
        r'\b@\w+\s+on\s+(?:twitter|instagram|x)\b',
        r'\bleave\s+(?:us|a)\s+review\b',
        r'\brate\s+(?:us|the\s+show|this\s+podcast)\b',
        r'\bon\s+apple\s+podcasts\b',
        r'\bon\s+spotify\b',
    ]
    
    # Ad-free subscription patterns (podcast's own premium offering)
    AD_FREE_PATTERNS = [
        r'\bad[- ]?free\b',
        r'\bpremium\s+(?:version|subscription|members?)\b',
        r'\bsubscribe\s+(?:for|to\s+get)\s+ad[- ]?free\b',
        r'\bremove\s+(?:the\s+)?ads\b',
        r'\bwithout\s+(?:the\s+)?ads\b',
        r'\bno\s+ads\b',
        r'\bbonus\s+(?:episodes?|content)\b',
        r'\bexclusive\s+(?:episodes?|content)\s+(?:for|on)\b',
        r'\b(?:apple|spotify)\s+premium\b',
        r'\bpodcast\s+(?:premium|plus)\b',
        r'\bsupercast\b',
        r'\bmembership\s+(?:tier|level)\b',
    ]
    
    # Self-promotion patterns (promoting own products/services)
    SELF_PROMOTION_PATTERNS = [
        r'\b(?:my|our)\s+(?:new\s+)?(?:book|course|workshop|merch)\b',
        r'\bcheck\s+out\s+(?:my|our)\s+(?:website|store|shop)\b',
        r'\b(?:my|our)\s+(?:other\s+)?podcast\b',
        r'\blive\s+show\s+tickets\b',
        r'\btour\s+dates\b',
    ]
    
    # Compile all exclusion patterns
    EXCLUSION_COMPILED = {
        ExclusionReason.PATREON_SUBSTACK: [re.compile(p, re.IGNORECASE) for p in PATREON_SUBSTACK_PATTERNS],
        ExclusionReason.SOCIAL_MEDIA_PLUG: [re.compile(p, re.IGNORECASE) for p in SOCIAL_MEDIA_PATTERNS],
        ExclusionReason.AD_FREE_SUBSCRIPTION: [re.compile(p, re.IGNORECASE) for p in AD_FREE_PATTERNS],
        ExclusionReason.SELF_PROMOTION: [re.compile(p, re.IGNORECASE) for p in SELF_PROMOTION_PATTERNS],
    }
    
    # =========================================================================
    # BRAND LOVE DETECTION - Shoutouts without commercial intent
    # =========================================================================
    
    # Phrases indicating genuine brand love (not sponsored)
    BRAND_LOVE_PHRASES = [
        r'\bi\s+(?:really\s+)?love\s+(?:using\s+)?(?:my\s+)?\w+\b',
        r'\bi\'ve\s+been\s+(?:using|loving)\b',
        r'\bshoutout\s+to\b',
        r'\bshout\s+out\s+to\b',
        r'\bbig\s+fan\s+of\b',
        r'\bi\'m\s+a\s+(?:huge\s+)?fan\b',
        r'\bpersonally\s+(?:use|recommend)\b',
        r'\bjust\s+(?:wanted\s+to\s+)?mention\b',
        r'\bquick\s+(?:shoutout|mention)\b',
        r'\bnot\s+(?:a\s+)?sponsor(?:ed)?\b',
        r'\bno\s+one\s+(?:paid|asked)\b',
        r'\bthey\s+(?:didn\'t|did\s+not)\s+(?:pay|sponsor)\b',
    ]
    
    BRAND_LOVE_COMPILED = [re.compile(p, re.IGNORECASE) for p in BRAND_LOVE_PHRASES]
    
    # =========================================================================
    # PARTIAL AD DETECTION - Signs of incomplete context
    # =========================================================================
    
    # Patterns indicating we might have partial/incomplete ad text
    PARTIAL_AD_INDICATORS = [
        # Starts mid-sentence
        r'^[a-z]',  # Starts with lowercase
        r'^(?:and|but|so|or|also|plus)\s',  # Starts with conjunction
        # Ends mid-sentence
        r'[,]\s*$',  # Ends with comma
        r'\b(?:and|but|so|or|also|with|for|to)\s*$',  # Ends with conjunction/preposition
        # Truncated offer mentions
        r'\buse\s+(?:the\s+)?code\s*$',
        r'\bgo\s+to\s*$',
        r'\bvisit\s*$',
        r'\bcheck\s+out\s*$',
        r'percent\s+off\s*$',
        r'\bfor\s+(?:a\s+)?free\s*$',
        # Missing URL completion
        r'\b\w+\.(?:com|co|io|org)\s*/',  # URL with path but possibly truncated
    ]
    
    PARTIAL_AD_COMPILED = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in PARTIAL_AD_INDICATORS]
    
    # Entity labels with descriptions for better GLiNER2 accuracy
    ENTITY_SCHEMA = {
        "company": "A business, corporation, or company name that is being promoted or advertised",
        "brand": "A brand name, trademark, or product line being marketed",
        "product": "A specific product or service being offered or sold",
        "sponsor": "The company or brand sponsoring/paying for this advertisement",
        "promo_code": "A promotional code, discount code, or coupon code for purchases",
        "website": "A website URL or domain name mentioned for the offer",
        "discount": "A discount percentage, price reduction, or special offer amount",
        "host_name": "The name of the podcast host reading the advertisement",
    }
    
    # JSON schema for structured ad data extraction
    AD_JSON_SCHEMA = {
        "sponsor_name": {
            "type": "string",
            "description": "The name of the company or brand sponsoring this advertisement"
        },
        "sponsor_url": {
            "type": "string",
            "description": "The website URL mentioned for the sponsor (e.g., example.com/podcast)"
        },
        "promo_code": {
            "type": "string",
            "description": "The promotional or discount code mentioned (e.g., PODCAST20)"
        },
        "discount_offer": {
            "type": "string",
            "description": "The discount or special offer mentioned (e.g., '20% off', 'free trial')"
        },
        "product_name": {
            "type": "string",
            "description": "The specific product or service being advertised"
        },
        "call_to_action": {
            "type": "string",
            "description": "The action listeners are asked to take (e.g., 'visit', 'sign up', 'use code')"
        },
        "ad_type": {
            "type": "string",
            "description": "Type of advertisement",
            "choices": ["host_read", "pre_recorded", "dynamic_insertion", "unknown"]
        }
    }
    
    # Relation types for sponsor relationships
    RELATION_TYPES = [
        "sponsors",           # Company sponsors the podcast
        "offers_product",     # Sponsor offers a product
        "has_promo_code",     # Product/sponsor has a promo code
        "provides_discount",  # Sponsor provides discount
        "has_website",        # Sponsor has website
    ]
    
    # Classification labels for direct ad detection
    CLASSIFICATION_LABELS = {
        "advertisement": "This text is a sponsored advertisement, promotion, or paid endorsement for a product or service",
        "regular_content": "This text is regular podcast content, discussion, or conversation that is not sponsored",
    }
    
    # Ad-specific phrases (weighted by specificity) - kept as backup signal
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
    
    # URL pattern for detecting website mentions (backup)
    URL_PATTERN = re.compile(
        r'\b(?:https?://)?(?:www\.)?'
        r'([a-zA-Z0-9][-a-zA-Z0-9]*\.(com|co|io|org|net|app|ai|xyz|ly|me|tv|fm))'
        r'(?:/[^\s]*)?',
        re.IGNORECASE
    )
    
    # Promo code pattern (backup)
    PROMO_CODE_PATTERN = re.compile(
        r'\b(?:code|promo|coupon)[:\s]+([A-Z0-9]{3,15})\b',
        re.IGNORECASE
    )
    
    # Discount pattern (backup)
    DISCOUNT_PATTERN = re.compile(
        r'\b(?:(\d{1,2})\s*%\s*off|save\s*\$?\s*(\d+)|(\d{1,2})\s*percent\s*off)\b',
        re.IGNORECASE
    )
    
    def __init__(self, model_name: str = "fastino/gliner2-base-v1"):
        """Initialize the ad detector with GLiNER2 model"""
        self.model = GLiNER2.from_pretrained(model_name)
        # Keep backward compatibility
        self.extractor = self.model
        self.entity_labels = list(self.ENTITY_SCHEMA.keys())
    
    # =========================================================================
    # EXCLUSION DETECTION
    # =========================================================================
    
    def _check_exclusions(self, text: str) -> tuple[bool, ExclusionReason]:
        """
        Check if text matches any exclusion patterns.
        Returns (should_exclude, reason)
        """
        for reason, patterns in self.EXCLUSION_COMPILED.items():
            for pattern in patterns:
                if pattern.search(text):
                    return True, reason
        return False, ExclusionReason.NONE
    
    def _is_brand_love(self, text: str, signals: dict, structured_data: dict) -> bool:
        """
        Detect if this is a brand love/shoutout rather than a sponsored ad.
        
        Brand love characteristics:
        - Mentions a brand/company
        - No URL, promo code, or discount
        - Uses personal endorsement language
        - Short mention, quickly moves on
        """
        # Check for brand love phrases
        has_brand_love_phrase = any(p.search(text) for p in self.BRAND_LOVE_COMPILED)
        
        # Check for commercial signals
        has_url = bool(signals.get("urls")) or bool(structured_data.get("sponsor_url"))
        has_promo = bool(signals.get("promo_codes")) or bool(structured_data.get("promo_code"))
        has_discount = bool(signals.get("discounts")) or bool(structured_data.get("discount_offer"))
        has_cta = bool(structured_data.get("call_to_action"))
        
        # Brand love = brand mentioned but no commercial intent signals
        has_brand = bool(signals.get("companies")) or bool(signals.get("products"))
        no_commercial_signals = not (has_url or has_promo or has_discount or has_cta)
        
        # If brand love phrase detected and no commercial signals, it's brand love
        if has_brand_love_phrase and no_commercial_signals:
            return True
        
        # If brand mentioned but absolutely no commercial signals and short text
        if has_brand and no_commercial_signals and len(text.split()) < 50:
            return True
        
        return False
    
    def _is_partial_ad(self, text: str, signals: dict) -> bool:
        """
        Detect if this might be a partial/incomplete ad that needs context expansion.
        """
        # Check partial ad indicators
        for pattern in self.PARTIAL_AD_COMPILED:
            if pattern.search(text):
                return True
        
        # Check if we have strong ad signals but missing key info
        has_sponsor_phrase = signals.get("phrase_score", 0) > 0.7
        has_url = bool(signals.get("urls"))
        has_promo = bool(signals.get("promo_codes"))
        has_company = bool(signals.get("companies"))
        
        # Strong ad phrase but missing URL or promo code might mean partial
        if has_sponsor_phrase and not (has_url or has_promo) and not has_company:
            return True
        
        # Has URL but no company name detected - might be truncated
        if has_url and not has_company:
            return True
        
        return False
    
    # =========================================================================
    # GLiNER2 ENTITY EXTRACTION WITH DESCRIPTIONS
    # =========================================================================
    
    def _extract_entities_with_descriptions(self, text: str) -> dict:
        """
        Extract named entities using GLiNER2 with descriptions for better accuracy.
        Descriptions help the model understand exactly what to look for.
        """
        try:
            # Use entity extraction with descriptions for better accuracy
            result = self.model.extract_entities(
                text,
                labels=self.ENTITY_SCHEMA,  # Pass dict with descriptions
                threshold=0.3  # Lower threshold to catch more potential matches
            )
            
            if isinstance(result, dict) and "entities" in result:
                return result["entities"]
            elif isinstance(result, list):
                # Convert list format to dict
                entities = {}
                for ent in result:
                    label = ent.get("label", "unknown")
                    if label not in entities:
                        entities[label] = []
                    entities[label].append(ent.get("text", ""))
                return entities
        except Exception as e:
            # Fallback to basic extraction
            try:
                result = self.model.extract_entities(text, list(self.ENTITY_SCHEMA.keys()))
                if isinstance(result, dict) and "entities" in result:
                    return result["entities"]
            except:
                pass
        return {}
    
    # =========================================================================
    # GLiNER2 CLASSIFICATION
    # =========================================================================
    
    def _classify_as_ad(self, text: str) -> tuple[bool, float]:
        """
        Use GLiNER2 classification to determine if text is an advertisement.
        Returns (is_ad: bool, confidence: float)
        """
        try:
            # Single-label classification with descriptions
            result = self.model.classify(
                text,
                labels=self.CLASSIFICATION_LABELS,
            )
            
            if isinstance(result, dict):
                # Check if "advertisement" has higher score
                ad_score = result.get("advertisement", 0)
                regular_score = result.get("regular_content", 0)
                
                is_ad = ad_score > regular_score
                confidence = ad_score if is_ad else (1 - regular_score)
                return is_ad, confidence
            elif isinstance(result, str):
                # Label-only result
                is_ad = result == "advertisement"
                return is_ad, 0.7 if is_ad else 0.3
                
        except Exception as e:
            # Classification not available, return neutral
            pass
        
        return None, 0.0
    
    # =========================================================================
    # GLiNER2 JSON STRUCTURE EXTRACTION
    # =========================================================================
    
    def _extract_structured_data(self, text: str) -> dict:
        """
        Use GLiNER2 JSON extraction to get structured ad data in one pass.
        This is more efficient than multiple separate extractions.
        """
        try:
            result = self.model.extract_json(
                text,
                schema=self.AD_JSON_SCHEMA,
            )
            
            if isinstance(result, dict):
                # Clean up empty values
                return {k: v for k, v in result.items() if v}
                
        except Exception as e:
            # JSON extraction not available
            pass
        
        return {}
    
    # =========================================================================
    # GLiNER2 RELATION EXTRACTION
    # =========================================================================
    
    def _extract_relations(self, text: str, entities: dict) -> list:
        """
        Extract relationships between entities (e.g., sponsor -> product).
        """
        relations = []
        
        try:
            # Flatten entities for relation extraction
            all_entities = []
            for label, items in entities.items():
                if isinstance(items, list):
                    for item in items:
                        all_entities.append({"text": item, "label": label})
            
            if len(all_entities) >= 2:
                result = self.model.extract_relations(
                    text,
                    entities=all_entities,
                    relation_types=self.RELATION_TYPES,
                )
                
                if isinstance(result, list):
                    relations = result
                    
        except Exception as e:
            # Relation extraction not available
            pass
        
        return relations
    
    # =========================================================================
    # GLiNER2 COMBINED SCHEMA (MOST EFFICIENT)
    # =========================================================================
    
    def _extract_all_combined(self, text: str) -> dict:
        """
        Use GLiNER2 combined schema to do classification, entity extraction,
        and JSON extraction in a single inference pass. Most efficient approach.
        """
        combined_result = {
            "classification": None,
            "entities": {},
            "structured": {},
            "relations": [],
        }
        
        try:
            # Build combined schema
            combined_schema = {
                "classification": {
                    "labels": self.CLASSIFICATION_LABELS,
                    "multi_label": False
                },
                "entities": {
                    "labels": self.ENTITY_SCHEMA
                },
                "json": self.AD_JSON_SCHEMA
            }
            
            result = self.model.extract(
                text,
                schema=combined_schema,
            )
            
            if isinstance(result, dict):
                # Parse classification result
                if "classification" in result:
                    cls_result = result["classification"]
                    if isinstance(cls_result, dict):
                        ad_score = cls_result.get("advertisement", 0)
                        combined_result["classification"] = {
                            "is_ad": ad_score > 0.5,
                            "confidence": ad_score
                        }
                
                # Parse entities
                if "entities" in result:
                    combined_result["entities"] = result["entities"]
                
                # Parse JSON structure
                if "json" in result:
                    combined_result["structured"] = {
                        k: v for k, v in result["json"].items() if v
                    }
                    
        except Exception as e:
            # Combined extraction not available, fall back to individual calls
            pass
        
        return combined_result
    
    # =========================================================================
    # LEGACY PATTERN-BASED DETECTION (BACKUP SIGNALS)
    # =========================================================================
    
    def _extract_entities(self, text: str) -> dict:
        """Legacy entity extraction for backward compatibility"""
        return self._extract_entities_with_descriptions(text)
    
    def _detect_ad_phrases(self, text: str) -> tuple[float, list[str]]:
        """Detect ad-specific phrases in text (backup signal)"""
        text_lower = text.lower()
        matched = []
        max_weight = 0.0
        
        for phrase, weight in self.AD_PHRASES.items():
            if phrase in text_lower:
                matched.append(phrase)
                max_weight = max(max_weight, weight)
        
        return max_weight, matched
    
    def _detect_urls(self, text: str) -> list[str]:
        """Detect URLs/website mentions in text (backup)"""
        matches = self.URL_PATTERN.findall(text)
        return [m[0] if isinstance(m, tuple) else m for m in matches]
    
    def _detect_promo_codes(self, text: str) -> list[str]:
        """Detect promo/coupon codes (backup)"""
        return self.PROMO_CODE_PATTERN.findall(text)
    
    def _detect_discounts(self, text: str) -> list[str]:
        """Detect discount mentions (backup)"""
        matches = self.DISCOUNT_PATTERN.findall(text)
        discounts = []
        for m in matches:
            val = next((v for v in m if v), None)
            if val:
                discounts.append(val)
        return discounts
    
    def _calculate_entity_density(self, text: str, entities: dict) -> float:
        """Calculate entity density score"""
        word_count = len(text.split())
        if word_count < 5:
            return 0.0
        
        entity_count = sum(
            len(v) if isinstance(v, list) else 0 
            for v in entities.values()
        )
        
        density = entity_count / (word_count / 20)
        return min(density, 1.0)
    
    # =========================================================================
    # CONFIDENCE SCORING
    # =========================================================================
    
    def _compute_confidence_score(self, signals: dict, 
                                   classification_result: dict = None,
                                   structured_data: dict = None,
                                   is_brand_love: bool = False) -> float:
        """
        Compute overall confidence score from all signals.
        Now incorporates GLiNER2 classification, structured data, and brand love penalty.
        """
        score = 0.0
        
        # PENALTY: Brand love gets heavily penalized
        if is_brand_love:
            return 0.15  # Very low confidence for brand love/shoutouts
        
        # NEW: Direct classification signal (strongest if available)
        if classification_result and classification_result.get("is_ad"):
            cls_confidence = classification_result.get("confidence", 0)
            score += cls_confidence * 0.4  # Up to 0.4 from classification alone
        
        # NEW: Structured data signals (GLiNER2 JSON extraction)
        if structured_data:
            if structured_data.get("sponsor_name"):
                score += 0.2
            if structured_data.get("promo_code"):
                score += 0.25
            if structured_data.get("sponsor_url"):
                score += 0.15
            if structured_data.get("discount_offer"):
                score += 0.15
        
        # Base NER signal (companies + products present)
        has_company = bool(signals.get("companies")) or bool(signals.get("sponsors"))
        has_product = bool(signals.get("products"))
        
        if has_company and has_product:
            score += 0.15
        elif has_company or has_product:
            score += 0.08
        
        # Ad phrase signal
        phrase_score = signals.get("phrase_score", 0.0)
        if phrase_score > 0:
            score += phrase_score * 0.3
        
        # URL signal (from patterns if not in structured data)
        if signals.get("urls") and not (structured_data and structured_data.get("sponsor_url")):
            score += 0.12
            if has_company:
                score += 0.05
        
        # Promo code signal (from patterns if not in structured data)
        if signals.get("promo_codes") and not (structured_data and structured_data.get("promo_code")):
            score += 0.2
        
        # Discount signal (from patterns)
        if signals.get("discounts") and not (structured_data and structured_data.get("discount_offer")):
            score += 0.1
        
        # Entity density bonus
        density = signals.get("entity_density", 0.0)
        if density > 0.5:
            score += density * 0.08
        
        # Multi-signal boost
        signal_count = sum([
            has_company or has_product,
            phrase_score > 0,
            bool(signals.get("urls")) or bool(structured_data and structured_data.get("sponsor_url")),
            bool(signals.get("promo_codes")) or bool(structured_data and structured_data.get("promo_code")),
            bool(signals.get("discounts")) or bool(structured_data and structured_data.get("discount_offer")),
            bool(classification_result and classification_result.get("is_ad")),
            bool(structured_data and structured_data.get("sponsor_name")),
        ])
        
        if signal_count >= 5:
            score *= 1.25  # 25% boost for very strong multi-signal
        elif signal_count >= 4:
            score *= 1.15
        elif signal_count >= 3:
            score *= 1.08
        
        return min(score, 1.0)
    
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
    
    # =========================================================================
    # MAIN ANALYSIS METHODS
    # =========================================================================
    
    def analyze_segment(self, text: str, start_time: float = 0.0, 
                        end_time: float = 0.0, 
                        use_combined: bool = True,
                        skip_exclusion_check: bool = False) -> AdSegment:
        """
        Analyze a single transcript segment for ad content.
        
        Args:
            text: The transcript text
            start_time: Start timestamp in seconds
            end_time: End timestamp in seconds
            use_combined: Use combined schema for efficiency (recommended)
            skip_exclusion_check: Skip exclusion patterns (for re-analysis)
            
        Returns:
            AdSegment with comprehensive detection results
        """
        # First, check exclusions (unless skipped)
        if not skip_exclusion_check:
            excluded, exclusion_reason = self._check_exclusions(text)
            if excluded:
                return AdSegment(
                    start_time=start_time,
                    end_time=end_time,
                    text=text,
                    confidence=AdConfidence.NONE,
                    confidence_score=0.0,
                    excluded=True,
                    exclusion_reason=exclusion_reason,
                )
        
        signals = {}
        entities = {}
        structured_data = {}
        relations = []
        classification_result = {}
        
        if use_combined:
            # Try combined extraction first (most efficient)
            combined = self._extract_all_combined(text)
            
            if combined.get("entities"):
                entities = combined["entities"]
            if combined.get("structured"):
                structured_data = combined["structured"]
            if combined.get("classification"):
                classification_result = combined["classification"]
            
            # Extract relations if we have entities
            if entities:
                relations = self._extract_relations(text, entities)
        
        # Fallback or supplement with individual extractions if combined didn't work
        if not entities:
            entities = self._extract_entities_with_descriptions(text)
        
        if not structured_data:
            structured_data = self._extract_structured_data(text)
        
        if not classification_result:
            is_ad, cls_conf = self._classify_as_ad(text)
            if is_ad is not None:
                classification_result = {"is_ad": is_ad, "confidence": cls_conf}
        
        # Build signals dict (combining GLiNER2 extraction + pattern backup)
        signals["companies"] = (
            entities.get("company", []) + 
            entities.get("organization", []) +
            entities.get("sponsor", [])
        )
        signals["sponsors"] = entities.get("sponsor", [])
        signals["products"] = entities.get("product", []) + entities.get("brand", [])
        
        # Pattern-based signals (backup/supplement)
        phrase_score, matched_phrases = self._detect_ad_phrases(text)
        signals["phrase_score"] = phrase_score
        signals["matched_phrases"] = matched_phrases
        
        # URLs - prefer structured, fall back to patterns
        structured_url = structured_data.get("sponsor_url", "")
        pattern_urls = self._detect_urls(text)
        signals["urls"] = [structured_url] if structured_url else pattern_urls
        
        # Promo codes - prefer structured, fall back to patterns
        structured_code = structured_data.get("promo_code", "")
        entity_codes = entities.get("promo_code", [])
        pattern_codes = self._detect_promo_codes(text)
        signals["promo_codes"] = (
            [structured_code] if structured_code 
            else entity_codes if entity_codes 
            else pattern_codes
        )
        
        # Discounts - prefer structured, fall back to patterns
        structured_discount = structured_data.get("discount_offer", "")
        entity_discounts = entities.get("discount", [])
        pattern_discounts = self._detect_discounts(text)
        signals["discounts"] = (
            [structured_discount] if structured_discount 
            else entity_discounts if entity_discounts 
            else pattern_discounts
        )
        
        # Entity density
        signals["entity_density"] = self._calculate_entity_density(text, entities)
        
        # Check for brand love/shoutout
        is_brand_love = self._is_brand_love(text, signals, structured_data)
        
        # Check if partial ad (for context expansion)
        is_partial = self._is_partial_ad(text, signals)
        signals["is_partial"] = is_partial
        
        # Compute final score
        confidence_score = self._compute_confidence_score(
            signals, 
            classification_result, 
            structured_data,
            is_brand_love
        )
        confidence = self._score_to_confidence(confidence_score)
        
        return AdSegment(
            start_time=start_time,
            end_time=end_time,
            text=text,
            confidence=confidence,
            confidence_score=confidence_score,
            signals=signals,
            entities=entities,
            structured_data=structured_data,
            relations=relations,
            is_ad_classification=classification_result.get("is_ad"),
            classification_confidence=classification_result.get("confidence", 0),
            is_brand_love=is_brand_love,
        )
    
    def analyze_segment_quick(self, text: str, start_time: float = 0.0,
                               end_time: float = 0.0) -> AdSegment:
        """
        Quick analysis using only classification and basic patterns.
        Use this for initial filtering before deeper analysis.
        """
        # First check exclusions
        excluded, exclusion_reason = self._check_exclusions(text)
        if excluded:
            return AdSegment(
                start_time=start_time,
                end_time=end_time,
                text=text,
                confidence=AdConfidence.NONE,
                confidence_score=0.0,
                excluded=True,
                exclusion_reason=exclusion_reason,
            )
        
        signals = {}
        
        # Quick classification
        is_ad, cls_conf = self._classify_as_ad(text)
        classification_result = {"is_ad": is_ad, "confidence": cls_conf} if is_ad is not None else {}
        
        # Quick pattern matching
        phrase_score, matched_phrases = self._detect_ad_phrases(text)
        signals["phrase_score"] = phrase_score
        signals["matched_phrases"] = matched_phrases
        signals["urls"] = self._detect_urls(text)
        signals["promo_codes"] = self._detect_promo_codes(text)
        signals["discounts"] = self._detect_discounts(text)
        
        # Quick brand love check (no URL, no promo, has brand phrase)
        has_brand_love_phrase = any(p.search(text) for p in self.BRAND_LOVE_COMPILED)
        no_commercial = not (signals["urls"] or signals["promo_codes"] or signals["discounts"])
        is_brand_love = has_brand_love_phrase and no_commercial
        
        # Check partial
        is_partial = self._is_partial_ad(text, signals)
        signals["is_partial"] = is_partial
        
        # Simple scoring
        if is_brand_love:
            score = 0.15
        else:
            score = 0.0
            if classification_result.get("is_ad"):
                score += cls_conf * 0.5
            score += phrase_score * 0.3
            if signals["promo_codes"]:
                score += 0.25
            if signals["urls"]:
                score += 0.15
        
        score = min(score, 1.0)
        
        return AdSegment(
            start_time=start_time,
            end_time=end_time,
            text=text,
            confidence=self._score_to_confidence(score),
            confidence_score=score,
            signals=signals,
            is_ad_classification=is_ad,
            classification_confidence=cls_conf if cls_conf else 0.0,
            is_brand_love=is_brand_love,
        )
    
    def detect_ads(self, transcript_data: dict, 
                   min_confidence: AdConfidence = AdConfidence.MEDIUM,
                   merge_adjacent: bool = True,
                   merge_gap_seconds: float = 5.0,
                   quick_filter: bool = True) -> list[AdSegment]:
        """
        Detect ads in a full transcript.
        
        Args:
            transcript_data: Dict with "transcriptByWords" key containing segments
            min_confidence: Minimum confidence level to report
            merge_adjacent: Whether to merge adjacent ad segments
            merge_gap_seconds: Maximum gap between segments to merge
            quick_filter: Use quick analysis first, then deep analysis on candidates
            
        Returns:
            List of detected AdSegments with full extraction data
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
            
            if quick_filter:
                # Quick filter pass
                quick_result = self.analyze_segment_quick(text, start, end)
                
                # Skip if excluded
                if quick_result.excluded:
                    continue
                
                # Only do deep analysis if quick pass suggests it might be an ad
                if quick_result.confidence.value >= AdConfidence.LOW.value:
                    result = self.analyze_segment(text, start, end, use_combined=True)
                else:
                    result = quick_result
            else:
                result = self.analyze_segment(text, start, end, use_combined=True)
            
            # Skip excluded segments
            if result.excluded:
                continue
            
            if result.confidence.value >= min_confidence.value:
                detected.append(result)
        
        # Merge adjacent segments if requested
        if merge_adjacent and detected:
            detected = self._merge_adjacent_segments(detected, merge_gap_seconds)
        
        return detected
    
    def get_expanded_context(self, transcript_data: dict, 
                              ad_start: float, ad_end: float,
                              context_segments: int = 1) -> tuple[str, float, float]:
        """
        Get expanded context around an ad segment.
        Returns (expanded_text, new_start, new_end)
        
        Args:
            transcript_data: Full transcript data
            ad_start: Original ad start time
            ad_end: Original ad end time
            context_segments: Number of segments before/after to include
        """
        segments = transcript_data.get("transcriptByWords", [])
        
        # Find the segment(s) that overlap with the ad
        ad_indices = []
        for i, seg in enumerate(segments):
            seg_start = seg.get("start", 0)
            seg_end = seg.get("end", 0)
            
            # Check if segment overlaps with ad
            if seg_end >= ad_start and seg_start <= ad_end:
                ad_indices.append(i)
        
        if not ad_indices:
            return "", ad_start, ad_end
        
        # Expand range
        first_idx = max(0, min(ad_indices) - context_segments)
        last_idx = min(len(segments) - 1, max(ad_indices) + context_segments)
        
        # Collect expanded text and times
        expanded_texts = []
        new_start = None
        new_end = None
        
        for i in range(first_idx, last_idx + 1):
            seg = segments[i]
            text = seg.get("words", "")
            if text:
                expanded_texts.append(text)
                
                seg_start = seg.get("start", 0)
                seg_end = seg.get("end", 0)
                
                if new_start is None:
                    new_start = seg_start
                new_end = seg_end
        
        expanded_text = " ".join(expanded_texts)
        return expanded_text, new_start or ad_start, new_end or ad_end
    
    def reanalyze_with_expanded_context(self, transcript_data: dict,
                                         ad_segment: AdSegment,
                                         context_segments: int = 1) -> AdSegment:
        """
        Re-analyze an ad segment with expanded context (sliding window).
        Use this when a partial ad is detected.
        
        Args:
            transcript_data: Full transcript data
            ad_segment: Original ad segment
            context_segments: Number of segments before/after to include
            
        Returns:
            New AdSegment with expanded context analysis
        """
        # Get expanded context
        expanded_text, new_start, new_end = self.get_expanded_context(
            transcript_data,
            ad_segment.start_time,
            ad_segment.end_time,
            context_segments
        )
        
        if not expanded_text or expanded_text == ad_segment.text:
            return ad_segment  # No expansion possible
        
        # Re-analyze with expanded context
        new_result = self.analyze_segment(
            expanded_text, 
            new_start, 
            new_end, 
            use_combined=True,
            skip_exclusion_check=False  # Re-check exclusions with full context
        )
        
        # Mark as context-expanded
        new_result.context_expanded = True
        new_result.original_start = ad_segment.start_time
        new_result.original_end = ad_segment.end_time
        
        return new_result
    
    def extract_ad_data_batch(self, texts: list[str]) -> list[dict]:
        """
        Batch extraction of ad data from multiple text segments.
        More efficient than processing one at a time.
        """
        results = []
        
        for text in texts:
            # Check exclusions first
            excluded, reason = self._check_exclusions(text)
            if excluded:
                results.append({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "excluded": True,
                    "exclusion_reason": reason.value,
                })
                continue
            
            structured = self._extract_structured_data(text)
            entities = self._extract_entities_with_descriptions(text)
            
            results.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "structured": structured,
                "entities": entities,
                "excluded": False,
            })
        
        return results
    
    # =========================================================================
    # MERGING UTILITIES
    # =========================================================================
    
    def _merge_adjacent_segments(self, segments: list[AdSegment], 
                                  max_gap: float) -> list[AdSegment]:
        """Merge adjacent ad segments that are close together"""
        if not segments:
            return segments
        
        sorted_segs = sorted(segments, key=lambda s: s.start_time)
        merged = [sorted_segs[0]]
        
        for seg in sorted_segs[1:]:
            last = merged[-1]
            gap = seg.start_time - last.end_time
            
            if gap <= max_gap:
                # Merge segments
                merged[-1] = AdSegment(
                    start_time=last.start_time,
                    end_time=seg.end_time,
                    text=last.text + " " + seg.text,
                    confidence=max(last.confidence, seg.confidence, key=lambda c: c.value),
                    confidence_score=max(last.confidence_score, seg.confidence_score),
                    signals=self._merge_signals(last.signals, seg.signals),
                    entities=self._merge_entities(last.entities, seg.entities),
                    structured_data=self._merge_structured(last.structured_data, seg.structured_data),
                    relations=last.relations + seg.relations,
                    is_ad_classification=last.is_ad_classification or seg.is_ad_classification,
                    classification_confidence=max(
                        last.classification_confidence, 
                        seg.classification_confidence
                    ),
                    is_brand_love=last.is_brand_love and seg.is_brand_love,  # Only if both are
                    context_expanded=last.context_expanded or seg.context_expanded,
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
            elif isinstance(v1, bool) and isinstance(v2, bool):
                merged[key] = v1 or v2
            else:
                merged[key] = v1 or v2
        return merged
    
    def _merge_entities(self, e1: dict, e2: dict) -> dict:
        """Merge entity dictionaries"""
        merged = {}
        for key in set(e1.keys()) | set(e2.keys()):
            v1 = e1.get(key, [])
            v2 = e2.get(key, [])
            if isinstance(v1, list) and isinstance(v2, list):
                merged[key] = list(set(v1 + v2))
            else:
                merged[key] = v1 or v2
        return merged
    
    def _merge_structured(self, s1: dict, s2: dict) -> dict:
        """Merge structured data, preferring non-empty values"""
        merged = {}
        for key in set(s1.keys()) | set(s2.keys()):
            v1 = s1.get(key, "")
            v2 = s2.get(key, "")
            merged[key] = v1 or v2
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
    print("Loading ad detection model (GLiNER2 with full capabilities)...")
    detector = AdDetector()
    
    # Fetch transcript
    url = URL
    print(f"Fetching transcript from: {url[:60]}...")
    
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    
    # Detect ads
    print("\nAnalyzing transcript for ads (using classification + NER + JSON extraction)...\n")
    print("=" * 70)
    
    ads = detector.detect_ads(
        data,
        min_confidence=AdConfidence.MEDIUM,
        merge_adjacent=True,
        merge_gap_seconds=10.0,
        quick_filter=True  # Use two-pass for efficiency
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
            
            # Show exclusion status
            if ad.excluded:
                print(f"  ❌ EXCLUDED: {ad.exclusion_reason.value}")
                continue
            
            # Show brand love status
            if ad.is_brand_love:
                print(f"  ⚠️ BRAND LOVE/SHOUTOUT (not a sponsored ad)")
            
            # Show context expansion status
            if ad.context_expanded:
                print(f"  🔄 Context expanded from original segment")
            
            # Show classification result
            if ad.is_ad_classification is not None:
                print(f"  GLiNER2 Classification: {'AD' if ad.is_ad_classification else 'NOT AD'} ({ad.classification_confidence:.0%})")
            
            # Show structured data from JSON extraction
            if ad.structured_data:
                print(f"\n  📋 Structured Data (GLiNER2 JSON extraction):")
                for key, value in ad.structured_data.items():
                    if value:
                        print(f"     {key}: {value}")
            
            # Show detected signals
            if ad.signals.get("matched_phrases"):
                print(f"\n  🔍 Phrase signals: {', '.join(ad.signals['matched_phrases'][:5])}")
            if ad.signals.get("companies"):
                print(f"  🏢 Companies:  {', '.join(ad.signals['companies'][:5])}")
            if ad.signals.get("sponsors"):
                print(f"  💰 Sponsors:   {', '.join(ad.signals['sponsors'][:5])}")
            if ad.signals.get("products"):
                print(f"  📦 Products:   {', '.join(ad.signals['products'][:5])}")
            if ad.signals.get("urls"):
                print(f"  🔗 URLs:       {', '.join(ad.signals['urls'][:3])}")
            if ad.signals.get("promo_codes"):
                print(f"  🎟️  Promo codes: {', '.join(ad.signals['promo_codes'])}")
            if ad.signals.get("discounts"):
                print(f"  💸 Discounts:  {', '.join(str(d) for d in ad.signals['discounts'])}")
            
            # Show relations if any
            if ad.relations:
                print(f"\n  🔗 Relations:")
                for rel in ad.relations[:5]:
                    print(f"     {rel}")
            
            # Show entities from NER
            if ad.entities:
                print(f"\n  🏷️  All Entities (GLiNER2 NER):")
                for label, items in ad.entities.items():
                    if items:
                        print(f"     {label}: {', '.join(items[:5])}")
            
            # Show text preview
            preview = ad.text[:200] + "..." if len(ad.text) > 200 else ad.text
            print(f"\n  📝 Text preview:")
            print(f"  \"{preview}\"")
            print()
    
    print("=" * 70)
    print("Detection complete.")
    
    # Also show lower confidence detections for reference
    low_conf_ads = detector.detect_ads(
        data,
        min_confidence=AdConfidence.LOW,
        merge_adjacent=True,
        quick_filter=True,
    )
    
    low_only = [a for a in low_conf_ads if a.confidence == AdConfidence.LOW]
    if low_only:
        print(f"\n[Note: {len(low_only)} additional low-confidence segment(s) detected]")
        brand_love_count = sum(1 for a in low_only if a.is_brand_love)
        if brand_love_count:
            print(f"  - {brand_love_count} are brand love/shoutouts (not sponsored)")