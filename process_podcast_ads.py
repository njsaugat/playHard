"""
Podcast Ad Detection Pipeline
Fetches episodes from database, runs ad detection, and saves results.
Uses GLiNER2's structured extraction + Gemini for verification.

Key features:
- Exclusion filters: Patreon, Substack, social media plugs, ad-free subscriptions
- Brand love detection: Low confidence for shoutouts without CTAs
- Sliding window context expansion for partial ads
- GLiNER2 re-analysis on expanded context
"""

import os
import json
import uuid
import re
import requests
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from urllib.parse import urlparse
from datetime import datetime
from dotenv import load_dotenv
from ad_detector import AdDetector, AdConfidence, format_timestamp, ExclusionReason

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold


# =============================================================================
# ROBUST JSON PARSING UTILITIES
# =============================================================================

def repair_json(text: str) -> str:
    """
    Attempt to repair common JSON issues from LLM responses.
    
    Fixes:
    - Single quotes -> double quotes
    - Trailing commas
    - Unquoted keys
    - Truncated strings (close them)
    - Missing closing braces
    """
    if not text:
        return text
    
    # Remove any markdown code blocks
    text = re.sub(r'^```(?:json)?\s*', '', text.strip())
    text = re.sub(r'\s*```$', '', text.strip())
    
    # Replace single quotes with double quotes (but be careful with apostrophes in text)
    # This is a simple heuristic - replace 'key': with "key":
    text = re.sub(r"'(\w+)'(\s*:)", r'"\1"\2', text)
    
    # Replace trailing single quotes for values that look like strings
    # 'value' -> "value" but only in JSON context
    text = re.sub(r":\s*'([^']*)'(\s*[,}\]])", r': "\1"\2', text)
    
    # Remove trailing commas before } or ]
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Count braces to see if we need to close any
    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')
    
    # Try to fix unterminated strings by finding the last unclosed quote
    # This is tricky - if we have an odd number of quotes, try to close
    quote_count = text.count('"') - text.count('\\"')
    if quote_count % 2 == 1:
        # Find the last quote and close the string
        last_quote_idx = text.rfind('"')
        if last_quote_idx > 0:
            # Check if this quote is starting a string (not ending one)
            # by seeing if there's a colon before it
            before_quote = text[:last_quote_idx].rstrip()
            if before_quote.endswith(':') or before_quote.endswith(',') or before_quote.endswith('['):
                # This is an unclosed string value, close it
                text = text + '"'
    
    # Close any unclosed braces/brackets
    text = text + ('}' * open_braces) + (']' * open_brackets)
    
    return text


def extract_json_from_text(text: str) -> str:
    """
    Extract JSON object from text that may contain extra content.
    Uses regex to find the outermost { } pair.
    """
    if not text:
        return text
    
    # Try to find a JSON object
    # Match from first { to the last }
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    
    # If no complete object found, try to find start of JSON
    start_idx = text.find('{')
    if start_idx >= 0:
        return text[start_idx:]
    
    return text


def safe_json_parse(text: str) -> tuple[dict | None, str | None]:
    """
    Safely parse JSON with multiple fallback strategies.
    
    Returns:
        (parsed_dict, error_message) - error_message is None on success
    """
    if not text or not text.strip():
        return None, "Empty response"
    
    original_text = text
    
    # Strategy 1: Direct parse
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result, None
        elif isinstance(result, list) and result and isinstance(result[0], dict):
            return result[0], None
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract JSON object and parse
    try:
        extracted = extract_json_from_text(text)
        result = json.loads(extracted)
        if isinstance(result, dict):
            return result, None
        elif isinstance(result, list) and result and isinstance(result[0], dict):
            return result[0], None
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Repair and parse
    try:
        repaired = repair_json(text)
        result = json.loads(repaired)
        if isinstance(result, dict):
            return result, None
        elif isinstance(result, list) and result and isinstance(result[0], dict):
            return result[0], None
    except json.JSONDecodeError:
        pass
    
    # Strategy 4: Extract and repair
    try:
        extracted = extract_json_from_text(text)
        repaired = repair_json(extracted)
        result = json.loads(repaired)
        if isinstance(result, dict):
            return result, None
        elif isinstance(result, list) and result and isinstance(result[0], dict):
            return result[0], None
    except json.JSONDecodeError as e:
        pass
    
    # Strategy 5: Try to build a minimal valid response from regex patterns
    # This is the last resort - extract key fields manually
    fallback = try_extract_fields_manually(original_text)
    if fallback:
        return fallback, None
    
    return None, "All JSON parsing strategies failed"


def try_extract_fields_manually(text: str) -> dict | None:
    """
    Last resort: try to extract key fields from malformed JSON using regex.
    """
    result = {
        "is_ad": False,
        "is_brand_love": False,
        "confidence": 0.0,
        "sponsor": None,
        "sponsor_url": None,
        "promo_code": None,
        "discount_offer": None,
        "product_name": None,
        "ad_type": "unknown",
        "call_to_action": None,
        "brand_love_reason": None,
    }
    
    found_something = False
    
    # Try to extract is_ad
    is_ad_match = re.search(r'"is_ad"\s*:\s*(true|false)', text, re.IGNORECASE)
    if is_ad_match:
        result["is_ad"] = is_ad_match.group(1).lower() == "true"
        found_something = True
    
    # Try to extract is_brand_love
    brand_love_match = re.search(r'"is_brand_love"\s*:\s*(true|false)', text, re.IGNORECASE)
    if brand_love_match:
        result["is_brand_love"] = brand_love_match.group(1).lower() == "true"
        found_something = True
    
    # Try to extract confidence
    conf_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
    if conf_match:
        try:
            result["confidence"] = float(conf_match.group(1))
            found_something = True
        except ValueError:
            pass
    
    # Try to extract sponsor
    sponsor_match = re.search(r'"sponsor"\s*:\s*"([^"]*)"', text)
    if sponsor_match:
        result["sponsor"] = sponsor_match.group(1)
        found_something = True
    
    # Try to extract sponsor_url
    url_match = re.search(r'"sponsor_url"\s*:\s*"([^"]*)"', text)
    if url_match:
        result["sponsor_url"] = url_match.group(1)
        found_something = True
    
    # Try to extract promo_code
    promo_match = re.search(r'"promo_code"\s*:\s*"([^"]*)"', text)
    if promo_match:
        result["promo_code"] = promo_match.group(1)
        found_something = True
    
    # Try to extract ad_type
    type_match = re.search(r'"ad_type"\s*:\s*"([^"]*)"', text)
    if type_match:
        result["ad_type"] = type_match.group(1)
        found_something = True
    
    return result if found_something else None

# Load environment variables from .env.local
load_dotenv('.env.local')

DATABASE_URL = os.getenv('DATABASE_URL')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env.local")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env.local")

# Setup Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Safety settings to avoid blocking podcast content
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

gemini_model = genai.GenerativeModel("gemini-2.5-pro")

# Weights for confidence scoring
LLM_WEIGHT = 0.5
MODEL_WEIGHT = 0.35
GLINER_CLASSIFICATION_WEIGHT = 0.15
MIN_CONFIDENCE_TO_SAVE = 0.75
HIGH_CONFIDENCE_THRESHOLD = 0.90
BRAND_LOVE_CONFIDENCE = 0.20  # Very low confidence for brand love


def get_context_text(transcript_data: dict, ad_start: float, ad_end: float, context_seconds: float = 30) -> str:
    """Get the ad text plus surrounding context from transcript."""
    segments = transcript_data.get("transcriptByWords", [])
    context_start = ad_start - context_seconds
    context_end = ad_end + context_seconds
    
    relevant_texts = []
    for seg in segments:
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)
        if seg_end >= context_start and seg_start <= context_end:
            text = seg.get("words", "")
            if text:
                relevant_texts.append(text)
    
    return " ".join(relevant_texts)


def verify_ad_with_gemini(text: str, gliner_data: dict = None) -> dict:
    """
    Verify ad with Gemini, detecting brand love/shoutouts vs real ads.
    
    Brand love indicators (LOW CONFIDENCE):
    - No URL, promo code, or discount mentioned
    - Just a casual mention/shoutout
    - No call to action
    """
    generation_config = {
        "temperature": 0.1,
        "max_output_tokens": 1500,
        "response_mime_type": "application/json",
    }

    system_instruction = (
        "You are a specialized classifier for podcast advertisements. "
        "You must distinguish between REAL SPONSORED ADS and BRAND LOVE/SHOUTOUTS. "
        "BRAND LOVE is when someone casually mentions a brand they like WITHOUT: "
        "1) A specific URL or website to visit "
        "2) A promo/discount code "
        "3) A discount offer (like '20% off') "
        "4) A clear call-to-action from the sponsor "
        "If it's just a casual mention with no commercial elements, mark is_brand_love=true. "
        "You must respond ONLY with a JSON object."
    )
    
    if gliner_data and gliner_data.get("sponsor_name"):
        system_instruction += (
            f" GLiNER2 detected potential sponsor: {gliner_data.get('sponsor_name')}. "
            "Verify if this is a real sponsored ad or just brand love/shoutout."
        )
    
    model = genai.GenerativeModel(
        model_name="gemini-2.5-pro",
        generation_config=generation_config,
        system_instruction=system_instruction
    )

    prompt = f"""Analyze the following transcript excerpt. Determine if it's a REAL SPONSORED AD or just BRAND LOVE/SHOUTOUT.

TRANSCRIPT:
\"\"\"{text}\"\"\"

JSON Schema:
{{
  "is_ad": boolean (true if sponsored ad, false if not or if brand love),
  "is_brand_love": boolean (true if just a casual mention/shoutout without commercial elements),
  "confidence": float (0.0-1.0, VERY LOW for brand love),
  "sponsor": string or null (company/brand name),
  "sponsor_url": string or null (website mentioned - REQUIRED for real ads),
  "promo_code": string or null (discount code if mentioned),
  "discount_offer": string or null (e.g., "20% off", "free trial"),
  "product_name": string or null (specific product advertised),
  "ad_type": "host_read" | "pre_recorded" | "unknown",
  "call_to_action": string or null (what listeners are asked to do),
  "brand_love_reason": string or null (why this is brand love, not a real ad)
}}

IMPORTANT: If there's NO URL, NO promo code, NO discount, and NO clear call-to-action,
this is likely BRAND LOVE, not a real ad. Set is_brand_love=true and confidence=0.2 or lower."""

    # Default fallback response
    default_response = {
        "is_ad": False, "is_brand_love": False, "confidence": 0, 
        "sponsor": None, "sponsor_url": None, "promo_code": None,
        "discount_offer": None, "product_name": None, "ad_type": "unknown",
        "call_to_action": None, "brand_love_reason": None,
    }
    
    try:
        response = model.generate_content(prompt)
        
        # 1. Check if we have candidates
        if not response.candidates:
            print(f"    ⚠️ No candidates in Gemini response")
            return default_response
        
        candidate = response.candidates[0]
        finish_reason = getattr(candidate.finish_reason, "name", "UNKNOWN")

        if finish_reason == "SAFETY":
            print(f"    ⚠️ Blocked by Safety Filters")
            return {**default_response, "sponsor": "BLOCKED_BY_SAFETY"}
        
        # 2. Check if we have content and parts
        if not hasattr(candidate, 'content') or candidate.content is None:
            print(f"    ⚠️ No content in Gemini candidate")
            return default_response
            
        if not hasattr(candidate.content, 'parts') or not candidate.content.parts:
            print(f"    ⚠️ No parts in Gemini candidate content")
            return default_response

        # 3. Get response text safely
        try:
            response_text = candidate.content.parts[0].text
        except (IndexError, AttributeError) as e:
            print(f"    ⚠️ Error accessing response text: {str(e)}")
            return default_response
        
        if not response_text or not response_text.strip():
            print(f"    ⚠️ Empty response text from Gemini")
            return default_response
        
        # 4. Use robust JSON parsing with multiple fallback strategies
        result, error = safe_json_parse(response_text)
        
        if result is None:
            print(f"    ⚠️ Gemini JSON Parse Failed: {error}")
            return default_response
        
        # 5. If brand love detected, force low confidence
        is_brand_love = bool(result.get("is_brand_love", False))
        if is_brand_love:
            conf = 0.2
            try:
                raw_conf = result.get("confidence", 0.2)
                conf = float(raw_conf) if raw_conf is not None else 0.2
            except (ValueError, TypeError):
                pass
            result["confidence"] = min(conf, BRAND_LOVE_CONFIDENCE)
            result["is_ad"] = False
        
        # 6. Final Schema Enforcement with Safe Type Conversions
        def safe_str(val):
            if val is None:
                return None
            return str(val).strip() if str(val).strip() else None

        def safe_float(val, default=0.0):
            if val is None:
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default

        def safe_bool(val, default=False):
            if val is None:
                return default
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower() in ('true', '1', 'yes')
            return bool(val)

        return {
            "is_ad": safe_bool(result.get("is_ad")),
            "is_brand_love": safe_bool(result.get("is_brand_love")),
            "confidence": safe_float(result.get("confidence")),
            "sponsor": safe_str(result.get("sponsor")),
            "sponsor_url": safe_str(result.get("sponsor_url")),
            "promo_code": safe_str(result.get("promo_code")),
            "discount_offer": safe_str(result.get("discount_offer")),
            "product_name": safe_str(result.get("product_name")),
            "ad_type": safe_str(result.get("ad_type")) or "unknown",
            "call_to_action": safe_str(result.get("call_to_action")),
            "brand_love_reason": safe_str(result.get("brand_love_reason")),
        }

    except Exception as e:
        print(f"    ⚠️ Gemini unexpected error: {str(e)}")
        return default_response


def get_db_connection():
    """Create and return a database connection"""
    return psycopg2.connect(DATABASE_URL)


def extract_domain_from_url(url: str) -> str:
    """Extract clean domain from URL"""
    if not url:
        return ""
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except Exception:
        return url.lower()


def find_or_create_brand(cursor, brand_name: str, brand_url: str) -> str:
    """Find existing brand by domain or create a new one."""
    domain = extract_domain_from_url(brand_url)
    if not domain:
        domain = brand_name.lower().replace(' ', '').replace('.', '') + '.com'
    
    cursor.execute('''
        SELECT id FROM "BrandProfile" 
        WHERE LOWER(domain) = LOWER(%s)
        LIMIT 1
    ''', (domain,))
    
    result = cursor.fetchone()
    if result:
        return result['id'] if isinstance(result, dict) else result[0]
    
    cursor.execute('''
        SELECT id FROM "BrandProfile" 
        WHERE LOWER(name) = LOWER(%s)
        LIMIT 1
    ''', (brand_name,))
    
    result = cursor.fetchone()
    if result:
        return result['id'] if isinstance(result, dict) else result[0]
    
    brand_id = str(uuid.uuid4())
    cursor.execute('''
        INSERT INTO "BrandProfile" (
            id, status, origin, claimed_by_brand, name, domain, 
            details, social_links, other_details,
            created_at, updated_at
        ) VALUES (
            %s, 'INACTIVE', 'AUTO', false, %s, %s,
            '{}', '{}', '{}',
            NOW(), NOW()
        )
        ON CONFLICT (domain) DO UPDATE SET updated_at = NOW()
        RETURNING id
    ''', (brand_id, brand_name, domain))
    
    result = cursor.fetchone()
    return result['id'] if isinstance(result, dict) else result[0]


def save_podcast_ad(cursor, episode_id: str, brand_id: str, ad_segment, ad_content: dict):
    """Save a detected ad to the PodcastAd table"""
    ad_id = str(uuid.uuid4())
    start_time = int(ad_segment.start_time)
    end_time = int(ad_segment.end_time)
    
    ad_type_map = {
        "host_read": "HOST_READ",
        "pre_recorded": "PRE_PRODUCED",
        "dynamic_insertion": "DYNAMICALLY_INSERTED",
        "unknown": "UNKNOWN",
    }
    extracted_type = ad_content.get("adType", "unknown")
    ad_type = ad_type_map.get(extracted_type, "HOST_READ")
    ad_format = 'UNCLASSIFIED'
    confidence_score = round(ad_segment.confidence_score, 4)
    
    cursor.execute('''
        INSERT INTO "PodcastAd" (
            id, "episodeId", "brandId",
            start_time, end_time,
            ad_type, ad_format, ad_content,
            "confidenceScore",
            created_at, updated_at
        ) VALUES (
            %s, %s, %s,
            %s, %s,
            %s, %s, %s,
            %s,
            NOW(), NOW()
        )
        RETURNING id
    ''', (
        ad_id, episode_id, brand_id,
        start_time, end_time,
        ad_type, ad_format, Json(ad_content),
        confidence_score
    ))
    
    return cursor.fetchone()


def fetch_episodes(cursor, limit: int = 100000):
    """Fetch podcast episodes that need ad detection"""
    query = '''
        SELECT 
            PodcastEpi.id as "episodeId", 
            PodcastEpi."transcriptUrl"
        FROM public."PodcastEpisode" as PodcastEpi
        JOIN public."Podcast" as Podcast ON Podcast."id" = PodcastEpi."podcastId"
        WHERE PodcastEpi.duration IS NOT NULL 
            AND PodcastEpi.duration >= 5400
            AND PodcastEpi.duration < 6000
            AND PodcastEpi."transcriptUrl" IS NOT NULL 
            AND PodcastEpi."audioUrl" IS NOT NULL 
            AND PodcastEpi."transcriptStatus" = 'COMPLETED' 
            AND PodcastEpi."title" NOT LIKE '%%12 Hours%%'
        ORDER BY Podcast."audienceSize" DESC
        LIMIT %s
    '''
    cursor.execute(query, (limit,))
    return cursor.fetchall()


def fetch_transcript(transcript_url: str) -> dict:
    """Fetch transcript JSON from URL"""
    try:
        resp = requests.get(transcript_url, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  ❌ Error fetching transcript: {e}")
        return None


def merge_extraction_data(gliner_data: dict, llm_data: dict) -> dict:
    """Merge GLiNER2 structured data with Gemini extraction."""
    merged = {}
    
    # Ensure inputs are dictionaries
    if not isinstance(gliner_data, dict):
        gliner_data = {}
    if not isinstance(llm_data, dict):
        llm_data = {}
        
    fields = [
        "sponsor_name", "sponsor_url", "promo_code", "discount_offer",
        "product_name", "ad_type", "call_to_action"
    ]
    llm_key_map = {
        "sponsor_name": "sponsor",
        "sponsor_url": "sponsor_url",
        "promo_code": "promo_code",
        "discount_offer": "discount_offer",
        "product_name": "product_name",
        "ad_type": "ad_type",
        "call_to_action": "call_to_action",
    }
    for field in fields:
        # Use .get() safely
        llm_val = llm_data.get(llm_key_map.get(field, field))
        gliner_val = gliner_data.get(field)
        
        # Prefer non-empty LLM value if available
        if llm_val and str(llm_val).strip():
            merged[field] = llm_val
        elif gliner_val and str(gliner_val).strip():
            merged[field] = gliner_val
    return merged


def build_ad_content(ad, llm_result: dict, combined_score: float) -> dict:
    """Build comprehensive ad_content using GLiNER2 structured data + LLM verification."""
    gliner_structured = ad.structured_data or {}
    merged = merge_extraction_data(gliner_structured, llm_result)
    
    companies = ad.signals.get("companies", [])
    sponsors = ad.signals.get("sponsors", [])
    urls = ad.signals.get("urls", [])
    promo_codes = ad.signals.get("promo_codes", [])
    
    sponsor_name = (
        merged.get("sponsor_name") or
        (sponsors[0] if sponsors else None) or
        (companies[0] if companies else None) or
        "Unknown Brand"
    )
    sponsor_url = merged.get("sponsor_url") or (urls[0] if urls else "")
    sponsor_code = merged.get("promo_code") or (promo_codes[0] if promo_codes else "")
    
    return {
        "sponsorName": sponsor_name,
        "sponsorUrl": sponsor_url,
        "sponsorCode": sponsor_code,
        "productName": merged.get("product_name", ""),
        "discountOffer": merged.get("discount_offer", ""),
        "callToAction": merged.get("call_to_action", ""),
        "adType": merged.get("ad_type", "unknown"),
        "allCompanies": companies,
        "allSponsors": sponsors,
        "allUrls": urls,
        "allPromoCodes": promo_codes,
        "discounts": ad.signals.get("discounts", []),
        "matchedPhrases": ad.signals.get("matched_phrases", []),
        "glinerStructured": gliner_structured,
        "entities": ad.entities,
        "relations": ad.relations if ad.relations else [],
        "glinerClassification": {
            "isAd": ad.is_ad_classification,
            "confidence": ad.classification_confidence,
        },
        "modelConfidence": ad.confidence_score,
        "llmConfidence": llm_result.get("confidence", 0),
        "combinedConfidence": combined_score,
        "isBrandLove": llm_result.get("is_brand_love", False),
        "brandLoveReason": llm_result.get("brand_love_reason"),
        "contextExpanded": ad.context_expanded,
    }


def process_episode(detector: AdDetector, conn, cursor, episode_id: str, transcript_url: str):
    """
    Process a single episode for ad detection.
    
    Features:
    - Exclusion filtering (Patreon, social plugs, ad-free mentions)
    - Brand love detection
    - Sliding window context expansion for partial ads
    - GLiNER2 re-analysis on expanded context
    """
    print(f"\n{'='*70}")
    print(f"Processing episode: {episode_id}")
    print(f"Transcript URL: {transcript_url[:80]}...")
    
    transcript_data = fetch_transcript(transcript_url)
    if not transcript_data:
        return {"detected": 0, "saved": 0, "excluded": 0, "brand_love": 0}
    
    # Detect ads using GLiNER2's full capabilities
    ads = detector.detect_ads(
        transcript_data,
        min_confidence=AdConfidence.MEDIUM,
        merge_adjacent=True,
        merge_gap_seconds=10.0,
        quick_filter=True,
    )
    
    if not ads:
        print(f"  No ads detected with medium or higher confidence.")
        return {"detected": 0, "saved": 0, "excluded": 0, "brand_love": 0}
    
    print(f"  Found {len(ads)} candidate ad(s)")
    
    saved_count = 0
    excluded_count = 0
    brand_love_count = 0
    
    for i, ad in enumerate(ads, 1):
        print(f"\n  {'─'*60}")
        print(f"  AD #{i}")
        print(f"  Time:        {format_timestamp(ad.start_time)} → {format_timestamp(ad.end_time)}")
        print(f"  Model Score: {ad.confidence_score:.0%}")
        
        # Check if excluded by GLiNER2
        if ad.excluded:
            print(f"  ⛔ EXCLUDED by GLiNER2: {ad.exclusion_reason.value}")
            excluded_count += 1
            continue
        
        # Check if GLiNER2 detected brand love
        if ad.is_brand_love:
            print(f"  💝 BRAND LOVE detected by GLiNER2 (not a real ad)")
            brand_love_count += 1
            continue
        
        # Check if partial ad - needs context expansion
        is_partial = ad.signals.get("is_partial", False)
        if is_partial:
            print(f"  🔄 Partial ad detected - expanding context...")
            # Expand context with sliding window (1 segment before/after)
            expanded_ad = detector.reanalyze_with_expanded_context(
                transcript_data, ad, context_segments=1
            )
            
            if expanded_ad.context_expanded:
                print(f"  ✅ Context expanded: {format_timestamp(expanded_ad.start_time)} → {format_timestamp(expanded_ad.end_time)}")
                print(f"  📊 New Model Score: {expanded_ad.confidence_score:.0%}")
                
                # Check if expanded context is now excluded
                if expanded_ad.excluded:
                    print(f"  ⛔ EXCLUDED after expansion: {expanded_ad.exclusion_reason.value}")
                    excluded_count += 1
                    continue
                
                # Check if expanded context is brand love
                if expanded_ad.is_brand_love:
                    print(f"  💝 BRAND LOVE after expansion (not a real ad)")
                    brand_love_count += 1
                    continue
                
                # Use expanded result
                ad = expanded_ad
        
        # Show GLiNER2 classification
        if ad.is_ad_classification is not None:
            print(f"  GLiNER2 Class: {'AD' if ad.is_ad_classification else 'NOT AD'} ({ad.classification_confidence:.0%})")
        
        # Show GLiNER2 structured data
        if ad.structured_data:
            gliner_sponsor = ad.structured_data.get("sponsor_name", "")
            if gliner_sponsor:
                print(f"  GLiNER2 Sponsor: {gliner_sponsor}")
        
        # Decide if we need Gemini verification
        skip_llm = (
            ad.confidence_score >= HIGH_CONFIDENCE_THRESHOLD and
            ad.is_ad_classification is True and
            ad.classification_confidence >= 0.8 and
            ad.structured_data.get("sponsor_name") and
            (ad.structured_data.get("sponsor_url") or ad.structured_data.get("promo_code"))
        )
        
        if skip_llm:
            print(f"  ⚡ HIGH CONFIDENCE - Skipping LLM verification")
            llm_result = {
                "is_ad": True,
                "is_brand_love": False,
                "confidence": ad.classification_confidence,
                "sponsor": ad.structured_data.get("sponsor_name"),
                "sponsor_url": ad.structured_data.get("sponsor_url"),
                "promo_code": ad.structured_data.get("promo_code"),
                "discount_offer": ad.structured_data.get("discount_offer"),
            }
            combined_score = ad.confidence_score
        else:
            # Get ad text with surrounding context for Gemini
            context_text = get_context_text(transcript_data, ad.start_time, ad.end_time)
            
            # Pass GLiNER2 data to Gemini for verification
            llm_result = verify_ad_with_gemini(context_text, ad.structured_data)
            
            # Check for brand love from Gemini
            if llm_result.get("is_brand_love"):
                print(f"  💝 BRAND LOVE detected by Gemini")
                if llm_result.get("brand_love_reason"):
                    print(f"     Reason: {llm_result['brand_love_reason']}")
                brand_love_count += 1
                continue
            
            if llm_result["confidence"] == 0 and not llm_result["is_ad"]:
                combined_score = ad.confidence_score * 0.5
                print(f"  LLM Score:   N/A (error/blocked)")
            else:
                print(f"  LLM Score:   {llm_result['confidence']:.0%} (is_ad={llm_result['is_ad']})")
                
                if not llm_result["is_ad"]:
                    # LLM says NOT an ad - but check if GLiNER2 strongly disagrees
                    if ad.is_ad_classification and ad.classification_confidence > 0.7:
                        print(f"  ⚠️ GLiNER2 disagrees with LLM - using hybrid score")
                        combined_score = (
                            ad.confidence_score * MODEL_WEIGHT +
                            ad.classification_confidence * GLINER_CLASSIFICATION_WEIGHT
                        )
                    else:
                        print(f"  ⏭️  SKIPPED - LLM says not an ad")
                        continue
                else:
                    # Calculate weighted average
                    if ad.is_ad_classification is not None:
                        combined_score = (
                            llm_result["confidence"] * LLM_WEIGHT +
                            ad.confidence_score * MODEL_WEIGHT +
                            ad.classification_confidence * GLINER_CLASSIFICATION_WEIGHT
                        )
                    else:
                        combined_score = (
                            llm_result["confidence"] * (LLM_WEIGHT + GLINER_CLASSIFICATION_WEIGHT) +
                            ad.confidence_score * MODEL_WEIGHT
                        )
        
        print(f"  Combined:    {combined_score:.0%}")
        
        # Check threshold
        if combined_score < MIN_CONFIDENCE_TO_SAVE:
            print(f"  ⏭️  SKIPPED - Below {MIN_CONFIDENCE_TO_SAVE:.0%} threshold")
            continue
        
        # Build comprehensive ad content
        ad_content = build_ad_content(ad, llm_result, combined_score)
        
        sponsor_name = ad_content["sponsorName"]
        sponsor_url = ad_content["sponsorUrl"]
        
        print(f"  Sponsor:     {sponsor_name}")
        if ad_content.get("productName"):
            print(f"  Product:     {ad_content['productName']}")
        if ad_content.get("sponsorCode"):
            print(f"  Promo Code:  {ad_content['sponsorCode']}")
        if ad_content.get("sponsorUrl"):
            print(f"  URL:         {ad_content['sponsorUrl']}")
        
        try:
            cursor.execute("SAVEPOINT ad_save")
            brand_id = find_or_create_brand(cursor, sponsor_name, sponsor_url)
            ad.confidence_score = combined_score
            save_podcast_ad(cursor, episode_id, brand_id, ad, ad_content)
            cursor.execute("RELEASE SAVEPOINT ad_save")
            saved_count += 1
            print(f"  ✅ Saved (combined confidence: {combined_score:.0%})")
        except Exception as e:
            cursor.execute("ROLLBACK TO SAVEPOINT ad_save")
            print(f"  ❌ Error saving: {e}")
            continue
    
    return {
        "detected": len(ads),
        "saved": saved_count,
        "excluded": excluded_count,
        "brand_love": brand_love_count
    }


def main():
    """Main execution function"""
    print("=" * 70)
    print("🎙️  Podcast Ad Detection Pipeline")
    print("   Powered by GLiNER2 + Gemini Verification")
    print("=" * 70)
    print(f"   LLM Weight: {LLM_WEIGHT:.0%} | Model Weight: {MODEL_WEIGHT:.0%} | GLiNER Class: {GLINER_CLASSIFICATION_WEIGHT:.0%}")
    print(f"   Min Confidence: {MIN_CONFIDENCE_TO_SAVE:.0%}")
    print(f"   Skip LLM Threshold: {HIGH_CONFIDENCE_THRESHOLD:.0%}")
    print("=" * 70)
    print("\n📋 Exclusion Filters Active:")
    print("   • Patreon/Substack (creator support)")
    print("   • Social media plugs (self-promotion)")
    print("   • Ad-free subscription offers")
    print("   • Brand love/shoutouts (no commercial intent)")
    print("=" * 70)
    
    print("\n📦 Loading ad detection model (GLiNER2 with full capabilities)...")
    detector = AdDetector()
    
    print("🔌 Connecting to database...")
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        print("📋 Fetching episodes...")
        episodes = fetch_episodes(cursor)
        print(f"   Found {len(episodes)} episodes to process")
        
        if not episodes:
            print("No episodes found matching criteria.")
            return
        
        total_detected = 0
        total_saved = 0
        total_excluded = 0
        total_brand_love = 0
        episodes_processed = 0
        episodes_with_ads = 0
        
        for episode in episodes:
            episode_id = episode['episodeId']
            transcript_url = episode['transcriptUrl']
            
            try:
                result = process_episode(detector, conn, cursor, episode_id, transcript_url)
                
                total_detected += result["detected"]
                total_saved += result["saved"]
                total_excluded += result["excluded"]
                total_brand_love += result["brand_love"]
                
                if result["saved"] > 0:
                    episodes_with_ads += 1
                
                conn.commit()
                episodes_processed += 1
                
            except Exception as e:
                print(f"  ❌ Error processing episode {episode_id}: {e}")
                conn.rollback()
                continue
        
        # Final summary
        print("\n" + "=" * 70)
        print("📊 SUMMARY")
        print("=" * 70)
        print(f"   Episodes processed:  {episodes_processed}")
        print(f"   Episodes with ads:   {episodes_with_ads}")
        print(f"   Ads detected:        {total_detected}")
        print(f"   Ads saved:           {total_saved}")
        print(f"   Excluded:            {total_excluded}")
        print(f"   Brand love/shoutouts: {total_brand_love}")
        if total_detected > 0:
            print(f"   Filter rate:         {(1 - total_saved/total_detected)*100:.1f}% noise removed")
        print("=" * 70)
        print("✅ Pipeline complete!")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        conn.rollback()
        raise
    
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    main()
