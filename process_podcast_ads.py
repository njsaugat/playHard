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
import requests
import psycopg2
from psycopg2.extras import RealDictCursor, Json
from urllib.parse import urlparse
from datetime import datetime
from dotenv import load_dotenv
from ad_detector import AdDetector, AdConfidence, format_timestamp, ExclusionReason

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

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

    try:
        response = model.generate_content(prompt)
        candidate = response.candidates[0]
        finish_reason = candidate.finish_reason.name

        if finish_reason == "SAFETY":
            print(f"    ⚠️ Blocked by Safety Filters")
            return {"is_ad": False, "confidence": 0, "sponsor": "BLOCKED_BY_SAFETY"}
        
        response_text = candidate.content.parts[0].text
        result = json.loads(response_text)
        
        # If brand love detected, force low confidence
        is_brand_love = result.get("is_brand_love", False)
        if is_brand_love:
            result["confidence"] = min(result.get("confidence", 0.2), BRAND_LOVE_CONFIDENCE)
            result["is_ad"] = False
        
        return {
            "is_ad": bool(result.get("is_ad", False)),
            "is_brand_love": bool(result.get("is_brand_love", False)),
            "confidence": float(result.get("confidence", 0)),
            "sponsor": result.get("sponsor"),
            "sponsor_url": result.get("sponsor_url"),
            "promo_code": result.get("promo_code"),
            "discount_offer": result.get("discount_offer"),
            "product_name": result.get("product_name"),
            "ad_type": result.get("ad_type", "unknown"),
            "call_to_action": result.get("call_to_action"),
            "brand_love_reason": result.get("brand_love_reason"),
        }

    except Exception as e:
        print(f"    ⚠️ Gemini error: {str(e)}")
        return {"is_ad": False, "confidence": 0, "sponsor": None}


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
        "pre_recorded": "PRE_RECORDED",
        "dynamic_insertion": "DYNAMIC",
        "unknown": "UNCLASSIFIED",
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
            AND PodcastEpi.duration >= 7000
            AND PodcastEpi.duration < 7200
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
        llm_val = llm_data.get(llm_key_map.get(field, field))
        gliner_val = gliner_data.get(field)
        if llm_val:
            merged[field] = llm_val
        elif gliner_val:
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
