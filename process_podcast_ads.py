"""
Podcast Ad Detection Pipeline
Fetches episodes from database, runs ad detection, and saves results.
Uses Gemini to verify ads before saving.
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
from ad_detector import AdDetector, AdConfidence, format_timestamp

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
LLM_WEIGHT = 0.6
MODEL_WEIGHT = 0.4
MIN_CONFIDENCE_TO_SAVE = 0.75  # Only save ads with 75%+ combined confidence


def get_context_text(transcript_data: dict, ad_start: float, ad_end: float, context_seconds: float = 30) -> str:
    """
    Get the ad text plus surrounding context from transcript.
    Pulls segments within context_seconds before and after the ad.
    """
    segments = transcript_data.get("transcriptByWords", [])
    
    context_start = ad_start - context_seconds
    context_end = ad_end + context_seconds
    
    relevant_texts = []
    for seg in segments:
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)
        
        # Check if segment overlaps with our context window
        if seg_end >= context_start and seg_start <= context_end:
            text = seg.get("words", "")
            if text:
                relevant_texts.append(text)
    
    return " ".join(relevant_texts)


def verify_ad_with_gemini(text: str) -> dict:
    # 1. Configuration: Ensure enough headroom for tokens
    generation_config = {
        "temperature": 0.1,
        "max_output_tokens": 1000, # Increased slightly to avoid Reason: 2
        "response_mime_type": "application/json",
    }

    # 2. System Instruction: Best practice for consistent JSON
    model = genai.GenerativeModel(
        model_name="gemini-2.5-pro", # Or your preferred version
        generation_config=generation_config,
        system_instruction="You are a specialized classifier. Your task is to identify podcast ads. "
                           "You must respond ONLY with a JSON object."
    )

    prompt = f"""Analyze the following transcript excerpt for sponsored content or product promotions.
    
    TRANSCRIPT:
    \"\"\"{text}\"\"\"

    JSON Schema:
    {{
      "is_ad": boolean,
      "confidence": float,
      "sponsor": string or null
    }}"""

    try:
        response = model.generate_content(prompt)
        
        # 3. Enhanced Error Handling for finish_reason
        candidate = response.candidates[0]
        finish_reason = candidate.finish_reason.name # e.g., 'MAX_TOKENS', 'SAFETY', 'STOP'

        if finish_reason == "SAFETY":
            print(f"    ⚠️ Blocked by Safety Filters")
            return {"is_ad": False, "confidence": 0, "sponsor": "BLOCKED_BY_SAFETY"}
        
        if finish_reason == "MAX_TOKENS":
            print(f"    ⚠️ Truncated (Reason 2: Max Tokens). Consider simplifying prompt.")
            # We can still try to parse what we got, or fail gracefully
        
        # Extract text safely
        response_text = candidate.content.parts[0].text
        result = json.loads(response_text)
        
        return {
            "is_ad": bool(result.get("is_ad", False)),
            "confidence": float(result.get("confidence", 0)),
            "sponsor": result.get("sponsor")
        }

    except Exception as e:
        print(f"    ⚠️ Gemini error: {str(e)}")
        return {"is_ad": False, "confidence": 0, "sponsor": None}

def get_db_connection():
    """Create and return a database connection"""
    return psycopg2.connect(DATABASE_URL)


def extract_domain_from_url(url: str) -> str:
    """Extract clean domain from URL (e.g., 'example.com' from 'https://www.example.com/path')"""
    if not url:
        return ""
    
    # Add scheme if missing for proper parsing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove www. prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except Exception:
        return url.lower()


def find_or_create_brand(cursor, brand_name: str, brand_url: str) -> str:
    """
    Find existing brand by domain or create a new one.
    Returns the brand ID.
    """
    domain = extract_domain_from_url(brand_url)
    
    if not domain:
        # If no URL, try to create domain from brand name
        domain = brand_name.lower().replace(' ', '').replace('.', '') + '.com'
    
    # First, try to find existing brand by domain
    cursor.execute('''
        SELECT id FROM "BrandProfile" 
        WHERE LOWER(domain) = LOWER(%s)
        LIMIT 1
    ''', (domain,))
    
    result = cursor.fetchone()
    if result:
        return result['id'] if isinstance(result, dict) else result[0]
    
    # Also try to find by name (fuzzy match)
    cursor.execute('''
        SELECT id FROM "BrandProfile" 
        WHERE LOWER(name) = LOWER(%s)
        LIMIT 1
    ''', (brand_name,))
    
    result = cursor.fetchone()
    if result:
        return result['id'] if isinstance(result, dict) else result[0]
    
    # Brand doesn't exist, create it
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
    
    # Convert times to integers (seconds)
    start_time = int(ad_segment.start_time)
    end_time = int(ad_segment.end_time)
    
    # Default ad type and format - adjust based on your enum values
    ad_type = 'HOST_READ'  # Common podcast ad type
    ad_format = 'UNCLASSIFIED'
    
    # Confidence score as decimal
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


def process_episode(detector: AdDetector, conn, cursor, episode_id: str, transcript_url: str):
    """Process a single episode for ad detection with Gemini verification"""
    print(f"\n{'='*70}")
    print(f"Processing episode: {episode_id}")
    print(f"Transcript URL: {transcript_url[:80]}...")
    
    # Fetch transcript
    transcript_data = fetch_transcript(transcript_url)
    if not transcript_data:
        return {"detected": 0, "saved": 0}
    
    # Detect ads
    ads = detector.detect_ads(
        transcript_data,
        min_confidence=AdConfidence.MEDIUM,
        merge_adjacent=True,
        merge_gap_seconds=10.0
    )
    
    if not ads:
        print(f"  No ads detected with medium or higher confidence.")
        return {"detected": 0, "saved": 0}
    
    print(f"  Found {len(ads)} candidate ad(s), verifying with Gemini...")
    
    saved_count = 0
    
    for i, ad in enumerate(ads, 1):
        print(f"\n  {'─'*60}")
        print(f"  AD #{i}")
        print(f"  Time:        {format_timestamp(ad.start_time)} → {format_timestamp(ad.end_time)}")
        print(f"  Model Score: {ad.confidence_score:.0%}")
        
        # Get ad text with surrounding context
        context_text = get_context_text(transcript_data, ad.start_time, ad.end_time)
        # Verify with Gemini
        llm_result = verify_ad_with_gemini(context_text)
        
        if llm_result["confidence"] == 0 and not llm_result["is_ad"]:
            # Gemini failed or blocked - use model score only with penalty
            combined_score = ad.confidence_score * 0.5
            print(f"  LLM Score:   N/A (error/blocked)")
        else:
            print(f"  LLM Score:   {llm_result['confidence']:.0%} (is_ad={llm_result['is_ad']})")
            
            if not llm_result["is_ad"]:
                # LLM explicitly says NOT an ad - skip it
                print(f"  ⏭️  SKIPPED - LLM says not an ad")
                continue
            
            # Calculate weighted average
            combined_score = (llm_result["confidence"] * LLM_WEIGHT) + (ad.confidence_score * MODEL_WEIGHT)
        
        print(f"  Combined:    {combined_score:.0%}")
        
        # Check threshold
        if combined_score < MIN_CONFIDENCE_TO_SAVE:
            print(f"  ⏭️  SKIPPED - Below {MIN_CONFIDENCE_TO_SAVE:.0%} threshold")
            continue
        
        # Extract brand info (prefer LLM-detected sponsor)
        companies = ad.signals.get("companies", [])
        urls = ad.signals.get("urls", [])
        promo_codes = ad.signals.get("promo_codes", [])
        
        sponsor_name = llm_result.get("sponsor") or (companies[0] if companies else "Unknown Brand")
        sponsor_url = urls[0] if urls else ""
        sponsor_code = promo_codes[0] if promo_codes else ""
        
        print(f"  Sponsor:     {sponsor_name}")
        
        # Prepare ad content
        ad_content = {
            "sponsorName": sponsor_name,
            "sponsorUrl": sponsor_url,
            "sponsorCode": sponsor_code,
            "allCompanies": companies,
            "allUrls": urls,
            "allPromoCodes": promo_codes,
            "discounts": ad.signals.get("discounts", []),
            "matchedPhrases": ad.signals.get("matched_phrases", []),
            "modelConfidence": ad.confidence_score,
            "llmConfidence": llm_result["confidence"],
            "combinedConfidence": combined_score,
        }
        
        try:
            cursor.execute("SAVEPOINT ad_save")
            
            brand_id = find_or_create_brand(cursor, sponsor_name, sponsor_url)
            
            # Update ad's confidence to combined score before saving
            ad.confidence_score = combined_score
            save_podcast_ad(cursor, episode_id, brand_id, ad, ad_content)
            
            cursor.execute("RELEASE SAVEPOINT ad_save")
            saved_count += 1
            print(f"  ✅ Saved (combined confidence: {combined_score:.0%})")
            
        except Exception as e:
            cursor.execute("ROLLBACK TO SAVEPOINT ad_save")
            print(f"  ❌ Error saving: {e}")
            continue
    
    return {"detected": len(ads), "saved": saved_count}


def main():
    """Main execution function"""
    print("=" * 70)
    print("🎙️  Podcast Ad Detection Pipeline + Gemini Verification")
    print(f"   LLM Weight: {LLM_WEIGHT:.0%} | Model Weight: {MODEL_WEIGHT:.0%}")
    print(f"   Min Confidence: {MIN_CONFIDENCE_TO_SAVE:.0%}")
    print("=" * 70)
    
    # Initialize detector
    print("\n📦 Loading ad detection model...")
    detector = AdDetector()
    
    # Connect to database
    print("🔌 Connecting to database...")
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Fetch episodes
        print("📋 Fetching episodes...")
        episodes = fetch_episodes(cursor)
        print(f"   Found {len(episodes)} episodes to process")
        
        if not episodes:
            print("No episodes found matching criteria.")
            return
        
        total_detected = 0
        total_saved = 0
        episodes_processed = 0
        episodes_with_ads = 0
        
        for episode in episodes:
            episode_id = episode['episodeId']
            transcript_url = episode['transcriptUrl']
            
            try:
                result = process_episode(detector, conn, cursor, episode_id, transcript_url)
                
                total_detected += result["detected"]
                total_saved += result["saved"]
                
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
