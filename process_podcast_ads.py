"""
Podcast Ad Detection Pipeline
Fetches episodes from database, runs ad detection, and saves results.
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

# Load environment variables from .env.local
load_dotenv('.env.local')

DATABASE_URL = os.getenv('DATABASE_URL')

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env.local")


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


def fetch_episodes(cursor, limit: int = 1000):
    """Fetch podcast episodes that need ad detection"""
    query = '''
        SELECT 
            PodcastEpi.id as "episodeId", 
            PodcastEpi."transcriptUrl"
        FROM public."PodcastEpisode" as PodcastEpi
        WHERE PodcastEpi.duration IS NOT NULL 
            AND PodcastEpi.duration >= 7200
            AND PodcastEpi.duration < 7300
            AND PodcastEpi."transcriptUrl" IS NOT NULL 
            AND PodcastEpi."audioUrl" IS NOT NULL 
            AND PodcastEpi."transcriptStatus" = 'COMPLETED' 
            AND PodcastEpi."title" NOT LIKE '%%12 Hours%%'
        ORDER BY PodcastEpi.duration desc
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
    """Process a single episode for ad detection"""
    print(f"\n{'='*70}")
    print(f"Processing episode: {episode_id}")
    print(f"Transcript URL: {transcript_url[:80]}...")
    
    # Fetch transcript
    transcript_data = fetch_transcript(transcript_url)
    if not transcript_data:
        return 0
    
    # Detect ads
    ads = detector.detect_ads(
        transcript_data,
        min_confidence=AdConfidence.MEDIUM,
        merge_adjacent=True,
        merge_gap_seconds=10.0
    )
    
    if not ads:
        print(f"  No ads detected with medium or higher confidence.")
        return 0
    
    print(f"  Found {len(ads)} ad segment(s)")
    
    saved_count = 0
    
    for i, ad in enumerate(ads, 1):
        print(f"\n  {'─'*60}")
        print(f"  AD #{i}")
        print(f"  Time:       {format_timestamp(ad.start_time)} → {format_timestamp(ad.end_time)}")
        print(f"  Confidence: {ad.confidence.name} ({ad.confidence_score:.0%})")
        
        # Extract brand information
        companies = ad.signals.get("companies", [])
        urls = ad.signals.get("urls", [])
        promo_codes = ad.signals.get("promo_codes", [])
        discounts = ad.signals.get("discounts", [])
        matched_phrases = ad.signals.get("matched_phrases", [])
        
        # Get primary brand name and URL
        sponsor_name = companies[0] if companies else "Unknown Brand"
        sponsor_url = urls[0] if urls else ""
        sponsor_code = promo_codes[0] if promo_codes else ""
        
        print(f"  Sponsor:    {sponsor_name}")
        if sponsor_url:
            print(f"  URL:        {sponsor_url}")
        if sponsor_code:
            print(f"  Promo Code: {sponsor_code}")
        
        # Prepare ad content JSON
        ad_content = {
            "sponsorName": sponsor_name,
            "sponsorUrl": sponsor_url,
            "sponsorCode": sponsor_code,
            "allCompanies": companies,
            "allUrls": urls,
            "allPromoCodes": promo_codes,
            "discounts": discounts,
            "matchedPhrases": matched_phrases,
            "products": ad.signals.get("products", []),
            "entityDensity": ad.signals.get("entity_density", 0),
            "confidenceLevel": ad.confidence.name
        }
        
        try:
            # Create a savepoint so we can rollback just this ad if it fails
            cursor.execute("SAVEPOINT ad_save")
            
            # Find or create brand
            brand_id = find_or_create_brand(cursor, sponsor_name, sponsor_url)
            print(f"  Brand ID:   {brand_id}")
            
            # Save the ad
            save_podcast_ad(cursor, episode_id, brand_id, ad, ad_content)
            
            # Release savepoint on success
            cursor.execute("RELEASE SAVEPOINT ad_save")
            saved_count += 1
            print(f"  ✅ Saved to database")
            
        except Exception as e:
            # Rollback to savepoint - this clears the aborted transaction state
            # and allows subsequent operations to continue
            cursor.execute("ROLLBACK TO SAVEPOINT ad_save")
            print(f"  ❌ Error saving ad: {e}")
            # Continue with next ad instead of failing completely
            continue
    
    return saved_count


def main():
    """Main execution function"""
    print("=" * 70)
    print("🎙️  Podcast Ad Detection Pipeline")
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
        
        total_ads_saved = 0
        episodes_processed = 0
        episodes_with_ads = 0
        
        for episode in episodes:
            episode_id = episode['episodeId']
            transcript_url = episode['transcriptUrl']
            
            try:
                ads_saved = process_episode(detector, conn, cursor, episode_id, transcript_url)
                
                if ads_saved > 0:
                    episodes_with_ads += 1
                    total_ads_saved += ads_saved
                
                # Commit after each episode (even if no ads, to clear transaction state)
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
        print(f"   Total ads saved:     {total_ads_saved}")
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
