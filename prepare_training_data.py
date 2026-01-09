"""
Prepare Training Data for GLiNER2 Ad Classifier

Fetches verified sponsor data from database, processes transcripts,
and creates a training dataset in GLiNER2-compatible format.

Data source: VerifiedEpisodeSponsor table with manually verified labels
Output: CSV for inspection + JSONL for GLiNER2 training
"""

import os
import json
import csv
import requests
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

# Load environment
load_dotenv('.env.local')

DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env.local")


@dataclass
class TrainingExample:
    """Represents a single training example for ad classification"""
    episode_id: str
    transcript_blurb: str
    start_time: float
    end_time: float
    is_ad: bool  # True = ad, False = not ad (is_deleted=True means not ad)
    label: str  # "ad" or "no_ad"
    deletion_reason: Optional[str] = None
    deletion_description: Optional[str] = None
    # Source of timing info
    timing_source: str = "db_column"  # "manually_updated", "ai_updated", or "db_column"
    # Original values for debugging
    original_start: Optional[float] = None
    original_end: Optional[float] = None


def get_db_connection():
    """Create database connection"""
    return psycopg2.connect(DATABASE_URL)


def fetch_verified_sponsors(cursor, limit: int = None) -> List[Dict]:
    """
    Fetch verified sponsor data from database.
    Uses the exact query provided by user.
    """
    query = '''
        SELECT 
            PodcastEpi.id as "episodeId", 
            PodcastEpi."transcriptUrl",
            EpiSponsor.start_time,
            EpiSponsor.end_time,
            VerifiedSpo.ai_updated_details,
            VerifiedSpo.is_deleted,
            VerifiedSpo.deletion_reason,
            VerifiedSpo.deletion_description, 
            VerifiedSpo.is_manually_verified,
            VerifiedSpo.manually_updated_details 
        FROM public."PodcastEpisode" as PodcastEpi
        JOIN public."EpisodeSponsor" as EpiSponsor ON PodcastEpi."id" = EpiSponsor."episodeId"
        JOIN public."VerifiedEpisodeSponsor" as VerifiedSpo ON EpiSponsor."id" = VerifiedSpo."episodeSponsorId"
        JOIN public."Podcast" as Podcast ON Podcast."id" = PodcastEpi."podcastId"
        WHERE PodcastEpi.duration IS NOT NULL 
            AND PodcastEpi."transcriptUrl" IS NOT NULL 
            AND PodcastEpi."audioUrl" IS NOT NULL 
            AND VerifiedSpo.is_manually_verified = true
    '''
    
    if limit:
        query += f' LIMIT {limit}'
    
    cursor.execute(query)
    return cursor.fetchall()


def fetch_transcript(transcript_url: str) -> Optional[Dict]:
    """Fetch and parse transcript JSON from URL"""
    try:
        resp = requests.get(transcript_url, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  ⚠️ Error fetching transcript: {e}")
        return None


def safe_parse_json(data: Any) -> Dict:
    """Safely parse JSON field that might be string or dict"""
    if data is None:
        return {}
    if isinstance(data, dict):
        return data
    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return {}
    return {}


def get_timing_with_priority(record: Dict) -> Tuple[float, float, str]:
    """
    Get start_time and end_time with priority:
    1. manually_updated_details
    2. ai_updated_details  
    3. Direct start_time/end_time columns
    
    Returns: (start_time, end_time, source)
    """
    # Default values from DB columns
    db_start = float(record.get('start_time', 0) or 0)
    db_end = float(record.get('end_time', 0) or 0)
    
    # 1. Check manually_updated_details first (highest priority)
    manually_updated = safe_parse_json(record.get('manually_updated_details'))
    if manually_updated:
        manual_start = manually_updated.get('start_time') or manually_updated.get('startTime')
        manual_end = manually_updated.get('end_time') or manually_updated.get('endTime')
        
        if manual_start is not None and manual_end is not None:
            try:
                return float(manual_start), float(manual_end), "manually_updated"
            except (ValueError, TypeError):
                pass
        # Partial override - check individual fields
        start = float(manual_start) if manual_start is not None else None
        end = float(manual_end) if manual_end is not None else None
        if start is not None or end is not None:
            return (start if start is not None else db_start, 
                    end if end is not None else db_end, 
                    "manually_updated")
    
    # 2. Check ai_updated_details (second priority)
    ai_updated = safe_parse_json(record.get('ai_updated_details'))
    if ai_updated:
        ai_start = ai_updated.get('start_time') or ai_updated.get('startTime')
        ai_end = ai_updated.get('end_time') or ai_updated.get('endTime')
        
        if ai_start is not None and ai_end is not None:
            try:
                return float(ai_start), float(ai_end), "ai_updated"
            except (ValueError, TypeError):
                pass
        # Partial override
        start = float(ai_start) if ai_start is not None else None
        end = float(ai_end) if ai_end is not None else None
        if start is not None or end is not None:
            return (start if start is not None else db_start,
                    end if end is not None else db_end,
                    "ai_updated")
    
    # 3. Use direct DB columns (default)
    return db_start, db_end, "db_column"


def find_closest_word_segment(
    transcript_by_words: List[Dict], 
    target_start: float, 
    target_end: float
) -> Tuple[str, float, float]:
    """
    Find the closest matching transcript segment(s) for given time range.
    Since we can't always have exact matches, find the closest ones.
    
    Returns: (transcript_text, actual_start, actual_end)
    """
    if not transcript_by_words:
        return "", target_start, target_end
    
    matching_words = []
    actual_start = None
    actual_end = None
    
    # Find all segments that overlap with our time range
    # Use a tolerance of 1 second for boundary matching
    tolerance = 1.0
    
    for segment in transcript_by_words:
        seg_start = float(segment.get('start', 0))
        seg_end = float(segment.get('end', 0))
        words = segment.get('words', '')
        
        if not words:
            continue
        
        # Check if segment overlaps with target range (with tolerance)
        # Segment overlaps if: seg_start <= target_end + tol AND seg_end >= target_start - tol
        if seg_start <= target_end + tolerance and seg_end >= target_start - tolerance:
            matching_words.append(words)
            
            if actual_start is None:
                actual_start = seg_start
            actual_end = seg_end
    
    if matching_words:
        transcript_text = ' '.join(matching_words)
        return transcript_text, actual_start or target_start, actual_end or target_end
    
    # If no overlap found, find closest segments
    # Find the segment with start time closest to target_start
    closest_start_idx = None
    closest_start_diff = float('inf')
    
    for i, segment in enumerate(transcript_by_words):
        seg_start = float(segment.get('start', 0))
        diff = abs(seg_start - target_start)
        if diff < closest_start_diff:
            closest_start_diff = diff
            closest_start_idx = i
    
    if closest_start_idx is None:
        return "", target_start, target_end
    
    # Find the segment with end time closest to target_end
    closest_end_idx = closest_start_idx
    closest_end_diff = float('inf')
    
    for i, segment in enumerate(transcript_by_words):
        seg_end = float(segment.get('end', 0))
        diff = abs(seg_end - target_end)
        if diff < closest_end_diff:
            closest_end_diff = diff
            closest_end_idx = i
    
    # Ensure start_idx <= end_idx
    if closest_end_idx < closest_start_idx:
        closest_start_idx, closest_end_idx = closest_end_idx, closest_start_idx
    
    # Collect words from start to end
    words_collected = []
    for i in range(closest_start_idx, closest_end_idx + 1):
        words = transcript_by_words[i].get('words', '')
        if words:
            words_collected.append(words)
    
    if words_collected:
        transcript_text = ' '.join(words_collected)
        actual_start = float(transcript_by_words[closest_start_idx].get('start', target_start))
        actual_end = float(transcript_by_words[closest_end_idx].get('end', target_end))
        return transcript_text, actual_start, actual_end
    
    return "", target_start, target_end


def process_record(record: Dict, transcript_cache: Dict) -> Optional[TrainingExample]:
    """
    Process a single database record into a training example.
    
    Steps:
    1. Get timing with priority (manual > ai > db)
    2. Fetch transcript
    3. Find closest matching words
    4. Determine label based on is_deleted
    """
    episode_id = record['episodeId']
    transcript_url = record.get('transcriptUrl')
    
    if not transcript_url:
        return None
    
    # Get timing with priority
    start_time, end_time, timing_source = get_timing_with_priority(record)
    
    # Validate timing
    if start_time >= end_time or end_time <= 0:
        print(f"  ⚠️ Invalid timing for episode {episode_id}: {start_time} - {end_time}")
        return None
    
    # Fetch transcript (with caching)
    if transcript_url in transcript_cache:
        transcript_data = transcript_cache[transcript_url]
    else:
        transcript_data = fetch_transcript(transcript_url)
        transcript_cache[transcript_url] = transcript_data
    
    if not transcript_data:
        return None
    
    # Get transcript by words
    transcript_by_words = transcript_data.get('transcriptByWords', [])
    if not transcript_by_words:
        # Fallback: use full transcript if no word-level data
        full_transcript = transcript_data.get('transcript', '')
        if not full_transcript:
            return None
        # Can't accurately match timing without word-level data
        print(f"  ⚠️ No word-level transcript for episode {episode_id}")
        return None
    
    # Find closest matching transcript segment
    transcript_blurb, actual_start, actual_end = find_closest_word_segment(
        transcript_by_words, start_time, end_time
    )
    
    if not transcript_blurb or len(transcript_blurb.strip()) < 10:
        print(f"  ⚠️ Empty/short transcript blurb for episode {episode_id}")
        return None
    
    # Determine label based on is_deleted
    # is_deleted = True means it's NOT an ad (was marked for deletion)
    # is_deleted = False means it IS an ad
    is_deleted = bool(record.get('is_deleted', False))
    is_ad = not is_deleted
    label = "ad" if is_ad else "no_ad"
    
    # Get deletion reason if applicable
    deletion_reason = record.get('deletion_reason') if is_deleted else None
    deletion_description = record.get('deletion_description') if is_deleted else None
    
    return TrainingExample(
        episode_id=episode_id,
        transcript_blurb=transcript_blurb.strip(),
        start_time=actual_start,
        end_time=actual_end,
        is_ad=is_ad,
        label=label,
        deletion_reason=deletion_reason,
        deletion_description=deletion_description,
        timing_source=timing_source,
        original_start=start_time if start_time != actual_start else None,
        original_end=end_time if end_time != actual_end else None
    )


def create_gliner2_jsonl_record(example: TrainingExample) -> Dict:
    """
    Convert training example to GLiNER2 JSONL format for classification.
    
    Format:
    {
        "input": "transcript text",
        "output": {
            "classifications": [
                {
                    "task": "ad_detection",
                    "labels": ["ad", "no_ad"],
                    "true_label": ["ad"]  # or ["no_ad"]
                }
            ]
        }
    }
    """
    return {
        "input": example.transcript_blurb,
        "output": {
            "classifications": [
                {
                    "task": "ad_detection",
                    "labels": ["ad", "no_ad"],
                    "true_label": [example.label]
                }
            ]
        }
    }


def create_gliner2_jsonl_with_reason(example: TrainingExample) -> Dict:
    """
    Alternative format that includes deletion reason as a separate classification task.
    Useful for multi-task learning.
    """
    classifications = [
        {
            "task": "ad_detection",
            "labels": ["ad", "no_ad"],
            "true_label": [example.label]
        }
    ]
    
    # Add deletion reason classification if applicable
    if example.deletion_reason:
        # Common deletion reasons based on your system
        reason_labels = [
            "not_sponsor",
            "brand_love",
            "self_promotion",
            "patreon_substack",
            "social_media_plug",
            "ad_free_offer",
            "duplicate",
            "other"
        ]
        
        # Normalize the deletion reason
        reason = example.deletion_reason.lower().replace(' ', '_')
        if reason not in reason_labels:
            reason = "other"
        
        classifications.append({
            "task": "deletion_reason",
            "labels": reason_labels,
            "true_label": [reason]
        })
    
    return {
        "input": example.transcript_blurb,
        "output": {
            "classifications": classifications
        }
    }


def save_to_csv(examples: List[TrainingExample], output_path: str):
    """Save examples to CSV for easy inspection"""
    if not examples:
        print("No examples to save to CSV")
        return
    
    fieldnames = [
        'episode_id', 'label', 'is_ad', 'start_time', 'end_time',
        'timing_source', 'deletion_reason', 'deletion_description',
        'transcript_blurb', 'original_start', 'original_end'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for ex in examples:
            row = {
                'episode_id': ex.episode_id,
                'label': ex.label,
                'is_ad': ex.is_ad,
                'start_time': ex.start_time,
                'end_time': ex.end_time,
                'timing_source': ex.timing_source,
                'deletion_reason': ex.deletion_reason or '',
                'deletion_description': ex.deletion_description or '',
                'transcript_blurb': ex.transcript_blurb[:500] + '...' if len(ex.transcript_blurb) > 500 else ex.transcript_blurb,
                'original_start': ex.original_start or '',
                'original_end': ex.original_end or ''
            }
            writer.writerow(row)
    
    print(f"✅ Saved {len(examples)} examples to {output_path}")


def save_to_jsonl(examples: List[TrainingExample], output_path: str, include_reason: bool = False):
    """Save examples to JSONL for GLiNER2 training"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            if include_reason:
                record = create_gliner2_jsonl_with_reason(ex)
            else:
                record = create_gliner2_jsonl_record(ex)
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"✅ Saved {len(examples)} examples to {output_path}")


def save_full_data_jsonl(examples: List[TrainingExample], output_path: str):
    """Save full training data with all metadata for reference"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            record = {
                'episode_id': ex.episode_id,
                'transcript_blurb': ex.transcript_blurb,
                'start_time': ex.start_time,
                'end_time': ex.end_time,
                'is_ad': ex.is_ad,
                'label': ex.label,
                'deletion_reason': ex.deletion_reason,
                'deletion_description': ex.deletion_description,
                'timing_source': ex.timing_source,
                'gliner2_format': create_gliner2_jsonl_record(ex)
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"✅ Saved {len(examples)} full records to {output_path}")


def prepare_training_data(
    limit: int = None,
    output_dir: str = "./training_data",
    train_split: float = 0.8,
    include_deletion_reason: bool = False
) -> Tuple[List[TrainingExample], List[TrainingExample]]:
    """
    Main function to prepare training data.
    
    Args:
        limit: Maximum number of records to process (None for all)
        output_dir: Directory to save output files
        train_split: Fraction of data for training (rest for validation)
        include_deletion_reason: Include deletion reason as multi-task
    
    Returns:
        (train_examples, val_examples)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("🎯 GLiNER2 Ad Classifier - Training Data Preparation")
    print("=" * 70)
    
    # Connect to database
    print("\n🔌 Connecting to database...")
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Fetch verified sponsor records
        print("📋 Fetching verified sponsor records...")
        records = fetch_verified_sponsors(cursor, limit=limit)
        print(f"   Found {len(records)} verified sponsor records")
        
        if not records:
            print("❌ No records found!")
            return [], []
        
        # Process records
        print("\n📝 Processing records...")
        examples = []
        transcript_cache = {}  # Cache transcripts to avoid re-fetching
        
        ad_count = 0
        no_ad_count = 0
        error_count = 0
        
        for i, record in enumerate(records):
            if (i + 1) % 100 == 0:
                print(f"   Processed {i + 1}/{len(records)} records...")
            
            try:
                example = process_record(record, transcript_cache)
                if example:
                    examples.append(example)
                    if example.is_ad:
                        ad_count += 1
                    else:
                        no_ad_count += 1
                else:
                    error_count += 1
            except Exception as e:
                print(f"  ❌ Error processing record: {e}")
                error_count += 1
                continue
        
        print(f"\n📊 Processing Complete:")
        print(f"   Total examples: {len(examples)}")
        print(f"   Ads: {ad_count}")
        print(f"   Non-ads: {no_ad_count}")
        print(f"   Errors/Skipped: {error_count}")
        
        if not examples:
            print("❌ No valid examples generated!")
            return [], []
        
        # Shuffle and split
        import random
        random.seed(42)  # For reproducibility
        random.shuffle(examples)
        
        split_idx = int(len(examples) * train_split)
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]
        
        print(f"\n📁 Dataset Split:")
        print(f"   Training: {len(train_examples)} examples")
        print(f"   Validation: {len(val_examples)} examples")
        
        # Count labels in each split
        train_ads = sum(1 for ex in train_examples if ex.is_ad)
        train_no_ads = len(train_examples) - train_ads
        val_ads = sum(1 for ex in val_examples if ex.is_ad)
        val_no_ads = len(val_examples) - val_ads
        
        print(f"   Training - Ads: {train_ads}, Non-ads: {train_no_ads}")
        print(f"   Validation - Ads: {val_ads}, Non-ads: {val_no_ads}")
        
        # Save outputs
        print("\n💾 Saving datasets...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # CSV for inspection
        save_to_csv(train_examples, os.path.join(output_dir, f"train_{timestamp}.csv"))
        save_to_csv(val_examples, os.path.join(output_dir, f"val_{timestamp}.csv"))
        
        # JSONL for GLiNER2 training
        save_to_jsonl(
            train_examples, 
            os.path.join(output_dir, f"train_{timestamp}.jsonl"),
            include_reason=include_deletion_reason
        )
        save_to_jsonl(
            val_examples, 
            os.path.join(output_dir, f"val_{timestamp}.jsonl"),
            include_reason=include_deletion_reason
        )
        
        # Full data JSONL for reference
        save_full_data_jsonl(
            examples, 
            os.path.join(output_dir, f"full_data_{timestamp}.jsonl")
        )
        
        # Also save latest versions for easy access
        save_to_jsonl(
            train_examples,
            os.path.join(output_dir, "train_latest.jsonl"),
            include_reason=include_deletion_reason
        )
        save_to_jsonl(
            val_examples,
            os.path.join(output_dir, "val_latest.jsonl"),
            include_reason=include_deletion_reason
        )
        
        # Save dataset stats
        stats = {
            'timestamp': timestamp,
            'total_records': len(records),
            'total_examples': len(examples),
            'train_examples': len(train_examples),
            'val_examples': len(val_examples),
            'train_ads': train_ads,
            'train_no_ads': train_no_ads,
            'val_ads': val_ads,
            'val_no_ads': val_no_ads,
            'error_count': error_count,
            'train_split': train_split,
            'include_deletion_reason': include_deletion_reason
        }
        
        with open(os.path.join(output_dir, "dataset_stats.json"), 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n✅ Dataset preparation complete!")
        print(f"   Output directory: {output_dir}")
        
        return train_examples, val_examples
        
    finally:
        cursor.close()
        conn.close()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare training data for GLiNER2 ad classifier')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of records to process')
    parser.add_argument('--output-dir', type=str, default='./training_data', help='Output directory')
    parser.add_argument('--train-split', type=float, default=0.8, help='Fraction for training (0.8 = 80%)')
    parser.add_argument('--include-reason', action='store_true', help='Include deletion reason as multi-task')
    
    args = parser.parse_args()
    
    prepare_training_data(
        limit=args.limit,
        output_dir=args.output_dir,
        train_split=args.train_split,
        include_deletion_reason=args.include_reason
    )


if __name__ == "__main__":
    main()
