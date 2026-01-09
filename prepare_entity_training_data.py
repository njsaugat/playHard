"""
Prepare Entity Extraction Training Data for GLiNER2

This script creates training data for the entity extraction task:
- Sponsor names
- Product names  
- Promo codes
- URLs/websites
- Discount offers

Uses the same VerifiedEpisodeSponsor data but formats it for NER training.

Usage:
    python prepare_entity_training_data.py
    python prepare_entity_training_data.py --limit 1000
"""

import os
import sys
import json
import csv
import re
import requests
import argparse
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from dotenv import load_dotenv

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("❌ psycopg2 not installed. Run: pip install psycopg2-binary")
    sys.exit(1)

# Load environment
load_dotenv('.env.local')

DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env.local")


@dataclass
class EntityAnnotation:
    """Single entity annotation"""
    text: str
    label: str
    start: int
    end: int
    
    def to_dict(self):
        return {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end
        }


@dataclass
class EntityTrainingExample:
    """Training example for entity extraction"""
    episode_id: str
    transcript_blurb: str
    entities: List[EntityAnnotation]
    is_ad: bool
    
    def to_gliner2_format(self) -> Dict:
        """Convert to GLiNER2 JSONL format"""
        return {
            "input": self.transcript_blurb,
            "output": {
                "entities": [e.to_dict() for e in self.entities]
            }
        }
    
    def to_combined_format(self) -> Dict:
        """Combined classification + entity extraction format"""
        return {
            "input": self.transcript_blurb,
            "output": {
                "classifications": [
                    {
                        "task": "ad_detection",
                        "labels": ["ad", "no_ad"],
                        "true_label": ["ad" if self.is_ad else "no_ad"]
                    }
                ],
                "entities": [e.to_dict() for e in self.entities]
            }
        }


def get_db_connection():
    """Create database connection"""
    return psycopg2.connect(DATABASE_URL)


def fetch_verified_sponsors(cursor, limit: int = None) -> List[Dict]:
    """Fetch verified sponsor data with entity information"""
    query = '''
        SELECT 
            PodcastEpi.id as "episodeId", 
            PodcastEpi."transcriptUrl",
            EpiSponsor.start_time,
            EpiSponsor.end_time,
            EpiSponsor.sponsor_name,
            EpiSponsor.sponsor_url,
            EpiSponsor.product_name,
            VerifiedSpo.ai_updated_details,
            VerifiedSpo.is_deleted,
            VerifiedSpo.is_manually_verified,
            VerifiedSpo.manually_updated_details 
        FROM public."PodcastEpisode" as PodcastEpi
        JOIN public."EpisodeSponsor" as EpiSponsor ON PodcastEpi."id" = EpiSponsor."episodeId"
        JOIN public."VerifiedEpisodeSponsor" as VerifiedSpo ON EpiSponsor."id" = VerifiedSpo."episodeSponsorId"
        WHERE PodcastEpi.duration IS NOT NULL 
            AND PodcastEpi."transcriptUrl" IS NOT NULL 
            AND VerifiedSpo.is_manually_verified = true
            AND VerifiedSpo.is_deleted = false
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
        return None


def safe_parse_json(data: Any) -> Dict:
    """Safely parse JSON field"""
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


def get_timing_with_priority(record: Dict) -> Tuple[float, float]:
    """Get start_time and end_time with priority (manual > ai > db)"""
    db_start = float(record.get('start_time', 0) or 0)
    db_end = float(record.get('end_time', 0) or 0)
    
    # Check manually_updated_details first
    manually_updated = safe_parse_json(record.get('manually_updated_details'))
    if manually_updated:
        manual_start = manually_updated.get('start_time') or manually_updated.get('startTime')
        manual_end = manually_updated.get('end_time') or manually_updated.get('endTime')
        
        if manual_start is not None and manual_end is not None:
            try:
                return float(manual_start), float(manual_end)
            except (ValueError, TypeError):
                pass
    
    # Check ai_updated_details
    ai_updated = safe_parse_json(record.get('ai_updated_details'))
    if ai_updated:
        ai_start = ai_updated.get('start_time') or ai_updated.get('startTime')
        ai_end = ai_updated.get('end_time') or ai_updated.get('endTime')
        
        if ai_start is not None and ai_end is not None:
            try:
                return float(ai_start), float(ai_end)
            except (ValueError, TypeError):
                pass
    
    return db_start, db_end


def find_entity_positions(text: str, entity_value: str) -> List[Tuple[int, int]]:
    """
    Find all positions of an entity value in text.
    Returns list of (start, end) tuples.
    """
    if not entity_value or not text:
        return []
    
    positions = []
    text_lower = text.lower()
    entity_lower = entity_value.lower()
    
    start = 0
    while True:
        pos = text_lower.find(entity_lower, start)
        if pos == -1:
            break
        positions.append((pos, pos + len(entity_value)))
        start = pos + 1
    
    return positions


def find_promo_code_positions(text: str) -> List[Tuple[str, int, int]]:
    """
    Find promo codes using regex patterns.
    Returns list of (code, start, end) tuples.
    """
    patterns = [
        r'\b(?:code|promo|coupon)[:\s]+([A-Z0-9]{3,15})\b',
        r'\buse\s+(?:code\s+)?([A-Z0-9]{3,15})\b',
        r'\b([A-Z0-9]{3,15})\s+(?:at\s+checkout|for\s+\d+%?\s+off)\b',
    ]
    
    results = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            code = match.group(1)
            if code.upper() == code:  # Promo codes are usually uppercase
                results.append((code, match.start(1), match.end(1)))
    
    return results


def find_url_positions(text: str) -> List[Tuple[str, int, int]]:
    """
    Find URLs in text.
    Returns list of (url, start, end) tuples.
    """
    pattern = r'\b(?:https?://)?(?:www\.)?([a-zA-Z0-9][-a-zA-Z0-9]*\.(?:com|co|io|org|net|app|ai|xyz|ly|me|tv|fm)(?:/[^\s]*)?)'
    
    results = []
    for match in re.finditer(pattern, text, re.IGNORECASE):
        results.append((match.group(0), match.start(), match.end()))
    
    return results


def extract_transcript_segment(
    transcript_by_words: List[Dict], 
    target_start: float, 
    target_end: float
) -> Tuple[str, float, float]:
    """Extract transcript segment for given time range"""
    if not transcript_by_words:
        return "", target_start, target_end
    
    matching_words = []
    actual_start = None
    actual_end = None
    tolerance = 1.0
    
    for segment in transcript_by_words:
        seg_start = float(segment.get('start', 0))
        seg_end = float(segment.get('end', 0))
        words = segment.get('words', '')
        
        if not words:
            continue
        
        if seg_start <= target_end + tolerance and seg_end >= target_start - tolerance:
            matching_words.append(words)
            
            if actual_start is None:
                actual_start = seg_start
            actual_end = seg_end
    
    if matching_words:
        transcript_text = ' '.join(matching_words)
        return transcript_text, actual_start or target_start, actual_end or target_end
    
    return "", target_start, target_end


def get_entity_values(record: Dict) -> Dict[str, str]:
    """
    Extract entity values from record, prioritizing manually updated details.
    """
    # Priority: manually_updated > ai_updated > db columns
    manually_updated = safe_parse_json(record.get('manually_updated_details'))
    ai_updated = safe_parse_json(record.get('ai_updated_details'))
    
    entities = {}
    
    # Sponsor name
    sponsor_name = (
        manually_updated.get('sponsorName') or 
        manually_updated.get('sponsor_name') or
        ai_updated.get('sponsorName') or 
        ai_updated.get('sponsor_name') or
        record.get('sponsor_name')
    )
    if sponsor_name:
        entities['sponsor'] = sponsor_name
    
    # Product name
    product_name = (
        manually_updated.get('productName') or 
        manually_updated.get('product_name') or
        ai_updated.get('productName') or 
        ai_updated.get('product_name') or
        record.get('product_name')
    )
    if product_name:
        entities['product'] = product_name
    
    # Sponsor URL
    sponsor_url = (
        manually_updated.get('sponsorUrl') or 
        manually_updated.get('sponsor_url') or
        ai_updated.get('sponsorUrl') or 
        ai_updated.get('sponsor_url') or
        record.get('sponsor_url')
    )
    if sponsor_url:
        entities['website'] = sponsor_url
    
    # Promo code
    promo_code = (
        manually_updated.get('promoCode') or 
        manually_updated.get('promo_code') or
        ai_updated.get('promoCode') or 
        ai_updated.get('promo_code')
    )
    if promo_code:
        entities['promo_code'] = promo_code
    
    # Discount offer
    discount = (
        manually_updated.get('discountOffer') or 
        manually_updated.get('discount_offer') or
        ai_updated.get('discountOffer') or 
        ai_updated.get('discount_offer')
    )
    if discount:
        entities['discount'] = discount
    
    return entities


def create_entity_annotations(text: str, entity_values: Dict[str, str]) -> List[EntityAnnotation]:
    """
    Create entity annotations by finding positions in text.
    """
    annotations = []
    
    for label, value in entity_values.items():
        if not value:
            continue
        
        positions = find_entity_positions(text, value)
        for start, end in positions:
            annotations.append(EntityAnnotation(
                text=text[start:end],  # Use actual text from transcript
                label=label,
                start=start,
                end=end
            ))
    
    # Also find promo codes using patterns (might catch ones not in entity_values)
    if 'promo_code' not in entity_values:
        for code, start, end in find_promo_code_positions(text):
            annotations.append(EntityAnnotation(
                text=code,
                label='promo_code',
                start=start,
                end=end
            ))
    
    # Sort by position and remove duplicates
    annotations = sorted(annotations, key=lambda x: (x.start, -x.end))
    
    # Remove overlapping annotations (keep first/longer)
    filtered = []
    last_end = -1
    for ann in annotations:
        if ann.start >= last_end:
            filtered.append(ann)
            last_end = ann.end
    
    return filtered


def process_record(record: Dict, transcript_cache: Dict) -> Optional[EntityTrainingExample]:
    """Process a single record into an entity training example"""
    episode_id = record['episodeId']
    transcript_url = record.get('transcriptUrl')
    
    if not transcript_url:
        return None
    
    # Get timing
    start_time, end_time = get_timing_with_priority(record)
    
    if start_time >= end_time or end_time <= 0:
        return None
    
    # Fetch transcript
    if transcript_url in transcript_cache:
        transcript_data = transcript_cache[transcript_url]
    else:
        transcript_data = fetch_transcript(transcript_url)
        transcript_cache[transcript_url] = transcript_data
    
    if not transcript_data:
        return None
    
    # Get transcript segment
    transcript_by_words = transcript_data.get('transcriptByWords', [])
    if not transcript_by_words:
        return None
    
    transcript_blurb, _, _ = extract_transcript_segment(
        transcript_by_words, start_time, end_time
    )
    
    if not transcript_blurb or len(transcript_blurb.strip()) < 20:
        return None
    
    # Get entity values from record
    entity_values = get_entity_values(record)
    
    if not entity_values:
        return None  # No entities to annotate
    
    # Create annotations
    annotations = create_entity_annotations(transcript_blurb, entity_values)
    
    if not annotations:
        return None  # Couldn't find entities in text
    
    return EntityTrainingExample(
        episode_id=episode_id,
        transcript_blurb=transcript_blurb.strip(),
        entities=annotations,
        is_ad=True  # These are verified ads
    )


def prepare_entity_training_data(
    limit: int = None,
    output_dir: str = "./training_data",
    train_split: float = 0.8,
    combined_format: bool = False
) -> Tuple[List[EntityTrainingExample], List[EntityTrainingExample]]:
    """
    Main function to prepare entity extraction training data.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("🎯 GLiNER2 Entity Extraction - Training Data Preparation")
    print("=" * 70)
    
    # Connect to database
    print("\n🔌 Connecting to database...")
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Fetch records
        print("📋 Fetching verified sponsor records...")
        records = fetch_verified_sponsors(cursor, limit=limit)
        print(f"   Found {len(records)} verified sponsor records")
        
        if not records:
            print("❌ No records found!")
            return [], []
        
        # Process records
        print("\n📝 Processing records for entity extraction...")
        examples = []
        transcript_cache = {}
        error_count = 0
        
        entity_counts = {
            'sponsor': 0,
            'product': 0,
            'website': 0,
            'promo_code': 0,
            'discount': 0,
        }
        
        for i, record in enumerate(records):
            if (i + 1) % 100 == 0:
                print(f"   Processed {i + 1}/{len(records)} records...")
            
            try:
                example = process_record(record, transcript_cache)
                if example:
                    examples.append(example)
                    for entity in example.entities:
                        entity_counts[entity.label] = entity_counts.get(entity.label, 0) + 1
                else:
                    error_count += 1
            except Exception as e:
                error_count += 1
                continue
        
        print(f"\n📊 Processing Complete:")
        print(f"   Total examples: {len(examples)}")
        print(f"   Errors/Skipped: {error_count}")
        print(f"\n   Entity counts:")
        for label, count in sorted(entity_counts.items(), key=lambda x: -x[1]):
            print(f"      {label}: {count}")
        
        if not examples:
            print("❌ No valid examples generated!")
            return [], []
        
        # Shuffle and split
        import random
        random.seed(42)
        random.shuffle(examples)
        
        split_idx = int(len(examples) * train_split)
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]
        
        print(f"\n📁 Dataset Split:")
        print(f"   Training: {len(train_examples)} examples")
        print(f"   Validation: {len(val_examples)} examples")
        
        # Save outputs
        print("\n💾 Saving datasets...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Entity-only format
        train_entity_path = os.path.join(output_dir, f"train_entities_{timestamp}.jsonl")
        val_entity_path = os.path.join(output_dir, f"val_entities_{timestamp}.jsonl")
        
        with open(train_entity_path, 'w', encoding='utf-8') as f:
            for ex in train_examples:
                f.write(json.dumps(ex.to_gliner2_format(), ensure_ascii=False) + '\n')
        print(f"✅ Saved {len(train_examples)} examples to {train_entity_path}")
        
        with open(val_entity_path, 'w', encoding='utf-8') as f:
            for ex in val_examples:
                f.write(json.dumps(ex.to_gliner2_format(), ensure_ascii=False) + '\n')
        print(f"✅ Saved {len(val_examples)} examples to {val_entity_path}")
        
        # Combined format (classification + entities)
        if combined_format:
            train_combined_path = os.path.join(output_dir, f"train_combined_{timestamp}.jsonl")
            val_combined_path = os.path.join(output_dir, f"val_combined_{timestamp}.jsonl")
            
            with open(train_combined_path, 'w', encoding='utf-8') as f:
                for ex in train_examples:
                    f.write(json.dumps(ex.to_combined_format(), ensure_ascii=False) + '\n')
            print(f"✅ Saved combined format to {train_combined_path}")
            
            with open(val_combined_path, 'w', encoding='utf-8') as f:
                for ex in val_examples:
                    f.write(json.dumps(ex.to_combined_format(), ensure_ascii=False) + '\n')
            print(f"✅ Saved combined format to {val_combined_path}")
        
        # Save latest versions
        with open(os.path.join(output_dir, "train_entities_latest.jsonl"), 'w', encoding='utf-8') as f:
            for ex in train_examples:
                f.write(json.dumps(ex.to_gliner2_format(), ensure_ascii=False) + '\n')
        
        with open(os.path.join(output_dir, "val_entities_latest.jsonl"), 'w', encoding='utf-8') as f:
            for ex in val_examples:
                f.write(json.dumps(ex.to_gliner2_format(), ensure_ascii=False) + '\n')
        
        # Save stats
        stats = {
            'timestamp': timestamp,
            'total_records': len(records),
            'total_examples': len(examples),
            'train_examples': len(train_examples),
            'val_examples': len(val_examples),
            'entity_counts': entity_counts,
            'error_count': error_count,
        }
        
        with open(os.path.join(output_dir, "entity_dataset_stats.json"), 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n✅ Entity extraction dataset preparation complete!")
        print(f"   Output directory: {output_dir}")
        
        return train_examples, val_examples
        
    finally:
        cursor.close()
        conn.close()


def main():
    parser = argparse.ArgumentParser(description='Prepare entity extraction training data')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of records')
    parser.add_argument('--output-dir', type=str, default='./training_data', help='Output directory')
    parser.add_argument('--train-split', type=float, default=0.8, help='Training data fraction')
    parser.add_argument('--combined', action='store_true', 
                        help='Also create combined classification + entity format')
    
    args = parser.parse_args()
    
    prepare_entity_training_data(
        limit=args.limit,
        output_dir=args.output_dir,
        train_split=args.train_split,
        combined_format=args.combined
    )


if __name__ == "__main__":
    main()
