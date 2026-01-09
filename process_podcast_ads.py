"""
Podcast Ad Detection Pipeline
Fetches episodes from database, runs ad detection, and saves results.
Uses GLiNER2's structured extraction + Gemini for verification.

Architecture: Event-driven with async batched DB writes
- ML inference emits events to an in-memory queue
- Background thread consumes events and batches DB writes
- Decouples CPU-bound model work from I/O-bound database operations

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
from psycopg2.extras import RealDictCursor, Json, execute_batch
from urllib.parse import urlparse
from datetime import datetime
from dotenv import load_dotenv
from ad_detector import AdDetector, AdConfidence, format_timestamp, ExclusionReason
from dataclasses import dataclass, field
from typing import Optional, Any, Callable
from enum import Enum
import threading
import queue
import time
import atexit
from contextlib import contextmanager
import difflib

# Multiprocessing for true parallelism (bypasses GIL)
from multiprocessing import Pool, cpu_count, Manager, current_process
import multiprocessing

# =============================================================================
# MULTIPROCESSING WORKER STATE
# =============================================================================
# Each worker process gets its own detector and Gemini model instance.
# These are initialized once per process via worker_init().

_worker_detector: Optional[AdDetector] = None
_worker_gemini_model = None
_worker_process_name: str = ""

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold


# =============================================================================
# EVENT SYSTEM - Decouples ML inference from DB writes
# =============================================================================

class EventType(Enum):
    """Types of events that can be emitted by the processing pipeline"""
    AD_DETECTED = "ad_detected"
    BRAND_LOVE_DETECTED = "brand_love_detected"
    SHUTDOWN = "shutdown"
    FLUSH = "flush"


@dataclass
class AdEvent:
    """
    Event emitted when an ad is detected.
    Contains all data needed to persist to database.
    """
    event_type: EventType
    episode_id: str
    sponsor_name: str
    sponsor_url: str
    start_time: int
    end_time: int
    confidence_score: float
    ad_content: dict
    product_name: str = ""
    ad_type: str = "HOST_READ"
    ad_format: str = "UNCLASSIFIED"
    is_brand_love: bool = False
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        # Ensure times are integers
        self.start_time = int(self.start_time)
        self.end_time = int(self.end_time)
        self.confidence_score = round(self.confidence_score, 4)


@dataclass
class ShutdownEvent:
    """Signal to shutdown the writer thread"""
    event_type: EventType = EventType.SHUTDOWN


@dataclass
class FlushEvent:
    """Signal to flush pending writes immediately"""
    event_type: EventType = EventType.FLUSH
    wait_event: threading.Event = field(default_factory=threading.Event)


# =============================================================================
# ASYNC DATABASE WRITER - Background thread for batched writes
# =============================================================================

class AsyncDBWriter:
    """
    Asynchronous database writer with batched operations.
    
    Design:
    - Runs in a background thread, consuming events from a queue
    - Batches writes to minimize database round-trips
    - Flushes on batch size threshold OR time interval (whichever comes first)
    - Thread-safe queue operations
    - Graceful shutdown with final flush
    
    Trade-offs (In-Memory Queue):
    + No external dependencies (Redis/Kafka)
    + Low latency for high-throughput
    + Simple incremental adoption
    - Data loss on crash (mitigated by periodic commits)
    - Single-process only (fine for this pipeline)
    """
    
    def __init__(
        self,
        database_url: str,
        batch_size: int = 50,
        flush_interval_seconds: float = 5.0,
        max_queue_size: int = 10000,
    ):
        """
        Initialize the async DB writer.
        
        Args:
            database_url: PostgreSQL connection string
            batch_size: Number of events to batch before writing
            flush_interval_seconds: Max time to wait before flushing
            max_queue_size: Max queue size (backpressure mechanism)
        """
        self.database_url = database_url
        self.batch_size = batch_size
        self.flush_interval = flush_interval_seconds
        self.max_queue_size = max_queue_size
        
        # Thread-safe queue for events
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        
        # Background writer thread
        self._writer_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Stats for monitoring
        self._stats = {
            "events_received": 0,
            "events_written": 0,
            "batches_written": 0,
            "errors": 0,
            "last_flush_time": None,
        }
        self._stats_lock = threading.Lock()
        
        # Brand cache to avoid repeated lookups
        self._brand_cache: dict[str, str] = {}  # domain/name -> brand_id
        self._brand_cache_lock = threading.Lock()
    
    def start(self):
        """Start the background writer thread"""
        if self._running:
            return
        
        self._running = True
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="AsyncDBWriter",
            daemon=True  # Dies with main thread
        )
        self._writer_thread.start()
        
        # Register cleanup on exit
        atexit.register(self.shutdown)
        
        print("  ✅ AsyncDBWriter started (background thread)")
    
    def shutdown(self, timeout: float = 30.0):
        """
        Graceful shutdown: flush remaining events and stop thread.
        
        Args:
            timeout: Max seconds to wait for flush
        """
        if not self._running:
            return
        
        print("\n  🛑 AsyncDBWriter shutting down...")
        
        # Signal shutdown
        self._running = False
        
        # Send shutdown event to unblock the queue
        try:
            self._queue.put_nowait(ShutdownEvent())
        except queue.Full:
            pass
        
        # Wait for thread to finish
        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=timeout)
        
        # Print final stats
        with self._stats_lock:
            print(f"  📊 Final stats: {self._stats['events_written']}/{self._stats['events_received']} events written, "
                  f"{self._stats['batches_written']} batches, {self._stats['errors']} errors")
    
    def emit(self, event: AdEvent) -> bool:
        """
        Emit an event to the queue (non-blocking).
        
        Args:
            event: AdEvent to queue for writing
            
        Returns:
            True if queued successfully, False if queue is full
        """
        if not self._running:
            print("  ⚠️ AsyncDBWriter not running, event dropped")
            return False
        
        try:
            self._queue.put_nowait(event)
            with self._stats_lock:
                self._stats["events_received"] += 1
            return True
        except queue.Full:
            print("  ⚠️ Event queue full, applying backpressure...")
            # Block with timeout as backpressure
            try:
                self._queue.put(event, timeout=5.0)
                with self._stats_lock:
                    self._stats["events_received"] += 1
                return True
            except queue.Full:
                print("  ❌ Event dropped due to queue overflow")
                return False
    
    def flush_sync(self, timeout: float = 10.0) -> bool:
        """
        Synchronously flush all pending events.
        Blocks until flush is complete or timeout.
        
        Args:
            timeout: Max seconds to wait
            
        Returns:
            True if flush completed, False on timeout
        """
        if not self._running:
            return False
        
        flush_event = FlushEvent()
        try:
            self._queue.put(flush_event, timeout=1.0)
            return flush_event.wait_event.wait(timeout=timeout)
        except queue.Full:
            return False
    
    def get_stats(self) -> dict:
        """Get current writer statistics"""
        with self._stats_lock:
            return dict(self._stats)
    
    def _writer_loop(self):
        """
        Main writer loop - runs in background thread.
        Collects events into batches and writes them to the database.
        """
        conn = None
        pending_events: list[AdEvent] = []
        last_flush_time = time.time()
        
        try:
            # Create dedicated connection for this thread
            conn = psycopg2.connect(self.database_url)
            conn.set_session(autocommit=False)
            
            while self._running or not self._queue.empty():
                try:
                    # Calculate time until next forced flush
                    time_since_flush = time.time() - last_flush_time
                    wait_time = max(0.1, self.flush_interval - time_since_flush)
                    
                    # Get event from queue with timeout
                    try:
                        event = self._queue.get(timeout=wait_time)
                    except queue.Empty:
                        event = None
                    
                    # Handle special events
                    if isinstance(event, ShutdownEvent):
                        break
                    
                    if isinstance(event, FlushEvent):
                        # Immediate flush requested
                        if pending_events:
                            self._write_batch(conn, pending_events)
                            pending_events = []
                            last_flush_time = time.time()
                        event.wait_event.set()
                        continue
                    
                    # Add regular event to pending batch
                    if event and isinstance(event, AdEvent):
                        pending_events.append(event)
                    
                    # Check if we should flush
                    should_flush = (
                        len(pending_events) >= self.batch_size or
                        (pending_events and time.time() - last_flush_time >= self.flush_interval)
                    )
                    
                    if should_flush and pending_events:
                        self._write_batch(conn, pending_events)
                        pending_events = []
                        last_flush_time = time.time()
                    
                except Exception as e:
                    print(f"  ❌ Writer loop error: {e}")
                    with self._stats_lock:
                        self._stats["errors"] += 1
                    
                    # Reconnect on connection errors
                    if conn is None or conn.closed:
                        try:
                            conn = psycopg2.connect(self.database_url)
                            conn.set_session(autocommit=False)
                        except Exception as reconnect_error:
                            print(f"  ❌ Reconnect failed: {reconnect_error}")
                            time.sleep(1)
            
            # Final flush on shutdown
            if pending_events:
                print(f"  💾 Final flush: {len(pending_events)} events...")
                self._write_batch(conn, pending_events)
                
        except Exception as e:
            print(f"  ❌ Writer thread fatal error: {e}")
        finally:
            if conn and not conn.closed:
                conn.close()
    
    def _write_batch(self, conn, events: list[AdEvent]):
        """
        Write a batch of events to the database.
        Uses execute_batch for efficient bulk inserts.
        """
        if not events:
            return
        
        cursor = None
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Prepare batch data
            brand_ids = {}  # sponsor_name -> brand_id
            
            # Step 1: Find or create brands (with caching)
            for event in events:
                cache_key = self._get_brand_cache_key(event.sponsor_name, event.sponsor_url)
                
                with self._brand_cache_lock:
                    if cache_key in self._brand_cache:
                        brand_ids[event.sponsor_name] = self._brand_cache[cache_key]
                        continue
                
                # Look up or create brand
                brand_id = self._find_or_create_brand(cursor, event.sponsor_name, event.sponsor_url)
                brand_ids[event.sponsor_name] = brand_id
                
                with self._brand_cache_lock:
                    self._brand_cache[cache_key] = brand_id
            
            # Step 2: Batch insert ads
            ad_records = []
            for event in events:
                ad_id = str(uuid.uuid4())
                brand_id = brand_ids.get(event.sponsor_name)
                
                if not brand_id:
                    continue
                
                # Clean ad_content: remove sponsorName and productName (now separate columns)
                # These are stored in dedicated columns, not in JSON
                clean_ad_content = {k: v for k, v in event.ad_content.items() 
                                   if k not in ("sponsorName", "productName")}
                
                ad_records.append((
                    ad_id,
                    event.episode_id,
                    brand_id,
                    event.start_time,
                    event.end_time,
                    event.ad_type,
                    event.ad_format,
                    Json(clean_ad_content),
                    event.confidence_score,
                    event.sponsor_name,
                    event.product_name or None,
                ))
            
            if ad_records:
                execute_batch(
                    cursor,
                    '''
                    INSERT INTO "PodcastAd" (
                        id, "episodeId", "brandId",
                        start_time, end_time,
                        ad_type, ad_format, ad_content,
                        "confidenceScore",
                        sponsor_name, product_name,
                        created_at, updated_at
                    ) VALUES (
                        %s, %s, %s,
                        %s, %s,
                        %s, %s, %s,
                        %s,
                        %s, %s,
                        NOW(), NOW()
                    )
                    ''',
                    ad_records,
                    page_size=100
                )
            
            conn.commit()
            
            # Update stats
            with self._stats_lock:
                self._stats["events_written"] += len(events)
                self._stats["batches_written"] += 1
                self._stats["last_flush_time"] = datetime.now().isoformat()
            
        except Exception as e:
            print(f"  ❌ Batch write error: {e}")
            if conn and not conn.closed:
                conn.rollback()
            with self._stats_lock:
                self._stats["errors"] += 1
            raise
        finally:
            if cursor:
                cursor.close()
    
    def _get_brand_cache_key(self, name: str, url: str) -> str:
        """Generate cache key for brand lookup"""
        domain = extract_domain_from_url(url) if url else ""
        return f"{name.lower().strip()}|{domain.lower()}"
    
    def _find_or_create_brand(self, cursor, brand_name: str, brand_url: str) -> str:
        """Find existing brand by domain or create a new one."""
        domain = extract_domain_from_url(brand_url)
        if not domain:
            domain = brand_name.lower().replace(' ', '').replace('.', '') + '.com'
        
        # Check by domain
        cursor.execute('''
            SELECT id FROM "BrandProfile" 
            WHERE LOWER(domain) = LOWER(%s)
            LIMIT 1
        ''', (domain,))
        
        result = cursor.fetchone()
        if result:
            return result['id'] if isinstance(result, dict) else result[0]
        
        # Check by name
        cursor.execute('''
            SELECT id FROM "BrandProfile" 
            WHERE LOWER(name) = LOWER(%s)
            LIMIT 1
        ''', (brand_name,))
        
        result = cursor.fetchone()
        if result:
            return result['id'] if isinstance(result, dict) else result[0]
        
        # Create new brand
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


# =============================================================================
# EVENT EMITTER - Helper class for emitting events from processing
# =============================================================================

class AdEventEmitter:
    """
    Helper class for emitting ad detection events.
    Provides a clean interface between the processing logic and the queue.
    """
    
    def __init__(self, writer: AsyncDBWriter):
        self.writer = writer
        self._local_stats = {
            "ads_emitted": 0,
            "brand_love_emitted": 0,
            "duplicates_skipped": 0,
        }
    
    def emit_ad(
        self,
        episode_id: str,
        ad_segment,  # AdSegment from detector
        ad_content: dict,
        is_brand_love: bool = False,
    ) -> bool:
        """
        Emit an ad detection event.
        
        Args:
            episode_id: Episode ID
            ad_segment: AdSegment object from detector
            ad_content: Full ad content dict
            is_brand_love: Whether this is a brand love mention
            
        Returns:
            True if event was queued successfully
        """
        # Determine ad type and format
        if is_brand_love:
            ad_type = "BRAND_LOVE"
            ad_format = "ENDORSEMENT"
        else:
            # Use HOST_READ as default since adType is no longer in ad_content
            ad_type = "HOST_READ"
            ad_format = "UNCLASSIFIED"
        
        event = AdEvent(
            event_type=EventType.BRAND_LOVE_DETECTED if is_brand_love else EventType.AD_DETECTED,
            episode_id=episode_id,
            sponsor_name=ad_content.get("sponsorName", "Unknown Brand"),
            sponsor_url=ad_content.get("sponsorUrl", ""),
            start_time=int(ad_segment.start_time),
            end_time=int(ad_segment.end_time),
            confidence_score=ad_segment.confidence_score,
            ad_content=ad_content,
            product_name=ad_content.get("productName", ""),
            ad_type=ad_type,
            ad_format=ad_format,
            is_brand_love=is_brand_love,
        )
        
        success = self.writer.emit(event)
        
        if success:
            if is_brand_love:
                self._local_stats["brand_love_emitted"] += 1
            else:
                self._local_stats["ads_emitted"] += 1
        
        return success
    
    def get_local_stats(self) -> dict:
        """Get stats for this emitter instance"""
        return dict(self._local_stats)


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

# Deduplication: Skip ads from the same sponsor within this time window (seconds)
AD_DEDUP_WINDOW_SECONDS = 300  # 5 minutes
SPONSOR_SIMILARITY_THRESHOLD = 0.75  # Minimum similarity score to consider sponsors as same

# Multiprocessing configuration
USE_MULTIPROCESSING = True  # Set to False to use single-process mode
NUM_WORKERS = None  # None = use all available CPU cores
WORKER_CHUNK_SIZE = 1  # Episodes per worker task (1 = best for load balancing)


# =============================================================================
# MULTIPROCESSING WORKER FUNCTIONS
# =============================================================================

def worker_init():
    """
    Initialize worker process with its own model instances.
    Called once when each worker process starts.
    
    This loads the GLiNER2 model and Gemini model in each worker,
    ensuring true parallelism without shared state.
    """
    global _worker_detector, _worker_gemini_model, _worker_process_name
    
    _worker_process_name = current_process().name
    print(f"  [{_worker_process_name}] Initializing worker...")
    
    # Load GLiNER2 model (CPU-bound, ~1-2 seconds)
    _worker_detector = AdDetector()
    print(f"  [{_worker_process_name}] ✅ GLiNER2 model loaded")
    
    # Initialize Gemini model for this worker
    _worker_gemini_model = genai.GenerativeModel("gemini-2.5-pro")
    print(f"  [{_worker_process_name}] ✅ Gemini model ready")


def worker_process_episode(episode_data: dict) -> dict:
    """
    Worker function to process a single episode.
    Runs in a separate process with its own model instances.
    
    Args:
        episode_data: Dict with 'episodeId' and 'transcriptUrl'
    
    Returns:
        Dict with processing results and ad events to emit
    """
    global _worker_detector, _worker_gemini_model, _worker_process_name
    
    episode_id = episode_data['episodeId']
    transcript_url = episode_data['transcriptUrl']
    
    result = {
        "episode_id": episode_id,
        "process_name": _worker_process_name,
        "detected": 0,
        "emitted": 0,
        "excluded": 0,
        "brand_love": 0,
        "duplicates": 0,
        "ad_events": [],  # List of ad events to emit
        "error": None,
    }
    
    try:
        # Fetch transcript
        transcript_data = fetch_transcript(transcript_url)
        if not transcript_data:
            result["error"] = "Failed to fetch transcript"
            return result
        
        # Detect ads using worker's GLiNER2 detector (CPU-bound)
        ads = _worker_detector.detect_ads(
            transcript_data,
            min_confidence=AdConfidence.MEDIUM,
            merge_adjacent=True,
            merge_gap_seconds=10.0,
            quick_filter=True,
        )
        
        if not ads:
            return result
        
        result["detected"] = len(ads)
        
        # Track recently processed ads for deduplication
        recent_ads = {}
        
        for ad in ads:
            # Check if excluded by GLiNER2
            if ad.excluded:
                result["excluded"] += 1
                continue
            
            # Check if GLiNER2 detected brand love
            if ad.is_brand_love:
                ad_content = build_brand_love_content(ad, source="gliner")
                if ad_content["sponsorName"] != "Unknown Brand":
                    ad.confidence_score = BRAND_LOVE_CONFIDENCE
                    result["ad_events"].append({
                        "episode_id": episode_id,
                        "ad_segment": {
                            "start_time": ad.start_time,
                            "end_time": ad.end_time,
                            "confidence_score": ad.confidence_score,
                        },
                        "ad_content": ad_content,
                        "is_brand_love": True,
                    })
                    result["brand_love"] += 1
                continue
            
            # Check if partial ad - needs context expansion
            is_partial = ad.signals.get("is_partial", False)
            if is_partial:
                expanded_ad = _worker_detector.reanalyze_with_expanded_context(
                    transcript_data, ad, context_segments=1
                )
                if expanded_ad.context_expanded:
                    if expanded_ad.excluded:
                        result["excluded"] += 1
                        continue
                    if expanded_ad.is_brand_love:
                        ad_content = build_brand_love_content(expanded_ad, source="gliner")
                        if ad_content["sponsorName"] != "Unknown Brand":
                            expanded_ad.confidence_score = BRAND_LOVE_CONFIDENCE
                            result["ad_events"].append({
                                "episode_id": episode_id,
                                "ad_segment": {
                                    "start_time": expanded_ad.start_time,
                                    "end_time": expanded_ad.end_time,
                                    "confidence_score": expanded_ad.confidence_score,
                                },
                                "ad_content": ad_content,
                                "is_brand_love": True,
                            })
                            result["brand_love"] += 1
                        continue
                    ad = expanded_ad
            
            # Decide if we need Gemini verification
            skip_llm = (
                ad.confidence_score >= HIGH_CONFIDENCE_THRESHOLD and
                ad.is_ad_classification is True and
                ad.classification_confidence >= 0.8 and
                ad.structured_data.get("sponsor_name") and
                (ad.structured_data.get("sponsor_url") or ad.structured_data.get("promo_code"))
            )
            
            if skip_llm:
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
                # Get context for Gemini (uses worker's model)
                context_text = get_context_text(transcript_data, ad.start_time, ad.end_time)
                llm_result = verify_ad_with_gemini(context_text, ad.structured_data)
                
                # Check for brand love from Gemini
                if llm_result.get("is_brand_love"):
                    ad_content = build_brand_love_content(ad, source="gemini", reason=llm_result.get("brand_love_reason"))
                    if ad_content["sponsorName"] != "Unknown Brand":
                        ad.confidence_score = BRAND_LOVE_CONFIDENCE
                        result["ad_events"].append({
                            "episode_id": episode_id,
                            "ad_segment": {
                                "start_time": ad.start_time,
                                "end_time": ad.end_time,
                                "confidence_score": ad.confidence_score,
                            },
                            "ad_content": ad_content,
                            "is_brand_love": True,
                        })
                        result["brand_love"] += 1
                    continue
                
                if llm_result["confidence"] == 0 and not llm_result["is_ad"]:
                    combined_score = ad.confidence_score * 0.5
                elif not llm_result["is_ad"]:
                    if ad.is_ad_classification and ad.classification_confidence > 0.7:
                        combined_score = (
                            ad.confidence_score * MODEL_WEIGHT +
                            ad.classification_confidence * GLINER_CLASSIFICATION_WEIGHT
                        )
                    else:
                        continue
                else:
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
            
            # Check threshold
            if combined_score < MIN_CONFIDENCE_TO_SAVE:
                continue
            
            # Build ad content
            ad_content = build_ad_content(ad, llm_result, combined_score)
            sponsor_name = ad_content["sponsorName"]
            
            # Deduplication check
            normalized_sponsor = sponsor_name.lower().strip() if sponsor_name else ""
            similar_match = find_similar_sponsor(sponsor_name, recent_ads)
            if similar_match:
                matched_sponsor, last_end_time = similar_match
                time_since_last = ad.start_time - last_end_time
                if 0 < time_since_last < AD_DEDUP_WINDOW_SECONDS:
                    result["duplicates"] += 1
                    continue
            
            # Add to events list
            result["ad_events"].append({
                "episode_id": episode_id,
                "ad_segment": {
                    "start_time": ad.start_time,
                    "end_time": ad.end_time,
                    "confidence_score": combined_score,
                },
                "ad_content": ad_content,
                "is_brand_love": False,
            })
            result["emitted"] += 1
            
            if normalized_sponsor and normalized_sponsor != "unknown brand":
                recent_ads[normalized_sponsor] = ad.end_time
            
            # Handle multi-ad detection
            likely_multiple = llm_result.get("likely_multiple_ads", False)
            other_sponsors_hint = llm_result.get("other_sponsors_hint") or []
            
            if likely_multiple and other_sponsors_hint:
                processed_hints = {normalized_sponsor}
                EXPANDED_CONTEXT_SECONDS = 60
                expanded_context_text = get_context_text(
                    transcript_data, ad.start_time, ad.end_time,
                    context_seconds=EXPANDED_CONTEXT_SECONDS
                )
                expanded_analysis = _worker_detector.analyze_segment(
                    expanded_context_text,
                    ad.start_time - EXPANDED_CONTEXT_SECONDS,
                    ad.end_time + EXPANDED_CONTEXT_SECONDS,
                    use_combined=True,
                    skip_exclusion_check=True
                )
                
                for hint_sponsor in other_sponsors_hint:
                    hint_normalized = hint_sponsor.lower().strip()
                    if are_sponsors_similar(hint_normalized, normalized_sponsor):
                        continue
                    if any(are_sponsors_similar(hint_normalized, p) for p in processed_hints):
                        continue
                    
                    similar_match = find_similar_sponsor(hint_sponsor, recent_ads)
                    if similar_match:
                        matched_sponsor, last_end_time = similar_match
                        if 0 < ad.start_time - last_end_time < AD_DEDUP_WINDOW_SECONDS:
                            result["duplicates"] += 1
                            processed_hints.add(hint_normalized)
                            continue
                    
                    additional_ad_content = build_additional_ad_content(
                        hint_sponsor, ad, ad_content, expanded_analysis,
                        combined_score * 0.9
                    )
                    
                    if additional_ad_content:
                        result["ad_events"].append({
                            "episode_id": episode_id,
                            "ad_segment": {
                                "start_time": ad.start_time,
                                "end_time": ad.end_time,
                                "confidence_score": additional_ad_content["combinedConfidence"],
                            },
                            "ad_content": additional_ad_content,
                            "is_brand_love": False,
                        })
                        result["emitted"] += 1
                        if hint_normalized != "unknown brand":
                            recent_ads[hint_normalized] = ad.end_time
                    
                    processed_hints.add(hint_normalized)
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def get_optimal_worker_count() -> int:
    """
    Get optimal number of worker processes.
    Uses all available CPU cores minus 1 (for main process overhead).
    """
    if NUM_WORKERS is not None:
        return NUM_WORKERS
    
    cores = cpu_count()
    # Leave 1 core for main process and OS overhead
    optimal = max(1, cores - 1)
    return optimal


def are_sponsors_similar(sponsor1: str, sponsor2: str, threshold: float = SPONSOR_SIMILARITY_THRESHOLD) -> bool:
    """
    Determine if two sponsor names are semantically similar.
    
    Uses a multi-layered approach:
    1. Exact match (after normalization)
    2. Containment check (e.g., "Lenovo" is in "Lenovo Pro")
    3. Token overlap (core brand words match)
    4. Fuzzy string matching (handles typos, variations)
    
    Examples that should match:
    - "Lenovo" and "Lenovo Pro" -> True (containment)
    - "Better Help" and "BetterHelp" -> True (fuzzy match)
    - "Athletic Greens" and "AG1 by Athletic Greens" -> True (token overlap)
    
    Args:
        sponsor1: First sponsor name
        sponsor2: Second sponsor name
        threshold: Minimum similarity score (0.0 to 1.0)
    
    Returns:
        True if sponsors are considered the same, False otherwise
    """
    if not sponsor1 or not sponsor2:
        return False
    
    # Normalize: lowercase, strip whitespace
    s1 = sponsor1.lower().strip()
    s2 = sponsor2.lower().strip()
    
    # 1. Exact match
    if s1 == s2:
        return True
    
    # 2. Containment check - if one fully contains the other
    # Handles: "Lenovo" vs "Lenovo Pro", "AG1" vs "AG1 Athletic Greens"
    if s1 in s2 or s2 in s1:
        return True
    
    # 3. Token-based overlap - check if core brand words match
    # Remove common suffixes/modifiers that don't change brand identity
    noise_words = {'inc', 'llc', 'ltd', 'co', 'company', 'corp', 'pro', 'plus', 
                   'premium', 'business', 'enterprise', 'by', 'the', 'and', 'for'}
    
    tokens1 = set(s1.replace('-', ' ').replace('.', ' ').split()) - noise_words
    tokens2 = set(s2.replace('-', ' ').replace('.', ' ').split()) - noise_words
    
    if tokens1 and tokens2:
        # If any significant token matches, high chance it's the same brand
        common_tokens = tokens1 & tokens2
        if common_tokens:
            # At least one meaningful word matches - likely same sponsor
            # Calculate overlap ratio for confidence
            overlap_ratio = len(common_tokens) / min(len(tokens1), len(tokens2))
            if overlap_ratio >= 0.5:  # At least half the tokens match
                return True
    
    # 4. Fuzzy string matching - handles minor variations
    # Uses SequenceMatcher which is good at finding "gestalt pattern matching"
    # Handles: "BetterHelp" vs "Better Help", "HelloFresh" vs "Hello Fresh"
    ratio = difflib.SequenceMatcher(None, s1, s2).ratio()
    if ratio >= threshold:
        return True
    
    # 5. First-word match (often the brand name)
    # Handles: "Lenovo" vs "Lenovo Pro Solutions", "Nike" vs "Nike Running"
    first_word1 = s1.split()[0] if s1.split() else ""
    first_word2 = s2.split()[0] if s2.split() else ""
    if first_word1 and first_word2 and len(first_word1) >= 3 and len(first_word2) >= 3:
        if first_word1 == first_word2:
            return True
        # Also check fuzzy match on first words (e.g., "betterhelp" vs "better")
        first_word_ratio = difflib.SequenceMatcher(None, first_word1, first_word2).ratio()
        if first_word_ratio >= 0.8:
            return True
    
    return False


def find_similar_sponsor(new_sponsor: str, recent_ads: dict) -> tuple[str, float] | None:
    """
    Find a similar sponsor in the recent_ads dictionary.
    
    Args:
        new_sponsor: The new sponsor name to check
        recent_ads: Dictionary of {normalized_sponsor: end_time}
    
    Returns:
        Tuple of (matching_sponsor_key, end_time) if found, None otherwise
    """
    if not new_sponsor:
        return None
    
    normalized_new = new_sponsor.lower().strip()
    if normalized_new == "unknown brand":
        return None
    
    for existing_sponsor, end_time in recent_ads.items():
        if are_sponsors_similar(normalized_new, existing_sponsor):
            return (existing_sponsor, end_time)
    
    return None


def extract_distinct_sponsors_from_segment(ad_content: dict) -> list[dict]:
    """
    Extract distinct sponsors from a single segment that might contain multiple ads.
    
    Uses the same similarity logic as deduplication to identify when a segment
    contains 2+ different sponsors (e.g., "BetterHelp" and "Athletic Greens" 
    mentioned in the same ad break).
    
    Args:
        ad_content: The ad content dict with allSponsors, allCompanies, allUrls, etc.
    
    Returns:
        List of distinct sponsor dicts. If only one sponsor is found, returns a list
        with just the original ad_content. If multiple distinct sponsors are found,
        returns multiple dicts with sponsor-specific information.
    """
    all_sponsors = ad_content.get("allSponsors", [])
    all_companies = ad_content.get("allCompanies", [])
    all_urls = ad_content.get("allUrls", [])
    all_promo_codes = ad_content.get("allPromoCodes", [])
    
    # Combine sponsors and companies as potential distinct advertisers
    all_potential_sponsors = []
    for s in all_sponsors:
        if s and s.strip() and s.lower() not in ["unknown brand", "unknown"]:
            all_potential_sponsors.append(s.strip())
    for c in all_companies:
        if c and c.strip() and c.lower() not in ["unknown brand", "unknown"]:
            # Only add company if it's not already similar to an existing sponsor
            is_duplicate = False
            for existing in all_potential_sponsors:
                if are_sponsors_similar(c, existing):
                    is_duplicate = True
                    break
            if not is_duplicate:
                all_potential_sponsors.append(c.strip())
    
    # If we have 0 or 1 sponsors, return the original content
    if len(all_potential_sponsors) <= 1:
        return [ad_content]
    
    # Group sponsors by similarity
    sponsor_groups = []  # List of lists, each inner list is similar sponsors
    
    for sponsor in all_potential_sponsors:
        found_group = False
        for group in sponsor_groups:
            # Check if this sponsor is similar to any in the group
            if any(are_sponsors_similar(sponsor, existing) for existing in group):
                group.append(sponsor)
                found_group = True
                break
        if not found_group:
            sponsor_groups.append([sponsor])
    
    # If all sponsors group into one, return original
    if len(sponsor_groups) <= 1:
        return [ad_content]
    
    # Multiple distinct sponsors detected - create separate ad_content for each
    print(f"    🔀 Detected {len(sponsor_groups)} distinct ads in segment:")
    distinct_ads = []
    
    for i, group in enumerate(sponsor_groups):
        # Pick the first (likely best) sponsor name from the group
        primary_sponsor = group[0]
        print(f"       • Ad {i+1}: {primary_sponsor}")
        
        # Try to find a URL that matches this sponsor
        matched_url = ""
        for url in all_urls:
            if url:
                url_lower = url.lower()
                sponsor_lower = primary_sponsor.lower().replace(" ", "").replace("-", "")
                # Check if sponsor name is in the URL
                if sponsor_lower[:4] in url_lower or any(part in url_lower for part in sponsor_lower.split() if len(part) > 3):
                    matched_url = url
                    break
        
        # If no specific URL match, don't assign any URL (avoid wrong attribution)
        sponsor_url = matched_url
        
        # Try to find a promo code that might be for this sponsor
        # Promo codes often contain brand name or first letters
        sponsor_code = ""
        for code in all_promo_codes:
            if code:
                code_lower = code.lower()
                sponsor_lower = primary_sponsor.lower()
                # Check if promo code relates to this sponsor
                if sponsor_lower[:3] in code_lower or code_lower in sponsor_lower:
                    sponsor_code = code
                    break
        
        # Create a new ad_content for this specific sponsor
        new_ad_content = {
            "sponsorName": primary_sponsor,
            "productName": ad_content.get("productName", ""),  # Keep original if any
            "sponsorUrl": sponsor_url,
            "sponsorCode": sponsor_code,
            "discountOffer": ad_content.get("discountOffer", ""),
            "callToAction": ad_content.get("callToAction", ""),
            # Keep original lists for reference
            "allCompanies": group,  # Just this group
            "allSponsors": group,
            "allUrls": [matched_url] if matched_url else [],
            "allPromoCodes": [sponsor_code] if sponsor_code else [],
            "discounts": ad_content.get("discounts", []),
            "matchedPhrases": ad_content.get("matchedPhrases", []),
            "entities": ad_content.get("entities", {}),
            "relations": ad_content.get("relations", []),
            "modelConfidence": ad_content.get("modelConfidence", 0),
            "llmConfidence": ad_content.get("llmConfidence", 0),
            "combinedConfidence": ad_content.get("combinedConfidence", 0),
            "isBrandLove": ad_content.get("isBrandLove", False),
            "contextExpanded": ad_content.get("contextExpanded", False),
            "splitFromMultiAd": True,  # Flag to indicate this was split
            "totalAdsInSegment": len(sponsor_groups),
        }
        distinct_ads.append(new_ad_content)
    
    return distinct_ads


# Async writer configuration
BATCH_SIZE = 50  # Flush after this many events
FLUSH_INTERVAL_SECONDS = 5.0  # Or flush after this many seconds
MAX_QUEUE_SIZE = 10000  # Backpressure threshold


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
  "sponsor": string or null (company/brand name - the PRIMARY sponsor being advertised),
  "sponsor_url": string or null (website mentioned - REQUIRED for real ads),
  "promo_code": string or null (discount code if mentioned),
  "discount_offer": string or null (e.g., "20% off", "free trial"),
  "product_name": string or null (specific product advertised),
  "ad_type": "host_read" | "pre_recorded" | "unknown",
  "call_to_action": string or null (what listeners are asked to do),
  "brand_love_reason": string or null (why this is brand love, not a real ad),
  "likely_multiple_ads": boolean (true if this segment appears to contain ads for MULTIPLE DIFFERENT sponsors/brands),
  "other_sponsors_hint": array of strings or null (if likely_multiple_ads is true, list other sponsor names you detected)
}}

IMPORTANT NOTES:
1. If there's NO URL, NO promo code, NO discount, and NO clear call-to-action,
   this is likely BRAND LOVE, not a real ad. Set is_brand_love=true and confidence=0.2 or lower.

2. MULTIPLE ADS DETECTION: Sometimes a transcript segment contains back-to-back ads for different brands.
   If you notice mentions of multiple distinct sponsors with their own URLs/promo codes/offers,
   set likely_multiple_ads=true and list the other sponsor names in other_sponsors_hint.
   Focus on the PRIMARY sponsor for the main fields, and flag the others for further analysis."""

    # Default fallback response
    default_response = {
        "is_ad": False, "is_brand_love": False, "confidence": 0, 
        "sponsor": None, "sponsor_url": None, "promo_code": None,
        "discount_offer": None, "product_name": None, "ad_type": "unknown",
        "call_to_action": None, "brand_love_reason": None,
        "likely_multiple_ads": False, "other_sponsors_hint": None,
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

        def safe_list(val):
            if val is None:
                return None
            if isinstance(val, list):
                return [str(v).strip() for v in val if v]
            return None

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
            "likely_multiple_ads": safe_bool(result.get("likely_multiple_ads")),
            "other_sponsors_hint": safe_list(result.get("other_sponsors_hint")),
        }

    except Exception as e:
        print(f"    ⚠️ Gemini unexpected error: {str(e)}")
        return default_response


def get_db_connection():
    """Create and return a database connection"""
    conn = psycopg2.connect(DATABASE_URL)
    # Set a longer timeout for the connection
    conn.set_session(autocommit=False)
    return conn


def is_connection_alive(conn) -> bool:
    """Check if database connection is still alive"""
    if conn is None or conn.closed:
        return False
    try:
        # Simple query to test connection
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except Exception:
        return False


def safe_rollback(conn):
    """Safely rollback a connection, handling already-closed connections"""
    if conn is None:
        return
    try:
        if not conn.closed:
            conn.rollback()
    except Exception:
        pass  # Connection already closed or invalid


def safe_close(conn, cursor=None):
    """Safely close cursor and connection"""
    if cursor is not None:
        try:
            cursor.close()
        except Exception:
            pass
    if conn is not None:
        try:
            if not conn.closed:
                conn.close()
        except Exception:
            pass


def reconnect_if_needed(conn, cursor) -> tuple:
    """
    Check if connection is alive, reconnect if needed.
    Returns (conn, cursor) tuple - either existing or new.
    """
    if is_connection_alive(conn):
        return conn, cursor
    
    print("  🔄 Database connection lost, reconnecting...")
    safe_close(conn, cursor)
    
    try:
        new_conn = get_db_connection()
        new_cursor = new_conn.cursor(cursor_factory=RealDictCursor)
        print("  ✅ Reconnected to database")
        return new_conn, new_cursor
    except Exception as e:
        print(f"  ❌ Failed to reconnect: {e}")
        raise


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


def fetch_episodes(cursor, limit: int = 100000):
    """Fetch podcast episodes that need ad detection"""
    query = '''
        SELECT 
            PodcastEpi.id as "episodeId", 
            PodcastEpi."transcriptUrl"
        FROM public."PodcastEpisode" as PodcastEpi
        JOIN public."Podcast" as Podcast ON Podcast."id" = PodcastEpi."podcastId"
        WHERE PodcastEpi.duration IS NOT NULL 
            AND PodcastEpi.duration >= 7400
            AND PodcastEpi.duration < 7900
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
    """
    Build comprehensive ad_content using GLiNER2 structured data + LLM verification.
    
    Returns a dict with two top-level keys for DB columns (sponsorName, productName)
    and cleaned ad_content (without redundant fields like adType, brandLoveReason,
    glinerStructured, glinerClassification).
    """
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
    product_name = merged.get("product_name", "")
    
    # These fields go directly to DB columns, not in ad_content JSON
    # sponsorName -> sponsor_name column
    # productName -> product_name column
    
    # ad_content contains supporting data only - NO redundant fields:
    # Removed: adType, brandLoveReason, glinerStructured, glinerClassification
    return {
        # Top-level fields for separate DB columns
        "sponsorName": sponsor_name,
        "productName": product_name,
        # Fields that go into ad_content JSON column
        "sponsorUrl": sponsor_url,
        "sponsorCode": sponsor_code,
        "discountOffer": merged.get("discount_offer", ""),
        "callToAction": merged.get("call_to_action", ""),
        "allCompanies": companies,
        "allSponsors": sponsors,
        "allUrls": urls,
        "allPromoCodes": promo_codes,
        "discounts": ad.signals.get("discounts", []),
        "matchedPhrases": ad.signals.get("matched_phrases", []),
        "entities": ad.entities,
        "relations": ad.relations if ad.relations else [],
        "modelConfidence": ad.confidence_score,
        "llmConfidence": llm_result.get("confidence", 0),
        "combinedConfidence": combined_score,
        "isBrandLove": llm_result.get("is_brand_love", False),
        "contextExpanded": ad.context_expanded,
    }


def build_brand_love_content(ad, source: str = "gliner", reason: str = None) -> dict:
    """
    Build ad_content for brand love/endorsement mentions.
    
    Returns clean ad_content without redundant fields (adType, brandLoveReason,
    glinerStructured, glinerClassification removed).
    """
    gliner_structured = ad.structured_data or {}
    
    companies = ad.signals.get("companies", [])
    sponsors = ad.signals.get("sponsors", [])
    urls = ad.signals.get("urls", [])
    
    sponsor_name = (
        gliner_structured.get("sponsor_name") or
        (sponsors[0] if sponsors else None) or
        (companies[0] if companies else None) or
        "Unknown Brand"
    )
    sponsor_url = gliner_structured.get("sponsor_url") or (urls[0] if urls else "")
    product_name = gliner_structured.get("product_name", "")
    
    # Top-level fields for DB columns + cleaned ad_content
    # Removed: adType, brandLoveReason, glinerStructured, glinerClassification
    return {
        # Top-level fields for separate DB columns
        "sponsorName": sponsor_name,
        "productName": product_name,
        # Fields that go into ad_content JSON column
        "sponsorUrl": sponsor_url,
        "sponsorCode": "",
        "discountOffer": "",
        "callToAction": "",
        "isBrandLove": True,
        "brandLoveSource": source,  # "gliner" or "gemini"
        "allCompanies": companies,
        "allSponsors": sponsors,
        "allUrls": urls,
        "allPromoCodes": [],
        "discounts": [],
        "matchedPhrases": ad.signals.get("matched_phrases", []),
        "entities": ad.entities,
        "modelConfidence": ad.confidence_score,
        "combinedConfidence": BRAND_LOVE_CONFIDENCE,
        "contextExpanded": ad.context_expanded if hasattr(ad, 'context_expanded') else False,
    }


def build_additional_ad_content(
    hint_sponsor: str,
    original_ad,
    primary_ad_content: dict,
    expanded_analysis,
    base_confidence: float
) -> dict:
    """
    Build ad_content for an additional sponsor found in a multi-ad segment.
    
    Uses the hint_sponsor name from LLM and tries to match it with data
    from the expanded GLiNER analysis.
    
    Args:
        hint_sponsor: Sponsor name from LLM's other_sponsors_hint
        original_ad: The original AdSegment object
        primary_ad_content: The primary ad's content dict (for reference)
        expanded_analysis: AdSegment from expanded context GLiNER analysis
        base_confidence: Base confidence score to use
    
    Returns:
        Ad content dict for the additional sponsor, or None if invalid
    """
    if not hint_sponsor or hint_sponsor.lower().strip() in ["unknown", "unknown brand"]:
        return None
    
    # Try to find matching URL/promo code from expanded analysis
    matched_url = ""
    matched_code = ""
    
    # Check expanded analysis signals for URL matching this sponsor
    if expanded_analysis and expanded_analysis.signals:
        all_urls = expanded_analysis.signals.get("urls", [])
        all_codes = expanded_analysis.signals.get("promo_codes", [])
        
        sponsor_lower = hint_sponsor.lower().replace(" ", "").replace("-", "")
        
        # Find URL that matches the hint sponsor
        for url in all_urls:
            if url:
                url_lower = url.lower()
                # Check if sponsor name appears in URL
                if sponsor_lower[:4] in url_lower or any(
                    part in url_lower for part in sponsor_lower.split() if len(part) > 3
                ):
                    matched_url = url
                    break
        
        # Find promo code that matches the hint sponsor  
        for code in all_codes:
            if code:
                code_lower = code.lower()
                if sponsor_lower[:3] in code_lower or code_lower in sponsor_lower:
                    matched_code = code
                    break
    
    # Build the ad content for this additional sponsor
    return {
        "sponsorName": hint_sponsor,
        "productName": "",  # Unknown for derived ads
        "sponsorUrl": matched_url,
        "sponsorCode": matched_code,
        "discountOffer": "",
        "callToAction": "",
        "allCompanies": [hint_sponsor],
        "allSponsors": [hint_sponsor],
        "allUrls": [matched_url] if matched_url else [],
        "allPromoCodes": [matched_code] if matched_code else [],
        "discounts": [],
        "matchedPhrases": primary_ad_content.get("matchedPhrases", []),
        "entities": expanded_analysis.entities if expanded_analysis else {},
        "relations": [],
        "modelConfidence": base_confidence,
        "llmConfidence": base_confidence,  # Derived from LLM hint
        "combinedConfidence": base_confidence,
        "isBrandLove": False,
        "contextExpanded": True,
        "derivedFromMultiAd": True,  # Flag to indicate this was derived from multi-ad detection
        "primarySponsor": primary_ad_content.get("sponsorName", ""),  # Reference to primary
    }


def process_episode(
    detector: AdDetector,
    emitter: AdEventEmitter,
    episode_id: str,
    transcript_url: str,
    transcript_data: dict = None,
) -> dict:
    """
    Process a single episode for ad detection.
    
    CHANGED: Now emits events to the async queue instead of direct DB writes.
    CPU-bound ML inference is decoupled from I/O-bound database operations.
    
    Features:
    - Exclusion filtering (Patreon, social plugs, ad-free mentions)
    - Brand love detection
    - Sliding window context expansion for partial ads
    - GLiNER2 re-analysis on expanded context
    """
    print(f"\n{'='*70}")
    print(f"Processing episode: {episode_id}")
    print(f"Transcript URL: {transcript_url[:80]}...")
    
    # Fetch transcript if not provided
    if transcript_data is None:
        transcript_data = fetch_transcript(transcript_url)
    
    if not transcript_data:
        return {"detected": 0, "emitted": 0, "excluded": 0, "brand_love": 0, "duplicates": 0}
    
    # Detect ads using GLiNER2's full capabilities (CPU-bound)
    ads = detector.detect_ads(
        transcript_data,
        min_confidence=AdConfidence.MEDIUM,
        merge_adjacent=True,
        merge_gap_seconds=10.0,
        quick_filter=True,
    )
    
    if not ads:
        print(f"  No ads detected with medium or higher confidence.")
        return {"detected": 0, "emitted": 0, "excluded": 0, "brand_love": 0, "duplicates": 0}
    
    print(f"  Found {len(ads)} candidate ad(s)")
    
    emitted_count = 0
    excluded_count = 0
    brand_love_count = 0
    duplicate_count = 0
    
    # Track recently processed ads for deduplication: {normalized_sponsor: end_time}
    recent_ads = {}
    
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
        
        # Check if GLiNER2 detected brand love - emit as endorsement event
        if ad.is_brand_love:
            print(f"  💝 BRAND LOVE detected by GLiNER2")
            ad_content = build_brand_love_content(ad, source="gliner")
            
            if ad_content["sponsorName"] != "Unknown Brand":
                ad.confidence_score = BRAND_LOVE_CONFIDENCE
                if emitter.emit_ad(episode_id, ad, ad_content, is_brand_love=True):
                    brand_love_count += 1
                    print(f"  ✅ Emitted as endorsement (queued)")
                else:
                    print(f"  ⚠️ Failed to emit event")
            else:
                print(f"  ⏭️  SKIPPED - Could not identify brand")
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
                
                # Check if expanded context is brand love - emit as endorsement
                if expanded_ad.is_brand_love:
                    print(f"  💝 BRAND LOVE after expansion")
                    ad_content = build_brand_love_content(expanded_ad, source="gliner")
                    
                    if ad_content["sponsorName"] != "Unknown Brand":
                        expanded_ad.confidence_score = BRAND_LOVE_CONFIDENCE
                        if emitter.emit_ad(episode_id, expanded_ad, ad_content, is_brand_love=True):
                            brand_love_count += 1
                            print(f"  ✅ Emitted as endorsement (queued)")
                        else:
                            print(f"  ⚠️ Failed to emit event")
                    else:
                        print(f"  ⏭️  SKIPPED - Could not identify brand")
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
            # Get ad text with surrounding context for Gemini (I/O-bound but necessary)
            context_text = get_context_text(transcript_data, ad.start_time, ad.end_time)
            
            # Pass GLiNER2 data to Gemini for verification
            llm_result = verify_ad_with_gemini(context_text, ad.structured_data)
            
            # Check for brand love from Gemini - emit as endorsement
            if llm_result.get("is_brand_love"):
                print(f"  💝 BRAND LOVE detected by Gemini")
                reason = llm_result.get("brand_love_reason")
                if reason:
                    print(f"     Reason: {reason}")
                
                ad_content = build_brand_love_content(ad, source="gemini", reason=reason)
                
                if ad_content["sponsorName"] != "Unknown Brand":
                    ad.confidence_score = BRAND_LOVE_CONFIDENCE
                    if emitter.emit_ad(episode_id, ad, ad_content, is_brand_love=True):
                        brand_love_count += 1
                        print(f"  ✅ Emitted as endorsement (queued)")
                    else:
                        print(f"  ⚠️ Failed to emit event")
                else:
                    print(f"  ⏭️  SKIPPED - Could not identify brand")
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
        sponsor_url = ad_content.get("sponsorUrl", "")
        
        print(f"  Sponsor:     {sponsor_name}")
        if ad_content.get("productName"):
            print(f"  Product:     {ad_content['productName']}")
        if ad_content.get("sponsorCode"):
            print(f"  Promo Code:  {ad_content['sponsorCode']}")
        if ad_content.get("sponsorUrl"):
            print(f"  URL:         {ad_content['sponsorUrl']}")
        
        # === STEP 1: ALWAYS EMIT THE PRIMARY DETECTED AD FIRST ===
        # This ensures we don't lose the sponsor info that LLM verified
        normalized_sponsor = sponsor_name.lower().strip() if sponsor_name else ""
        
        # Deduplication check for the primary ad
        primary_emitted = False
        similar_match = find_similar_sponsor(sponsor_name, recent_ads)
        if similar_match:
            matched_sponsor, last_end_time = similar_match
            time_since_last = ad.start_time - last_end_time
            if 0 < time_since_last < AD_DEDUP_WINDOW_SECONDS:
                print(f"  ⏭️  SKIPPED primary ad ({sponsor_name}) - Duplicate of {matched_sponsor}")
                duplicate_count += 1
            else:
                primary_emitted = True
        else:
            primary_emitted = True
        
        if primary_emitted:
            ad.confidence_score = combined_score
            if emitter.emit_ad(episode_id, ad, ad_content):
                emitted_count += 1
                print(f"  ✅ Emitted primary ad: {sponsor_name} (confidence: {combined_score:.0%})")
                # Track for deduplication
                if normalized_sponsor and normalized_sponsor != "unknown brand":
                    recent_ads[normalized_sponsor] = ad.end_time
            else:
                print(f"  ❌ Failed to emit primary ad")
        
        # === STEP 2: CHECK IF LLM HINTS AT MULTIPLE ADS ===
        # Only expand context and re-analyze if LLM suggests there are more ads
        likely_multiple = llm_result.get("likely_multiple_ads", False)
        other_sponsors_hint = llm_result.get("other_sponsors_hint") or []
        
        if likely_multiple and other_sponsors_hint:
            print(f"  🔀 LLM hints at {len(other_sponsors_hint)} additional ad(s): {', '.join(other_sponsors_hint)}")
            
            # Track which hinted sponsors we've processed
            processed_hints = set()
            processed_hints.add(normalized_sponsor)  # Already processed primary
            
            # Expand context for re-analysis with GLiNER
            EXPANDED_CONTEXT_SECONDS = 60  # Larger window for multi-ad segments
            expanded_context_text = get_context_text(
                transcript_data, ad.start_time, ad.end_time, 
                context_seconds=EXPANDED_CONTEXT_SECONDS
            )
            
            # Re-analyze expanded context with GLiNER to get structured data for additional sponsors
            expanded_analysis = detector.analyze_segment(
                expanded_context_text,
                ad.start_time - EXPANDED_CONTEXT_SECONDS,
                ad.end_time + EXPANDED_CONTEXT_SECONDS,
                use_combined=True,
                skip_exclusion_check=True  # Don't re-check exclusions, already validated
            )
            
            for hint_sponsor in other_sponsors_hint:
                hint_normalized = hint_sponsor.lower().strip()
                
                # Skip if similar to primary sponsor
                if are_sponsors_similar(hint_normalized, normalized_sponsor):
                    continue
                
                # Skip if already processed a similar sponsor
                already_processed = False
                for processed in processed_hints:
                    if are_sponsors_similar(hint_normalized, processed):
                        already_processed = True
                        break
                if already_processed:
                    continue
                
                # Deduplication check
                similar_match = find_similar_sponsor(hint_sponsor, recent_ads)
                if similar_match:
                    matched_sponsor, last_end_time = similar_match
                    time_since_last = ad.start_time - last_end_time
                    if 0 < time_since_last < AD_DEDUP_WINDOW_SECONDS:
                        print(f"    ⏭️  SKIPPED additional ad ({hint_sponsor}) - Duplicate")
                        duplicate_count += 1
                        processed_hints.add(hint_normalized)
                        continue
                
                # Build ad content for the additional sponsor using expanded GLiNER analysis
                additional_ad_content = build_additional_ad_content(
                    hint_sponsor, 
                    ad, 
                    ad_content, 
                    expanded_analysis,
                    combined_score * 0.9  # Slightly lower confidence for derived ads
                )
                
                if additional_ad_content:
                    ad.confidence_score = additional_ad_content["combinedConfidence"]
                    if emitter.emit_ad(episode_id, ad, additional_ad_content):
                        emitted_count += 1
                        print(f"    ✅ Emitted additional ad: {hint_sponsor} (confidence: {additional_ad_content['combinedConfidence']:.0%})")
                        # Track for deduplication
                        if hint_normalized and hint_normalized != "unknown brand":
                            recent_ads[hint_normalized] = ad.end_time
                    else:
                        print(f"    ❌ Failed to emit additional ad: {hint_sponsor}")
                
                processed_hints.add(hint_normalized)
    
    return {
        "detected": len(ads),
        "emitted": emitted_count,
        "excluded": excluded_count,
        "brand_love": brand_love_count,
        "duplicates": duplicate_count
    }


def run_parallel_processing(episodes: list, writer: AsyncDBWriter) -> dict:
    """
    Process episodes in parallel using multiprocessing Pool.
    
    Each worker process has its own GLiNER2 and Gemini model instances.
    Results are collected and written to DB from the main process.
    
    Args:
        episodes: List of episode dicts with 'episodeId' and 'transcriptUrl'
        writer: AsyncDBWriter instance for database writes
    
    Returns:
        Dict with aggregated statistics
    """
    num_workers = get_optimal_worker_count()
    total_episodes = len(episodes)
    
    print(f"\n🚀 Starting parallel processing with {num_workers} worker processes")
    print(f"   Total CPU cores: {cpu_count()}")
    print(f"   Episodes to process: {total_episodes}")
    print("=" * 70)
    
    # Create emitter for main process
    emitter = AdEventEmitter(writer)
    
    # Statistics
    stats = {
        "detected": 0,
        "emitted": 0,
        "excluded": 0,
        "brand_love": 0,
        "duplicates": 0,
        "episodes_processed": 0,
        "episodes_with_ads": 0,
        "errors": 0,
    }
    
    # Convert episodes to list of dicts for multiprocessing
    episode_dicts = [dict(ep) for ep in episodes]
    
    start_time = time.time()
    
    # Use Pool with worker initialization
    # Each worker initializes its own model via worker_init()
    with Pool(
        processes=num_workers,
        initializer=worker_init,
    ) as pool:
        
        # Use imap_unordered for better load balancing
        # Results come back as soon as any worker finishes
        results_iter = pool.imap_unordered(
            worker_process_episode,
            episode_dicts,
            chunksize=WORKER_CHUNK_SIZE,
        )
        
        # Process results as they arrive
        for i, result in enumerate(results_iter, 1):
            stats["episodes_processed"] += 1
            
            if result.get("error"):
                stats["errors"] += 1
                print(f"  [{result.get('process_name', '?')}] ❌ Episode {result['episode_id'][:8]}...: {result['error']}")
                continue
            
            # Update statistics
            stats["detected"] += result["detected"]
            stats["excluded"] += result["excluded"]
            stats["brand_love"] += result["brand_love"]
            stats["duplicates"] += result["duplicates"]
            
            # Emit ad events to the async writer
            ad_events = result.get("ad_events", [])
            events_emitted = 0
            
            for event_data in ad_events:
                # Create a simple AdSegment-like object for emitter
                class SimpleAdSegment:
                    def __init__(self, data):
                        self.start_time = data["start_time"]
                        self.end_time = data["end_time"]
                        self.confidence_score = data["confidence_score"]
                
                ad_segment = SimpleAdSegment(event_data["ad_segment"])
                
                if emitter.emit_ad(
                    event_data["episode_id"],
                    ad_segment,
                    event_data["ad_content"],
                    is_brand_love=event_data["is_brand_love"],
                ):
                    events_emitted += 1
            
            stats["emitted"] += events_emitted
            
            if events_emitted > 0:
                stats["episodes_with_ads"] += 1
            
            # Progress logging
            if i % 10 == 0 or i == total_episodes:
                elapsed = time.time() - start_time
                eps_per_sec = i / elapsed if elapsed > 0 else 0
                writer_stats = writer.get_stats()
                print(f"\n  📊 Progress: {i}/{total_episodes} ({i/total_episodes*100:.1f}%) | "
                      f"{eps_per_sec:.2f} eps/sec | "
                      f"Written: {writer_stats['events_written']}")
    
    elapsed_total = time.time() - start_time
    print(f"\n  ⏱️ Parallel processing completed in {elapsed_total:.1f}s")
    print(f"     Throughput: {total_episodes/elapsed_total:.2f} episodes/second")
    
    return stats


def run_single_process(episodes: list, writer: AsyncDBWriter) -> dict:
    """
    Process episodes sequentially in single process mode.
    Original implementation - used as fallback.
    """
    print("\n📦 Loading ad detection model (GLiNER2 with full capabilities)...")
    detector = AdDetector()
    
    emitter = AdEventEmitter(writer)
    
    stats = {
        "detected": 0,
        "emitted": 0,
        "excluded": 0,
        "brand_love": 0,
        "duplicates": 0,
        "episodes_processed": 0,
        "episodes_with_ads": 0,
        "errors": 0,
    }
    
    for episode in episodes:
        episode_id = episode['episodeId']
        transcript_url = episode['transcriptUrl']
        
        try:
            result = process_episode(
                detector,
                emitter,
                episode_id,
                transcript_url,
            )
            
            stats["detected"] += result["detected"]
            stats["emitted"] += result["emitted"]
            stats["excluded"] += result["excluded"]
            stats["brand_love"] += result["brand_love"]
            stats["duplicates"] += result["duplicates"]
            
            if result["emitted"] > 0:
                stats["episodes_with_ads"] += 1
            
            stats["episodes_processed"] += 1
            
            if stats["episodes_processed"] % 10 == 0:
                writer_stats = writer.get_stats()
                print(f"\n  📊 Writer stats: {writer_stats['events_written']}/{writer_stats['events_received']} written, "
                      f"{writer_stats['batches_written']} batches")
            
        except Exception as e:
            print(f"  ❌ Error processing episode {episode_id}: {e}")
            stats["errors"] += 1
            continue
    
    return stats


def main():
    """Main execution function with multiprocessing support"""
    print("=" * 70)
    print("🎙️  Podcast Ad Detection Pipeline")
    print("   Powered by GLiNER2 + Gemini Verification")
    if USE_MULTIPROCESSING:
        print(f"   Architecture: MULTIPROCESSING with {get_optimal_worker_count()} workers")
    else:
        print("   Architecture: Event-driven with async batched DB writes")
    print("=" * 70)
    print(f"   LLM Weight: {LLM_WEIGHT:.0%} | Model Weight: {MODEL_WEIGHT:.0%} | GLiNER Class: {GLINER_CLASSIFICATION_WEIGHT:.0%}")
    print(f"   Min Confidence: {MIN_CONFIDENCE_TO_SAVE:.0%}")
    print(f"   Skip LLM Threshold: {HIGH_CONFIDENCE_THRESHOLD:.0%}")
    print(f"   Dedup Window: {AD_DEDUP_WINDOW_SECONDS}s ({AD_DEDUP_WINDOW_SECONDS // 60} min)")
    print("=" * 70)
    print("\n📋 Async Writer Configuration:")
    print(f"   Batch Size: {BATCH_SIZE} events")
    print(f"   Flush Interval: {FLUSH_INTERVAL_SECONDS}s")
    print(f"   Max Queue Size: {MAX_QUEUE_SIZE}")
    print("=" * 70)
    if USE_MULTIPROCESSING:
        print("\n🔧 Multiprocessing Configuration:")
        print(f"   Workers: {get_optimal_worker_count()} processes")
        print(f"   Chunk Size: {WORKER_CHUNK_SIZE} episode(s) per task")
        print("=" * 70)
    print("\n📋 Exclusion Filters Active:")
    print("   • Patreon/Substack/GoFundMe/BuyMeACoffee/PayPal (creator support)")
    print("   • Social media plugs (self-promotion)")
    print("   • Ad-free subscription offers")
    print("   • Brand love/shoutouts (no commercial intent)")
    print("=" * 70)
    
    print("\n🔌 Initializing async database writer...")
    writer = AsyncDBWriter(
        database_url=DATABASE_URL,
        batch_size=BATCH_SIZE,
        flush_interval_seconds=FLUSH_INTERVAL_SECONDS,
        max_queue_size=MAX_QUEUE_SIZE,
    )
    writer.start()
    
    print("🔌 Connecting to database for episode fetching...")
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        print("📋 Fetching episodes...")
        episodes = fetch_episodes(cursor)
        print(f"   Found {len(episodes)} episodes to process")
        
        if not episodes:
            print("No episodes found matching criteria.")
            return
        
        # Choose processing mode
        if USE_MULTIPROCESSING and len(episodes) > 1:
            stats = run_parallel_processing(episodes, writer)
        else:
            stats = run_single_process(episodes, writer)
        
        # Final flush - ensure all events are written
        print("\n💾 Flushing remaining events...")
        writer.flush_sync(timeout=30.0)
        
        # Get final writer stats
        writer_stats = writer.get_stats()
        
        # Final summary
        print("\n" + "=" * 70)
        print("📊 SUMMARY")
        print("=" * 70)
        print(f"   Processing mode:     {'PARALLEL' if USE_MULTIPROCESSING else 'SINGLE'}")
        if USE_MULTIPROCESSING:
            print(f"   Worker processes:    {get_optimal_worker_count()}")
        print(f"   Episodes processed:  {stats['episodes_processed']}")
        print(f"   Episodes with ads:   {stats['episodes_with_ads']}")
        print(f"   Ads detected:        {stats['detected']}")
        print(f"   Events emitted:      {stats['emitted']}")
        print(f"   Events written:      {writer_stats['events_written']}")
        print(f"   Batches written:     {writer_stats['batches_written']}")
        print(f"   Write errors:        {writer_stats['errors']}")
        print(f"   Processing errors:   {stats['errors']}")
        print(f"   Excluded:            {stats['excluded']}")
        print(f"   Brand endorsements:  {stats['brand_love']}")
        print(f"   Duplicates skipped:  {stats['duplicates']}")
        if stats["detected"] > 0:
            print(f"   Filter rate:         {(1 - stats['emitted']/stats['detected'])*100:.1f}% noise removed")
        print("=" * 70)
        print("✅ Pipeline complete!")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        raise
    
    finally:
        # Graceful shutdown
        writer.shutdown(timeout=30.0)
        safe_close(conn, cursor)


if __name__ == "__main__":
    main()
