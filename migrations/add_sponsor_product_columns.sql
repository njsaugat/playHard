-- Migration: Add sponsor_name and product_name columns to PodcastAd table
-- These fields are extracted from ad_content for easier querying
-- Both columns are NULLABLE since existing rows won't have values initially

ALTER TABLE "PodcastAd" 
ADD COLUMN IF NOT EXISTS sponsor_name TEXT DEFAULT NULL,
ADD COLUMN IF NOT EXISTS product_name TEXT DEFAULT NULL;

-- Backfill existing rows from ad_content JSON
-- Extract sponsorName and productName from the JSON and populate the new columns
UPDATE "PodcastAd"
SET 
    sponsor_name = ad_content->>'sponsorName',
    product_name = NULLIF(ad_content->>'productName', '')
WHERE sponsor_name IS NULL 
  AND ad_content->>'sponsorName' IS NOT NULL;

-- Create partial indexes that only index non-null values
-- This is more efficient for nullable columns - saves space and query time
CREATE INDEX IF NOT EXISTS idx_podcast_ad_sponsor_name 
    ON "PodcastAd"(sponsor_name) 
    WHERE sponsor_name IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_podcast_ad_product_name 
    ON "PodcastAd"(product_name) 
    WHERE product_name IS NOT NULL;

-- Add comments
COMMENT ON COLUMN "PodcastAd".sponsor_name IS 'Primary sponsor/brand name for this ad (nullable, extracted from ad_content)';
COMMENT ON COLUMN "PodcastAd".product_name IS 'Product name being advertised (nullable, may not always be present)';
