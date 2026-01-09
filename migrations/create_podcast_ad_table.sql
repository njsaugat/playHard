-- Migration: Create PodcastAd table

-- Ensure "AdType" enum has the required values
-- Note: These might fail if already present, but that's okay
ALTER TYPE "AdType" ADD VALUE IF NOT EXISTS 'HOST_READ';
ALTER TYPE "AdType" ADD VALUE IF NOT EXISTS 'PRE_PRODUCED';
ALTER TYPE "AdType" ADD VALUE IF NOT EXISTS 'DYNAMICALLY_INSERTED';
ALTER TYPE "AdType" ADD VALUE IF NOT EXISTS 'BRAND_LOVE';
ALTER TYPE "AdType" ADD VALUE IF NOT EXISTS 'UNKNOWN';

-- Ensure "AdFormat" enum has the required values
ALTER TYPE "AdFormat" ADD VALUE IF NOT EXISTS 'PRE_ROLL';
ALTER TYPE "AdFormat" ADD VALUE IF NOT EXISTS 'MID_ROLL';
ALTER TYPE "AdFormat" ADD VALUE IF NOT EXISTS 'POST_ROLL';
ALTER TYPE "AdFormat" ADD VALUE IF NOT EXISTS 'ENDORSEMENT';
ALTER TYPE "AdFormat" ADD VALUE IF NOT EXISTS 'UNCLASSIFIED';

-- Create PodcastAd table
CREATE TABLE IF NOT EXISTS "PodcastAd" (
    id TEXT PRIMARY KEY DEFAULT gen_random_uuid()::text,
    "episodeId" TEXT NOT NULL,
    "brandId" TEXT NOT NULL,
    
    start_time INTEGER NOT NULL,
    end_time INTEGER NOT NULL,
    
    ad_type "AdType" NOT NULL DEFAULT 'HOST_READ',
    ad_format "AdFormat" NOT NULL DEFAULT 'UNCLASSIFIED',
    ad_content JSONB NOT NULL DEFAULT '{}',
    "confidenceScore" DECIMAL(10, 4) NOT NULL DEFAULT 0,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Foreign key constraints
    CONSTRAINT fk_episode 
        FOREIGN KEY ("episodeId") 
        REFERENCES "PodcastEpisode"(id) 
        ON DELETE CASCADE,
    
    CONSTRAINT fk_brand 
        FOREIGN KEY ("brandId") 
        REFERENCES "BrandProfile"(id) 
        ON DELETE CASCADE
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_podcast_ad_episode_id ON "PodcastAd"("episodeId");
CREATE INDEX IF NOT EXISTS idx_podcast_ad_brand_id ON "PodcastAd"("brandId");
CREATE INDEX IF NOT EXISTS idx_podcast_ad_confidence ON "PodcastAd"("confidenceScore" DESC);
CREATE INDEX IF NOT EXISTS idx_podcast_ad_created_at ON "PodcastAd"(created_at DESC);

-- Add comment
COMMENT ON TABLE "PodcastAd" IS 'Detected advertisements in podcast episodes';