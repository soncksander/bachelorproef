CREATE EXTENSION IF NOT EXISTS vector;

-- belangrijk om de extension te activeren moest ze nog niet bestaan
CREATE TABLE
    IF NOT EXISTS EMBEDDING (
        id UUID PRIMARY KEY DEFAULT uuidv7 (),
        document TEXT NULL,
        embedding VECTOR (1024) NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
    );