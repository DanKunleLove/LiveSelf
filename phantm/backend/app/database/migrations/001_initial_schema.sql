-- LiveSelf Database Schema -- Migration 001
-- Run this in Supabase SQL Editor (Dashboard > SQL Editor > New Query)
-- Creates all core tables with RLS policies for multi-tenant security

-- ============================================
-- ENUM TYPES
-- ============================================

CREATE TYPE user_plan AS ENUM ('free', 'pro', 'enterprise');
CREATE TYPE llm_provider AS ENUM ('ollama', 'claude', 'openai');
CREATE TYPE document_file_type AS ENUM ('pdf', 'txt', 'md', 'qa_pairs');
CREATE TYPE document_status AS ENUM ('pending', 'indexing', 'indexed', 'failed');
CREATE TYPE session_status AS ENUM ('initializing', 'active', 'paused', 'ended', 'error');
CREATE TYPE target_platform AS ENUM ('zoom', 'meet', 'whatsapp', 'tiktok', 'youtube', 'other');


-- ============================================
-- USERS TABLE
-- ============================================
-- Extends Supabase auth.users with app-specific fields.
-- The id column references auth.users so Supabase Auth is the source of truth.

CREATE TABLE users (
    id              uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email           text UNIQUE NOT NULL,
    display_name    text,
    plan            user_plan NOT NULL DEFAULT 'free',
    stripe_customer_id text,
    minutes_used    int NOT NULL DEFAULT 0,
    minutes_limit   int NOT NULL DEFAULT 60,
    created_at      timestamptz NOT NULL DEFAULT now(),
    last_active_at  timestamptz
);

COMMENT ON TABLE users IS 'App-level user profile extending Supabase auth.users';


-- ============================================
-- KNOWLEDGE BASES TABLE
-- ============================================
-- Each user can have multiple knowledge bases containing documents
-- that get chunked and stored in ChromaDB for RAG retrieval.

CREATE TABLE knowledge_bases (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name                text NOT NULL,
    chroma_collection_id text NOT NULL,
    document_count      int NOT NULL DEFAULT 0,
    chunk_count         int NOT NULL DEFAULT 0,
    last_updated_at     timestamptz,
    created_at          timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_knowledge_bases_user_id ON knowledge_bases(user_id);

COMMENT ON TABLE knowledge_bases IS 'Knowledge bases for RAG -- metadata in Postgres, vectors in ChromaDB';


-- ============================================
-- PERSONAS TABLE
-- ============================================
-- A persona is a configured avatar: face photo + voice + knowledge + instructions.
-- Users can have multiple personas (e.g. "Professional Me", "Casual Me").

CREATE TABLE personas (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name                text NOT NULL,
    photo_url           text,
    voice_sample_url    text,
    voice_model_id      text,
    knowledge_base_id   uuid REFERENCES knowledge_bases(id) ON DELETE SET NULL,
    system_prompt       text,
    llm_provider        llm_provider NOT NULL DEFAULT 'ollama',
    is_active           boolean NOT NULL DEFAULT true,
    created_at          timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_personas_user_id ON personas(user_id);

COMMENT ON TABLE personas IS 'User-configured avatars with face, voice, knowledge, and LLM settings';


-- ============================================
-- DOCUMENTS TABLE
-- ============================================
-- Individual files uploaded to a knowledge base.
-- Status tracks the indexing pipeline: pending -> indexing -> indexed/failed.

CREATE TABLE documents (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    kb_id           uuid NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    filename        text NOT NULL,
    file_url        text NOT NULL,
    file_type       document_file_type NOT NULL,
    status          document_status NOT NULL DEFAULT 'pending',
    chunk_count     int,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_documents_kb_id ON documents(kb_id);

COMMENT ON TABLE documents IS 'Files uploaded to knowledge bases, tracked through indexing pipeline';


-- ============================================
-- SESSIONS TABLE
-- ============================================
-- Each live avatar session (a Zoom call, a WhatsApp video, etc.).
-- Tracks duration, status, and exchange count.

CREATE TABLE sessions (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             uuid NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    persona_id          uuid NOT NULL REFERENCES personas(id) ON DELETE CASCADE,
    status              session_status NOT NULL DEFAULT 'initializing',
    target_platform     target_platform,
    started_at          timestamptz,
    ended_at            timestamptz,
    duration_seconds    int,
    exchange_count      int NOT NULL DEFAULT 0,
    created_at          timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_sessions_persona_id ON sessions(persona_id);

COMMENT ON TABLE sessions IS 'Live avatar session records';


-- ============================================
-- SESSION EXCHANGES TABLE
-- ============================================
-- Q&A log for each session. Every question heard and answer generated
-- is stored here for review and knowledge base improvement.

CREATE TABLE session_exchanges (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      uuid NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    question_text   text NOT NULL,
    answer_text     text NOT NULL,
    kb_chunks_used  jsonb,
    latency_ms      int,
    user_edited     boolean NOT NULL DEFAULT false,
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_session_exchanges_session_id ON session_exchanges(session_id);

COMMENT ON TABLE session_exchanges IS 'Q&A exchange log for each session';


-- ============================================
-- ROW LEVEL SECURITY (RLS)
-- ============================================
-- Every table gets RLS so users can only access their own data,
-- even if backend code has a bug that forgets to filter by user_id.

ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE personas ENABLE ROW LEVEL SECURITY;
ALTER TABLE knowledge_bases ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE session_exchanges ENABLE ROW LEVEL SECURITY;

-- Users can only read/update their own row
CREATE POLICY "Users can access own profile"
    ON users FOR ALL
    USING (id = auth.uid());

-- Users can only CRUD their own personas
CREATE POLICY "Users can access own personas"
    ON personas FOR ALL
    USING (user_id = auth.uid());

-- Users can only CRUD their own knowledge bases
CREATE POLICY "Users can access own knowledge bases"
    ON knowledge_bases FOR ALL
    USING (user_id = auth.uid());

-- Documents: user access through knowledge_base ownership
CREATE POLICY "Users can access own documents"
    ON documents FOR ALL
    USING (
        kb_id IN (
            SELECT id FROM knowledge_bases WHERE user_id = auth.uid()
        )
    );

-- Users can only access their own sessions
CREATE POLICY "Users can access own sessions"
    ON sessions FOR ALL
    USING (user_id = auth.uid());

-- Session exchanges: user access through session ownership
CREATE POLICY "Users can access own session exchanges"
    ON session_exchanges FOR ALL
    USING (
        session_id IN (
            SELECT id FROM sessions WHERE user_id = auth.uid()
        )
    );


-- ============================================
-- FUNCTION: Auto-create user profile on signup
-- ============================================
-- When a new user signs up through Supabase Auth, automatically
-- create a row in our users table with their email.

CREATE OR REPLACE FUNCTION handle_new_user()
RETURNS trigger AS $$
BEGIN
    INSERT INTO public.users (id, email)
    VALUES (NEW.id, NEW.email);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW
    EXECUTE FUNCTION handle_new_user();
