-- Persist CRM analysis queries/responses for the analysis UI.
CREATE TABLE IF NOT EXISTS public.analysis_queries (
    id BIGSERIAL PRIMARY KEY,
    input_log TEXT NOT NULL,
    ai_response_log TEXT NOT NULL,
    created_by TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_analysis_queries_created_by ON public.analysis_queries (created_by);
CREATE INDEX IF NOT EXISTS idx_analysis_queries_created_at ON public.analysis_queries (created_at DESC);

ALTER TABLE public.analysis_queries ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE schemaname = 'public'
          AND tablename = 'analysis_queries'
          AND policyname = 'analysis_queries_insert_anon'
    ) THEN
        CREATE POLICY analysis_queries_insert_anon
            ON public.analysis_queries
            FOR INSERT
            TO anon
            WITH CHECK (true);
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE schemaname = 'public'
          AND tablename = 'analysis_queries'
          AND policyname = 'analysis_queries_select_anon'
    ) THEN
        CREATE POLICY analysis_queries_select_anon
            ON public.analysis_queries
            FOR SELECT
            TO anon
            USING (true);
    END IF;
END $$;
