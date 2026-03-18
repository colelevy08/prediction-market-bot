-- ============================================================================
-- Supabase Schema for Prediction Market Bot
-- ============================================================================
-- Run this SQL in the Supabase SQL Editor to set up all required tables.
-- Dashboard: https://supabase.com/dashboard/project/<YOUR_PROJECT_REF>/sql/new
--
-- Tables:
--   1. trades              — Paper and live trade records
--   2. scan_logs           — Scan history (events scanned, entries, exits)
--   3. paper_state         — Shadow trading state (balance, positions)
--   4. performance_snapshots — Periodic Sharpe/PnL/win-rate snapshots
--   5. model_training_runs — Model CV accuracy, OOB score, feature importance
--   6. training_samples    — Cumulative training data (deduped by ticker)
-- ============================================================================

-- 1. TRADES — stores every paper and live trade with full metrics
CREATE TABLE IF NOT EXISTS trades (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    mode        TEXT NOT NULL DEFAULT 'paper',        -- 'paper' or 'live'
    ticker      TEXT NOT NULL,
    side        TEXT NOT NULL,                         -- 'yes' or 'no'
    entry_price DOUBLE PRECISION DEFAULT 0,
    exit_price  DOUBLE PRECISION DEFAULT 0,
    contracts   INTEGER DEFAULT 1,
    pnl_cents   INTEGER DEFAULT 0,
    log_return  DOUBLE PRECISION DEFAULT 0,
    mae         DOUBLE PRECISION DEFAULT 0,           -- max adverse excursion
    mfe         DOUBLE PRECISION DEFAULT 0,           -- max favorable excursion
    model_probability            DOUBLE PRECISION DEFAULT 0,
    market_probability_at_entry  DOUBLE PRECISION DEFAULT 0,
    won         BOOLEAN DEFAULT FALSE,
    entry_time  TIMESTAMPTZ,
    exit_time   TIMESTAMPTZ,
    category    TEXT DEFAULT '',
    notes       TEXT DEFAULT '',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_mode ON trades(mode);
CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker);
CREATE INDEX IF NOT EXISTS idx_trades_created_at ON trades(created_at DESC);

-- 2. SCAN_LOGS — tracks each scan cycle (how many events, entries, exits)
CREATE TABLE IF NOT EXISTS scan_logs (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    scan_type       TEXT NOT NULL DEFAULT 'paper',    -- 'paper' or 'live'
    signals_found   INTEGER DEFAULT 0,
    entries         INTEGER DEFAULT 0,
    exits           INTEGER DEFAULT 0,
    open_positions  INTEGER DEFAULT 0,
    scanned_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_scan_logs_scanned_at ON scan_logs(scanned_at DESC);

-- 3. PAPER_STATE — singleton row (id=1) storing shadow trader state
CREATE TABLE IF NOT EXISTS paper_state (
    id              INTEGER PRIMARY KEY DEFAULT 1,
    balance_cents   INTEGER DEFAULT 10000,
    total_scans     INTEGER DEFAULT 0,
    signals_seen    INTEGER DEFAULT 0,
    positions       JSONB DEFAULT '{}',               -- open positions as JSON
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- 4. PERFORMANCE_SNAPSHOTS — periodic snapshots of trading metrics
CREATE TABLE IF NOT EXISTS performance_snapshots (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    mode                TEXT NOT NULL DEFAULT 'paper',
    total_trades        INTEGER DEFAULT 0,
    wins                INTEGER DEFAULT 0,
    losses              INTEGER DEFAULT 0,
    win_rate            DOUBLE PRECISION DEFAULT 0,
    total_pnl_cents     INTEGER DEFAULT 0,
    sharpe_ratio        DOUBLE PRECISION DEFAULT 0,
    sharpe_label        TEXT DEFAULT 'N/A',
    profit_factor       DOUBLE PRECISION DEFAULT 0,
    max_drawdown_cents  INTEGER DEFAULT 0,
    avg_mae             DOUBLE PRECISION DEFAULT 0,
    avg_mfe             DOUBLE PRECISION DEFAULT 0,
    avg_edge            DOUBLE PRECISION DEFAULT 0,
    avg_log_return      DOUBLE PRECISION DEFAULT 0,
    equity_curve        JSONB DEFAULT '[]',
    snapshot_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_perf_snapshots_mode ON performance_snapshots(mode);
CREATE INDEX IF NOT EXISTS idx_perf_snapshots_at ON performance_snapshots(snapshot_at DESC);

-- 5. MODEL_TRAINING_RUNS — logs every model retrain with accuracy metrics
CREATE TABLE IF NOT EXISTS model_training_runs (
    id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    samples_used        INTEGER DEFAULT 0,
    cv_accuracy         DOUBLE PRECISION DEFAULT 0,
    oob_score           DOUBLE PRECISION DEFAULT 0,
    n_features          INTEGER DEFAULT 0,
    n_estimators        INTEGER DEFAULT 0,
    feature_importance  JSONB DEFAULT '{}',
    trained_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_training_runs_at ON model_training_runs(trained_at DESC);

-- 6. TRAINING_SAMPLES — cumulative training data, deduplicated by ticker
--    Each settled market becomes one sample; new runs merge with existing data
CREATE TABLE IF NOT EXISTS training_samples (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ticker      TEXT NOT NULL UNIQUE,                 -- dedup key
    features    JSONB NOT NULL DEFAULT '{}',          -- 106-feature vector
    outcome     INTEGER NOT NULL DEFAULT 0,           -- 1 = yes won, 0 = no won
    fetched_at  TIMESTAMPTZ DEFAULT NOW(),
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_training_samples_ticker ON training_samples(ticker);

-- ============================================================================
-- Row Level Security (RLS) — disabled for service_role key access
-- If you enable RLS, add policies for the service_role to bypass.
-- ============================================================================
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE scan_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE paper_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE performance_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_training_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_samples ENABLE ROW LEVEL SECURITY;

-- Allow service_role full access (the key used by the backend)
DO $$
BEGIN
    -- trades
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'trades' AND policyname = 'service_role_all') THEN
        CREATE POLICY service_role_all ON trades FOR ALL USING (auth.role() = 'service_role');
    END IF;
    -- scan_logs
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'scan_logs' AND policyname = 'service_role_all') THEN
        CREATE POLICY service_role_all ON scan_logs FOR ALL USING (auth.role() = 'service_role');
    END IF;
    -- paper_state
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'paper_state' AND policyname = 'service_role_all') THEN
        CREATE POLICY service_role_all ON paper_state FOR ALL USING (auth.role() = 'service_role');
    END IF;
    -- performance_snapshots
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'performance_snapshots' AND policyname = 'service_role_all') THEN
        CREATE POLICY service_role_all ON performance_snapshots FOR ALL USING (auth.role() = 'service_role');
    END IF;
    -- model_training_runs
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'model_training_runs' AND policyname = 'service_role_all') THEN
        CREATE POLICY service_role_all ON model_training_runs FOR ALL USING (auth.role() = 'service_role');
    END IF;
    -- training_samples
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'training_samples' AND policyname = 'service_role_all') THEN
        CREATE POLICY service_role_all ON training_samples FOR ALL USING (auth.role() = 'service_role');
    END IF;
END $$;
