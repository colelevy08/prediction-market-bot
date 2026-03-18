/**
 * App.jsx - Root application component for the PredictionBot frontend.
 */
import { useState, useEffect, useCallback } from 'react';
import { api } from './api';
import Dashboard from './components/Dashboard';
import Signals from './components/Signals';
import Portfolio from './components/Portfolio';
import Performance from './components/Performance';
import Markets from './components/Markets';
import Testing from './components/Testing';
import Settings from './components/Settings';

const TABS = [
  { id: 'dashboard', label: 'Dashboard', icon: '◈' },
  { id: 'signals', label: 'Signals', icon: '◉' },
  { id: 'portfolio', label: 'Portfolio', icon: '◧' },
  { id: 'markets', label: 'Markets', icon: '◫' },
  { id: 'performance', label: 'Performance', icon: '◬' },
  { id: 'testing', label: 'Testing', icon: '◭' },
  { id: 'settings', label: 'Settings', icon: '◎' },
];

export default function App() {
  const [tab, setTab] = useState('dashboard');
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [scanning, setScanning] = useState(false);
  const [scanResult, setScanResult] = useState(null);
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'dark');
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    document.documentElement.classList.toggle('light', theme === 'light');
    localStorage.setItem('theme', theme);
  }, [theme]);

  const fetchStatus = useCallback(async () => {
    try {
      const data = await api.getStatus();
      setStatus(data);
      setError(null);
    } catch (e) {
      setError(`Backend offline: ${e.message}. Start the server with: uvicorn bot.server:app --reload --port 8000`);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  const handleScan = async (useAi = false) => {
    setScanning(true);
    setScanResult(null);
    const start = Date.now();
    try {
      const result = await api.runScan(status?.config?.max_events_to_analyze || 20, useAi);
      setScanResult(result);
      setError(null);
      const elapsed = Date.now() - start;
      if (elapsed < 1500) {
        await new Promise(r => setTimeout(r, 1500 - elapsed));
      }
    } catch (e) {
      setError(`Scan failed: ${e.message}`);
    } finally {
      setScanning(false);
    }
  };

  return (
    <div className="min-h-screen bg-surface">
      {/* ── Header ── */}
      <header className="border-b border-border sticky top-0 z-50 bg-surface/90 backdrop-blur-xl">
        <div className="max-w-[1400px] mx-auto px-4 md:px-6 py-3 flex items-center justify-between">
          {/* Brand */}
          <div className="flex items-center gap-3">
            <h1 className="text-base md:text-lg font-bold tracking-tight text-text-primary">
              PREDICTION<span className="text-accent-green">BOT</span>
            </h1>
            {status && (
              <span className={`badge ${
                status.environment === 'demo' ? 'badge-yellow' : 'badge-red'
              }`}>
                {status.environment}
              </span>
            )}
          </div>

          {/* Right side controls */}
          <div className="flex items-center gap-3 md:gap-4">
            {/* Connection indicators */}
            <div className="hidden md:flex items-center gap-3">
              {[
                { label: 'Kalshi', ok: status?.kalshi_connected, color: 'bg-accent-green' },
                { label: 'AI', ok: status?.anthropic_connected, color: 'bg-accent-purple' },
                { label: `RF(${status?.model?.n_features || 0})`, ok: status?.model?.is_trained, color: 'bg-accent-cyan' },
                status?.supabase_connected && { label: 'DB', ok: true, color: 'bg-accent-blue' },
              ].filter(Boolean).map((item, i) => (
                <div key={i} className="flex items-center gap-1.5">
                  <span className={`w-1.5 h-1.5 rounded-full ${item.ok ? `${item.color} pulse-dot` : 'bg-text-muted'}`} />
                  <span className={`text-[10px] font-medium tracking-wide ${item.ok ? 'text-text-secondary' : 'text-text-muted'}`}>
                    {item.label}
                  </span>
                </div>
              ))}
            </div>

            {/* Divider */}
            <div className="hidden md:block w-px h-5 bg-border" />

            {/* Theme toggle */}
            <button
              onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}
              className="w-8 h-8 rounded-lg bg-surface-2 border border-border hover:bg-card-hover flex items-center justify-center transition-all text-text-secondary text-sm"
            >
              {theme === 'dark' ? '☀' : '☾'}
            </button>

            {/* Scan button */}
            <button
              onClick={() => handleScan(false)}
              disabled={scanning || !status?.kalshi_connected}
              className={`px-4 py-2 text-[11px] font-semibold tracking-wide rounded-lg transition-all uppercase ${
                scanning
                  ? 'btn-primary animate-pulse'
                  : 'btn-primary'
              }`}
            >
              {scanning ? 'Scanning...' : scanResult ? `Scan (${scanResult.markets_scanned || 0})` : 'Scan'}
            </button>

            {/* Mobile hamburger */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden w-8 h-8 rounded-lg bg-surface-2 border border-border flex items-center justify-center text-text-secondary text-sm"
            >
              {mobileMenuOpen ? '✕' : '☰'}
            </button>
          </div>
        </div>

        {/* ── Tab navigation ── */}
        <div className="hidden md:block max-w-[1400px] mx-auto px-6">
          <nav className="flex gap-0">
            {TABS.map(t => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`px-4 py-2.5 text-[11px] font-medium tracking-wide uppercase border-b-2 transition-all whitespace-nowrap flex items-center gap-1.5 ${
                  tab === t.id
                    ? 'border-accent-green text-accent-green'
                    : 'border-transparent text-text-muted hover:text-text-secondary'
                }`}
              >
                <span className="opacity-60">{t.icon}</span>
                {t.label}
              </button>
            ))}
          </nav>
        </div>

        {/* ── Mobile menu ── */}
        {mobileMenuOpen && (
          <div className="md:hidden border-t border-border bg-card animate-fade-in">
            <div className="p-2 space-y-0.5">
              {TABS.map(t => (
                <button
                  key={t.id}
                  onClick={() => { setTab(t.id); setMobileMenuOpen(false); }}
                  className={`w-full text-left px-4 py-2.5 text-xs font-medium tracking-wide uppercase rounded-lg transition-all flex items-center gap-2 ${
                    tab === t.id
                      ? 'bg-accent-green/10 text-accent-green'
                      : 'text-text-secondary hover:bg-surface-2'
                  }`}
                >
                  <span className="opacity-50">{t.icon}</span>
                  {t.label}
                </button>
              ))}
              {/* Mobile status indicators */}
              <div className="px-4 py-2 flex items-center gap-4 text-[10px] text-text-muted border-t border-border mt-1 pt-2">
                {[
                  { label: 'Kalshi', ok: status?.kalshi_connected },
                  { label: 'AI', ok: status?.anthropic_connected },
                  { label: 'RF', ok: status?.model?.is_trained },
                ].map((item, i) => (
                  <span key={i} className="flex items-center gap-1">
                    <span className={`w-1.5 h-1.5 rounded-full ${item.ok ? 'bg-accent-green' : 'bg-text-muted'}`} />
                    {item.label}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}
      </header>

      {/* ── Error banner ── */}
      {error && (
        <div className="max-w-[1400px] mx-auto px-4 md:px-6 mt-4 animate-slide-up">
          <div className="bg-accent-red/5 border border-accent-red/20 rounded-xl p-3 text-xs text-accent-red flex items-center justify-between">
            <span>{error}</span>
            <button onClick={() => setError(null)} className="text-accent-red/60 hover:text-accent-red ml-3 shrink-0">✕</button>
          </div>
        </div>
      )}

      {/* ── Scan result banner ── */}
      {scanResult && !scanning && (
        <div className="max-w-[1400px] mx-auto px-4 md:px-6 mt-4 animate-slide-up">
          <div className="bg-accent-green/5 border border-accent-green/20 rounded-xl p-3 text-xs text-text-secondary flex items-center justify-between">
            <span>
              Scanned <span className="text-text-primary font-semibold">{scanResult.events_scanned}</span> events
              {' / '}<span className="text-text-primary font-semibold">{scanResult.markets_scanned}</span> markets
              {scanResult.rf_signals > 0 && (
                <> — <span className="text-accent-green font-semibold">{scanResult.rf_signals} signals found</span></>
              )}
              {scanResult.rf_signals === 0 && (
                <> — no signals</>
              )}
            </span>
            <div className="flex items-center gap-3">
              <span className="text-text-muted hidden md:inline">{new Date(scanResult.scan_time).toLocaleTimeString()}</span>
              <button onClick={() => setScanResult(null)} className="text-text-muted hover:text-text-secondary">✕</button>
            </div>
          </div>
        </div>
      )}

      {/* ── Main content ── */}
      <main className="max-w-[1400px] mx-auto px-4 md:px-6 py-5 md:py-6">
        {tab === 'dashboard' && <Dashboard status={status} scanResult={scanResult} onScan={handleScan} scanning={scanning} />}
        {tab === 'signals' && <Signals scanResult={scanResult} onScan={handleScan} scanning={scanning} />}
        {tab === 'portfolio' && <Portfolio />}
        {tab === 'markets' && <Markets />}
        {tab === 'performance' && <Performance />}
        {tab === 'testing' && <Testing />}
        {tab === 'settings' && <Settings status={status} onRefresh={fetchStatus} />}
      </main>
    </div>
  );
}
