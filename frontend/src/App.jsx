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
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'signals', label: 'Signals' },
  { id: 'portfolio', label: 'Portfolio' },
  { id: 'markets', label: 'Markets' },
  { id: 'performance', label: 'Performance' },
  { id: 'testing', label: 'Testing' },
  { id: 'settings', label: 'Settings' },
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
      {/* Header */}
      <header className="border-b border-border sticky top-0 z-50 bg-surface/95 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-4 md:px-6 py-3 md:py-4 flex items-center justify-between">
          <div className="flex items-center gap-3 md:gap-4">
            <h1 className="text-base md:text-lg font-bold tracking-tight text-text-primary">
              PREDICTION<span className="text-accent-green">BOT</span>
            </h1>
            {status && (
              <span className={`text-[10px] px-2 py-0.5 md:px-2.5 md:py-1 rounded-full font-semibold tracking-widest uppercase ${
                status.environment === 'demo'
                  ? 'bg-accent-yellow/10 text-accent-yellow border border-accent-yellow/20'
                  : 'bg-accent-red/10 text-accent-red border border-accent-red/20'
              }`}>
                {status.environment}
              </span>
            )}
          </div>

          <div className="flex items-center gap-3 md:gap-5">
            {/* Connection indicators - hidden on mobile */}
            <div className="hidden md:flex items-center gap-4 text-xs text-text-secondary">
              <span className="flex items-center gap-1.5">
                <span className={`w-1.5 h-1.5 rounded-full ${status?.kalshi_connected ? 'bg-accent-green pulse-dot' : 'bg-accent-red'}`} />
                Kalshi
              </span>
              <span className="flex items-center gap-1.5">
                <span className={`w-1.5 h-1.5 rounded-full ${status?.anthropic_connected ? 'bg-accent-green pulse-dot' : 'bg-text-muted'}`} />
                AI
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-accent-green pulse-dot" />
                RF ({status?.model?.n_features || 0})
              </span>
            </div>

            {/* Theme toggle */}
            <button onClick={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}
              className="p-1.5 rounded-lg bg-card border border-border hover:border-text-muted transition-all text-text-secondary text-sm">
              {theme === 'dark' ? '☀️' : '🌙'}
            </button>

            <button
              onClick={() => handleScan(false)}
              disabled={scanning || !status?.kalshi_connected}
              className={`px-3 md:px-5 py-1.5 md:py-2 text-[10px] md:text-xs font-semibold tracking-wide rounded-lg transition-all uppercase ${
                scanning
                  ? 'bg-accent-green text-black animate-pulse'
                  : 'bg-accent text-surface hover:opacity-80 disabled:opacity-30 disabled:cursor-not-allowed'
              }`}
            >
              {scanning ? 'Scanning...' : scanResult ? `Scan (${scanResult.markets_scanned || 0})` : 'Scan'}
            </button>

            {/* Mobile hamburger */}
            <button onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden p-1.5 rounded-lg bg-card border border-border text-text-secondary text-sm">
              {mobileMenuOpen ? '✕' : '☰'}
            </button>
          </div>
        </div>

        {/* Tabs - desktop */}
        <div className="hidden md:block max-w-7xl mx-auto px-6">
          <nav className="flex gap-0 overflow-x-auto">
            {TABS.map(t => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`px-5 py-2.5 text-xs font-medium tracking-wide uppercase border-b-2 transition-all whitespace-nowrap ${
                  tab === t.id
                    ? 'border-accent-green text-accent-green'
                    : 'border-transparent text-text-secondary hover:text-text-primary'
                }`}
              >
                {t.label}
              </button>
            ))}
          </nav>
        </div>

        {/* Mobile menu */}
        {mobileMenuOpen && (
          <div className="md:hidden border-t border-border bg-card">
            <div className="p-2 space-y-1">
              {TABS.map(t => (
                <button
                  key={t.id}
                  onClick={() => { setTab(t.id); setMobileMenuOpen(false); }}
                  className={`w-full text-left px-4 py-2.5 text-xs font-medium tracking-wide uppercase rounded-lg transition-all ${
                    tab === t.id
                      ? 'bg-accent-green/10 text-accent-green'
                      : 'text-text-secondary hover:bg-surface'
                  }`}
                >
                  {t.label}
                </button>
              ))}
              {/* Mobile connection indicators */}
              <div className="px-4 py-2 flex items-center gap-4 text-[10px] text-text-muted">
                <span className="flex items-center gap-1.5">
                  <span className={`w-1.5 h-1.5 rounded-full ${status?.kalshi_connected ? 'bg-accent-green' : 'bg-accent-red'}`} />
                  Kalshi
                </span>
                <span className="flex items-center gap-1.5">
                  <span className={`w-1.5 h-1.5 rounded-full ${status?.anthropic_connected ? 'bg-accent-green' : 'bg-text-muted'}`} />
                  AI
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-accent-green" />
                  RF
                </span>
              </div>
            </div>
          </div>
        )}
      </header>

      {/* Error banner */}
      {error && (
        <div className="max-w-7xl mx-auto px-4 md:px-6 mt-4">
          <div className="bg-accent-red/5 border border-accent-red/20 rounded-lg p-3 text-xs text-accent-red">
            {error}
          </div>
        </div>
      )}

      {/* Scan result banner */}
      {scanResult && !scanning && (
        <div className="max-w-7xl mx-auto px-4 md:px-6 mt-4">
          <div className="bg-accent-green/5 border border-accent-green/20 rounded-lg p-3 text-xs text-text-secondary flex items-center justify-between">
            <span>
              Scanned <span className="text-text-primary font-semibold">{scanResult.events_scanned}</span> events / <span className="text-text-primary font-semibold">{scanResult.markets_scanned}</span> markets
              {scanResult.rf_signals > 0 && <> — <span className="text-accent-green font-semibold">{scanResult.rf_signals} signals found</span></>}
              {scanResult.rf_signals === 0 && <> — no signals (model needs training or markets don't meet 2x undervalue criteria)</>}
            </span>
            <span className="text-text-muted hidden md:inline">{new Date(scanResult.scan_time).toLocaleTimeString()}</span>
          </div>
        </div>
      )}

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-4 md:px-6 py-4 md:py-6">
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
