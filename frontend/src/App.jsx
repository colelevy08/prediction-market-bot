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
    try {
      const result = await api.runScan(status?.config?.max_events_to_analyze || 20, useAi);
      setScanResult(result);
      setError(null);
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
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <h1 className="text-lg font-bold tracking-tight text-white">
              PREDICTION<span className="text-accent-green">BOT</span>
            </h1>
            {status && (
              <span className={`text-[10px] px-2.5 py-1 rounded-full font-semibold tracking-widest uppercase ${
                status.environment === 'demo'
                  ? 'bg-accent-yellow/10 text-accent-yellow border border-accent-yellow/20'
                  : 'bg-accent-red/10 text-accent-red border border-accent-red/20'
              }`}>
                {status.environment}
              </span>
            )}
          </div>

          <div className="flex items-center gap-5">
            {/* Connection indicators */}
            <div className="flex items-center gap-4 text-xs text-text-secondary">
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

            <button
              onClick={() => handleScan(false)}
              disabled={scanning || !status?.kalshi_connected}
              className="px-5 py-2 bg-white text-black text-xs font-semibold tracking-wide rounded-lg
                         hover:bg-gray-200 disabled:opacity-30 disabled:cursor-not-allowed
                         transition-all uppercase"
            >
              {scanning ? 'Scanning...' : 'Scan'}
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="max-w-7xl mx-auto px-6">
          <nav className="flex gap-0">
            {TABS.map(t => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`px-5 py-2.5 text-xs font-medium tracking-wide uppercase border-b-2 transition-all ${
                  tab === t.id
                    ? 'border-accent-green text-accent-green'
                    : 'border-transparent text-text-secondary hover:text-white'
                }`}
              >
                {t.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      {/* Error banner */}
      {error && (
        <div className="max-w-7xl mx-auto px-6 mt-4">
          <div className="bg-accent-red/5 border border-accent-red/20 rounded-lg p-3 text-xs text-accent-red">
            {error}
          </div>
        </div>
      )}

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-6 py-6">
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
