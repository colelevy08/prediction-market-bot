import { useState } from 'react';
import { api } from '../api';

export default function Settings({ status, onRefresh }) {
  const [config, setConfig] = useState({
    max_bet_amount_cents: status?.config?.max_bet_amount_cents || 2500,
    min_edge_threshold: status?.config?.min_edge_threshold || 0.08,
    max_daily_loss_cents: status?.config?.max_daily_loss_cents || 10000,
    max_open_positions: status?.config?.max_open_positions || 10,
  });
  const [saved, setSaved] = useState(false);

  const handleSave = async () => {
    try {
      await api.updateConfig(config);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
      onRefresh();
    } catch (e) { alert(`Save failed: ${e.message}`); }
  };

  return (
    <div className="space-y-6 max-w-2xl">
      {/* Connection status */}
      <div className="bg-card border border-border rounded-lg p-5">
        <h3 className="text-[10px] uppercase tracking-widest text-text-secondary font-semibold mb-4">Connections</h3>
        <div className="space-y-3">
          {[
            { label: 'Kalshi API', ok: status?.kalshi_connected },
            { label: 'Claude AI', ok: status?.anthropic_connected, optional: true },
            { label: 'Environment', value: status?.environment?.toUpperCase(), warn: status?.environment !== 'demo' },
            { label: 'RF Model', value: `${status?.model?.n_features || 0} features, ${status?.model?.n_estimators || 0} trees${status?.model?.is_trained ? '' : ' (heuristic)'}` },
          ].map((item, i) => (
            <div key={i} className="flex items-center justify-between">
              <span className="text-xs">{item.label}</span>
              {item.value ? (
                <span className={`text-[10px] px-2.5 py-1 rounded font-semibold uppercase tracking-wide ${
                  item.warn ? 'bg-accent-red/10 text-accent-red' : 'bg-white/5 text-text-secondary'
                }`}>{item.value}</span>
              ) : (
                <span className={`text-[10px] px-2.5 py-1 rounded font-semibold uppercase tracking-wide ${
                  item.ok ? 'bg-accent-green/10 text-accent-green' : item.optional ? 'bg-white/5 text-text-muted' : 'bg-accent-red/10 text-accent-red'
                }`}>{item.ok ? 'Connected' : item.optional ? 'Not Configured' : 'Disconnected'}</span>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Parameters */}
      <div className="bg-card border border-border rounded-lg p-5">
        <h3 className="text-[10px] uppercase tracking-widest text-text-secondary font-semibold mb-4">Trading Parameters</h3>
        <div className="space-y-4">
          {[
            { label: 'Max Bet ($)', key: 'max_bet_amount_cents', div: 100, step: 1, min: 1 },
            { label: 'Min Edge (%)', key: 'min_edge_threshold', div: 0.01, step: 0.5, min: 0, max: 50, note: 'Guide: buy at 2x undervalue' },
            { label: 'Daily Loss Limit ($)', key: 'max_daily_loss_cents', div: 100, step: 10, min: 10 },
            { label: 'Max Positions', key: 'max_open_positions', div: 1, step: 1, min: 1, max: 50 },
          ].map(f => (
            <div key={f.key}>
              <label className="text-[10px] uppercase tracking-widest text-text-secondary block mb-1">
                {f.label}
                {f.note && <span className="normal-case tracking-normal text-text-muted ml-2">— {f.note}</span>}
              </label>
              <input type="number" step={f.step} min={f.min} max={f.max}
                value={(config[f.key] / f.div).toFixed(f.div === 1 ? 0 : f.div === 100 ? 2 : 1)}
                onChange={e => setConfig(prev => ({ ...prev, [f.key]: Math.round(parseFloat(e.target.value) * f.div) }))}
                className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm font-mono text-white focus:border-accent-green focus:outline-none transition-colors" />
            </div>
          ))}
          <button onClick={handleSave}
            className="w-full py-2.5 bg-white text-black text-xs font-semibold tracking-wide rounded-lg hover:bg-gray-200 transition-all uppercase">
            {saved ? 'Saved!' : 'Save Settings'}
          </button>
        </div>
      </div>

      {/* Strategy reference */}
      <div className="bg-card border border-border rounded-lg p-5">
        <h3 className="text-[10px] uppercase tracking-widest text-text-secondary font-semibold mb-3">Strategy Reference</h3>
        <div className="space-y-1.5 text-xs font-mono">
          {[
            { phase: '1', text: `Random Forest — ${status?.model?.n_estimators || 200} trees + GB(150), sqrt(features) per tree` },
            { phase: '4', text: 'Entry: market_price ≤ model_prob × 0.5 (buy at 2x undervalue)' },
            { phase: '5', text: 'Sharpe Ratio = (Rp - Rf) / σ  |  <1 bad, 1-2 good, >2 excellent' },
            { phase: '6', text: 'Log returns = ln(P1/P0) — additive, correct for big moves' },
            { phase: '7', text: 'Exit: market_price ≥ model_prob × 0.9 or days_to_expiry ≤ 7' },
            { phase: '8', text: 'Complete: predict → filter → lock profit → evaluate Sharpe' },
          ].map(({ phase, text }) => (
            <div key={phase} className="bg-surface border border-border rounded p-2.5 card-hover">
              <span className="text-accent-green">#{phase}</span>
              <span className="text-text-secondary ml-2">{text}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Setup */}
      <div className="bg-card border border-border rounded-lg p-5">
        <h3 className="text-[10px] uppercase tracking-widest text-text-secondary font-semibold mb-3">Setup</h3>
        <div className="space-y-2 text-xs text-text-secondary">
          <p>1. Copy <code className="text-accent-green bg-surface px-1.5 py-0.5 rounded">.env.example</code> → <code className="text-accent-green bg-surface px-1.5 py-0.5 rounded">.env</code></p>
          <p>2. Add Kalshi API key + private key from <code className="text-white">demo.kalshi.co</code></p>
          <p>3. Add Anthropic API key (optional, for AI+RF hybrid)</p>
          <p>4. Backend: <code className="text-accent-green bg-surface px-1.5 py-0.5 rounded">uvicorn bot.server:app --reload --port 8000</code></p>
          <p>5. Frontend: <code className="text-accent-green bg-surface px-1.5 py-0.5 rounded">cd frontend && npm run dev</code></p>
        </div>
      </div>
    </div>
  );
}
