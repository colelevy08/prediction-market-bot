import { useState, useEffect } from 'react';
import { api } from '../api';

export default function Settings({ status, onRefresh }) {
  const [config, setConfig] = useState({
    max_bet_amount_cents: status?.config?.max_bet_amount_cents || 2500,
    min_edge_threshold: status?.config?.min_edge_threshold || 0.08,
    max_daily_loss_cents: status?.config?.max_daily_loss_cents || 10000,
    max_open_positions: status?.config?.max_open_positions || 10,
  });
  const [saved, setSaved] = useState(false);
  const [autoScan, setAutoScan] = useState(false);
  const [autoTrade, setAutoTrade] = useState(false);
  const [scanInterval, setScanInterval] = useState(60);
  const [scanLog, setScanLog] = useState([]);
  const [notifConfig, setNotifConfig] = useState({});
  const [notifTestResult, setNotifTestResult] = useState('');
  const [retrainSchedule, setRetrainSchedule] = useState({ days: 'mon,wed,fri', hour: 3, active: false, next_run: null });
  const [retrainDays, setRetrainDays] = useState('mon,wed,fri');
  const [retrainHour, setRetrainHour] = useState(3);

  useEffect(() => {
    Promise.all([
      api.getAutoScanStatus().then(s => {
        setAutoScan(s.auto_scan_enabled || false);
        setAutoTrade(s.auto_trade_enabled || false);
        setScanLog(s.log || []);
      }),
      api.getNotificationConfig().then(setNotifConfig).catch(() => {}),
      api.getRetrainSchedule().then(s => {
        setRetrainSchedule(s);
        setRetrainDays(s.days || 'mon,wed,fri');
        setRetrainHour(s.hour || 3);
      }).catch(() => {}),
    ]).catch(() => {});
  }, []);

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

      {/* Auto-Scan / Auto-Trade */}
      <div className="bg-card border border-border rounded-lg p-5">
        <h3 className="text-[10px] uppercase tracking-widest text-text-secondary font-semibold mb-4">24/7 Automation</h3>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <span className="text-xs font-semibold">Auto-Scan (Paper)</span>
              <p className="text-[10px] text-text-muted mt-0.5">Scans markets and records shadow trades automatically</p>
            </div>
            <button
              onClick={async () => {
                const next = !autoScan;
                await api.toggleAutoScan(next, scanInterval);
                setAutoScan(next);
              }}
              className={`text-[10px] px-3 py-1.5 rounded font-semibold uppercase tracking-wide transition-all ${
                autoScan ? 'bg-accent-green/10 text-accent-green' : 'bg-white/5 text-text-muted hover:bg-white/10'
              }`}
            >
              {autoScan ? 'Running' : 'Off'}
            </button>
          </div>

          <div>
            <label className="text-[10px] uppercase tracking-widest text-text-secondary block mb-1">Scan Interval (seconds)</label>
            <input type="number" min={10} max={600} step={10} value={scanInterval}
              onChange={e => setScanInterval(parseInt(e.target.value) || 60)}
              className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm font-mono text-white focus:border-accent-green focus:outline-none transition-colors" />
          </div>

          <div className="flex items-center justify-between pt-2 border-t border-border">
            <div>
              <span className="text-xs font-semibold">Auto-Trade (Live)</span>
              <p className="text-[10px] text-accent-red mt-0.5">Executes real trades on Kalshi — use with caution</p>
            </div>
            <button
              onClick={async () => {
                if (!autoTrade && !confirm('This will place REAL trades on Kalshi. Continue?')) return;
                const next = !autoTrade;
                await api.toggleAutoTrade(next);
                setAutoTrade(next);
              }}
              className={`text-[10px] px-3 py-1.5 rounded font-semibold uppercase tracking-wide transition-all ${
                autoTrade ? 'bg-accent-red/10 text-accent-red' : 'bg-white/5 text-text-muted hover:bg-white/10'
              }`}
            >
              {autoTrade ? 'Live' : 'Off'}
            </button>
          </div>

          {scanLog.length > 0 && (
            <div className="mt-3">
              <h4 className="text-[10px] uppercase tracking-widest text-text-muted mb-2">Recent Scans</h4>
              <div className="space-y-1 max-h-32 overflow-y-auto">
                {scanLog.slice().reverse().map((entry, i) => (
                  <div key={i} className="text-[10px] font-mono text-text-secondary bg-surface rounded px-2 py-1 flex justify-between">
                    <span>{new Date(entry.time).toLocaleTimeString()}</span>
                    <span className={entry.type === 'live' ? 'text-accent-red' : 'text-accent-green'}>
                      {entry.type === 'live' ? `${entry.signals || 0} signals` : `${entry.entries || 0} entries, ${entry.exits || 0} exits`}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Notifications */}
      <div className="bg-card border border-border rounded-lg p-5">
        <h3 className="text-[10px] uppercase tracking-widest text-text-secondary font-semibold mb-4">Notifications</h3>
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs">Slack</span>
            <span className={`text-[10px] px-2.5 py-1 rounded font-semibold uppercase tracking-wide ${
              notifConfig.slack_configured ? 'bg-accent-green/10 text-accent-green' : 'bg-white/5 text-text-muted'
            }`}>{notifConfig.slack_configured ? 'Configured' : 'Not Set'}</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs">Discord</span>
            <span className={`text-[10px] px-2.5 py-1 rounded font-semibold uppercase tracking-wide ${
              notifConfig.discord_configured ? 'bg-accent-green/10 text-accent-green' : 'bg-white/5 text-text-muted'
            }`}>{notifConfig.discord_configured ? 'Configured' : 'Not Set'}</span>
          </div>
          <p className="text-[10px] text-text-muted">Set SLACK_WEBHOOK_URL and/or DISCORD_WEBHOOK_URL in .env</p>
          <div className="flex gap-2">
            <button onClick={async () => {
              try {
                const r = await api.testNotifications();
                setNotifTestResult(r.status === 'sent' ? 'Test sent!' : 'Not configured');
                setTimeout(() => setNotifTestResult(''), 3000);
              } catch (e) { setNotifTestResult('Failed'); }
            }}
              className="px-3 py-1.5 bg-white/5 text-text-secondary text-[10px] font-semibold uppercase tracking-wide rounded hover:bg-white/10 transition-all">
              Test Notifications
            </button>
            {notifTestResult && <span className="text-[10px] text-accent-green self-center">{notifTestResult}</span>}
          </div>
        </div>
      </div>

      {/* Retrain Schedule */}
      <div className="bg-card border border-border rounded-lg p-5">
        <h3 className="text-[10px] uppercase tracking-widest text-text-secondary font-semibold mb-4">Model Retrain Schedule</h3>
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs">Status</span>
            <span className={`text-[10px] px-2.5 py-1 rounded font-semibold uppercase tracking-wide ${
              retrainSchedule.active ? 'bg-accent-green/10 text-accent-green' : 'bg-white/5 text-text-muted'
            }`}>{retrainSchedule.active ? 'Active' : 'Inactive'}</span>
          </div>
          {retrainSchedule.next_run && (
            <div className="text-[10px] text-text-secondary">Next: {new Date(retrainSchedule.next_run).toLocaleString()}</div>
          )}
          <div>
            <label className="text-[10px] uppercase tracking-widest text-text-secondary block mb-1">Days (comma-separated)</label>
            <input type="text" value={retrainDays} onChange={e => setRetrainDays(e.target.value)}
              placeholder="mon,wed,fri"
              className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm font-mono text-white focus:border-accent-green focus:outline-none transition-colors" />
          </div>
          <div>
            <label className="text-[10px] uppercase tracking-widest text-text-secondary block mb-1">Hour (0-23 UTC)</label>
            <input type="number" min={0} max={23} value={retrainHour} onChange={e => setRetrainHour(parseInt(e.target.value) || 0)}
              className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm font-mono text-white focus:border-accent-green focus:outline-none transition-colors" />
          </div>
          <div className="flex gap-2">
            <button onClick={async () => {
              await api.updateRetrainSchedule(retrainDays, retrainHour);
              const s = await api.getRetrainSchedule();
              setRetrainSchedule(s);
            }}
              className="px-3 py-1.5 bg-white text-black text-[10px] font-semibold uppercase tracking-wide rounded hover:bg-gray-200 transition-all">
              Update Schedule
            </button>
            <button onClick={async () => {
              try { await api.retrainNow(); alert('Retrain complete!'); } catch (e) { alert(`Retrain failed: ${e.message}`); }
            }}
              className="px-3 py-1.5 bg-accent-green/10 text-accent-green text-[10px] font-semibold uppercase tracking-wide rounded hover:bg-accent-green/20 transition-all">
              Retrain Now
            </button>
          </div>
        </div>
      </div>

      {/* Strategy reference */}
      <div className="bg-card border border-border rounded-lg p-5">
        <h3 className="text-[10px] uppercase tracking-widest text-text-secondary font-semibold mb-3">Strategy Reference</h3>
        <div className="space-y-1.5 text-xs font-mono">
          {[
            { phase: '1', text: `Random Forest — ${status?.model?.n_estimators || 500} trees + GB(150), sqrt(features) per tree` },
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
