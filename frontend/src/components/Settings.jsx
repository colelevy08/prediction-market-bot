/**
 * Settings -- Main configuration and monitoring dashboard for the prediction market bot.
 *
 * This component serves as the central control panel, providing real-time status
 * monitoring, trading parameter configuration, automation controls, and trade logs.
 *
 * Key sections:
 *   - Connections: Displays live connection status for Kalshi API, Claude AI,
 *     Supabase DB, current environment (demo/live), and RF model info.
 *   - Paper Trader Status: Shows balance, model state, training samples, open
 *     positions, total scans, and auto-scan status.
 *   - Trading Parameters: Editable form for max bet, min edge threshold, daily
 *     loss limit, max positions, markets to scan, and Kelly fraction. Includes
 *     a daily P&L reset button.
 *   - 24/7 Automation: Toggle controls for auto-scan (paper/shadow trading) and
 *     auto-trade (live Kalshi orders), configurable scan interval, and a log of
 *     recent scan results.
 *   - Trade Logs: Tabbed view (Shadow vs Live) showing open positions, completed
 *     trades with entry/exit prices and P&L, and recent activity feeds. Auto-
 *     refreshes every 30 seconds when auto-scan is active.
 *   - Notifications: Shows Slack/Discord webhook configuration status with a
 *     test button.
 *   - Model Retrain Schedule: Configure retrain days/hour (UTC), view next
 *     scheduled run, and trigger immediate retraining.
 *   - Strategy Reference: Read-only summary of the bot's trading strategy
 *     (RF model, Kelly sizing, risk limits, entry/exit rules, Sharpe ratio).
 *   - Setup: Step-by-step environment setup instructions.
 *
 * API endpoints called:
 *   - api.getAutoScanStatus()        -- Fetch auto-scan/trade state and scan log
 *   - api.getNotificationConfig()    -- Fetch Slack/Discord configuration status
 *   - api.getRetrainSchedule()       -- Fetch model retrain schedule
 *   - api.getShadowTradeLog()        -- Fetch shadow/paper trade log
 *   - api.getLiveTradeLog()          -- Fetch live trade log
 *   - api.updateConfig(config)       -- Save trading parameter changes
 *   - api.resetDailyPnl()            -- Reset the daily P&L counter
 *   - api.toggleAutoScan(enabled, interval) -- Enable/disable auto-scan
 *   - api.toggleAutoTrade(enabled)   -- Enable/disable live auto-trading
 *   - api.testNotifications()        -- Send a test notification to configured channels
 *   - api.updateRetrainSchedule(days, hour) -- Update retrain schedule
 *   - api.retrainNow()               -- Trigger immediate model retrain
 *
 * Props:
 *   @param {Object} status   -- Bot status object (connections, config, model info, paper trader state)
 *   @param {Function} onRefresh -- Callback to refresh parent status after config saves or retrains
 */
import { useState, useEffect } from 'react';
import { api } from '../api';

export default function Settings({ status, onRefresh }) {
  const [config, setConfig] = useState({
    max_bet_amount_cents: status?.config?.max_bet_amount_cents || 2500,
    min_edge_threshold: status?.config?.min_edge_threshold || 0.08,
    max_daily_loss_cents: status?.config?.max_daily_loss_cents || 10000,
    max_open_positions: status?.config?.max_open_positions || 10,
    max_events_to_analyze: status?.config?.max_events_to_analyze || 20,
    kelly_fraction: status?.config?.kelly_fraction || 0.5,
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
  const [retrainLoading, setRetrainLoading] = useState(false);
  const [shadowLog, setShadowLog] = useState(null);
  const [liveLog, setLiveLog] = useState(null);
  const [logTab, setLogTab] = useState('shadow');

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
      api.getShadowTradeLog().then(setShadowLog).catch(() => {}),
      api.getLiveTradeLog().then(setLiveLog).catch(() => {}),
    ]).catch(() => {});
  }, []);

  // Auto-refresh trade logs every 30s when auto-scan is active
  useEffect(() => {
    if (!autoScan) return;
    const interval = setInterval(() => {
      api.getShadowTradeLog().then(setShadowLog).catch(() => {});
      api.getLiveTradeLog().then(setLiveLog).catch(() => {});
      api.getAutoScanStatus().then(s => setScanLog(s.log || [])).catch(() => {});
    }, 30000);
    return () => clearInterval(interval);
  }, [autoScan]);

  // Sync config when status prop updates
  useEffect(() => {
    if (status?.config) {
      setConfig(prev => ({
        ...prev,
        max_bet_amount_cents: status.config.max_bet_amount_cents ?? prev.max_bet_amount_cents,
        min_edge_threshold: status.config.min_edge_threshold ?? prev.min_edge_threshold,
        max_daily_loss_cents: status.config.max_daily_loss_cents ?? prev.max_daily_loss_cents,
        max_open_positions: status.config.max_open_positions ?? prev.max_open_positions,
        max_events_to_analyze: status.config.max_events_to_analyze ?? prev.max_events_to_analyze,
        kelly_fraction: status.config.kelly_fraction ?? prev.kelly_fraction,
      }));
    }
  }, [status?.config]);

  const handleSave = async () => {
    try {
      await api.updateConfig(config);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
      onRefresh();
    } catch (e) { alert(`Save failed: ${e.message}`); }
  };

  const pt = status?.paper_trader;

  return (
    <div className="space-y-6 max-w-2xl">
      {/* Connection status */}
      <div className="bg-card border border-border rounded-lg p-5">
        <h3 className="text-[10px] uppercase tracking-widest text-text-secondary font-semibold mb-4">Connections</h3>
        <div className="space-y-3">
          {[
            { label: 'Kalshi API', ok: status?.kalshi_connected },
            { label: 'Claude AI', ok: status?.anthropic_connected, optional: true },
            { label: 'Supabase DB', ok: status?.supabase_connected, optional: true },
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

      {/* Paper Trader Status */}
      {pt && (
        <div className="bg-card border border-border rounded-lg p-5">
          <h3 className="text-[10px] uppercase tracking-widest text-text-secondary font-semibold mb-4">Paper Trader Status</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {[
              { label: 'Balance', value: `$${(pt.balance_cents / 100).toFixed(2)}`, color: 'text-accent-green', border: 'border-l-green', icon: '💰' },
              { label: 'Model', value: pt.model_trained ? 'Trained' : 'Heuristic', color: pt.model_trained ? 'text-accent-green' : 'text-accent-yellow', border: pt.model_trained ? 'border-l-green' : 'border-l-yellow', icon: '🧠' },
              { label: 'Training Samples', value: pt.training_samples?.toLocaleString() || '0', border: 'border-l-purple', icon: '📚' },
              { label: 'Open Positions', value: pt.open_positions, color: pt.open_positions > 0 ? 'text-accent-blue' : 'text-white', border: 'border-l-blue', icon: '📊' },
              { label: 'Total Scans', value: pt.total_scans, border: 'border-l-cyan', icon: '🔍' },
              { label: 'Auto-Scan', value: status?.auto_scan_enabled ? 'Active' : 'Off', color: status?.auto_scan_enabled ? 'text-accent-green' : 'text-text-muted', border: status?.auto_scan_enabled ? 'border-l-green' : '', icon: status?.auto_scan_enabled ? '🟢' : '⏸️' },
            ].map((item, i) => (
              <div key={i} className={`bg-surface border border-border rounded-lg p-3 ${item.border || ''}`}>
                <div className="flex items-center justify-between mb-1">
                  <div className="text-[10px] uppercase tracking-widest text-text-muted">{item.label}</div>
                  {item.icon && <span className="text-xs opacity-40">{item.icon}</span>}
                </div>
                <div className={`text-lg font-bold font-mono ${item.color || 'text-white'}`}>{item.value}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Trading Parameters */}
      <div className="bg-card border border-border rounded-lg p-5">
        <h3 className="text-[10px] uppercase tracking-widest text-text-secondary font-semibold mb-4">Trading Parameters</h3>
        <div className="space-y-4">
          {[
            { label: 'Max Bet ($)', key: 'max_bet_amount_cents', div: 100, step: 1, min: 1, note: 'Maximum per-trade wager' },
            { label: 'Min Edge (%)', key: 'min_edge_threshold', div: 0.01, step: 0.5, min: 0, max: 50, note: 'Minimum model vs market edge to enter' },
            { label: 'Daily Loss Limit ($)', key: 'max_daily_loss_cents', div: 100, step: 10, min: 10, note: 'Stops trading when daily losses exceed this' },
            { label: 'Max Positions', key: 'max_open_positions', div: 1, step: 1, min: 1, max: 50, note: 'Maximum simultaneous open positions' },
            { label: 'Markets to Scan', key: 'max_events_to_analyze', div: 1, step: 5, min: 5, max: 200, note: 'Number of events fetched per scan cycle' },
            { label: 'Kelly Fraction', key: 'kelly_fraction', div: 1, step: 0.05, min: 0.1, max: 1.0, note: 'Position sizing (0.5 = Half-Kelly, safer)' },
          ].map(f => (
            <div key={f.key}>
              <label className="text-[10px] uppercase tracking-widest text-text-secondary block mb-1">
                {f.label}
                {f.note && <span className="normal-case tracking-normal text-text-muted ml-2">-- {f.note}</span>}
              </label>
              <input type="number" step={f.step} min={f.min} max={f.max}
                value={f.div === 1 ? config[f.key] : (config[f.key] / f.div).toFixed(f.div === 100 ? 2 : 1)}
                onChange={e => {
                  const val = parseFloat(e.target.value);
                  if (!isNaN(val)) {
                    setConfig(prev => ({ ...prev, [f.key]: f.div === 1 ? val : Math.round(val * f.div) }));
                  }
                }}
                className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm font-mono text-white focus:border-accent-green focus:outline-none transition-colors" />
            </div>
          ))}
          <div className="flex gap-2">
            <button onClick={handleSave}
              className="flex-1 py-2.5 bg-white text-black text-xs font-semibold tracking-wide rounded-lg hover:bg-gray-200 transition-all uppercase">
              {saved ? 'Saved!' : 'Save Settings'}
            </button>
            <button onClick={async () => {
              try {
                await api.resetDailyPnl();
                alert('Daily P&L counter reset');
              } catch (e) { alert(`Reset failed: ${e.message}`); }
            }}
              className="px-4 py-2.5 bg-accent-yellow/10 border border-accent-yellow/20 text-accent-yellow text-xs font-semibold tracking-wide rounded-lg hover:bg-accent-yellow/20 transition-all uppercase">
              Reset Daily P&L
            </button>
          </div>
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
                try {
                  const next = !autoScan;
                  await api.toggleAutoScan(next, scanInterval);
                  setAutoScan(next);
                } catch (e) { alert(`Toggle failed: ${e.message}`); }
              }}
              className={`text-[10px] px-3 py-1.5 rounded font-semibold uppercase tracking-wide transition-all ${
                autoScan ? 'bg-accent-green/10 text-accent-green' : 'bg-white/5 text-text-muted hover:bg-white/10'
              }`}
            >
              {autoScan ? 'Running' : 'Off'}
            </button>
          </div>

          <div>
            <label className="text-[10px] uppercase tracking-widest text-text-secondary block mb-1">
              Scan Interval (seconds)
              <span className="normal-case tracking-normal text-text-muted ml-2">-- scans ALL of Kalshi (~5000+ events, ~40000 markets). Min 30s recommended.</span>
            </label>
            <input type="number" min={30} max={600} step={10} value={scanInterval}
              onChange={e => setScanInterval(Math.max(30, parseInt(e.target.value) || 60))}
              className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm font-mono text-white focus:border-accent-green focus:outline-none transition-colors" />
          </div>

          <div className="flex items-center justify-between pt-2 border-t border-border">
            <div>
              <span className="text-xs font-semibold">Auto-Trade (Live)</span>
              <p className="text-[10px] text-accent-red mt-0.5">Executes real trades on Kalshi -- use with caution</p>
            </div>
            <button
              onClick={async () => {
                if (!autoTrade && !confirm('This will place REAL trades on Kalshi. Continue?')) return;
                try {
                  const next = !autoTrade;
                  await api.toggleAutoTrade(next);
                  setAutoTrade(next);
                } catch (e) { alert(`Toggle failed: ${e.message}`); }
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
              <h4 className="text-[10px] uppercase tracking-widest text-text-muted mb-2">Recent Scans ({scanLog.length})</h4>
              <div className="space-y-1 max-h-48 overflow-y-auto">
                {scanLog.slice().reverse().map((entry, i) => (
                  <div key={i} className="text-[10px] font-mono text-text-secondary bg-surface rounded px-2 py-1.5 flex justify-between items-center gap-2">
                    <span className="shrink-0">{entry.time ? new Date(entry.time).toLocaleTimeString() : '--'}</span>
                    <span className="text-text-muted shrink-0">
                      {entry.events_scanned ? `${entry.events_scanned.toLocaleString()} events` : ''}
                      {entry.markets_scanned ? ` / ${entry.markets_scanned.toLocaleString()} mkts` : ''}
                    </span>
                    <span className={entry.type === 'live' ? 'text-accent-red' : 'text-accent-green'}>
                      {entry.type === 'live'
                        ? `${entry.signals || 0} signals, ${entry.exits || 0} exits`
                        : `+${entry.entries || 0} / -${entry.exits || 0} / ${entry.open_positions ?? 0} open`
                      }
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Trade Logs — Shadow vs Live */}
      <div className="bg-card border border-border rounded-lg p-5">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-[10px] uppercase tracking-widest text-text-secondary font-semibold">Trade Logs</h3>
          <div className="flex gap-1">
            {[
              { id: 'shadow', label: 'Shadow', color: 'accent-green' },
              { id: 'live', label: 'Live', color: 'accent-red' },
            ].map(tab => (
              <button key={tab.id} onClick={() => setLogTab(tab.id)}
                className={`text-[10px] px-3 py-1 rounded font-semibold uppercase tracking-wide transition-all ${
                  logTab === tab.id
                    ? `bg-${tab.color}/10 text-${tab.color}`
                    : 'bg-white/5 text-text-muted hover:bg-white/10'
                }`}>
                {tab.label}
                {tab.id === 'shadow' && shadowLog ? ` (${shadowLog.total_shadow_trades})` : ''}
                {tab.id === 'live' && liveLog ? ` (${liveLog.total_live_trades})` : ''}
              </button>
            ))}
            <button onClick={() => {
              api.getShadowTradeLog().then(setShadowLog).catch(() => {});
              api.getLiveTradeLog().then(setLiveLog).catch(() => {});
            }}
              className="text-[10px] px-2 py-1 rounded bg-white/5 text-text-muted hover:bg-white/10 transition-all ml-1">
              Refresh
            </button>
          </div>
        </div>

        {logTab === 'shadow' && (
          <div className="space-y-3">
            {/* Open shadow positions */}
            {shadowLog?.open_positions?.length > 0 && (
              <div>
                <h4 className="text-[10px] uppercase tracking-widest text-accent-green mb-2">
                  Open Positions ({shadowLog.open_positions.length})
                </h4>
                <div className="space-y-1">
                  {shadowLog.open_positions.map((p, i) => (
                    <div key={i} className="text-[10px] font-mono bg-surface rounded px-2 py-1.5 flex justify-between items-center">
                      <span className="text-white">{p.ticker}</span>
                      <div className="flex items-center gap-3 text-text-secondary">
                        <span className={p.side === 'yes' ? 'text-accent-green' : 'text-accent-red'}>{p.side?.toUpperCase()}</span>
                        <span>{p.contracts}x @ {(p.entry_price * 100).toFixed(0)}c</span>
                        <span>Model: {(p.model_prob * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Completed shadow trades */}
            {shadowLog?.completed_trades?.length > 0 ? (
              <div>
                <h4 className="text-[10px] uppercase tracking-widest text-text-muted mb-2">
                  Completed Trades ({shadowLog.completed_trades.length})
                </h4>
                <div className="overflow-x-auto">
                  <table className="w-full text-[10px] font-mono">
                    <thead>
                      <tr className="text-text-muted border-b border-border uppercase tracking-widest">
                        <th className="text-left py-1.5 px-2">Ticker</th>
                        <th className="text-center py-1.5 px-1">Side</th>
                        <th className="text-right py-1.5 px-1">Entry</th>
                        <th className="text-right py-1.5 px-1">Exit</th>
                        <th className="text-right py-1.5 px-2">P&L</th>
                        <th className="text-center py-1.5 px-1">Result</th>
                      </tr>
                    </thead>
                    <tbody>
                      {shadowLog.completed_trades.map((t, i) => (
                        <tr key={i} className={`border-b border-border/30 ${t.won ? 'row-win' : 'row-loss'}`}>
                          <td className="py-1 px-2 text-white max-w-[120px] truncate">{t.ticker}</td>
                          <td className="py-1 px-1 text-center">
                            <span className={`px-1 py-0.5 rounded text-[9px] font-semibold ${t.side === 'yes' ? 'bg-accent-green/10 text-accent-green' : 'bg-accent-red/10 text-accent-red'}`}>
                              {t.side?.toUpperCase()}
                            </span>
                          </td>
                          <td className="py-1 px-1 text-right text-text-secondary">{t.entry_price != null ? (t.entry_price * 100).toFixed(0) + 'c' : '--'}</td>
                          <td className="py-1 px-1 text-right text-text-secondary">{t.exit_price != null ? (t.exit_price * 100).toFixed(0) + 'c' : '--'}</td>
                          <td className="py-1 px-2 text-right">
                            <span className={`inline-block px-1 py-0.5 rounded font-semibold ${(t.pnl_cents ?? 0) >= 0 ? 'bg-accent-green/10 text-accent-green' : 'bg-accent-red/10 text-accent-red'}`}>
                              ${((t.pnl_cents ?? 0) / 100).toFixed(2)}
                            </span>
                          </td>
                          <td className="py-1 px-1 text-center">
                            <span className={`px-1.5 py-0.5 rounded ${t.won ? 'bg-accent-green/10 text-accent-green' : 'bg-accent-red/10 text-accent-red'}`}>
                              {t.won ? 'W' : 'L'}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : (
              <div className="text-[10px] text-text-muted text-center py-4">
                No shadow trades yet. The bot will trade when it finds edges above the {(config.min_edge_threshold * 100 / 0.01 * 0.01).toFixed(0)}% threshold.
              </div>
            )}

            {/* Live entry/exit log */}
            {shadowLog?.live_log?.length > 0 && (
              <div>
                <h4 className="text-[10px] uppercase tracking-widest text-text-muted mb-2">Recent Activity</h4>
                <div className="space-y-1 max-h-32 overflow-y-auto">
                  {shadowLog.live_log.slice().reverse().map((entry, i) => (
                    <div key={i} className="text-[10px] font-mono bg-surface rounded px-2 py-1 flex justify-between">
                      <span className="text-text-secondary">{entry.time ? new Date(entry.time).toLocaleTimeString() : '--'}</span>
                      <span className={entry.action === 'entry' ? 'text-accent-green' : 'text-accent-yellow'}>
                        {entry.action?.toUpperCase()} {entry.ticker} {entry.side?.toUpperCase()}
                        {entry.pnl_cents != null ? ` P&L: $${(entry.pnl_cents / 100).toFixed(2)}` : ''}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {logTab === 'live' && (
          <div className="space-y-3">
            {liveLog?.completed_trades?.length > 0 ? (
              <div>
                <h4 className="text-[10px] uppercase tracking-widest text-text-muted mb-2">
                  Completed Live Trades ({liveLog.completed_trades.length})
                </h4>
                <div className="overflow-x-auto">
                  <table className="w-full text-[10px] font-mono">
                    <thead>
                      <tr className="text-text-muted border-b border-border uppercase tracking-widest">
                        <th className="text-left py-1.5 px-2">Ticker</th>
                        <th className="text-center py-1.5 px-1">Side</th>
                        <th className="text-right py-1.5 px-1">Entry</th>
                        <th className="text-right py-1.5 px-1">Exit</th>
                        <th className="text-right py-1.5 px-2">P&L</th>
                        <th className="text-center py-1.5 px-1">Result</th>
                      </tr>
                    </thead>
                    <tbody>
                      {liveLog.completed_trades.map((t, i) => (
                        <tr key={i} className={`border-b border-border/30 ${t.won ? 'row-win' : 'row-loss'}`}>
                          <td className="py-1 px-2 text-white max-w-[120px] truncate">{t.ticker}</td>
                          <td className="py-1 px-1 text-center">
                            <span className={`px-1 py-0.5 rounded text-[9px] font-semibold ${t.side === 'yes' ? 'bg-accent-green/10 text-accent-green' : 'bg-accent-red/10 text-accent-red'}`}>
                              {t.side?.toUpperCase()}
                            </span>
                          </td>
                          <td className="py-1 px-1 text-right text-text-secondary">{t.entry_price != null ? (t.entry_price * 100).toFixed(0) + 'c' : '--'}</td>
                          <td className="py-1 px-1 text-right text-text-secondary">{t.exit_price != null ? (t.exit_price * 100).toFixed(0) + 'c' : '--'}</td>
                          <td className="py-1 px-2 text-right">
                            <span className={`inline-block px-1 py-0.5 rounded font-semibold ${(t.pnl_cents ?? 0) >= 0 ? 'bg-accent-green/10 text-accent-green' : 'bg-accent-red/10 text-accent-red'}`}>
                              ${((t.pnl_cents ?? 0) / 100).toFixed(2)}
                            </span>
                          </td>
                          <td className="py-1 px-1 text-center">
                            <span className={`px-1.5 py-0.5 rounded ${t.won ? 'bg-accent-green/10 text-accent-green' : 'bg-accent-red/10 text-accent-red'}`}>
                              {t.won ? 'W' : 'L'}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : (
              <div className="text-[10px] text-text-muted text-center py-4">
                No live trades yet. Enable Auto-Trade to execute real orders on Kalshi.
              </div>
            )}

            {liveLog?.live_log?.length > 0 && (
              <div>
                <h4 className="text-[10px] uppercase tracking-widest text-text-muted mb-2">Recent Live Activity</h4>
                <div className="space-y-1 max-h-32 overflow-y-auto">
                  {liveLog.live_log.slice().reverse().map((entry, i) => (
                    <div key={i} className="text-[10px] font-mono bg-surface rounded px-2 py-1 flex justify-between">
                      <span className="text-text-secondary">{entry.time ? new Date(entry.time).toLocaleTimeString() : '--'}</span>
                      <span className="text-accent-red">
                        {entry.action?.toUpperCase()} {entry.ticker} {entry.side?.toUpperCase()}
                        {entry.edge ? ` edge: ${(entry.edge * 100).toFixed(1)}%` : ''}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
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
          <p className="text-[10px] text-text-muted">
            Set <code className="text-accent-green bg-surface px-1 py-0.5 rounded">SLACK_WEBHOOK_URL</code> and/or{' '}
            <code className="text-accent-green bg-surface px-1 py-0.5 rounded">DISCORD_WEBHOOK_URL</code> in .env to enable notifications for trade entries, exits, and retrains.
          </p>
          <div className="flex gap-2">
            <button onClick={async () => {
              try {
                const r = await api.testNotifications();
                setNotifTestResult(r.status === 'sent' ? `Test sent to ${r.channels?.join(', ') || 'channels'}!` : 'Not configured -- set webhook URLs in .env');
                setTimeout(() => setNotifTestResult(''), 4000);
              } catch (e) { setNotifTestResult('Failed to send'); setTimeout(() => setNotifTestResult(''), 3000); }
            }}
              className="px-3 py-1.5 bg-white/5 text-text-secondary text-[10px] font-semibold uppercase tracking-wide rounded hover:bg-white/10 transition-all">
              Test Notifications
            </button>
            {notifTestResult && (
              <span className={`text-[10px] self-center ${notifTestResult.includes('sent') ? 'text-accent-green' : 'text-accent-yellow'}`}>
                {notifTestResult}
              </span>
            )}
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
            <div className="text-[10px] text-text-secondary">
              Next run: <span className="text-white font-mono">{new Date(retrainSchedule.next_run).toLocaleString()}</span>
            </div>
          )}
          <div>
            <label className="text-[10px] uppercase tracking-widest text-text-secondary block mb-1">
              Days <span className="normal-case tracking-normal text-text-muted">-- comma-separated: mon,tue,wed,thu,fri,sat,sun</span>
            </label>
            <input type="text" value={retrainDays} onChange={e => setRetrainDays(e.target.value)}
              placeholder="mon,wed,fri"
              className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm font-mono text-white focus:border-accent-green focus:outline-none transition-colors" />
          </div>
          <div>
            <label className="text-[10px] uppercase tracking-widest text-text-secondary block mb-1">
              Hour (0-23 UTC)
            </label>
            <input type="number" min={0} max={23} value={retrainHour} onChange={e => setRetrainHour(parseInt(e.target.value) || 0)}
              className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm font-mono text-white focus:border-accent-green focus:outline-none transition-colors" />
          </div>
          <div className="flex gap-2">
            <button onClick={async () => {
              try {
                await api.updateRetrainSchedule(retrainDays, retrainHour);
                const s = await api.getRetrainSchedule();
                setRetrainSchedule(s);
                alert('Schedule updated!');
              } catch (e) { alert(`Update failed: ${e.message}`); }
            }}
              className="px-3 py-1.5 bg-white text-black text-[10px] font-semibold uppercase tracking-wide rounded hover:bg-gray-200 transition-all">
              Update Schedule
            </button>
            <button onClick={async () => {
              setRetrainLoading(true);
              try {
                await api.retrainNow();
                alert('Retrain complete! Cumulative training data has been updated.');
                onRefresh();
              } catch (e) { alert(`Retrain failed: ${e.message}`); }
              finally { setRetrainLoading(false); }
            }}
              disabled={retrainLoading}
              className="px-3 py-1.5 bg-accent-green/10 text-accent-green text-[10px] font-semibold uppercase tracking-wide rounded hover:bg-accent-green/20 disabled:opacity-30 transition-all">
              {retrainLoading ? 'Retraining...' : 'Retrain Now'}
            </button>
          </div>
        </div>
      </div>

      {/* Strategy reference */}
      <div className="bg-card border border-border rounded-lg p-5">
        <h3 className="text-[10px] uppercase tracking-widest text-text-secondary font-semibold mb-3">Strategy Reference</h3>
        <div className="space-y-1.5 text-xs font-mono">
          {[
            { phase: '1', text: `Random Forest -- ${status?.model?.n_estimators || 500} trees + GB(150), sqrt(features) per tree` },
            { phase: '2', text: `Kelly sizing: f* = (bp - q) / b, using ${(config.kelly_fraction * 100).toFixed(0)}% Kelly (${config.kelly_fraction <= 0.25 ? 'quarter' : config.kelly_fraction <= 0.5 ? 'half' : 'full'})` },
            { phase: '3', text: `Risk: max $${(config.max_bet_amount_cents / 100).toFixed(0)}/bet, $${(config.max_daily_loss_cents / 100).toFixed(0)}/day loss limit, ${config.max_open_positions} max positions` },
            { phase: '4', text: `Entry: edge >= ${(config.min_edge_threshold * 100).toFixed(0)}% (model_prob - market_price)` },
            { phase: '5', text: 'Sharpe Ratio = (Rp - Rf) / \u03C3  |  <1 bad, 1-2 good, >2 excellent' },
            { phase: '6', text: 'Log returns = ln(P1/P0) -- additive, direction-adjusted for YES/NO sides' },
            { phase: '7', text: 'Exit: market_price >= model_prob * 0.9 or days_to_expiry <= 7' },
            { phase: '8', text: 'Training: cumulative data persistence -- each run builds on previous samples' },
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
          <p>1. Copy <code className="text-accent-green bg-surface px-1.5 py-0.5 rounded">.env.example</code> to <code className="text-accent-green bg-surface px-1.5 py-0.5 rounded">.env</code></p>
          <p>2. Add Kalshi API key + private key from <code className="text-white">demo.kalshi.co</code></p>
          <p>3. Add Anthropic API key (optional, for AI+RF hybrid analysis)</p>
          <p>4. Add Supabase URL + key (optional, for persistent trade history)</p>
          <p>5. Add Slack/Discord webhook URLs (optional, for trade notifications)</p>
          <p>6. Backend: <code className="text-accent-green bg-surface px-1.5 py-0.5 rounded">uvicorn bot.server:app --reload --port 8000</code></p>
          <p>7. Frontend: <code className="text-accent-green bg-surface px-1.5 py-0.5 rounded">cd frontend && npm run dev</code></p>
        </div>
      </div>
    </div>
  );
}
