import { useState, useEffect, useRef } from 'react';
import { api } from '../api';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts';

function StatCard({ label, value, sub, color = 'text-white' }) {
  return (
    <div className="bg-card border border-border rounded-lg p-4 card-hover">
      <div className="text-[10px] uppercase tracking-widest text-text-secondary mb-2">{label}</div>
      <div className={`text-2xl font-bold font-mono ${color}`}>{value}</div>
      {sub && <div className="text-[10px] text-text-secondary mt-1.5">{sub}</div>}
    </div>
  );
}

export default function Testing() {
  const [mode, setMode] = useState('backtest');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Backtest
  const [btResult, setBtResult] = useState(null);
  const [btConfig, setBtConfig] = useState({
    initial_balance_cents: 10000,
    entry_threshold: 0.5,
    exit_threshold: 0.9,
    min_confidence: 0.70,
    max_markets: 200,
  });
  const [sweepResult, setSweepResult] = useState(null);

  // Paper / Shadow trading
  const [paperState, setPaperState] = useState(null);
  const [paperBalance, setPaperBalance] = useState(10000);
  const [scanHistory, setScanHistory] = useState([]);
  const [autoScan, setAutoScan] = useState(false);
  const intervalRef = useRef(null);

  useEffect(() => {
    if (mode === 'shadow') {
      api.getPaperState().then(setPaperState).catch(() => {});
    }
  }, [mode]);

  // Auto-scan for shadow trading
  useEffect(() => {
    if (autoScan) {
      intervalRef.current = setInterval(async () => {
        try {
          const result = await api.paperScan();
          setScanHistory(prev => [result, ...prev].slice(0, 100));
          const state = await api.getPaperState();
          setPaperState(state);
        } catch (e) { /* silent */ }
      }, 60000); // every 60s
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [autoScan]);

  const runBacktest = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.runBacktest(btConfig);
      setBtResult(result);
    } catch (e) {
      setError(`Backtest failed: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const runSweep = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.runSweep({ max_markets: btConfig.max_markets });
      setSweepResult(result);
    } catch (e) {
      setError(`Sweep failed: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const initPaper = async () => {
    setLoading(true);
    try {
      const state = await api.configurePaper(paperBalance);
      setPaperState(state);
      setScanHistory([]);
      setAutoScan(false);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const trainPaper = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.paperTrain();
      setPaperState(prev => prev ? { ...prev, model_trained: true } : prev);
      alert(`Trained on ${result.samples} samples. CV: ${(result.cv_accuracy * 100).toFixed(1)}%`);
    } catch (e) { setError(`Training failed: ${e.message}`); }
    finally { setLoading(false); }
  };

  const paperScan = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.paperScan();
      setScanHistory(prev => [result, ...prev].slice(0, 100));
      const state = await api.getPaperState();
      setPaperState(state);
    } catch (e) { setError(`Scan failed: ${e.message}`); }
    finally { setLoading(false); }
  };

  const MODES = [
    { id: 'backtest', label: 'Backtester' },
    { id: 'shadow', label: 'Shadow Trading' },
  ];

  return (
    <div className="space-y-6">
      {/* Mode tabs */}
      <div className="flex items-center gap-2">
        {MODES.map(m => (
          <button
            key={m.id}
            onClick={() => setMode(m.id)}
            className={`px-5 py-2 rounded-lg text-xs font-semibold tracking-wide uppercase transition-all ${
              mode === m.id
                ? 'bg-white text-black'
                : 'bg-card border border-border text-text-secondary hover:text-white'
            }`}
          >
            {m.label}
          </button>
        ))}
      </div>

      {error && (
        <div className="bg-accent-red/5 border border-accent-red/20 rounded-lg p-3 text-xs text-accent-red">
          {error}
        </div>
      )}

      {/* ── BACKTESTER ── */}
      {mode === 'backtest' && (
        <>
          <div className="bg-card border border-border rounded-lg p-5">
            <h2 className="text-sm font-semibold uppercase tracking-widest text-text-secondary mb-4">Configuration</h2>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              {[
                { label: 'Balance ($)', key: 'initial_balance_cents', div: 100 },
                { label: 'Entry Threshold', key: 'entry_threshold', step: 0.05 },
                { label: 'Exit Threshold', key: 'exit_threshold', step: 0.05 },
                { label: 'Min Confidence', key: 'min_confidence', step: 0.05 },
                { label: 'Max Markets', key: 'max_markets' },
              ].map(f => (
                <div key={f.key}>
                  <label className="text-[10px] uppercase tracking-widest text-text-secondary block mb-1">{f.label}</label>
                  <input
                    type="number"
                    step={f.step || 1}
                    value={f.div ? btConfig[f.key] / f.div : btConfig[f.key]}
                    onChange={e => setBtConfig(c => ({ ...c, [f.key]: f.div ? Math.round(parseFloat(e.target.value) * f.div) : parseFloat(e.target.value) }))}
                    className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm font-mono text-white focus:border-accent-green focus:outline-none transition-colors"
                  />
                </div>
              ))}
            </div>
            <div className="flex gap-3 mt-4">
              <button onClick={runBacktest} disabled={loading}
                className="px-5 py-2 bg-white text-black text-xs font-semibold tracking-wide rounded-lg hover:bg-gray-200 disabled:opacity-30 transition-all uppercase">
                {loading ? 'Running...' : 'Run Backtest'}
              </button>
              <button onClick={runSweep} disabled={loading}
                className="px-5 py-2 bg-card border border-border text-white text-xs font-semibold tracking-wide rounded-lg hover:bg-surface disabled:opacity-30 transition-all uppercase">
                {loading ? 'Running...' : 'Parameter Sweep'}
              </button>
            </div>
          </div>

          {btResult && !btResult.config?.error && (
            <>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
                <StatCard label="Win Rate" value={`${(btResult.win_rate * 100).toFixed(1)}%`}
                  sub={`${btResult.wins}W / ${btResult.losses}L`}
                  color={btResult.win_rate >= 0.6 ? 'text-accent-green' : 'text-accent-red'} />
                <StatCard label="Sharpe" value={btResult.sharpe_ratio} sub={btResult.sharpe_label}
                  color={btResult.sharpe_ratio >= 2 ? 'text-accent-green' : btResult.sharpe_ratio >= 1 ? 'text-accent-yellow' : 'text-accent-red'} />
                <StatCard label="P&L" value={`$${(btResult.total_pnl_cents / 100).toFixed(2)}`}
                  color={btResult.total_pnl_cents > 0 ? 'text-accent-green' : 'text-accent-red'} />
                <StatCard label="Profit Factor" value={btResult.profit_factor}
                  color={btResult.profit_factor >= 1.5 ? 'text-accent-green' : 'text-white'} />
                <StatCard label="CV Accuracy" value={`${(btResult.cv_accuracy * 100).toFixed(1)}%`}
                  sub={`OOB: ${(btResult.oob_score * 100).toFixed(1)}%`} />
                <StatCard label="Max Drawdown" value={`$${(btResult.max_drawdown_cents / 100).toFixed(2)}`} color="text-accent-red" />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
                <div className="bg-card border border-border rounded-lg p-3">
                  <span className="text-text-secondary">Train / Test:</span>{' '}
                  <span className="font-mono">{btResult.train_samples} / {btResult.test_samples}</span>
                </div>
                <div className="bg-card border border-border rounded-lg p-3">
                  <span className="text-text-secondary">Signals:</span>{' '}
                  <span className="font-mono text-accent-green">{btResult.signals_generated}</span>
                  <span className="text-text-secondary ml-2">Filtered:</span>{' '}
                  <span className="font-mono text-accent-red">{btResult.signals_filtered}</span>
                </div>
                <div className="bg-card border border-border rounded-lg p-3">
                  <span className="text-text-secondary">Avg Edge:</span>{' '}
                  <span className="font-mono">{(btResult.avg_edge * 100).toFixed(1)}%</span>
                  <span className="text-text-secondary ml-2">Log Return:</span>{' '}
                  <span className="font-mono">{btResult.avg_log_return?.toFixed(4)}</span>
                </div>
              </div>

              {btResult.equity_curve?.length > 0 && (
                <div className="bg-card border border-border rounded-lg p-5">
                  <h3 className="text-[10px] uppercase tracking-widest text-text-secondary mb-4">Equity Curve</h3>
                  <ResponsiveContainer width="100%" height={250}>
                    <LineChart data={btResult.equity_curve}>
                      <XAxis dataKey="trade_num" tick={{ fontSize: 10, fill: '#666' }} axisLine={{ stroke: '#1a1a1a' }} />
                      <YAxis tick={{ fontSize: 10, fill: '#666' }} tickFormatter={v => `$${(v / 100).toFixed(0)}`} axisLine={{ stroke: '#1a1a1a' }} />
                      <Tooltip contentStyle={{ background: '#0a0a0a', border: '1px solid #1a1a1a', borderRadius: 6, fontSize: 11, color: '#fff' }}
                        formatter={v => [`$${(v / 100).toFixed(2)}`, 'Equity']} />
                      <Line type="monotone" dataKey="equity_cents" stroke="#00ff87" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}

              {btResult.feature_importance && Object.keys(btResult.feature_importance).length > 0 && (
                <div className="bg-card border border-border rounded-lg p-5">
                  <h3 className="text-[10px] uppercase tracking-widest text-text-secondary mb-4">Feature Importance (Top 15)</h3>
                  <ResponsiveContainer width="100%" height={320}>
                    <BarChart data={Object.entries(btResult.feature_importance).slice(0, 15).map(([name, imp]) => ({
                      name: name.replace(/_/g, ' '), importance: +(imp * 100).toFixed(1),
                    }))} layout="vertical">
                      <XAxis type="number" tick={{ fontSize: 9, fill: '#666' }} axisLine={{ stroke: '#1a1a1a' }} />
                      <YAxis type="category" dataKey="name" width={140} tick={{ fontSize: 9, fill: '#666' }} axisLine={{ stroke: '#1a1a1a' }} />
                      <Tooltip contentStyle={{ background: '#0a0a0a', border: '1px solid #1a1a1a', borderRadius: 6, fontSize: 11, color: '#fff' }}
                        formatter={v => [`${v}%`, 'Importance']} />
                      <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                        {Object.entries(btResult.feature_importance).slice(0, 15).map((_, i) => (
                          <Cell key={i} fill={i < 3 ? '#00ff87' : i < 7 ? '#ffffff' : '#444'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}

              {btResult.trades?.length > 0 && (
                <div className="bg-card border border-border rounded-lg p-5">
                  <h3 className="text-[10px] uppercase tracking-widest text-text-secondary mb-3">Trade Log</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="text-text-secondary border-b border-border text-[10px] uppercase tracking-widest">
                          <th className="text-left py-2 px-2">#</th>
                          <th className="text-left py-2 px-2">Ticker</th>
                          <th className="text-left py-2 px-2">Side</th>
                          <th className="text-right py-2 px-2">Entry</th>
                          <th className="text-right py-2 px-2">Exit</th>
                          <th className="text-right py-2 px-2">P&L</th>
                          <th className="text-right py-2 px-2">Edge</th>
                        </tr>
                      </thead>
                      <tbody>
                        {btResult.trades.map((t, i) => (
                          <tr key={i} className="border-b border-border/50 hover:bg-surface/50">
                            <td className="py-1.5 px-2 text-text-secondary font-mono">{i + 1}</td>
                            <td className="py-1.5 px-2 font-mono">{t.ticker?.slice(0, 20)}</td>
                            <td className="py-1.5 px-2 font-mono">{t.side}</td>
                            <td className="py-1.5 px-2 text-right font-mono">{t.entry_price?.toFixed(2)}</td>
                            <td className="py-1.5 px-2 text-right font-mono">{t.exit_price?.toFixed(2)}</td>
                            <td className={`py-1.5 px-2 text-right font-mono ${t.pnl_cents > 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                              ${(t.pnl_cents / 100).toFixed(2)}
                            </td>
                            <td className="py-1.5 px-2 text-right font-mono">{((t.model_probability - t.market_probability_at_entry) * 100).toFixed(1)}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </>
          )}

          {btResult?.config?.error && (
            <div className="bg-accent-yellow/5 border border-accent-yellow/20 rounded-lg p-4 text-accent-yellow text-xs">
              {btResult.config.error}
            </div>
          )}

          {sweepResult && (
            <div className="bg-card border border-border rounded-lg p-5">
              <h3 className="text-[10px] uppercase tracking-widest text-text-secondary mb-3">
                Parameter Sweep — {sweepResult.total_combinations} combos, ranked by Sharpe
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-text-secondary border-b border-border text-[10px] uppercase tracking-widest">
                      <th className="text-left py-2 px-2">Rank</th>
                      <th className="text-right py-2 px-2">Entry</th>
                      <th className="text-right py-2 px-2">Conf</th>
                      <th className="text-right py-2 px-2">Sharpe</th>
                      <th className="text-right py-2 px-2">Win%</th>
                      <th className="text-right py-2 px-2">P&L</th>
                      <th className="text-right py-2 px-2">Trades</th>
                      <th className="text-right py-2 px-2">PF</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sweepResult.results.map((r, i) => (
                      <tr key={i} className={`border-b border-border/50 ${i === 0 ? 'bg-accent-green/5' : 'hover:bg-surface/50'}`}>
                        <td className="py-1.5 px-2 font-mono">{i + 1}</td>
                        <td className="py-1.5 px-2 text-right font-mono">{r.config?.entry_threshold}</td>
                        <td className="py-1.5 px-2 text-right font-mono">{r.config?.min_confidence}</td>
                        <td className={`py-1.5 px-2 text-right font-mono font-bold ${r.sharpe_ratio >= 2 ? 'text-accent-green' : r.sharpe_ratio >= 1 ? 'text-accent-yellow' : 'text-accent-red'}`}>
                          {r.sharpe_ratio}
                        </td>
                        <td className="py-1.5 px-2 text-right font-mono">{(r.win_rate * 100).toFixed(1)}%</td>
                        <td className={`py-1.5 px-2 text-right font-mono ${r.total_pnl_cents > 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                          ${(r.total_pnl_cents / 100).toFixed(2)}
                        </td>
                        <td className="py-1.5 px-2 text-right font-mono">{r.total_trades}</td>
                        <td className="py-1.5 px-2 text-right font-mono">{r.profit_factor}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}

      {/* ── SHADOW TRADING ── */}
      {mode === 'shadow' && (
        <>
          <div className="bg-card border border-border rounded-lg p-5">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="text-sm font-semibold uppercase tracking-widest text-text-secondary">Shadow Trading</h2>
                <p className="text-xs text-text-secondary mt-1">
                  Records every trade the bot would make using live data — no real money at risk.
                  Enable auto-scan to run continuously.
                </p>
              </div>
              {autoScan && (
                <span className="flex items-center gap-2 text-xs text-accent-green">
                  <span className="w-2 h-2 rounded-full bg-accent-green pulse-dot" />
                  LIVE
                </span>
              )}
            </div>
            <div className="flex items-end gap-3 flex-wrap">
              <div>
                <label className="text-[10px] uppercase tracking-widest text-text-secondary block mb-1">Balance ($)</label>
                <input type="number" value={paperBalance / 100}
                  onChange={e => setPaperBalance(Math.round(e.target.value * 100))}
                  className="w-28 bg-surface border border-border rounded-lg px-3 py-2 text-sm font-mono text-white focus:border-accent-green focus:outline-none" />
              </div>
              <button onClick={initPaper} disabled={loading}
                className="px-4 py-2 bg-card border border-border text-white text-xs font-semibold rounded-lg hover:bg-surface disabled:opacity-30 transition-all uppercase tracking-wide">
                Reset
              </button>
              <button onClick={trainPaper} disabled={loading}
                className="px-4 py-2 bg-accent-yellow/10 border border-accent-yellow/20 text-accent-yellow text-xs font-semibold rounded-lg hover:bg-accent-yellow/20 disabled:opacity-30 transition-all uppercase tracking-wide">
                {loading ? 'Training...' : 'Train Model'}
              </button>
              <button onClick={paperScan} disabled={loading}
                className="px-4 py-2 bg-white text-black text-xs font-semibold rounded-lg hover:bg-gray-200 disabled:opacity-30 transition-all uppercase tracking-wide">
                {loading ? 'Scanning...' : 'Scan Once'}
              </button>
              <button
                onClick={() => setAutoScan(a => !a)}
                className={`px-4 py-2 text-xs font-semibold rounded-lg transition-all uppercase tracking-wide ${
                  autoScan
                    ? 'bg-accent-red/10 border border-accent-red/20 text-accent-red hover:bg-accent-red/20'
                    : 'bg-accent-green/10 border border-accent-green/20 text-accent-green hover:bg-accent-green/20'
                }`}
              >
                {autoScan ? 'Stop Auto' : 'Auto Scan (60s)'}
              </button>
            </div>
          </div>

          {paperState && (
            <>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
                <StatCard label="Balance" value={`$${(paperState.balance_cents / 100).toFixed(2)}`} color="text-accent-green" />
                <StatCard label="Positions" value={paperState.open_positions?.length ?? 0} />
                <StatCard label="Trades" value={paperState.metrics?.total_trades ?? 0}
                  sub={paperState.metrics?.total_trades > 0 ? `${paperState.metrics.wins}W / ${paperState.metrics.losses}L` : null} />
                <StatCard label="Sharpe" value={paperState.metrics?.sharpe_ratio ?? '--'}
                  sub={paperState.metrics?.sharpe_label}
                  color={(paperState.metrics?.sharpe_ratio ?? 0) >= 2 ? 'text-accent-green' : 'text-white'} />
                <StatCard label="P&L"
                  value={paperState.metrics?.total_pnl_cents != null ? `$${(paperState.metrics.total_pnl_cents / 100).toFixed(2)}` : '--'}
                  color={(paperState.metrics?.total_pnl_cents ?? 0) > 0 ? 'text-accent-green' : 'text-accent-red'} />
                <StatCard label="Model"
                  value={paperState.model_trained ? 'Trained' : 'Heuristic'}
                  color={paperState.model_trained ? 'text-accent-green' : 'text-accent-yellow'}
                  sub={`${paperState.total_scans} scans`} />
              </div>

              {paperState.open_positions?.length > 0 && (
                <div className="bg-card border border-border rounded-lg p-5">
                  <h3 className="text-[10px] uppercase tracking-widest text-text-secondary mb-3">Open Shadow Positions</h3>
                  <div className="space-y-2">
                    {paperState.open_positions.map((p, i) => (
                      <div key={i} className="flex items-center justify-between bg-surface border border-border rounded-lg p-3 text-xs card-hover">
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-white">{p.ticker}</span>
                          <span className={`text-[10px] px-1.5 py-0.5 rounded uppercase font-semibold ${p.side === 'yes' ? 'bg-accent-green/10 text-accent-green' : 'bg-accent-red/10 text-accent-red'}`}>
                            {p.side}
                          </span>
                        </div>
                        <div className="text-text-secondary font-mono">
                          {p.contracts}x @ {(p.entry_price * 100).toFixed(0)}c
                          <span className="ml-3">Model: {(p.model_prob * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {paperState.equity_curve?.length > 0 && (
                <div className="bg-card border border-border rounded-lg p-5">
                  <h3 className="text-[10px] uppercase tracking-widest text-text-secondary mb-4">Shadow Equity</h3>
                  <ResponsiveContainer width="100%" height={220}>
                    <LineChart data={paperState.equity_curve}>
                      <XAxis dataKey="trade_num" tick={{ fontSize: 10, fill: '#666' }} axisLine={{ stroke: '#1a1a1a' }} />
                      <YAxis tick={{ fontSize: 10, fill: '#666' }} tickFormatter={v => `$${(v / 100).toFixed(0)}`} axisLine={{ stroke: '#1a1a1a' }} />
                      <Tooltip contentStyle={{ background: '#0a0a0a', border: '1px solid #1a1a1a', borderRadius: 6, fontSize: 11, color: '#fff' }}
                        formatter={v => [`$${(v / 100).toFixed(2)}`, 'Equity']} />
                      <Line type="monotone" dataKey="equity_cents" stroke="#00ff87" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}

              {paperState.trades?.length > 0 && (
                <div className="bg-card border border-border rounded-lg p-5">
                  <h3 className="text-[10px] uppercase tracking-widest text-text-secondary mb-3">Shadow Trade Log</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="text-text-secondary border-b border-border text-[10px] uppercase tracking-widest">
                          <th className="text-left py-2 px-2">Ticker</th>
                          <th className="text-left py-2 px-2">Side</th>
                          <th className="text-right py-2 px-2">Entry</th>
                          <th className="text-right py-2 px-2">Exit</th>
                          <th className="text-right py-2 px-2">P&L</th>
                        </tr>
                      </thead>
                      <tbody>
                        {paperState.trades.map((t, i) => (
                          <tr key={i} className="border-b border-border/50 hover:bg-surface/50">
                            <td className="py-1.5 px-2 font-mono">{t.ticker?.slice(0, 25)}</td>
                            <td className="py-1.5 px-2 font-mono">{t.side}</td>
                            <td className="py-1.5 px-2 text-right font-mono">{t.entry_price?.toFixed(2)}</td>
                            <td className="py-1.5 px-2 text-right font-mono">{t.exit_price?.toFixed(2)}</td>
                            <td className={`py-1.5 px-2 text-right font-mono ${t.pnl_cents > 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                              ${(t.pnl_cents / 100).toFixed(2)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </>
          )}

          {scanHistory.length > 0 && (
            <div className="bg-card border border-border rounded-lg p-5">
              <h3 className="text-[10px] uppercase tracking-widest text-text-secondary mb-3">Scan Log</h3>
              <div className="space-y-1">
                {scanHistory.map((s, i) => (
                  <div key={i} className="bg-surface border border-border rounded-lg p-2.5 text-xs flex items-center justify-between font-mono card-hover">
                    <span className="text-text-secondary">#{s.scan_number}</span>
                    <div className="flex items-center gap-4">
                      <span className="text-accent-green">+{s.entries?.length ?? 0}</span>
                      <span className="text-accent-red">-{s.exits?.length ?? 0}</span>
                      <span className="text-text-secondary">{s.open_positions} open</span>
                      <span className="text-white">${(s.balance_cents / 100).toFixed(2)}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
