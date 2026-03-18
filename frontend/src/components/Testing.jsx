/**
 * Testing Component — Backtester and Shadow Trading
 *
 * Provides two modes for evaluating the prediction market bot's strategy
 * without risking real money, toggled via top-level tabs.
 *
 * Mode 1: Backtester
 *   - Configuration panel: initial balance, entry threshold, exit threshold,
 *     minimum confidence, and max markets.
 *   - "Run Backtest" executes a single backtest with the configured parameters.
 *   - "Parameter Sweep" runs multiple backtests across parameter combinations,
 *     ranked by Sharpe ratio.
 *   - Results display: win rate, Sharpe ratio, P&L, profit factor, CV accuracy,
 *     OOB score, max drawdown, train/test split, signals generated/filtered,
 *     average edge, and average log return.
 *   - Equity Curve line chart for the backtest run.
 *   - Feature Importance horizontal bar chart (top 15 features).
 *   - Trade Log table: ticker, side, entry/exit prices, P&L, and edge per trade.
 *   - Sweep results table: ranked parameter combos with Sharpe, win%, P&L,
 *     trade count, and profit factor.
 *
 * Mode 2: Shadow Trading
 *   - Simulates live trading using real market data but with virtual funds.
 *   - Add Demo Funds: preset buttons ($100-$10,000) or custom amount input.
 *   - Controls: Reset All (re-initialize), Train Model (incremental training),
 *     Scan Once (single market scan), Auto Scan (60-second polling interval).
 *   - State cards: balance, open positions count, total trades (W/L), Sharpe,
 *     P&L, and model status (trained vs heuristic with sample count).
 *   - Open Shadow Positions list: ticker, side, contracts, entry price,
 *     model probability.
 *   - Shadow Equity line chart over trade sequence.
 *   - Shadow Trade Log table: ticker, side, entry/exit, P&L.
 *   - Scan Log: reverse-chronological feed of scan results showing entries,
 *     exits, open positions, and balance after each scan.
 *
 * API endpoints called:
 *   - api.runBacktest(config)        — run a single backtest
 *   - api.runSweep({ max_markets })  — run parameter sweep
 *   - api.getPaperState()            — fetch current shadow trading state
 *   - api.configurePaper(balance)    — reset/initialize shadow trading
 *   - api.addPaperFunds(amountCents) — add virtual funds to shadow balance
 *   - api.paperTrain()               — train/retrain the ML model on paper data
 *   - api.paperScan()                — execute one market scan cycle
 *
 * Data displayed:
 *   - Backtest results: win_rate, sharpe_ratio, total_pnl_cents, profit_factor,
 *     cv_accuracy, oob_score, max_drawdown_cents, equity_curve, trades,
 *     feature_importance, train/test sample counts, signals generated/filtered
 *   - Sweep results[]: config (entry_threshold, min_confidence), sharpe_ratio,
 *     win_rate, total_pnl_cents, total_trades, profit_factor
 *   - Paper state: balance_cents, open_positions, metrics, model_trained,
 *     total_scans, training_samples_count, equity_curve, trades
 */
import { useState, useEffect, useRef } from 'react';
import { api } from '../api';
import { LineChart, Line, XAxis, YAxis, Tooltip as RechartsTooltip, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts';
import Tooltip from './Tooltip';

const CHART_TOOLTIP_STYLE = { background: '#0c0c0e', border: '1px solid #1e1e22', borderRadius: 8, fontSize: 11, color: '#fff' };

function StatCard({ label, value, sub, color = 'text-text-primary', accentColor, tooltip }) {
  const inner = (
    <div className="stat-card" style={accentColor ? { '--accent-color': accentColor } : undefined}>
      <div className="text-[10px] uppercase tracking-widest text-text-secondary mb-2">{label}</div>
      <div className={`text-2xl font-bold font-mono ${color}`}>{value}</div>
      {sub && <div className="text-[10px] text-text-secondary mt-1.5">{sub}</div>}
    </div>
  );
  if (tooltip) {
    return <Tooltip text={tooltip}>{inner}</Tooltip>;
  }
  return inner;
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
  const [customFunds, setCustomFunds] = useState('');
  const [fundingMsg, setFundingMsg] = useState(null);
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

  const addFunds = async (amountCents) => {
    try {
      const result = await api.addPaperFunds(amountCents);
      setPaperState(prev => prev ? { ...prev, balance_cents: result.balance_cents } : prev);
      setFundingMsg(`+$${(amountCents / 100).toFixed(0)} added`);
      setTimeout(() => setFundingMsg(null), 2000);
    } catch (e) { setError(`Failed to add funds: ${e.message}`); }
  };

  const trainPaper = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await api.paperTrain();
      setPaperState(prev => prev ? { ...prev, model_trained: true, training_samples_count: result.total_cumulative_samples } : prev);
      alert(`Trained on ${result.total_cumulative_samples || result.samples} cumulative samples (${result.new_samples_added ?? '?'} new). CV: ${(result.cv_accuracy * 100).toFixed(1)}%`);
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

  // Tooltip text constants
  const TIPS = {
    cvAccuracy: 'Cross-Validation Accuracy \u2014 model performance on held-out folds during training',
    oob: 'Out-of-Bag Score \u2014 Random Forest validation using samples not in each tree\u2019s bootstrap',
    sharpe: 'Sharpe Ratio \u2014 risk-adjusted return (excess return / volatility)',
    profitFactor: 'Profit Factor \u2014 gross profit / gross loss',
    maxDrawdown: 'Largest peak-to-trough equity decline',
    pf: 'Profit Factor \u2014 gross profit / gross loss',
    entryThreshold: 'Market price must be \u2264 model probability \u00d7 this value to trigger a buy',
    exitThreshold: 'Market price must be \u2265 model probability \u00d7 this value to trigger a sell',
    minConfidence: 'Minimum model confidence required to act on a signal',
  };

  const configFields = [
    { label: 'Balance ($)', key: 'initial_balance_cents', div: 100 },
    { label: 'Entry Threshold', key: 'entry_threshold', step: 0.05, tip: TIPS.entryThreshold },
    { label: 'Exit Threshold', key: 'exit_threshold', step: 0.05, tip: TIPS.exitThreshold },
    { label: 'Min Confidence', key: 'min_confidence', step: 0.05, tip: TIPS.minConfidence },
    { label: 'Max Markets', key: 'max_markets' },
  ];

  return (
    <div className="space-y-6">
      {/* Mode tabs — segmented pill control */}
      <div className="inline-flex items-center bg-card border border-border rounded-full p-1 gap-0.5">
        {MODES.map(m => (
          <button
            key={m.id}
            onClick={() => setMode(m.id)}
            className={`px-6 py-2 rounded-full text-xs font-semibold tracking-wide uppercase transition-all duration-200 ${
              mode === m.id
                ? 'bg-accent-green text-black shadow-sm'
                : 'text-text-secondary hover:text-text-primary'
            }`}
          >
            {m.label}
          </button>
        ))}
      </div>

      {error && (
        <div className="card bg-accent-red/5 border-accent-red/20 p-3 text-xs text-accent-red">
          {error}
        </div>
      )}

      {/* ── BACKTESTER ── */}
      {mode === 'backtest' && (
        <>
          <div className="card p-5 mb-0">
            <h2 className="section-title mb-2">How the Backtester Works</h2>
            <div className="text-xs text-text-secondary space-y-1.5 mb-4">
              <p>The backtester simulates your trading strategy against <strong className="text-text-primary">real historical Kalshi market data</strong> to evaluate performance before risking real money.</p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mt-2">
                {[
                  { step: '1', title: 'Fetch Data', desc: 'Pulls up to 1,000 settled markets from Kalshi with known outcomes' },
                  { step: '2', title: 'Train/Test Split', desc: 'Splits data 70/30 — trains the RF+GB ensemble on 70%, tests on 30%' },
                  { step: '3', title: 'Simulate Trades', desc: 'Walks through test markets chronologically, entering when edge > threshold and exiting at target or expiry' },
                  { step: '4', title: 'Calculate Metrics', desc: 'Computes win rate, Sharpe, P&L, drawdown, and equity curve from simulated trades' },
                ].map(s => (
                  <div key={s.step} className="bg-surface-2 border border-border-subtle rounded-lg p-2.5 flex items-start gap-2">
                    <span className="flex items-center justify-center w-5 h-5 rounded-full bg-accent-green/15 text-accent-green text-[10px] font-bold shrink-0">{s.step}</span>
                    <div>
                      <div className="text-text-primary font-semibold text-[11px]">{s.title}</div>
                      <div className="text-[10px] text-text-muted">{s.desc}</div>
                    </div>
                  </div>
                ))}
              </div>
              <p className="text-[10px] text-text-muted mt-2">
                <strong className="text-accent-yellow">Parameter Sweep</strong> runs multiple backtests across entry threshold and confidence combinations, ranking results by Sharpe ratio to find the optimal configuration.
              </p>
            </div>
          </div>

          <div className="card p-5">
            <h2 className="section-title mb-4">Configuration</h2>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              {configFields.map(f => (
                <div key={f.key}>
                  <label className="text-[10px] uppercase tracking-widest text-text-secondary block mb-1">
                    {f.tip ? <Tooltip text={f.tip}>{f.label}</Tooltip> : f.label}
                  </label>
                  <input
                    type="number"
                    step={f.step || 1}
                    value={f.div ? btConfig[f.key] / f.div : btConfig[f.key]}
                    onChange={e => setBtConfig(c => ({ ...c, [f.key]: f.div ? Math.round(parseFloat(e.target.value) * f.div) : parseFloat(e.target.value) }))}
                    className="input"
                  />
                </div>
              ))}
            </div>
            <div className="flex gap-3 mt-4">
              <button onClick={runBacktest} disabled={loading} className="btn-primary uppercase">
                {loading ? 'Running...' : 'Run Backtest'}
              </button>
              <button onClick={runSweep} disabled={loading} className="btn-secondary uppercase">
                {loading ? 'Running...' : 'Parameter Sweep'}
              </button>
            </div>
          </div>

          {btResult && !btResult.config?.error && (
            <>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
                <StatCard label="Win Rate" value={`${(btResult.win_rate * 100).toFixed(1)}%`}
                  tooltip="Percentage of backtest trades that were profitable"
                  sub={`${btResult.wins}W / ${btResult.losses}L`}
                  accentColor={btResult.win_rate >= 0.6 ? 'rgb(var(--color-green))' : 'rgb(var(--color-red))'}
                  color={btResult.win_rate >= 0.6 ? 'text-accent-green' : 'text-accent-red'} />
                <StatCard label="Sharpe" value={btResult.sharpe_ratio} sub={btResult.sharpe_label}
                  tooltip={TIPS.sharpe}
                  accentColor={btResult.sharpe_ratio >= 2 ? 'rgb(var(--color-green))' : btResult.sharpe_ratio >= 1 ? 'rgb(var(--color-yellow))' : 'rgb(var(--color-red))'}
                  color={btResult.sharpe_ratio >= 2 ? 'text-accent-green' : btResult.sharpe_ratio >= 1 ? 'text-accent-yellow' : 'text-accent-red'} />
                <StatCard label="P&L" value={`$${(btResult.total_pnl_cents / 100).toFixed(2)}`}
                  tooltip="Total simulated profit & loss from the backtest run"
                  accentColor={btResult.total_pnl_cents > 0 ? 'rgb(var(--color-green))' : 'rgb(var(--color-red))'}
                  color={btResult.total_pnl_cents > 0 ? 'text-accent-green' : 'text-accent-red'} />
                <StatCard label="Profit Factor" value={btResult.profit_factor}
                  tooltip={TIPS.profitFactor}
                  accentColor={btResult.profit_factor >= 1.5 ? 'rgb(var(--color-green))' : 'rgb(var(--color-text-muted))'}
                  color={btResult.profit_factor >= 1.5 ? 'text-accent-green' : 'text-text-primary'} />
                <StatCard label="CV Accuracy" value={`${(btResult.cv_accuracy * 100).toFixed(1)}%`}
                  tooltip={TIPS.cvAccuracy}
                  accentColor="var(--color-blue)"
                  sub={<><Tooltip text={TIPS.oob}>OOB</Tooltip>{`: ${(btResult.oob_score * 100).toFixed(1)}%`}</>} />
                <StatCard label="Max Drawdown" value={`$${(btResult.max_drawdown_cents / 100).toFixed(2)}`}
                  tooltip={TIPS.maxDrawdown}
                  accentColor="var(--color-red)"
                  color="text-accent-red" />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
                <div className="card p-3">
                  <span className="text-text-secondary">Train / Test:</span>{' '}
                  <span className="font-mono">{btResult.train_samples} / {btResult.test_samples}</span>
                </div>
                <div className="card p-3">
                  <span className="text-text-secondary">Signals:</span>{' '}
                  <span className="font-mono text-accent-green">{btResult.signals_generated}</span>
                  <span className="text-text-secondary ml-2">Filtered:</span>{' '}
                  <span className="font-mono text-accent-red">{btResult.signals_filtered}</span>
                </div>
                <div className="card p-3">
                  <span className="text-text-secondary">Avg Edge:</span>{' '}
                  <span className="font-mono">{(btResult.avg_edge * 100).toFixed(1)}%</span>
                  <span className="text-text-secondary ml-2">Log Return:</span>{' '}
                  <span className="font-mono">{btResult.avg_log_return?.toFixed(4)}</span>
                </div>
              </div>

              {btResult.equity_curve?.length > 0 && (
                <div className="card p-5">
                  <h3 className="section-title mb-4">Equity Curve</h3>
                  <ResponsiveContainer width="100%" height={250}>
                    <LineChart data={btResult.equity_curve}>
                      <XAxis dataKey="trade_num" tick={{ fontSize: 10, fill: '#71717a' }} axisLine={{ stroke: '#1e1e22' }} />
                      <YAxis tick={{ fontSize: 10, fill: '#71717a' }} tickFormatter={v => `$${(v / 100).toFixed(0)}`} axisLine={{ stroke: '#1e1e22' }} />
                      <RechartsTooltip contentStyle={CHART_TOOLTIP_STYLE}
                        formatter={v => [`$${(v / 100).toFixed(2)}`, 'Equity']} />
                      <Line type="monotone" dataKey="equity_cents" stroke="rgb(var(--color-green))" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}

              {btResult.feature_importance && Object.keys(btResult.feature_importance).length > 0 && (
                <div className="card p-5">
                  <h3 className="section-title mb-4">Feature Importance (Top 15)</h3>
                  <ResponsiveContainer width="100%" height={320}>
                    <BarChart data={Object.entries(btResult.feature_importance).slice(0, 15).map(([name, imp]) => ({
                      name: name.replace(/_/g, ' '), importance: +(imp * 100).toFixed(1),
                    }))} layout="vertical">
                      <XAxis type="number" tick={{ fontSize: 9, fill: '#71717a' }} axisLine={{ stroke: '#1e1e22' }} />
                      <YAxis type="category" dataKey="name" width={140} tick={{ fontSize: 9, fill: '#71717a' }} axisLine={{ stroke: '#1e1e22' }} />
                      <RechartsTooltip contentStyle={CHART_TOOLTIP_STYLE}
                        formatter={v => [`${v}%`, 'Importance']} />
                      <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                        {Object.entries(btResult.feature_importance).slice(0, 15).map((_, i) => (
                          <Cell key={i} fill={i < 3 ? 'rgb(var(--color-green))' : i < 7 ? 'rgb(var(--color-text-secondary))' : 'rgb(var(--color-text-muted))'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}

              {btResult.trades?.length > 0 && (
                <div className="card p-5">
                  <h3 className="section-title mb-3">Trade Log</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="table-header border-b border-border">
                          <th className="table-cell text-left">#</th>
                          <th className="table-cell text-left">Ticker</th>
                          <th className="table-cell text-left">Side</th>
                          <th className="table-cell text-right">Entry</th>
                          <th className="table-cell text-right">Exit</th>
                          <th className="table-cell text-right">P&L</th>
                          <th className="table-cell text-right">Edge</th>
                        </tr>
                      </thead>
                      <tbody>
                        {btResult.trades.map((t, i) => (
                          <tr key={i} className={`table-row ${t.pnl_cents > 0 ? 'row-win' : 'row-loss'}`}>
                            <td className="table-cell text-text-secondary font-mono">{i + 1}</td>
                            <td className="table-cell font-mono">{t.ticker?.slice(0, 20)}</td>
                            <td className="table-cell font-mono">{t.side}</td>
                            <td className="table-cell text-right font-mono">{t.entry_price?.toFixed(2)}</td>
                            <td className="table-cell text-right font-mono">{t.exit_price?.toFixed(2)}</td>
                            <td className={`table-cell text-right font-mono ${t.pnl_cents > 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                              ${(t.pnl_cents / 100).toFixed(2)}
                            </td>
                            <td className="table-cell text-right font-mono">{((t.model_probability - t.market_probability_at_entry) * 100).toFixed(1)}%</td>
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
            <div className="card bg-accent-yellow/5 border-accent-yellow/20 p-4 text-accent-yellow text-xs">
              {btResult.config.error}
            </div>
          )}

          {sweepResult && (
            <div className="card p-5">
              <h3 className="section-title mb-3">
                Parameter Sweep — {sweepResult.total_combinations} combos, ranked by Sharpe
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="table-header border-b border-border">
                      <th className="table-cell text-left">Rank</th>
                      <th className="table-cell text-right"><Tooltip text={TIPS.entryThreshold}>Entry</Tooltip></th>
                      <th className="table-cell text-right"><Tooltip text={TIPS.minConfidence}>Conf</Tooltip></th>
                      <th className="table-cell text-right"><Tooltip text={TIPS.sharpe}>Sharpe</Tooltip></th>
                      <th className="table-cell text-right">Win%</th>
                      <th className="table-cell text-right">P&L</th>
                      <th className="table-cell text-right">Trades</th>
                      <th className="table-cell text-right"><Tooltip text={TIPS.pf}>PF</Tooltip></th>
                    </tr>
                  </thead>
                  <tbody>
                    {sweepResult.results.map((r, i) => (
                      <tr key={i} className={`table-row ${i === 0 ? 'row-win' : ''}`}>
                        <td className="table-cell font-mono">{i + 1}</td>
                        <td className="table-cell text-right font-mono">{r.config?.entry_threshold}</td>
                        <td className="table-cell text-right font-mono">{r.config?.min_confidence}</td>
                        <td className={`table-cell text-right font-mono font-bold ${r.sharpe_ratio >= 2 ? 'text-accent-green' : r.sharpe_ratio >= 1 ? 'text-accent-yellow' : 'text-accent-red'}`}>
                          {r.sharpe_ratio}
                        </td>
                        <td className="table-cell text-right font-mono">{(r.win_rate * 100).toFixed(1)}%</td>
                        <td className={`table-cell text-right font-mono ${r.total_pnl_cents > 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                          ${(r.total_pnl_cents / 100).toFixed(2)}
                        </td>
                        <td className="table-cell text-right font-mono">{r.total_trades}</td>
                        <td className="table-cell text-right font-mono">{r.profit_factor}</td>
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
          <div className="card p-5">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h2 className="section-title">Shadow Trading</h2>
                <p className="text-xs text-text-secondary mt-1">
                  Records every trade the bot would make using live data — no real money at risk.
                  Enable auto-scan to run continuously.
                </p>
              </div>
              {autoScan && (
                <span className="flex items-center gap-2 text-xs text-accent-green font-semibold">
                  <span className="w-2 h-2 rounded-full bg-accent-green pulse-dot" />
                  LIVE
                </span>
              )}
            </div>

            {/* Add Demo Funds */}
            <div className="card p-4 mb-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-xs font-semibold uppercase tracking-widest text-accent-green">Add Demo Funds</h3>
                {fundingMsg && (
                  <span className="badge-green animate-pulse">{fundingMsg}</span>
                )}
              </div>
              <div className="flex items-center gap-2 flex-wrap">
                {[
                  { label: '$100', cents: 10000 },
                  { label: '$500', cents: 50000 },
                  { label: '$1,000', cents: 100000 },
                  { label: '$5,000', cents: 500000 },
                  { label: '$10,000', cents: 1000000 },
                ].map(preset => (
                  <button
                    key={preset.cents}
                    onClick={() => addFunds(preset.cents)}
                    className="px-4 py-2 bg-accent-green/10 border border-accent-green/20 text-accent-green text-xs font-semibold rounded-lg
                               hover:bg-accent-green/20 hover:border-accent-green/40 hover:shadow-[0_0_12px_rgba(52,211,153,0.15)]
                               active:scale-[0.97] transition-all duration-150"
                  >
                    {preset.label}
                  </button>
                ))}
                <div className="flex items-center gap-1.5 ml-2">
                  <span className="text-text-secondary text-xs">$</span>
                  <input
                    type="number"
                    placeholder="Custom"
                    value={customFunds}
                    onChange={e => setCustomFunds(e.target.value)}
                    className="input !w-24 text-xs"
                  />
                  <button
                    onClick={() => {
                      const amt = Math.round(parseFloat(customFunds) * 100);
                      if (amt > 0) { addFunds(amt); setCustomFunds(''); }
                    }}
                    disabled={!customFunds || parseFloat(customFunds) <= 0}
                    className="btn-primary"
                  >
                    Add
                  </button>
                </div>
              </div>
            </div>

            <div className="flex items-end gap-3 flex-wrap">
              <div>
                <label className="text-[10px] uppercase tracking-widest text-text-secondary block mb-1">Reset Balance ($)</label>
                <input type="number" value={paperBalance / 100}
                  onChange={e => setPaperBalance(Math.round(e.target.value * 100))}
                  className="input !w-28" />
              </div>
              <button onClick={initPaper} disabled={loading} className="btn-secondary uppercase">
                Reset All
              </button>
              <button onClick={trainPaper} disabled={loading}
                className="px-4 py-2 bg-accent-yellow/10 border border-accent-yellow/20 text-accent-yellow text-xs font-semibold rounded-lg
                           hover:bg-accent-yellow/20 hover:border-accent-yellow/40 disabled:opacity-40 disabled:cursor-not-allowed transition-all duration-150 uppercase tracking-wide">
                {loading ? 'Training...' : 'Train Model'}
              </button>
              <button onClick={paperScan} disabled={loading} className="btn-primary uppercase">
                {loading ? 'Scanning...' : 'Scan Once'}
              </button>
              <button
                onClick={() => setAutoScan(a => !a)}
                className={`px-4 py-2 text-xs font-semibold rounded-lg transition-all duration-150 uppercase tracking-wide ${
                  autoScan
                    ? 'btn-danger'
                    : 'bg-accent-green/10 border border-accent-green/20 text-accent-green hover:bg-accent-green/20 hover:border-accent-green/40 glow-pulse'
                }`}
              >
                {autoScan ? 'Stop Auto' : 'Auto Scan (60s)'}
              </button>
            </div>
          </div>

          {paperState && (
            <>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
                <StatCard label="Balance" value={`$${(paperState.balance_cents / 100).toFixed(2)}`}
                  tooltip="Virtual cash balance for shadow trading — not real money"
                  accentColor="var(--color-green)" color="text-accent-green" />
                <StatCard label="Positions" value={paperState.open_positions?.length ?? 0}
                  tooltip="Number of open shadow positions the bot is currently holding"
                  accentColor="var(--color-blue)" />
                <StatCard label="Trades" value={paperState.metrics?.total_trades ?? 0}
                  tooltip="Total completed shadow trades (entries that have been exited)"
                  accentColor="var(--color-purple)"
                  sub={paperState.metrics?.total_trades > 0 ? `${paperState.metrics.wins}W / ${paperState.metrics.losses}L` : null} />
                <StatCard label="Sharpe" value={paperState.metrics?.sharpe_ratio ?? '--'}
                  tooltip={TIPS.sharpe}
                  sub={paperState.metrics?.sharpe_label}
                  accentColor={(paperState.metrics?.sharpe_ratio ?? 0) >= 2 ? 'rgb(var(--color-green))' : 'rgb(var(--color-text-muted))'}
                  color={(paperState.metrics?.sharpe_ratio ?? 0) >= 2 ? 'text-accent-green' : 'text-text-primary'} />
                <StatCard label="P&L"
                  tooltip="Total profit & loss across all completed shadow trades"
                  value={paperState.metrics?.total_pnl_cents != null ? `$${(paperState.metrics.total_pnl_cents / 100).toFixed(2)}` : '--'}
                  accentColor={(paperState.metrics?.total_pnl_cents ?? 0) > 0 ? 'rgb(var(--color-green))' : 'rgb(var(--color-red))'}
                  color={(paperState.metrics?.total_pnl_cents ?? 0) > 0 ? 'text-accent-green' : 'text-accent-red'} />
                <StatCard label="Model"
                  tooltip="Trained = RF model active with learned patterns. Heuristic = using rule-based signals only"
                  value={paperState.model_trained ? 'Trained' : 'Heuristic'}
                  accentColor={paperState.model_trained ? 'rgb(var(--color-green))' : 'rgb(var(--color-yellow))'}
                  color={paperState.model_trained ? 'text-accent-green' : 'text-accent-yellow'}
                  sub={`${paperState.total_scans} scans${paperState.training_samples_count ? ` \u00b7 ${paperState.training_samples_count} samples` : ''}`} />
              </div>

              {paperState.open_positions?.length > 0 && (
                <div className="card p-5">
                  <h3 className="section-title mb-3">Open Shadow Positions</h3>
                  <div className="space-y-2">
                    {paperState.open_positions.map((p, i) => (
                      <div key={i} className="card-interactive flex items-center justify-between p-3 text-xs">
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-text-primary">{p.ticker}</span>
                          <span className={`badge ${p.side === 'yes' ? 'badge-green' : 'badge-red'}`}>
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
                <div className="card p-5">
                  <h3 className="section-title mb-4">Shadow Equity</h3>
                  <ResponsiveContainer width="100%" height={220}>
                    <LineChart data={paperState.equity_curve}>
                      <XAxis dataKey="trade_num" tick={{ fontSize: 10, fill: '#71717a' }} axisLine={{ stroke: '#1e1e22' }} />
                      <YAxis tick={{ fontSize: 10, fill: '#71717a' }} tickFormatter={v => `$${(v / 100).toFixed(0)}`} axisLine={{ stroke: '#1e1e22' }} />
                      <RechartsTooltip contentStyle={CHART_TOOLTIP_STYLE}
                        formatter={v => [`$${(v / 100).toFixed(2)}`, 'Equity']} />
                      <Line type="monotone" dataKey="equity_cents" stroke="rgb(var(--color-green))" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}

              {paperState.trades?.length > 0 && (
                <div className="card p-5">
                  <h3 className="section-title mb-3">Shadow Trade Log</h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="table-header border-b border-border">
                          <th className="table-cell text-left">Ticker</th>
                          <th className="table-cell text-left">Side</th>
                          <th className="table-cell text-right">Entry</th>
                          <th className="table-cell text-right">Exit</th>
                          <th className="table-cell text-right">P&L</th>
                        </tr>
                      </thead>
                      <tbody>
                        {paperState.trades.map((t, i) => (
                          <tr key={i} className={`table-row ${t.pnl_cents > 0 ? 'row-win' : 'row-loss'}`}>
                            <td className="table-cell font-mono">{t.ticker?.slice(0, 25)}</td>
                            <td className="table-cell font-mono">{t.side}</td>
                            <td className="table-cell text-right font-mono">{t.entry_price?.toFixed(2)}</td>
                            <td className="table-cell text-right font-mono">{t.exit_price?.toFixed(2)}</td>
                            <td className={`table-cell text-right font-mono ${t.pnl_cents > 0 ? 'text-accent-green' : 'text-accent-red'}`}>
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
            <div className="card p-5">
              <h3 className="section-title mb-3">Scan Log</h3>
              <div className="space-y-0.5">
                {scanHistory.map((s, i) => (
                  <div key={i} className={`rounded-lg p-2.5 text-xs flex items-center justify-between font-mono transition-colors duration-100
                    ${i % 2 === 0 ? 'bg-surface-2/50' : 'bg-transparent'} hover:bg-surface-2`}>
                    <span className="text-text-secondary">#{s.scan_number}</span>
                    <div className="flex items-center gap-4">
                      <span className="text-accent-green">+{s.entries?.length ?? 0}</span>
                      <span className="text-accent-red">-{s.exits?.length ?? 0}</span>
                      <span className="text-text-secondary">{s.open_positions} open</span>
                      <span className="text-text-primary">${(s.balance_cents / 100).toFixed(2)}</span>
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
