/**
 * Dashboard.jsx — Main overview dashboard for the prediction market bot.
 */
import { useState, useEffect } from 'react';
import { api } from '../api';
import Tooltip from './Tooltip';
import { LineChart, Line, XAxis, YAxis, Tooltip as RTooltip, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts';

const CHART_TOOLTIP = { background: '#0c0c0e', border: '1px solid #1e1e22', borderRadius: 8, fontSize: 11, color: '#fafafa' };

function StatCard({ label, tooltip, value, sub, color = 'text-text-primary', accentColor, progress = null, icon = null }) {
  return (
    <div className="stat-card" style={{ '--accent-color': accentColor || 'rgb(var(--color-text-muted))' }}>
      <div className="flex items-center justify-between mb-2">
        <Tooltip text={tooltip}>
          <span className="section-title">{label}</span>
        </Tooltip>
        {icon && <span className="text-sm opacity-30">{icon}</span>}
      </div>
      <div className={`text-2xl font-bold font-mono ${color}`}>{value}</div>
      {sub && <div className="text-[10px] text-text-secondary mt-1.5">{sub}</div>}
      {progress !== null && (
        <div className="progress-track">
          <div
            className="progress-fill"
            style={{
              width: `${Math.min(Math.max(progress, 0), 100)}%`,
              background: progress >= 60 ? 'rgb(var(--color-green))' : progress >= 50 ? 'rgb(var(--color-yellow))' : 'rgb(var(--color-red))',
            }}
          />
        </div>
      )}
    </div>
  );
}

const BET_AMOUNTS = [10, 25, 50, 100, 250, 500];

export default function Dashboard({ status, scanResult, onScan, scanning }) {
  const [portfolio, setPortfolio] = useState(null);
  const [perf, setPerf] = useState(null);
  const [features, setFeatures] = useState(null);
  const [signals, setSignals] = useState([]);
  const [betAmount, setBetAmount] = useState(100);

  const fetchAll = () => {
    api.getPortfolio().then(setPortfolio).catch(() => {});
    api.getPerformance().then(setPerf).catch(() => {});
    api.getFeatureImportance().then(setFeatures).catch(() => {});
    api.getSignals().then(data => setSignals(data.signals || [])).catch(() => {});
  };

  useEffect(() => {
    fetchAll();
  }, [scanResult]);

  // Auto-refresh every 30 seconds
  useEffect(() => {
    const interval = setInterval(fetchAll, 30000);
    return () => clearInterval(interval);
  }, []);

  // Top 3 opportunities sorted by edge
  const topOpportunities = [...signals]
    .filter(s => s.edge > 0 && s.confidence >= 0.7)
    .sort((a, b) => b.edge - a.edge)
    .slice(0, 3);

  const metrics = perf?.metrics || {};
  const equityCurve = perf?.equity_curve || [];

  const topFeatures = features?.features
    ? Object.entries(features.features).slice(0, 10).map(([name, imp]) => ({
        name: name.replace(/_/g, ' '),
        importance: +(imp * 100).toFixed(1),
      }))
    : [];

  return (
    <div className="space-y-5 animate-fade-in">
      {/* Strategy overview */}
      <div className="card p-5">
        <h2 className="section-title mb-4">Strategy Overview</h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          {[
            { label: 'Ensemble', value: `RF(${status?.model?.n_estimators || 500}) + GB(150) × ${status?.model?.n_features || 106}`, color: 'text-text-primary' },
            { label: 'Entry Rule', value: 'mkt ≤ model × 0.5', color: 'text-accent-green' },
            { label: 'Exit Rule', value: 'mkt ≥ model × 0.9', color: 'text-accent-red' },
            { label: 'Min Confidence', value: '70%+ required', color: 'text-accent-yellow' },
          ].map((item, i) => (
            <div key={i} className="bg-surface-2 border border-border-subtle rounded-lg p-3">
              <div className="text-[10px] uppercase tracking-widest text-text-muted mb-1">{item.label}</div>
              <div className={`font-mono text-sm ${item.color}`}>{item.value}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <StatCard
          label="Balance" tooltip="Current cash balance available for trading"
          value={portfolio ? `$${(portfolio.balance_cents / 100).toFixed(2)}` : '--'}
          color="text-accent-green" accentColor="var(--color-green)" icon="$"
        />
        <StatCard
          label="Positions" tooltip="Number of currently open market positions"
          value={portfolio?.positions?.length ?? '--'}
          accentColor="var(--color-blue)" icon="#"
        />
        <StatCard
          label="Win Rate" tooltip="Percentage of trades that were profitable"
          value={metrics.total_trades > 0 ? `${(metrics.win_rate * 100).toFixed(1)}%` : '--'}
          color={metrics.win_rate >= 0.6 ? 'text-accent-green' : metrics.win_rate >= 0.5 ? 'text-accent-yellow' : 'text-accent-red'}
          accentColor={metrics.win_rate >= 0.6 ? 'rgb(var(--color-green))' : metrics.win_rate >= 0.5 ? 'rgb(var(--color-yellow))' : 'rgb(var(--color-red))'}
          sub={metrics.total_trades > 0 ? `${metrics.wins}W / ${metrics.losses}L` : null}
          progress={metrics.total_trades > 0 ? metrics.win_rate * 100 : null}
        />
        <StatCard
          label="Sharpe" tooltip="Risk-adjusted return: (avg return - risk-free rate) / std deviation. >2 excellent, 1-2 good, <1 poor"
          value={metrics.sharpe_ratio ?? '--'}
          color={metrics.sharpe_ratio >= 2 ? 'text-accent-green' : metrics.sharpe_ratio >= 1 ? 'text-accent-yellow' : 'text-accent-red'}
          accentColor={metrics.sharpe_ratio >= 2 ? 'rgb(var(--color-green))' : metrics.sharpe_ratio >= 1 ? 'rgb(var(--color-yellow))' : 'rgb(var(--color-red))'}
          sub={metrics.sharpe_label ? (
            <span className={`badge ${
              metrics.sharpe_ratio >= 2 ? 'badge-green' : metrics.sharpe_ratio >= 1 ? 'badge-yellow' : 'badge-red'
            }`}>{metrics.sharpe_label}</span>
          ) : null}
        />
        <StatCard
          label="P&L" tooltip="Total Profit & Loss across all resolved trades"
          value={metrics.total_pnl_cents != null ? `$${(metrics.total_pnl_cents / 100).toFixed(2)}` : '--'}
          color={metrics.total_pnl_cents > 0 ? 'text-accent-green' : metrics.total_pnl_cents < 0 ? 'text-accent-red' : 'text-text-primary'}
          accentColor={metrics.total_pnl_cents > 0 ? 'rgb(var(--color-green))' : metrics.total_pnl_cents < 0 ? 'rgb(var(--color-red))' : 'rgb(var(--color-text-muted))'}
          icon={metrics.total_pnl_cents > 0 ? '↑' : metrics.total_pnl_cents < 0 ? '↓' : '–'}
        />
        <StatCard
          label="Profit Factor" tooltip="Gross profit divided by gross loss. >1.5 good, >2 excellent"
          value={metrics.profit_factor ?? '--'}
          color={metrics.profit_factor >= 2 ? 'text-accent-green' : metrics.profit_factor >= 1.5 ? 'text-accent-cyan' : metrics.profit_factor >= 1 ? 'text-accent-yellow' : 'text-accent-red'}
          accentColor={metrics.profit_factor >= 1.5 ? 'rgb(var(--color-green))' : metrics.profit_factor >= 1 ? 'rgb(var(--color-yellow))' : 'rgb(var(--color-red))'}
        />
      </div>

      {/* Top 3 Opportunities */}
      <div className="card overflow-hidden">
        <div className="px-5 py-3 border-b border-border flex items-center justify-between flex-wrap gap-3">
          <h3 className="section-title flex items-center gap-2">
            Top Opportunities
            <span className="w-1.5 h-1.5 rounded-full bg-accent-green pulse-dot" />
            <span className="font-normal normal-case tracking-normal text-text-muted text-[10px] ml-1">
              auto-refreshes every 30s
            </span>
          </h3>
          <div className="flex items-center gap-1 bg-surface-2 border border-border rounded-lg p-0.5">
            {BET_AMOUNTS.map(amt => (
              <button
                key={amt}
                onClick={() => setBetAmount(amt)}
                className={`px-2.5 py-1 text-[10px] font-semibold tracking-wide rounded-md transition-all ${
                  betAmount === amt
                    ? 'bg-accent-green text-black'
                    : 'text-text-muted hover:text-text-secondary'
                }`}
              >
                ${amt}
              </button>
            ))}
          </div>
        </div>

        {topOpportunities.length === 0 ? (
          <div className="p-10 text-center text-text-muted text-xs">
            No opportunities found. Run a scan to analyze all markets.
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-0 md:divide-x divide-border">
            {topOpportunities.map((sig, i) => {
              const edgePct = sig.edge * 100;
              const contractPrice = sig.market_probability;
              const validPrice = contractPrice > 0;
              const contracts = validPrice ? Math.floor(betAmount / (contractPrice * 100)) : 0;
              const costBasis = contracts * contractPrice * 100;
              const payout = contracts * 100;
              const profit = payout - costBasis;
              const roi = costBasis > 0 ? (profit / costBasis) * 100 : 0;
              const medal = i === 0 ? '🥇' : i === 1 ? '🥈' : '🥉';

              return (
                <div key={sig.ticker} className="p-5 relative">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <span className="text-lg">{medal}</span>
                      <div>
                        <div className="font-mono text-sm font-bold text-text-primary">{sig.ticker}</div>
                        <div className="text-[10px] text-text-muted max-w-[200px] truncate">{sig.market_title}</div>
                      </div>
                    </div>
                    <span className={`badge ${sig.side === 'yes' ? 'badge-green' : 'badge-red'}`}>{sig.side}</span>
                  </div>

                  <div className="grid grid-cols-2 gap-2 mb-4">
                    <div className="bg-surface-2 rounded-lg p-2">
                      <div className="text-[9px] uppercase tracking-widest text-text-muted">Edge</div>
                      <div className={`font-mono text-sm font-bold ${edgePct >= 15 ? 'text-accent-green' : edgePct >= 8 ? 'text-accent-yellow' : 'text-text-primary'}`}>
                        +{edgePct.toFixed(1)}%
                      </div>
                    </div>
                    <div className="bg-surface-2 rounded-lg p-2">
                      <div className="text-[9px] uppercase tracking-widest text-text-muted">Confidence</div>
                      <div className={`font-mono text-sm font-bold ${sig.confidence >= 0.8 ? 'text-accent-green' : 'text-accent-yellow'}`}>
                        {(sig.confidence * 100).toFixed(0)}%
                      </div>
                    </div>
                    <div className="bg-surface-2 rounded-lg p-2">
                      <div className="text-[9px] uppercase tracking-widest text-text-muted">Model Prob</div>
                      <div className="font-mono text-sm font-bold text-accent-blue">
                        {(sig.fair_probability * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="bg-surface-2 rounded-lg p-2">
                      <div className="text-[9px] uppercase tracking-widest text-text-muted">Market Price</div>
                      <div className="font-mono text-sm font-bold text-text-secondary">
                        {(sig.market_probability * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>

                  <div className="bg-accent-green/5 border border-accent-green/20 rounded-lg p-3">
                    <div className="text-[9px] uppercase tracking-widest text-accent-green/70 mb-2">
                      Earning Potential @ ${betAmount} bet
                    </div>
                    {!validPrice ? (
                      <div className="text-sm text-text-muted font-mono">N/A</div>
                    ) : (
                      <>
                        <div className="flex items-end justify-between">
                          <div>
                            <div className="text-xs text-text-muted">
                              {contracts} contracts @ {(contractPrice * 100).toFixed(0)}¢
                            </div>
                            <div className="text-2xl font-bold font-mono text-accent-green">
                              +${profit.toFixed(2)}
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="text-[10px] text-text-muted">ROI</div>
                            <div className={`font-mono text-lg font-bold ${roi >= 100 ? 'text-accent-green' : roi >= 50 ? 'text-accent-yellow' : 'text-text-primary'}`}>
                              {roi.toFixed(0)}%
                            </div>
                          </div>
                        </div>
                        <div className="text-[9px] text-text-muted mt-1.5">
                          If model is correct: ${payout.toFixed(2)} payout on ${costBasis.toFixed(2)} invested
                        </div>
                      </>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="card p-5">
          <h3 className="section-title mb-4">Equity Curve</h3>
          {equityCurve.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={equityCurve}>
                <XAxis dataKey="trade_num" tick={{ fontSize: 10, fill: '#71717a' }} axisLine={{ stroke: '#1e1e22' }} tickLine={false} />
                <YAxis tick={{ fontSize: 10, fill: '#71717a' }} tickFormatter={v => `$${(v / 100).toFixed(0)}`} axisLine={{ stroke: '#1e1e22' }} tickLine={false} />
                <RTooltip contentStyle={CHART_TOOLTIP} formatter={(v) => [`$${(v / 100).toFixed(2)}`, 'Equity']} />
                <Line type="monotone" dataKey="equity_cents" stroke="rgb(var(--color-green))" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[220px] flex items-center justify-center text-text-muted text-xs">
              No trade history yet. Run a scan to begin.
            </div>
          )}
        </div>

        <div className="card p-5">
          <h3 className="section-title mb-4">
            <Tooltip text="Shows how important each feature is in the model's predictions">
              Feature Importance ({features?.n_features || 106} total)
            </Tooltip>
          </h3>
          {topFeatures.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={topFeatures} layout="vertical">
                <XAxis type="number" tick={{ fontSize: 9, fill: '#71717a' }} axisLine={{ stroke: '#1e1e22' }} tickLine={false} />
                <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 9, fill: '#71717a' }} axisLine={{ stroke: '#1e1e22' }} tickLine={false} />
                <RTooltip contentStyle={CHART_TOOLTIP} formatter={(v) => [`${v}%`, 'Importance']} />
                <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                  {topFeatures.map((_, i) => (
                    <Cell key={i} fill={i < 3 ? 'rgb(var(--color-green))' : i < 6 ? 'rgb(var(--color-text-secondary))' : 'rgb(var(--color-text-muted))'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[220px] flex items-center justify-center text-text-muted text-xs">
              {features?.is_trained ? 'Loading...' : 'Model untrained. Using heuristic mode.'}
            </div>
          )}
        </div>
      </div>

      {/* Scan summary */}
      {scanResult && (
        <div className="card p-5 animate-slide-up">
          <h3 className="section-title mb-3">Latest Scan</h3>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            {[
              { label: 'Events', value: scanResult.events_scanned?.toLocaleString(), color: 'text-accent-blue', accent: 'rgb(var(--color-blue))', glow: false },
              { label: 'Markets', value: scanResult.markets_scanned?.toLocaleString(), color: 'text-accent-purple', accent: 'rgb(var(--color-purple))', glow: false },
              { label: 'RF Signals', value: scanResult.rf_signals, color: 'text-accent-green', accent: 'rgb(var(--color-green))', glow: scanResult.rf_signals > 0 },
              { label: 'AI Signals', value: scanResult.ai_signals, color: 'text-accent-cyan', accent: 'rgb(var(--color-cyan))', glow: false },
              { label: 'Exit Signals', value: scanResult.exit_signals, color: 'text-accent-red', accent: 'rgb(var(--color-red))', glow: scanResult.exit_signals > 0 },
            ].map((item, i) => (
              <div
                key={i}
                className={`stat-card p-3 ${item.glow ? 'shadow-glow-green' : ''}`}
                style={{ '--accent-color': item.accent }}
              >
                <span className="text-[10px] uppercase tracking-widest text-text-muted">{item.label}</span>
                <div className={`font-mono text-lg font-bold ${item.color}`}>{item.value}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Formulas */}
      <div className="card p-5">
        <h3 className="section-title mb-3">Strategy Formulas</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
          {[
            { label: '# Entry', code: 'if mkt ≤ prob * 0.5: buy()', color: 'text-accent-green' },
            { label: '# Exit', code: 'if mkt ≥ prob * 0.9: sell()', color: 'text-accent-red' },
            { label: '# Log Return', code: 'r = ln(P1 / P0)', color: 'text-text-primary' },
            { label: '# Sharpe Ratio', code: 'SR = (Rp - Rf) / σ', color: 'text-text-primary' },
          ].map((item, i) => (
            <div key={i} className="bg-surface-2 border border-border-subtle rounded-lg p-3 font-mono text-xs">
              <div className="text-text-muted mb-1">{item.label}</div>
              <div className={item.color}>{item.code}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
