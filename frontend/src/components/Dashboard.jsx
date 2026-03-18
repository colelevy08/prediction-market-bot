/*
 * Dashboard.jsx — Main overview dashboard for the prediction market bot.
 *
 * Displays a high-level summary of the trading strategy, portfolio health,
 * performance metrics, and model diagnostics.
 *
 * Key sections:
 *   - Strategy Overview: Shows the ensemble model configuration (Random Forest +
 *     Gradient Boosting), entry/exit thresholds, and confidence requirements.
 *   - Stat Cards: Balance, open positions count, win rate (W/L breakdown),
 *     Sharpe ratio, total P&L, and profit factor.
 *   - Equity Curve Chart: Line chart plotting cumulative equity over trade history.
 *   - Feature Importance Chart: Horizontal bar chart of the top 10 features used
 *     by the trained model, expressed as percentage importance.
 *   - Latest Scan Summary: Events scanned, markets scanned, RF signals, AI signals,
 *     and exit signals from the most recent scan run.
 *   - Strategy Formulas: Reference cards for entry/exit rules, log return, and
 *     Sharpe ratio calculations.
 *
 * API endpoints called (via the `api` module):
 *   - api.getPortfolio()        — Fetches current balance and open positions.
 *   - api.getPerformance()      — Fetches performance metrics and equity curve data.
 *   - api.getFeatureImportance() — Fetches model feature importance scores.
 *
 * Props:
 *   - status      — System status object containing model metadata (n_estimators, n_features).
 *   - scanResult  — Results from the latest market scan (triggers data refresh on change).
 *   - onScan      — Callback to initiate a new scan.
 *   - scanning    — Boolean indicating whether a scan is currently in progress.
 *
 * Sub-components:
 *   - StatCard — Reusable card displaying a labeled metric with optional subtitle
 *     and color-coded value.
 */
import { useState, useEffect } from 'react';
import { api } from '../api';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts';

function StatCard({ label, value, sub, color = 'text-white', borderColor = '', progress = null, icon = null }) {
  return (
    <div className={`bg-card border border-border rounded-lg p-4 card-hover ${borderColor}`}>
      <div className="flex items-center justify-between mb-2">
        <div className="text-[10px] uppercase tracking-widest text-text-secondary">{label}</div>
        {icon && <span className="text-sm opacity-40">{icon}</span>}
      </div>
      <div className={`text-2xl font-bold font-mono ${color}`}>{value}</div>
      {sub && <div className="text-[10px] text-text-secondary mt-1.5">{sub}</div>}
      {progress !== null && (
        <div className="progress-bar">
          <div className="fill" style={{
            width: `${Math.min(Math.max(progress, 0), 100)}%`,
            background: progress >= 60 ? '#00ff87' : progress >= 50 ? '#ffb800' : '#ff3b3b'
          }} />
        </div>
      )}
    </div>
  );
}

export default function Dashboard({ status, scanResult, onScan, scanning }) {
  const [portfolio, setPortfolio] = useState(null);
  const [perf, setPerf] = useState(null);
  const [features, setFeatures] = useState(null);

  useEffect(() => {
    api.getPortfolio().then(setPortfolio).catch(() => {});
    api.getPerformance().then(setPerf).catch(() => {});
    api.getFeatureImportance().then(setFeatures).catch(() => {});
  }, [scanResult]);

  const metrics = perf?.metrics || {};
  const equityCurve = perf?.equity_curve || [];

  const topFeatures = features?.features
    ? Object.entries(features.features).slice(0, 10).map(([name, imp]) => ({
        name: name.replace(/_/g, ' '),
        importance: +(imp * 100).toFixed(1),
      }))
    : [];

  return (
    <div className="space-y-6">
      {/* Strategy overview */}
      <div className="bg-card border border-border rounded-lg p-5">
        <h2 className="text-sm font-semibold uppercase tracking-widest text-text-secondary mb-4">Strategy</h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          <div className="bg-surface border border-border rounded-lg p-3">
            <div className="text-[10px] uppercase tracking-widest text-text-secondary mb-1">Ensemble</div>
            <div className="font-mono text-sm text-white">
              RF({status?.model?.n_estimators || 500}) + GB(150) &times; {status?.model?.n_features || 106}
            </div>
          </div>
          <div className="bg-surface border border-border rounded-lg p-3">
            <div className="text-[10px] uppercase tracking-widest text-text-secondary mb-1">Entry</div>
            <div className="font-mono text-sm text-accent-green">
              mkt &le; model &times; 0.5
            </div>
          </div>
          <div className="bg-surface border border-border rounded-lg p-3">
            <div className="text-[10px] uppercase tracking-widest text-text-secondary mb-1">Exit</div>
            <div className="font-mono text-sm text-accent-red">
              mkt &ge; model &times; 0.9
            </div>
          </div>
          <div className="bg-surface border border-border rounded-lg p-3">
            <div className="text-[10px] uppercase tracking-widest text-text-secondary mb-1">Confidence</div>
            <div className="font-mono text-sm text-white">70%+ required</div>
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <StatCard
          label="Balance"
          value={portfolio ? `$${(portfolio.balance_cents / 100).toFixed(2)}` : '--'}
          color="text-accent-green"
          borderColor="border-l-green"
          icon="💰"
        />
        <StatCard label="Positions" value={portfolio?.positions?.length ?? '--'} borderColor="border-l-blue" icon="📊" />
        <StatCard
          label="Win Rate"
          value={metrics.total_trades > 0 ? `${(metrics.win_rate * 100).toFixed(1)}%` : '--'}
          color={metrics.win_rate >= 0.6 ? 'text-accent-green' : metrics.win_rate >= 0.5 ? 'text-accent-yellow' : 'text-accent-red'}
          sub={metrics.total_trades > 0 ? `${metrics.wins}W / ${metrics.losses}L` : null}
          borderColor={metrics.win_rate >= 0.6 ? 'border-l-green' : metrics.win_rate >= 0.5 ? 'border-l-yellow' : 'border-l-red'}
          progress={metrics.total_trades > 0 ? metrics.win_rate * 100 : null}
        />
        <StatCard
          label="Sharpe"
          value={metrics.sharpe_ratio ?? '--'}
          color={metrics.sharpe_ratio >= 2 ? 'text-accent-green' : metrics.sharpe_ratio >= 1 ? 'text-accent-yellow' : 'text-accent-red'}
          sub={metrics.sharpe_label ? (
            <span className={`inline-block px-1.5 py-0.5 rounded text-[9px] font-semibold uppercase ${
              metrics.sharpe_ratio >= 2 ? 'bg-accent-green/10 text-accent-green' :
              metrics.sharpe_ratio >= 1 ? 'bg-accent-yellow/10 text-accent-yellow' :
              'bg-accent-red/10 text-accent-red'
            }`}>{metrics.sharpe_label}</span>
          ) : null}
          borderColor={metrics.sharpe_ratio >= 2 ? 'border-l-green' : metrics.sharpe_ratio >= 1 ? 'border-l-yellow' : 'border-l-red'}
        />
        <StatCard
          label="P&L"
          value={metrics.total_pnl_cents != null ? `$${(metrics.total_pnl_cents / 100).toFixed(2)}` : '--'}
          color={metrics.total_pnl_cents > 0 ? 'text-accent-green' : metrics.total_pnl_cents < 0 ? 'text-accent-red' : 'text-white'}
          borderColor={metrics.total_pnl_cents > 0 ? 'border-l-green' : metrics.total_pnl_cents < 0 ? 'border-l-red' : 'border-l-white'}
          icon={metrics.total_pnl_cents > 0 ? '📈' : metrics.total_pnl_cents < 0 ? '📉' : '➖'}
        />
        <StatCard
          label="Profit Factor"
          value={metrics.profit_factor ?? '--'}
          color={metrics.profit_factor >= 2 ? 'text-accent-green' : metrics.profit_factor >= 1.5 ? 'text-accent-cyan' : metrics.profit_factor >= 1 ? 'text-accent-yellow' : 'text-accent-red'}
          borderColor={metrics.profit_factor >= 1.5 ? 'border-l-green' : metrics.profit_factor >= 1 ? 'border-l-yellow' : 'border-l-red'}
          icon="⚖️"
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="bg-card border border-border rounded-lg p-5">
          <h3 className="text-[10px] uppercase tracking-widest text-text-secondary mb-4">Equity Curve</h3>
          {equityCurve.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={equityCurve}>
                <XAxis dataKey="trade_num" tick={{ fontSize: 10, fill: '#666' }} axisLine={{ stroke: '#1a1a1a' }} />
                <YAxis tick={{ fontSize: 10, fill: '#666' }} tickFormatter={v => `$${(v / 100).toFixed(0)}`} axisLine={{ stroke: '#1a1a1a' }} />
                <Tooltip
                  contentStyle={{ background: '#0a0a0a', border: '1px solid #1a1a1a', borderRadius: 6, fontSize: 11, color: '#fff' }}
                  formatter={(v) => [`$${(v / 100).toFixed(2)}`, 'Equity']}
                />
                <Line type="monotone" dataKey="equity_cents" stroke="#00ff87" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[220px] flex items-center justify-center text-text-secondary text-xs">
              No trade history yet. Run a scan to begin.
            </div>
          )}
        </div>

        <div className="bg-card border border-border rounded-lg p-5">
          <h3 className="text-[10px] uppercase tracking-widest text-text-secondary mb-4">
            Feature Importance ({features?.n_features || 106} total)
          </h3>
          {topFeatures.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={topFeatures} layout="vertical">
                <XAxis type="number" tick={{ fontSize: 9, fill: '#666' }} axisLine={{ stroke: '#1a1a1a' }} />
                <YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 9, fill: '#666' }} axisLine={{ stroke: '#1a1a1a' }} />
                <Tooltip
                  contentStyle={{ background: '#0a0a0a', border: '1px solid #1a1a1a', borderRadius: 6, fontSize: 11, color: '#fff' }}
                  formatter={(v) => [`${v}%`, 'Importance']}
                />
                <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                  {topFeatures.map((_, i) => (
                    <Cell key={i} fill={i < 3 ? '#00ff87' : i < 6 ? '#ffffff' : '#444'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[220px] flex items-center justify-center text-text-secondary text-xs">
              {features?.is_trained ? 'Loading...' : 'Model untrained. Using heuristic mode.'}
            </div>
          )}
        </div>
      </div>

      {/* Scan summary */}
      {scanResult && (
        <div className="bg-card border border-border rounded-lg p-5">
          <h3 className="text-[10px] uppercase tracking-widest text-text-secondary mb-3">Latest Scan</h3>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-sm">
            <div className="bg-surface border border-border rounded-lg p-2.5 border-l-blue">
              <span className="text-text-secondary text-[10px] uppercase tracking-widest">Events</span>
              <div className="font-mono text-accent-blue font-bold">{scanResult.events_scanned?.toLocaleString()}</div>
            </div>
            <div className="bg-surface border border-border rounded-lg p-2.5 border-l-purple">
              <span className="text-text-secondary text-[10px] uppercase tracking-widest">Markets</span>
              <div className="font-mono text-accent-purple font-bold">{scanResult.markets_scanned?.toLocaleString()}</div>
            </div>
            <div className={`bg-surface border border-border rounded-lg p-2.5 border-l-green ${scanResult.rf_signals > 0 ? 'glow-green' : ''}`}>
              <span className="text-text-secondary text-[10px] uppercase tracking-widest">RF Signals</span>
              <div className="font-mono text-accent-green font-bold">{scanResult.rf_signals}</div>
            </div>
            <div className={`bg-surface border border-border rounded-lg p-2.5 border-l-cyan`}>
              <span className="text-text-secondary text-[10px] uppercase tracking-widest">AI Signals</span>
              <div className="font-mono text-accent-cyan font-bold">{scanResult.ai_signals}</div>
            </div>
            <div className={`bg-surface border border-border rounded-lg p-2.5 border-l-red ${scanResult.exit_signals > 0 ? 'glow-red' : ''}`}>
              <span className="text-text-secondary text-[10px] uppercase tracking-widest">Exit Signals</span>
              <div className="font-mono text-accent-red font-bold">{scanResult.exit_signals}</div>
            </div>
          </div>
        </div>
      )}

      {/* Formulas */}
      <div className="bg-card border border-border rounded-lg p-5">
        <h3 className="text-[10px] uppercase tracking-widest text-text-secondary mb-3">Strategy Formulas</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
          <div className="bg-surface border border-border rounded-lg p-3 font-mono text-xs">
            <div className="text-text-muted mb-1"># Entry</div>
            <div className="text-accent-green">if mkt &le; prob * 0.5: buy()</div>
          </div>
          <div className="bg-surface border border-border rounded-lg p-3 font-mono text-xs">
            <div className="text-text-muted mb-1"># Exit</div>
            <div className="text-accent-red">if mkt &ge; prob * 0.9: sell()</div>
          </div>
          <div className="bg-surface border border-border rounded-lg p-3 font-mono text-xs">
            <div className="text-text-muted mb-1"># Log Return</div>
            <div className="text-white">r = ln(P1 / P0)</div>
          </div>
          <div className="bg-surface border border-border rounded-lg p-3 font-mono text-xs">
            <div className="text-text-muted mb-1"># Sharpe Ratio</div>
            <div className="text-white">SR = (Rp - Rf) / &sigma;</div>
          </div>
        </div>
      </div>
    </div>
  );
}
