import { useState, useEffect } from 'react';
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
              RF(200) + GB(150) &times; {status?.model?.n_features || 106}
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
        />
        <StatCard label="Positions" value={portfolio?.positions?.length ?? '--'} />
        <StatCard
          label="Win Rate"
          value={metrics.total_trades > 0 ? `${(metrics.win_rate * 100).toFixed(1)}%` : '--'}
          color={metrics.win_rate >= 0.6 ? 'text-accent-green' : metrics.win_rate >= 0.5 ? 'text-accent-yellow' : 'text-accent-red'}
          sub={metrics.total_trades > 0 ? `${metrics.wins}W / ${metrics.losses}L` : null}
        />
        <StatCard
          label="Sharpe"
          value={metrics.sharpe_ratio ?? '--'}
          color={metrics.sharpe_ratio >= 2 ? 'text-accent-green' : metrics.sharpe_ratio >= 1 ? 'text-accent-yellow' : 'text-accent-red'}
          sub={metrics.sharpe_label || null}
        />
        <StatCard
          label="P&L"
          value={metrics.total_pnl_cents != null ? `$${(metrics.total_pnl_cents / 100).toFixed(2)}` : '--'}
          color={metrics.total_pnl_cents > 0 ? 'text-accent-green' : 'text-accent-red'}
        />
        <StatCard
          label="Profit Factor"
          value={metrics.profit_factor ?? '--'}
          color={metrics.profit_factor >= 1.5 ? 'text-accent-green' : 'text-white'}
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
            <div>
              <span className="text-text-secondary text-xs">Events</span>
              <div className="font-mono">{scanResult.events_scanned}</div>
            </div>
            <div>
              <span className="text-text-secondary text-xs">Markets</span>
              <div className="font-mono">{scanResult.markets_scanned}</div>
            </div>
            <div>
              <span className="text-text-secondary text-xs">RF Signals</span>
              <div className="font-mono text-accent-green">{scanResult.rf_signals}</div>
            </div>
            <div>
              <span className="text-text-secondary text-xs">AI Signals</span>
              <div className="font-mono">{scanResult.ai_signals}</div>
            </div>
            <div>
              <span className="text-text-secondary text-xs">Exit Signals</span>
              <div className="font-mono text-accent-red">{scanResult.exit_signals}</div>
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
