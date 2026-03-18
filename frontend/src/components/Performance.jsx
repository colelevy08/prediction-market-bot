/**
 * Performance Dashboard Component
 *
 * Displays comprehensive trading performance analytics for resolved trades.
 *
 * Key sections:
 *   - Metric cards (top grid): Win rate, Sharpe ratio, total P&L, profit factor,
 *     average MAE (max adverse excursion), average MFE (max favorable excursion),
 *     average log return, average edge, max drawdown, and best/worst trade.
 *   - Equity Curve chart: Line chart showing cumulative equity over trade sequence.
 *   - P&L Per Trade chart: Bar chart of individual trade profit/loss, color-coded
 *     green (win) or red (loss).
 *   - P&L by Category: Breakdown of performance grouped by market category,
 *     showing total P&L, trade count, and win rate per category.
 *   - Trade History table: Full log of all resolved trades with ticker, side,
 *     category, entry/exit prices, log return, P&L, MAE, MFE, win/loss result,
 *     and editable notes. Includes CSV export.
 *   - Sharpe Ratio Guide: Visual reference for interpreting Sharpe values
 *     (< 1 bad, 1-2 good, > 2 excellent).
 *
 * API endpoints called:
 *   - api.getPerformance()          — fetches metrics, trades, and equity curve
 *   - api.getPerformanceByCategory() — fetches per-category performance breakdown
 *   - api.updateTradeNotes(idx, text) — saves user notes on individual trades
 *   - api.exportTradesCsv('paper')  — triggers CSV download of trade history
 *
 * Data displayed:
 *   - metrics: win_rate, sharpe_ratio, total_pnl_cents, profit_factor, avg_mae,
 *     avg_mfe, avg_log_return, avg_edge, max_drawdown_cents, best/worst_trade_pnl
 *   - trades[]: ticker, side, category, entry_price, exit_price, log_return,
 *     pnl_cents, mae, mfe, won, notes
 *   - equity_curve[]: trade_num, equity_cents
 *   - categories{}: keyed by category name, each with total_pnl_cents,
 *     total_trades, win_rate
 */
import { useState, useEffect } from 'react';
import { api } from '../api';
import { LineChart, Line, XAxis, YAxis, Tooltip as RechartsTooltip, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts';
import Tooltip from './Tooltip';
import { useToast } from './Toast';

function MetricCard({ label, tooltip, value, sub, color = 'text-text-primary', formula, accentColor, progress = null, icon = null }) {
  return (
    <div className="stat-card" style={{ '--accent-color': accentColor || 'rgb(var(--color-text-muted))' }}>
      <div className="flex items-center justify-between mb-2">
        <Tooltip text={tooltip}>
          <span className="text-[10px] uppercase tracking-widest text-text-secondary">{label}</span>
        </Tooltip>
        {icon && <span className="text-sm opacity-40">{icon}</span>}
      </div>
      <div className={`text-xl font-bold font-mono ${color}`}>{value}</div>
      {sub && <div className="text-[10px] text-text-secondary mt-1">{sub}</div>}
      {formula && <div className="text-[10px] font-mono text-text-muted mt-1">{formula}</div>}
      {progress !== null && (
        <div className="progress-track">
          <div className="progress-fill" style={{
            width: `${Math.min(Math.max(progress, 0), 100)}%`,
            background: progress >= 60 ? 'rgb(var(--color-green))' : progress >= 50 ? 'rgb(var(--color-yellow))' : 'rgb(var(--color-red))'
          }} />
        </div>
      )}
    </div>
  );
}

export default function Performance() {
  const toast = useToast();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [categoryData, setCategoryData] = useState({});
  const [editingNote, setEditingNote] = useState(null);
  const [noteText, setNoteText] = useState('');

  useEffect(() => {
    Promise.all([
      api.getPerformance().then(setData),
      api.getPerformanceByCategory().then(d => setCategoryData(d.categories || {})),
    ]).catch(() => {}).finally(() => setLoading(false));
  }, []);

  const handleSaveNote = async (idx) => {
    try {
      await api.updateTradeNotes(idx, noteText);
      setEditingNote(null);
      api.getPerformance().then(setData).catch(() => {});
    } catch (e) { toast.error('Failed to save note'); }
  };

  if (loading) return <div className="text-text-secondary text-xs p-8 text-center">Loading...</div>;

  const metrics = data?.metrics || {};
  const trades = data?.trades || [];
  const equityCurve = data?.equity_curve || [];
  const noData = trades.length === 0;

  const pnlBuckets = trades.map((t, i) => ({ trade: i + 1, pnl: t.pnl_cents, won: t.won }));

  const chartTooltipStyle = { background: '#0c0c0e', border: '1px solid #1e1e22', borderRadius: 8, fontSize: 11, color: '#fff' };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <MetricCard label="Win Rate"
          tooltip="Percentage of trades that were profitable"
          value={noData ? '--' : `${(metrics.win_rate * 100).toFixed(1)}%`}
          sub={noData ? null : `${metrics.wins}W / ${metrics.losses}L`}
          color={metrics.win_rate >= 0.6 ? 'text-accent-green' : metrics.win_rate >= 0.5 ? 'text-accent-yellow' : 'text-accent-red'}
          accentColor={metrics.win_rate >= 0.6 ? 'rgb(var(--color-green))' : metrics.win_rate >= 0.5 ? 'rgb(var(--color-yellow))' : 'rgb(var(--color-red))'}
          progress={noData ? null : metrics.win_rate * 100}
          icon="🎯" />
        <MetricCard label="Sharpe Ratio"
          tooltip="Risk-adjusted return: (avg return - risk-free rate) / std deviation. >2 is excellent, 1-2 is good, <1 is poor"
          value={noData ? '--' : metrics.sharpe_ratio}
          sub={metrics.sharpe_label ? (
            <span className={`badge ${
              metrics.sharpe_ratio >= 2 ? 'badge-green' :
              metrics.sharpe_ratio >= 1 ? 'badge-yellow' :
              'badge-red'
            }`}>{metrics.sharpe_label}</span>
          ) : null}
          formula="SR = (Rp - Rf) / σ"
          color={metrics.sharpe_ratio >= 2 ? 'text-accent-green' : metrics.sharpe_ratio >= 1 ? 'text-accent-yellow' : 'text-accent-red'}
          accentColor={metrics.sharpe_ratio >= 2 ? 'rgb(var(--color-green))' : metrics.sharpe_ratio >= 1 ? 'rgb(var(--color-yellow))' : 'rgb(var(--color-red))'}
          icon="📐" />
        <MetricCard label="P&L"
          tooltip="Total Profit & Loss across all resolved trades"
          value={noData ? '--' : `$${(metrics.total_pnl_cents / 100).toFixed(2)}`}
          color={metrics.total_pnl_cents > 0 ? 'text-accent-green' : metrics.total_pnl_cents < 0 ? 'text-accent-red' : 'text-accent-yellow'}
          accentColor={metrics.total_pnl_cents > 0 ? 'rgb(var(--color-green))' : metrics.total_pnl_cents < 0 ? 'rgb(var(--color-red))' : 'rgb(var(--color-yellow))'}
          icon={metrics.total_pnl_cents > 0 ? '📈' : '📉'} />
        <MetricCard label="Profit Factor"
          tooltip="Gross profit divided by gross loss. >1.5 is good, >2 is excellent"
          value={noData ? '--' : metrics.profit_factor}
          sub="Gross profit / loss"
          color={metrics.profit_factor >= 2 ? 'text-accent-green' : metrics.profit_factor >= 1.5 ? 'text-accent-cyan' : metrics.profit_factor >= 1 ? 'text-accent-yellow' : 'text-accent-red'}
          accentColor={metrics.profit_factor >= 1.5 ? 'rgb(var(--color-green))' : metrics.profit_factor >= 1 ? 'rgb(var(--color-yellow))' : 'rgb(var(--color-red))'}
          icon="⚖️" />
        <MetricCard label="Avg MAE"
          tooltip="Average Max Adverse Excursion — worst drawdown during each trade before exit"
          value={noData ? '--' : `${(metrics.avg_mae * 100).toFixed(1)}%`}
          color="text-accent-red"
          accentColor="var(--color-red)"
          icon="🔻" />
        <MetricCard label="Avg MFE"
          tooltip="Average Max Favorable Excursion — best unrealized gain during each trade"
          value={noData ? '--' : `${(metrics.avg_mfe * 100).toFixed(1)}%`}
          color="text-accent-green"
          accentColor="var(--color-green)"
          icon="🔺" />
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard label="Avg Log Return"
          tooltip="Average logarithmic return: ln(exit_price / entry_price). Used for additive compounding"
          value={noData ? '--' : metrics.avg_log_return?.toFixed(4)} formula="ln(P1/P0)"
          color={metrics.avg_log_return > 0 ? 'text-accent-green' : metrics.avg_log_return < 0 ? 'text-accent-red' : 'text-accent-yellow'}
          accentColor="var(--color-blue)" icon="📊" />
        <MetricCard label="Avg Edge"
          tooltip="Average difference between model probability and market price at entry"
          value={noData ? '--' : `${(metrics.avg_edge * 100).toFixed(1)}%`} sub="Model - market"
          color={metrics.avg_edge > 0.05 ? 'text-accent-green' : metrics.avg_edge > 0 ? 'text-accent-yellow' : 'text-accent-red'}
          accentColor="var(--color-purple)" icon="🔮" />
        <MetricCard label="Max Drawdown"
          tooltip="Largest peak-to-trough decline in equity during the trading period"
          value={noData ? '--' : `$${(metrics.max_drawdown_cents / 100).toFixed(2)}`}
          color="text-accent-red" accentColor="var(--color-red)" icon="⚠️" />
        <MetricCard label="Best / Worst"
          tooltip="Best single-trade profit and worst single-trade loss"
          value={noData ? '--' : (
            <span>
              <span className="text-accent-green">${(metrics.best_trade_pnl / 100).toFixed(2)}</span>
              <span className="text-text-muted mx-1">/</span>
              <span className="text-accent-red">${(metrics.worst_trade_pnl / 100).toFixed(2)}</span>
            </span>
          )}
          accentColor="var(--color-cyan)" icon="↕️" />
      </div>

      {noData ? (
        <div className="card p-12 text-center">
          <div className="text-text-secondary text-xs">No trades yet. Metrics populate as trades resolve.</div>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="card p-5">
              <h3 className="section-title mb-4">Equity Curve</h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={equityCurve}>
                  <XAxis dataKey="trade_num" tick={{ fontSize: 10, fill: '#71717a' }} axisLine={{ stroke: '#1e1e22' }} />
                  <YAxis tick={{ fontSize: 10, fill: '#71717a' }} tickFormatter={v => `$${(v / 100).toFixed(0)}`} axisLine={{ stroke: '#1e1e22' }} />
                  <RechartsTooltip contentStyle={chartTooltipStyle}
                    formatter={v => [`$${(v / 100).toFixed(2)}`, 'Equity']} />
                  <Line type="monotone" dataKey="equity_cents" stroke="rgb(var(--color-green))" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="card p-5">
              <h3 className="section-title mb-4">P&L Per Trade</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={pnlBuckets}>
                  <XAxis dataKey="trade" tick={{ fontSize: 10, fill: '#71717a' }} axisLine={{ stroke: '#1e1e22' }} />
                  <YAxis tick={{ fontSize: 10, fill: '#71717a' }} tickFormatter={v => `$${(v / 100).toFixed(0)}`} axisLine={{ stroke: '#1e1e22' }} />
                  <RechartsTooltip contentStyle={chartTooltipStyle}
                    formatter={v => [`$${(v / 100).toFixed(2)}`, 'P&L']} />
                  <Bar dataKey="pnl" radius={[2, 2, 0, 0]}>
                    {pnlBuckets.map((entry, i) => (
                      <Cell key={i} fill={entry.won ? 'rgb(var(--color-green))' : '#ff3b3b'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Category P&L Breakdown */}
          {Object.keys(categoryData).length > 0 && (
            <div className="card p-5">
              <h3 className="section-title mb-4">P&L by Category</h3>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                {Object.entries(categoryData).map(([cat, m]) => (
                  <div key={cat} className="stat-card" style={{ '--accent-color': m.total_pnl_cents >= 0 ? 'rgb(var(--color-green))' : 'rgb(var(--color-red))' }}>
                    <div className="text-[10px] uppercase tracking-widest text-text-muted mb-1">{cat}</div>
                    <div className={`text-lg font-bold font-mono ${m.total_pnl_cents >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                      ${(m.total_pnl_cents / 100).toFixed(2)}
                    </div>
                    <div className="text-[10px] text-text-secondary mt-1">
                      {m.total_trades} trades ·{' '}
                      <span className={m.win_rate >= 0.6 ? 'text-accent-green' : m.win_rate >= 0.5 ? 'text-accent-yellow' : 'text-accent-red'}>
                        {(m.win_rate * 100).toFixed(0)}% WR
                      </span>
                    </div>
                    <div className="progress-track">
                      <div className="progress-fill" style={{
                        width: `${m.win_rate * 100}%`,
                        background: m.win_rate >= 0.6 ? 'rgb(var(--color-green))' : m.win_rate >= 0.5 ? 'rgb(var(--color-yellow))' : 'rgb(var(--color-red))'
                      }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="card overflow-hidden">
            <div className="px-5 py-3 border-b border-border flex items-center justify-between">
              <h3 className="section-title">Trade History ({trades.length})</h3>
              <div className="flex gap-2">
                <button onClick={async () => {
                  try {
                    const res = await fetch(`${(import.meta.env.VITE_API_URL || '')}/api/export/trades?source=live`);
                    if (!res.ok) throw new Error('Export failed');
                    const blob = await res.blob();
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `trades_live_${new Date().toISOString().slice(0, 10)}.csv`;
                    document.body.appendChild(a);
                    a.click();
                    a.remove();
                    URL.revokeObjectURL(url);
                  } catch (e) { toast.error('Failed to export CSV'); }
                }}
                  className="btn-ghost !px-3 !py-1.5 !text-[10px] uppercase tracking-widest">
                  Export CSV
                </button>
                <button onClick={() => api.exportTradesCsv('paper')}
                  className="btn-secondary !px-3 !py-1.5 !text-[10px] uppercase tracking-widest">
                  Export Paper CSV
                </button>
              </div>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="table-header border-b border-border">
                    <th className="table-cell text-left">#</th>
                    <th className="table-cell text-left">Ticker</th>
                    <th className="table-cell text-center">Side</th>
                    <th className="table-cell text-left">Category</th>
                    <th className="table-cell text-right">
                      <Tooltip text="Price in cents when the position was opened">Entry</Tooltip>
                    </th>
                    <th className="table-cell text-right">
                      <Tooltip text="Price in cents when the position was closed">Exit</Tooltip>
                    </th>
                    <th className="table-cell text-right">
                      <Tooltip text="Logarithmic return: ln(exit/entry)">Log R</Tooltip>
                    </th>
                    <th className="table-cell text-right">
                      <Tooltip text="Total Profit & Loss across all resolved trades">P&L</Tooltip>
                    </th>
                    <th className="table-cell text-right">
                      <Tooltip text="Max Adverse Excursion — worst drawdown during this trade">MAE</Tooltip>
                    </th>
                    <th className="table-cell text-right">
                      <Tooltip text="Max Favorable Excursion — best unrealized gain during this trade">MFE</Tooltip>
                    </th>
                    <th className="table-cell text-center">Result</th>
                    <th className="table-cell text-left">Notes</th>
                  </tr>
                </thead>
                <tbody>
                  {trades.map((t, i) => (
                    <tr key={i} className={`table-row ${t.won ? 'row-win' : 'row-loss'}`}>
                      <td className="table-cell text-text-secondary font-mono">{i + 1}</td>
                      <td className="table-cell font-mono text-text-primary">{t.ticker}</td>
                      <td className="table-cell text-center">
                        <span className={`badge ${
                          t.side === 'yes' ? 'badge-green' : 'badge-red'
                        }`}>{t.side?.toUpperCase()}</span>
                      </td>
                      <td className="table-cell text-text-muted text-[10px]">{t.category || '--'}</td>
                      <td className="table-cell text-right font-mono">{t.entry_price != null ? (t.entry_price * 100).toFixed(0) + 'c' : '--'}</td>
                      <td className="table-cell text-right font-mono">{t.exit_price != null ? (t.exit_price * 100).toFixed(0) + 'c' : '--'}</td>
                      <td className="table-cell text-right font-mono">
                        <span className={t.log_return > 0 ? 'text-accent-green' : t.log_return < 0 ? 'text-accent-red' : ''}>
                          {t.log_return?.toFixed(4) ?? '--'}
                        </span>
                      </td>
                      <td className="table-cell text-right font-mono">
                        <span className={(t.pnl_cents ?? 0) >= 0 ? 'text-accent-green' : 'text-accent-red'}>
                          ${((t.pnl_cents ?? 0) / 100).toFixed(2)}
                        </span>
                      </td>
                      <td className="table-cell text-right font-mono text-accent-red">{t.mae != null ? (t.mae * 100).toFixed(1) + '%' : '--'}</td>
                      <td className="table-cell text-right font-mono text-accent-green">{t.mfe != null ? (t.mfe * 100).toFixed(1) + '%' : '--'}</td>
                      <td className="table-cell text-center">
                        <span className={`badge ${
                          t.won ? 'badge-green' : 'badge-red'
                        }`}>{t.won ? 'Win' : 'Loss'}</span>
                      </td>
                      <td className="table-cell max-w-[150px]">
                        {editingNote === i ? (
                          <div className="flex gap-1">
                            <input value={noteText} onChange={e => setNoteText(e.target.value)}
                              className="input !w-20 !px-1.5 !py-0.5 !text-[10px]" />
                            <button onClick={() => handleSaveNote(i)} className="text-accent-green text-[10px] font-semibold hover:text-white transition-colors">Save</button>
                          </div>
                        ) : (
                          <span onClick={() => { setEditingNote(i); setNoteText(t.notes || ''); }}
                            className="text-[10px] text-text-muted cursor-pointer hover:text-text-primary truncate block transition-colors">
                            {t.notes || 'Add note...'}
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* Sharpe guide */}
      <div className="card p-5">
        <h3 className="section-title mb-4">
          <Tooltip text="Risk-adjusted return: (avg return - risk-free rate) / std deviation. >2 is excellent, 1-2 is good, <1 is poor">
            Sharpe Ratio Guide
          </Tooltip>
        </h3>
        <div className="grid grid-cols-3 gap-3 text-center">
          <div className="rounded-xl p-5" style={{ background: 'linear-gradient(135deg, rgba(248,113,113,0.08), rgba(251,146,60,0.04))' }}>
            <div className="text-3xl font-bold font-mono text-accent-red mb-1">&lt; 1</div>
            <div className="badge badge-red">Bad</div>
          </div>
          <div className="rounded-xl p-5" style={{ background: 'linear-gradient(135deg, rgba(251,191,36,0.08), rgba(251,146,60,0.04))' }}>
            <div className="text-3xl font-bold font-mono text-accent-yellow mb-1">1 - 2</div>
            <div className="badge badge-yellow">Good</div>
          </div>
          <div className="rounded-xl p-5" style={{ background: 'linear-gradient(135deg, rgba(52,211,153,0.08), rgba(34,211,238,0.04))' }}>
            <div className="text-3xl font-bold font-mono text-accent-green mb-1">&gt; 2</div>
            <div className="badge badge-green">Excellent</div>
          </div>
        </div>
      </div>
    </div>
  );
}
