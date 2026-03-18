import { useState, useEffect } from 'react';
import { api } from '../api';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts';

function MetricCard({ label, value, sub, color = 'text-white', formula }) {
  return (
    <div className="bg-card border border-border rounded-lg p-4 card-hover">
      <div className="text-[10px] uppercase tracking-widest text-text-secondary mb-2">{label}</div>
      <div className={`text-xl font-bold font-mono ${color}`}>{value}</div>
      {sub && <div className="text-[10px] text-text-secondary mt-1">{sub}</div>}
      {formula && <div className="text-[10px] font-mono text-text-muted mt-1">{formula}</div>}
    </div>
  );
}

export default function Performance() {
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
      api.getPerformance().then(setData);
    } catch (e) { console.error(e); }
  };

  if (loading) return <div className="text-text-secondary text-xs p-8 text-center">Loading...</div>;

  const metrics = data?.metrics || {};
  const trades = data?.trades || [];
  const equityCurve = data?.equity_curve || [];
  const noData = trades.length === 0;

  const pnlBuckets = trades.map((t, i) => ({ trade: i + 1, pnl: t.pnl_cents, won: t.won }));

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        <MetricCard label="Win Rate" value={noData ? '--' : `${(metrics.win_rate * 100).toFixed(1)}%`}
          sub={noData ? null : `${metrics.wins}W / ${metrics.losses}L`}
          color={metrics.win_rate >= 0.6 ? 'text-accent-green' : metrics.win_rate >= 0.5 ? 'text-accent-yellow' : 'text-accent-red'} />
        <MetricCard label="Sharpe Ratio" value={noData ? '--' : metrics.sharpe_ratio}
          sub={metrics.sharpe_label} formula="SR = (Rp - Rf) / σ"
          color={metrics.sharpe_ratio >= 2 ? 'text-accent-green' : metrics.sharpe_ratio >= 1 ? 'text-accent-yellow' : 'text-accent-red'} />
        <MetricCard label="Total P&L" value={noData ? '--' : `$${(metrics.total_pnl_cents / 100).toFixed(2)}`}
          color={metrics.total_pnl_cents > 0 ? 'text-accent-green' : 'text-accent-red'} />
        <MetricCard label="Profit Factor" value={noData ? '--' : metrics.profit_factor}
          sub="Gross profit / loss" color={metrics.profit_factor >= 1.5 ? 'text-accent-green' : 'text-white'} />
        <MetricCard label="Avg MAE" value={noData ? '--' : `${(metrics.avg_mae * 100).toFixed(1)}%`}
          sub="Max adverse excursion" />
        <MetricCard label="Avg MFE" value={noData ? '--' : `${(metrics.avg_mfe * 100).toFixed(1)}%`}
          sub="Max favorable excursion" />
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricCard label="Avg Log Return" value={noData ? '--' : metrics.avg_log_return?.toFixed(4)} formula="ln(P1/P0)" />
        <MetricCard label="Avg Edge" value={noData ? '--' : `${(metrics.avg_edge * 100).toFixed(1)}%`} sub="Model - market" />
        <MetricCard label="Max Drawdown" value={noData ? '--' : `$${(metrics.max_drawdown_cents / 100).toFixed(2)}`} color="text-accent-red" />
        <MetricCard label="Best / Worst" value={noData ? '--' : `$${(metrics.best_trade_pnl / 100).toFixed(2)} / $${(metrics.worst_trade_pnl / 100).toFixed(2)}`} />
      </div>

      {noData ? (
        <div className="bg-card border border-border rounded-lg p-12 text-center">
          <div className="text-text-secondary text-xs">No trades yet. Metrics populate as trades resolve.</div>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="bg-card border border-border rounded-lg p-5">
              <h3 className="text-[10px] uppercase tracking-widest text-text-secondary mb-4">Equity Curve</h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={equityCurve}>
                  <XAxis dataKey="trade_num" tick={{ fontSize: 10, fill: '#666' }} axisLine={{ stroke: '#1a1a1a' }} />
                  <YAxis tick={{ fontSize: 10, fill: '#666' }} tickFormatter={v => `$${(v / 100).toFixed(0)}`} axisLine={{ stroke: '#1a1a1a' }} />
                  <Tooltip contentStyle={{ background: '#0a0a0a', border: '1px solid #1a1a1a', borderRadius: 6, fontSize: 11, color: '#fff' }}
                    formatter={v => [`$${(v / 100).toFixed(2)}`, 'Equity']} />
                  <Line type="monotone" dataKey="equity_cents" stroke="#00ff87" strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="bg-card border border-border rounded-lg p-5">
              <h3 className="text-[10px] uppercase tracking-widest text-text-secondary mb-4">P&L Per Trade</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={pnlBuckets}>
                  <XAxis dataKey="trade" tick={{ fontSize: 10, fill: '#666' }} axisLine={{ stroke: '#1a1a1a' }} />
                  <YAxis tick={{ fontSize: 10, fill: '#666' }} tickFormatter={v => `$${(v / 100).toFixed(0)}`} axisLine={{ stroke: '#1a1a1a' }} />
                  <Tooltip contentStyle={{ background: '#0a0a0a', border: '1px solid #1a1a1a', borderRadius: 6, fontSize: 11, color: '#fff' }}
                    formatter={v => [`$${(v / 100).toFixed(2)}`, 'P&L']} />
                  <Bar dataKey="pnl" radius={[2, 2, 0, 0]}>
                    {pnlBuckets.map((entry, i) => (
                      <Cell key={i} fill={entry.won ? '#00ff87' : '#ff3b3b'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Category P&L Breakdown */}
          {Object.keys(categoryData).length > 0 && (
            <div className="bg-card border border-border rounded-lg p-5">
              <h3 className="text-[10px] uppercase tracking-widest text-text-secondary mb-4">P&L by Category</h3>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                {Object.entries(categoryData).map(([cat, m]) => (
                  <div key={cat} className="bg-surface border border-border rounded-lg p-3">
                    <div className="text-[10px] uppercase tracking-widest text-text-muted mb-1">{cat}</div>
                    <div className={`text-lg font-bold font-mono ${m.total_pnl_cents >= 0 ? 'text-accent-green' : 'text-accent-red'}`}>
                      ${(m.total_pnl_cents / 100).toFixed(2)}
                    </div>
                    <div className="text-[10px] text-text-secondary mt-1">
                      {m.total_trades} trades · {(m.win_rate * 100).toFixed(0)}% WR
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="bg-card border border-border rounded-lg overflow-hidden">
            <div className="px-5 py-3 border-b border-border flex items-center justify-between">
              <h3 className="text-[10px] uppercase tracking-widest text-text-secondary font-semibold">Trade History ({trades.length})</h3>
              <button onClick={() => api.exportTradesCsv('paper')}
                className="text-[10px] uppercase tracking-widest text-accent-green hover:text-white transition-colors">
                Export CSV
              </button>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-text-secondary text-[10px] uppercase tracking-widest border-b border-border">
                    <th className="text-left px-4 py-2">#</th>
                    <th className="text-left px-4 py-2">Ticker</th>
                    <th className="text-center px-4 py-2">Side</th>
                    <th className="text-left px-4 py-2">Category</th>
                    <th className="text-right px-4 py-2">Entry</th>
                    <th className="text-right px-4 py-2">Exit</th>
                    <th className="text-right px-4 py-2">Log R</th>
                    <th className="text-right px-4 py-2">P&L</th>
                    <th className="text-right px-4 py-2">MAE</th>
                    <th className="text-right px-4 py-2">MFE</th>
                    <th className="text-center px-4 py-2">Result</th>
                    <th className="text-left px-4 py-2">Notes</th>
                  </tr>
                </thead>
                <tbody>
                  {trades.map((t, i) => (
                    <tr key={i} className="border-b border-border/50 hover:bg-surface/50">
                      <td className="px-4 py-2 text-text-secondary font-mono">{i + 1}</td>
                      <td className="px-4 py-2 font-mono text-white">{t.ticker}</td>
                      <td className="px-4 py-2 text-center font-mono">{t.side?.toUpperCase()}</td>
                      <td className="px-4 py-2 text-text-muted text-[10px]">{t.category || '--'}</td>
                      <td className="px-4 py-2 text-right font-mono">{(t.entry_price * 100).toFixed(0)}c</td>
                      <td className="px-4 py-2 text-right font-mono">{(t.exit_price * 100).toFixed(0)}c</td>
                      <td className="px-4 py-2 text-right font-mono">{t.log_return?.toFixed(4)}</td>
                      <td className="px-4 py-2 text-right font-mono">
                        <span className={t.pnl_cents >= 0 ? 'text-accent-green' : 'text-accent-red'}>
                          ${(t.pnl_cents / 100).toFixed(2)}
                        </span>
                      </td>
                      <td className="px-4 py-2 text-right font-mono text-accent-red">{(t.mae * 100).toFixed(1)}%</td>
                      <td className="px-4 py-2 text-right font-mono text-accent-green">{(t.mfe * 100).toFixed(1)}%</td>
                      <td className="px-4 py-2 text-center">
                        <span className={`text-[10px] px-2 py-0.5 rounded font-semibold uppercase ${
                          t.won ? 'bg-accent-green/10 text-accent-green' : 'bg-accent-red/10 text-accent-red'
                        }`}>{t.won ? 'Win' : 'Loss'}</span>
                      </td>
                      <td className="px-4 py-2 max-w-[150px]">
                        {editingNote === i ? (
                          <div className="flex gap-1">
                            <input value={noteText} onChange={e => setNoteText(e.target.value)}
                              className="bg-surface border border-border rounded px-1.5 py-0.5 text-[10px] text-white w-20 focus:outline-none focus:border-accent-green" />
                            <button onClick={() => handleSaveNote(i)} className="text-accent-green text-[10px]">Save</button>
                          </div>
                        ) : (
                          <span onClick={() => { setEditingNote(i); setNoteText(t.notes || ''); }}
                            className="text-[10px] text-text-muted cursor-pointer hover:text-white truncate block">
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
      <div className="bg-card border border-border rounded-lg p-5">
        <h3 className="text-[10px] uppercase tracking-widest text-text-secondary mb-3">Sharpe Ratio Guide</h3>
        <div className="grid grid-cols-3 gap-3 text-center text-sm">
          <div className="bg-accent-red/5 border border-accent-red/20 rounded-lg p-3">
            <div className="text-2xl font-bold font-mono text-accent-red">&lt; 1</div>
            <div className="text-accent-red text-xs">Bad</div>
          </div>
          <div className="bg-accent-yellow/5 border border-accent-yellow/20 rounded-lg p-3">
            <div className="text-2xl font-bold font-mono text-accent-yellow">1 - 2</div>
            <div className="text-accent-yellow text-xs">Good</div>
          </div>
          <div className="bg-accent-green/5 border border-accent-green/20 rounded-lg p-3">
            <div className="text-2xl font-bold font-mono text-accent-green">&gt; 2</div>
            <div className="text-accent-green text-xs">Excellent</div>
          </div>
        </div>
      </div>
    </div>
  );
}
