import { useState, useEffect } from 'react';
import { api } from '../api';

const HEATMAP_COLORS = ['#1a1a2e', '#16213e', '#0f3460', '#533483', '#e94560', '#00ff87'];

export default function Portfolio() {
  const [portfolio, setPortfolio] = useState(null);
  const [positions, setPositions] = useState([]);
  const [orders, setOrders] = useState([]);
  const [heatmap, setHeatmap] = useState({});
  const [loading, setLoading] = useState(true);

  const refresh = async () => {
    setLoading(true);
    try {
      const [p, pos, ord, hm] = await Promise.all([
        api.getPortfolio(), api.getPositions(), api.getOrders(), api.getPortfolioHeatmap(),
      ]);
      setPortfolio(p);
      setPositions(pos.positions || []);
      setOrders(ord.orders || []);
      setHeatmap(hm.categories || {});
    } catch (e) { console.error(e); }
    finally { setLoading(false); }
  };

  useEffect(() => { refresh(); }, []);

  const handleCancelOrder = async (orderId) => {
    try { await api.cancelOrder(orderId); refresh(); }
    catch (e) { alert(`Cancel failed: ${e.message}`); }
  };

  if (loading) return <div className="text-text-secondary text-xs p-8 text-center">Loading portfolio...</div>;

  return (
    <div className="space-y-6">
      {/* Balance */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <div className="bg-card border border-border rounded-lg p-5 card-hover">
          <div className="text-[10px] uppercase tracking-widest text-text-secondary mb-2">Cash Balance</div>
          <div className="text-3xl font-bold font-mono text-accent-green">
            ${portfolio ? (portfolio.balance_cents / 100).toFixed(2) : '0.00'}
          </div>
        </div>
        <div className="bg-card border border-border rounded-lg p-5 card-hover">
          <div className="text-[10px] uppercase tracking-widest text-text-secondary mb-2">Open Positions</div>
          <div className="text-3xl font-bold font-mono">{positions.length}</div>
        </div>
        <div className="bg-card border border-border rounded-lg p-5 card-hover">
          <div className="text-[10px] uppercase tracking-widest text-text-secondary mb-2">Open Orders</div>
          <div className="text-3xl font-bold font-mono">{orders.length}</div>
        </div>
      </div>

      {/* Positions */}
      <div className="bg-card border border-border rounded-lg overflow-hidden">
        <div className="px-5 py-3 border-b border-border flex items-center justify-between">
          <h3 className="text-[10px] uppercase tracking-widest text-text-secondary font-semibold">Positions</h3>
          <button onClick={refresh} className="text-[10px] uppercase tracking-widest text-accent-green hover:text-white transition-colors">Refresh</button>
        </div>
        {positions.length === 0 ? (
          <div className="p-8 text-center text-text-secondary text-xs">No open positions.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-text-secondary text-[10px] uppercase tracking-widest border-b border-border">
                  <th className="text-left px-4 py-2">Ticker</th>
                  <th className="text-left px-4 py-2">Market</th>
                  <th className="text-center px-4 py-2">Side</th>
                  <th className="text-right px-4 py-2">Qty</th>
                  <th className="text-right px-4 py-2">Avg</th>
                  <th className="text-right px-4 py-2">Current</th>
                  <th className="text-right px-4 py-2">P&L</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((pos, i) => (
                  <tr key={i} className="border-b border-border/50 hover:bg-surface/50">
                    <td className="px-4 py-2.5 font-mono text-white">{pos.ticker}</td>
                    <td className="px-4 py-2.5 max-w-[180px] truncate text-text-secondary">{pos.market_title || '--'}</td>
                    <td className="px-4 py-2.5 text-center">
                      <span className={`text-[10px] px-2 py-0.5 rounded font-semibold uppercase ${
                        pos.side === 'yes' ? 'bg-accent-green/10 text-accent-green' : 'bg-accent-red/10 text-accent-red'
                      }`}>{pos.side}</span>
                    </td>
                    <td className="px-4 py-2.5 text-right font-mono">{pos.quantity}</td>
                    <td className="px-4 py-2.5 text-right font-mono">{pos.avg_price_cents}c</td>
                    <td className="px-4 py-2.5 text-right font-mono">{pos.current_price_cents || '--'}c</td>
                    <td className="px-4 py-2.5 text-right font-mono">
                      {pos.unrealized_pnl_cents != null ? (
                        <span className={pos.unrealized_pnl_cents >= 0 ? 'text-accent-green' : 'text-accent-red'}>
                          {pos.unrealized_pnl_cents >= 0 ? '+' : ''}${(pos.unrealized_pnl_cents / 100).toFixed(2)}
                        </span>
                      ) : '--'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Category Heatmap */}
      {Object.keys(heatmap).length > 0 && (
        <div className="bg-card border border-border rounded-lg p-5">
          <h3 className="text-[10px] uppercase tracking-widest text-text-secondary font-semibold mb-4">Portfolio Heatmap</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
            {Object.entries(heatmap).map(([cat, data]) => {
              const totalValue = data.positions.reduce((s, p) => s + p.entry_value_cents, 0);
              return (
                <div key={cat} className="bg-surface border border-border rounded-lg p-3 card-hover">
                  <div className="text-[10px] uppercase tracking-widest text-text-muted mb-1">{cat}</div>
                  <div className="text-lg font-bold font-mono text-white">{data.count} pos</div>
                  <div className="text-[10px] text-text-secondary">${(totalValue / 100).toFixed(2)} invested</div>
                  <div className="mt-2 flex gap-1 flex-wrap">
                    {data.positions.map((p, i) => (
                      <span key={i} className="text-[9px] font-mono bg-accent-green/10 text-accent-green px-1.5 py-0.5 rounded">
                        {p.ticker}
                      </span>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Orders */}
      <div className="bg-card border border-border rounded-lg overflow-hidden">
        <div className="px-5 py-3 border-b border-border">
          <h3 className="text-[10px] uppercase tracking-widest text-text-secondary font-semibold">Open Orders</h3>
        </div>
        {orders.length === 0 ? (
          <div className="p-8 text-center text-text-secondary text-xs">No open orders.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-text-secondary text-[10px] uppercase tracking-widest border-b border-border">
                  <th className="text-left px-4 py-2">ID</th>
                  <th className="text-left px-4 py-2">Ticker</th>
                  <th className="text-center px-4 py-2">Side</th>
                  <th className="text-center px-4 py-2">Action</th>
                  <th className="text-right px-4 py-2">Price</th>
                  <th className="text-right px-4 py-2">Count</th>
                  <th className="text-center px-4 py-2"></th>
                </tr>
              </thead>
              <tbody>
                {orders.map((order, i) => (
                  <tr key={i} className="border-b border-border/50 hover:bg-surface/50">
                    <td className="px-4 py-2.5 font-mono text-text-secondary">{order.order_id?.slice(0, 12)}</td>
                    <td className="px-4 py-2.5 font-mono text-white">{order.ticker}</td>
                    <td className="px-4 py-2.5 text-center font-mono">{order.side?.toUpperCase()}</td>
                    <td className="px-4 py-2.5 text-center font-mono">{order.action?.toUpperCase()}</td>
                    <td className="px-4 py-2.5 text-right font-mono">{order.yes_price}c</td>
                    <td className="px-4 py-2.5 text-right font-mono">{order.remaining_count || order.count}</td>
                    <td className="px-4 py-2.5 text-center">
                      <button onClick={() => handleCancelOrder(order.order_id)}
                        className="px-3 py-1 bg-accent-red/10 text-accent-red text-[10px] font-semibold uppercase tracking-wide rounded hover:bg-accent-red/20 transition-all">
                        Cancel
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
