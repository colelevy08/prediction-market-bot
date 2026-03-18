/**
 * Portfolio.jsx — Portfolio overview with positions, orders, and category heatmap.
 */
import { useState, useEffect } from 'react';
import { api } from '../api';
import Tooltip from './Tooltip';
import { useToast } from './Toast';

export default function Portfolio() {
  const toast = useToast();
  const [portfolio, setPortfolio] = useState(null);
  const [positions, setPositions] = useState([]);
  const [orders, setOrders] = useState([]);
  const [heatmap, setHeatmap] = useState({});
  const [loading, setLoading] = useState(true);
  const [fetchError, setFetchError] = useState(null);

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
    } catch (e) { setFetchError('Failed to load portfolio'); }
    finally { setLoading(false); }
  };

  useEffect(() => { refresh(); }, []);

  const handleCancelOrder = async (orderId) => {
    try { await api.cancelOrder(orderId); refresh(); }
    catch (e) { toast.error(`Cancel failed: ${e.message}`); }
  };

  if (loading) return (
    <div className="card overflow-hidden">
      <div className="space-y-0">
        {[...Array(5)].map((_, i) => (
          <div key={i} className="flex items-center gap-4 px-4 py-3 border-b border-border last:border-0 animate-pulse">
            <div className="h-3 w-16 bg-surface-2 rounded" />
            <div className="h-3 w-32 bg-surface-2 rounded" />
            <div className="h-3 w-12 bg-surface-2 rounded ml-auto" />
            <div className="h-3 w-12 bg-surface-2 rounded" />
          </div>
        ))}
      </div>
    </div>
  );

  if (fetchError) return (
    <div className="card bg-accent-red/5 border-accent-red/20 p-4 text-xs text-accent-red text-center">
      {fetchError} <button onClick={refresh} className="underline ml-2">Retry</button>
    </div>
  );

  // Compute total unrealized P&L
  const totalUnrealizedPnl = positions.reduce((sum, p) => sum + (p.unrealized_pnl_cents || 0), 0);

  return (
    <div className="space-y-5 animate-fade-in">
      {/* Summary cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
        <div className="stat-card p-5" style={{ '--accent-color': 'rgb(var(--color-green))' }}>
          <div className="flex items-center justify-between mb-2">
            <Tooltip text="Cash available for trading, not including unrealized gains/losses">
              <span className="section-title">Cash Balance</span>
            </Tooltip>
            <span className="text-sm opacity-30">$</span>
          </div>
          <div className="text-3xl font-bold font-mono text-accent-green">
            ${portfolio ? (portfolio.balance_cents / 100).toFixed(2) : '0.00'}
          </div>
        </div>
        <div className="stat-card p-5" style={{ '--accent-color': 'rgb(var(--color-blue))' }}>
          <div className="flex items-center justify-between mb-2">
            <Tooltip text="Number of markets where the bot currently holds a position">
              <span className="section-title">Open Positions</span>
            </Tooltip>
            <span className="text-sm opacity-30">#</span>
          </div>
          <div className={`text-3xl font-bold font-mono ${positions.length > 0 ? 'text-accent-blue' : 'text-text-primary'}`}>
            {positions.length}
          </div>
        </div>
        <div className="stat-card p-5" style={{ '--accent-color': 'rgb(var(--color-purple))' }}>
          <div className="flex items-center justify-between mb-2">
            <Tooltip text="Pending limit orders waiting to be filled">
              <span className="section-title">Open Orders</span>
            </Tooltip>
            <span className="text-sm opacity-30">⌀</span>
          </div>
          <div className={`text-3xl font-bold font-mono ${orders.length > 0 ? 'text-accent-purple' : 'text-text-primary'}`}>
            {orders.length}
          </div>
        </div>
        <div className="stat-card p-5" style={{ '--accent-color': totalUnrealizedPnl >= 0 ? 'rgb(var(--color-green))' : 'rgb(var(--color-red))' }}>
          <div className="flex items-center justify-between mb-2">
            <Tooltip text="Sum of unrealized P&L across all open positions">
              <span className="section-title">Unrealized P&L</span>
            </Tooltip>
            <span className="text-sm opacity-30">{totalUnrealizedPnl >= 0 ? '↑' : '↓'}</span>
          </div>
          <div className={`text-3xl font-bold font-mono ${totalUnrealizedPnl > 0 ? 'text-accent-green' : totalUnrealizedPnl < 0 ? 'text-accent-red' : 'text-text-primary'}`}>
            {totalUnrealizedPnl >= 0 ? '+' : ''}${(totalUnrealizedPnl / 100).toFixed(2)}
          </div>
        </div>
      </div>

      {/* Positions */}
      <div className="card overflow-hidden">
        <div className="px-5 py-3 border-b border-border flex items-center justify-between">
          <h3 className="section-title">Positions</h3>
          <button onClick={refresh} className="btn-ghost">Refresh</button>
        </div>
        {positions.length === 0 ? (
          <div className="p-10 text-center text-text-muted text-xs">No open positions.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border">
                  <th className="table-header table-cell text-left">Ticker</th>
                  <th className="table-header table-cell text-left">Market</th>
                  <th className="table-header table-cell text-center">
                    <Tooltip text="YES = betting the event will happen, NO = betting it won't">Side</Tooltip>
                  </th>
                  <th className="table-header table-cell text-right">
                    <Tooltip text="Number of contracts held">Qty</Tooltip>
                  </th>
                  <th className="table-header table-cell text-right">
                    <Tooltip text="Average price paid per contract (in cents)">Avg</Tooltip>
                  </th>
                  <th className="table-header table-cell text-right">
                    <Tooltip text="Current market price per contract (in cents)">Current</Tooltip>
                  </th>
                  <th className="table-header table-cell text-right">
                    <Tooltip text="Unrealized profit/loss if the position were closed now">P&L</Tooltip>
                  </th>
                </tr>
              </thead>
              <tbody>
                {positions.map((pos, i) => (
                  <tr key={i} className={`table-row ${
                    pos.unrealized_pnl_cents > 0 ? 'row-win' : pos.unrealized_pnl_cents < 0 ? 'row-loss' : ''
                  }`}>
                    <td className="table-cell font-mono text-text-primary">{pos.ticker}</td>
                    <td className="table-cell max-w-[180px] truncate text-text-secondary">{pos.market_title || '--'}</td>
                    <td className="table-cell text-center">
                      <span className={`badge ${pos.side === 'yes' ? 'badge-green' : 'badge-red'}`}>{pos.side}</span>
                    </td>
                    <td className="table-cell text-right font-mono text-accent-blue">{pos.quantity}</td>
                    <td className="table-cell text-right font-mono text-text-secondary">{pos.avg_price_cents}¢</td>
                    <td className="table-cell text-right font-mono text-text-secondary">{pos.current_price_cents || '--'}¢</td>
                    <td className="table-cell text-right font-mono">
                      {pos.unrealized_pnl_cents != null ? (
                        <span className={`badge ${pos.unrealized_pnl_cents >= 0 ? 'badge-green' : 'badge-red'}`}>
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
        <div className="card p-5">
          <h3 className="section-title mb-4">
            <Tooltip text="Positions grouped by market category showing concentration and invested amounts">
              Portfolio Heatmap
            </Tooltip>
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
            {Object.entries(heatmap).map(([cat, data]) => {
              const totalValue = data.positions.reduce((s, p) => s + p.entry_value_cents, 0);
              return (
                <div key={cat} className="card-interactive p-3">
                  <div className="text-[10px] uppercase tracking-widest text-text-muted mb-1">{cat}</div>
                  <div className="text-lg font-bold font-mono text-text-primary">{data.count} <span className="text-xs text-text-secondary font-normal">pos</span></div>
                  <div className="text-[10px] text-text-secondary">${(totalValue / 100).toFixed(2)} invested</div>
                  <div className="mt-2 flex gap-1 flex-wrap">
                    {data.positions.map((p, i) => (
                      <span key={i} className="badge badge-green text-[9px]">{p.ticker}</span>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Orders */}
      <div className="card overflow-hidden">
        <div className="px-5 py-3 border-b border-border">
          <h3 className="section-title">Open Orders</h3>
        </div>
        {orders.length === 0 ? (
          <div className="p-10 text-center text-text-muted text-xs">No open orders.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border">
                  <th className="table-header table-cell text-left">
                    <Tooltip text="Unique identifier for this order">ID</Tooltip>
                  </th>
                  <th className="table-header table-cell text-left">Ticker</th>
                  <th className="table-header table-cell text-center">Side</th>
                  <th className="table-header table-cell text-center">
                    <Tooltip text="BUY = opening a new position, SELL = closing an existing one">Action</Tooltip>
                  </th>
                  <th className="table-header table-cell text-right">
                    <Tooltip text="Limit price for this order (in cents)">Price</Tooltip>
                  </th>
                  <th className="table-header table-cell text-right">
                    <Tooltip text="Number of contracts remaining to be filled">Count</Tooltip>
                  </th>
                  <th className="table-header table-cell text-center"></th>
                </tr>
              </thead>
              <tbody>
                {orders.map((order, i) => (
                  <tr key={i} className="table-row">
                    <td className="table-cell font-mono text-text-muted text-[10px]">{order.order_id?.slice(0, 12)}</td>
                    <td className="table-cell font-mono text-text-primary">{order.ticker}</td>
                    <td className="table-cell text-center">
                      <span className={`badge ${order.side === 'yes' ? 'badge-green' : 'badge-red'}`}>{order.side?.toUpperCase()}</span>
                    </td>
                    <td className="table-cell text-center font-mono text-text-secondary">{order.action?.toUpperCase()}</td>
                    <td className="table-cell text-right font-mono">{order.yes_price}¢</td>
                    <td className="table-cell text-right font-mono">{order.remaining_count || order.count}</td>
                    <td className="table-cell text-center">
                      <button onClick={() => handleCancelOrder(order.order_id)} className="btn-danger text-[10px] px-3 py-1">
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
