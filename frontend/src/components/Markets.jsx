/*
 * Markets Component
 *
 * Displays a browsable, sortable table of prediction market events and their
 * constituent markets. Includes an inline MarketDetail sub-component for
 * deep-diving into a single market's analysis and trading signals.
 *
 * Key Sections:
 *   - Markets Table: All markets across fetched events, flattened and sorted by
 *     volume descending. Columns: ticker, market title, parent event title,
 *     YES bid, YES ask, spread, volume, and open interest. Each row has an
 *     "Analyze" button that opens the detail view.
 *   - MarketDetail (inline sub-component): Shows model probability vs. market
 *     probability, calculated edge, bid/ask spread, entry/exit signal indicators,
 *     order book stats (YES bid, YES ask, volume, OI), and an expandable list
 *     of all model features.
 *   - Controls: A dropdown to set the event fetch limit (10/20/50) and a
 *     refresh button.
 *
 * API Endpoints Called:
 *   - api.getEvents(limit)   — Fetches a list of events (each containing markets)
 *                               with a configurable limit.
 *   - api.getMarket(ticker)  — Fetches detailed market data and model analysis
 *                               for a single ticker (used by MarketDetail).
 *
 * Data Displayed:
 *   - Events: title, nested markets array.
 *   - Markets: ticker, title, yes_bid, yes_ask, volume, open_interest.
 *   - Analysis (detail view): model_probability, market_probability, edge,
 *     entry_signal, exit_signal, entry_threshold, exit_threshold, and features map.
 */
import { useState, useEffect } from 'react';
import { api } from '../api';
import Tooltip from './Tooltip';

function MarketDetail({ ticker, onClose }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getMarket(ticker).then(setData).catch(() => {}).finally(() => setLoading(false));
  }, [ticker]);

  if (loading) return <div className="p-4 text-text-secondary text-xs">Loading...</div>;
  if (!data) return <div className="p-4 text-text-secondary text-xs">Failed to load.</div>;

  const { market, analysis } = data;

  return (
    <div className="card p-5 space-y-4 border-accent-green/30 shadow-[0_0_15px_rgba(52,211,153,0.08)]">
      <div className="flex items-start justify-between">
        <div>
          <div className="text-[10px] uppercase tracking-widest text-accent-green font-mono">{market.ticker}</div>
          <div className="text-lg font-semibold mt-1">{market.title}</div>
        </div>
        <button onClick={onClose} className="btn-ghost">Close</button>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="stat-card text-center" style={{ '--accent-color': 'rgb(var(--color-green))' }}>
          <div className="text-[10px] uppercase tracking-widest text-text-secondary">
            <Tooltip text="Probability estimated by the Random Forest model">Model</Tooltip>
          </div>
          <div className="text-2xl font-bold font-mono text-white">{(analysis.model_probability * 100).toFixed(1)}%</div>
        </div>
        <div className="stat-card text-center" style={{ '--accent-color': 'rgb(var(--color-blue))' }}>
          <div className="text-[10px] uppercase tracking-widest text-text-secondary">
            <Tooltip text="Current implied probability from market price">Market</Tooltip>
          </div>
          <div className="text-2xl font-bold font-mono">{(analysis.market_probability * 100).toFixed(1)}%</div>
        </div>
        <div className="stat-card text-center" style={{ '--accent-color': analysis.edge > 0 ? 'rgb(var(--color-green))' : 'rgb(var(--color-red))' }}>
          <div className="text-[10px] uppercase tracking-widest text-text-secondary">
            <Tooltip text="Model probability minus market price — positive means undervalued">Edge</Tooltip>
          </div>
          <div className={`text-2xl font-bold font-mono ${analysis.edge > 0 ? 'text-accent-green' : 'text-accent-red'}`}>
            {analysis.edge > 0 ? '+' : ''}{(analysis.edge * 100).toFixed(1)}%
          </div>
        </div>
        <div className="stat-card text-center" style={{ '--accent-color': 'rgb(var(--color-purple))' }}>
          <div className="text-[10px] uppercase tracking-widest text-text-secondary">
            <Tooltip text="Ask minus Bid in cents — lower means tighter market">Spread</Tooltip>
          </div>
          <div className="text-2xl font-bold font-mono">{market.yes_ask - market.yes_bid}c</div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className={`card p-3 ${analysis.entry_signal ? 'bg-accent-green/5 border-accent-green/20 glow-green' : ''}`}>
          <div className="text-[10px] uppercase tracking-widest text-text-secondary mb-1">Entry: mkt &le; model &times; 0.5</div>
          <div className="font-mono text-xs">
            {(analysis.market_probability * 100).toFixed(0)}% &le; {(analysis.entry_threshold * 100).toFixed(0)}%
            <span className={`ml-2 font-bold ${analysis.entry_signal ? 'text-accent-green' : 'text-text-muted'}`}>
              {analysis.entry_signal ? 'BUY SIGNAL' : 'No entry'}
            </span>
          </div>
        </div>
        <div className={`card p-3 ${analysis.exit_signal ? 'bg-accent-red/5 border-accent-red/20 glow-red' : ''}`}>
          <div className="text-[10px] uppercase tracking-widest text-text-secondary mb-1">Exit: mkt &ge; model &times; 0.9</div>
          <div className="font-mono text-xs">
            {(analysis.market_probability * 100).toFixed(0)}% &ge; {(analysis.exit_threshold * 100).toFixed(0)}%
            <span className={`ml-2 font-bold ${analysis.exit_signal ? 'text-accent-red' : 'text-text-muted'}`}>
              {analysis.exit_signal ? 'SELL SIGNAL' : 'No exit'}
            </span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-2 text-xs">
        {[
          { label: 'YES Bid', tooltip: 'Highest price a buyer will pay for YES shares (in cents)', val: `${market.yes_bid}c`, color: 'text-accent-green' },
          { label: 'YES Ask', tooltip: 'Lowest price a seller will accept for YES shares (in cents)', val: `${market.yes_ask}c`, color: 'text-accent-red' },
          { label: 'Volume', tooltip: 'Total number of contracts traded on this market', val: market.volume?.toLocaleString() },
          { label: 'OI', tooltip: 'Open Interest — total outstanding contracts not yet settled', val: market.open_interest?.toLocaleString() },
        ].map((item, i) => (
          <div key={i} className="card p-2 text-center">
            <div className="text-[9px] uppercase tracking-widest text-text-secondary">
              <Tooltip text={item.tooltip}>{item.label}</Tooltip>
            </div>
            <div className={`font-mono ${item.color || 'text-white'}`}>{item.val}</div>
          </div>
        ))}
      </div>

      <details className="text-xs">
        <summary className="text-text-secondary cursor-pointer hover:text-white transition-colors">
          All {Object.keys(analysis.features || {}).length} features
        </summary>
        <div className="mt-2 grid grid-cols-2 md:grid-cols-4 gap-1.5 font-mono">
          {Object.entries(analysis.features || {}).map(([k, v]) => (
            <div key={k} className="badge badge-muted gap-1">
              <span className="text-text-secondary">{k}:</span> <span className="text-text-primary">{v}</span>
            </div>
          ))}
        </div>
      </details>
    </div>
  );
}

function volumeColor(vol) {
  if (vol > 10000) return 'text-accent-green';
  if (vol > 1000) return 'text-accent-blue';
  return 'text-text-muted';
}

const SORT_OPTIONS = [
  { id: 'volume', label: 'Volume', fn: (a, b) => b.volume - a.volume },
  { id: 'oi', label: 'Open Interest', fn: (a, b) => (b.open_interest || 0) - (a.open_interest || 0) },
  { id: 'spread', label: 'Spread (tightest)', fn: (a, b) => (a.yes_ask - a.yes_bid) - (b.yes_ask - b.yes_bid) },
  { id: 'bid_high', label: 'Bid (highest)', fn: (a, b) => b.yes_bid - a.yes_bid },
  { id: 'bid_low', label: 'Bid (lowest)', fn: (a, b) => a.yes_bid - b.yes_bid },
  { id: 'status', label: 'Status', fn: (a, b) => (a.status || '').localeCompare(b.status || '') },
];

export default function Markets() {
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedTicker, setSelectedTicker] = useState(null);
  const [limit, setLimit] = useState(20);
  const [sortBy, setSortBy] = useState('volume');
  const [statusFilter, setStatusFilter] = useState('all');

  const loadEvents = async () => {
    setLoading(true);
    try { const data = await api.getEvents(limit); setEvents(data.events || []); }
    catch (e) { console.error(e); }
    finally { setLoading(false); }
  };

  useEffect(() => { loadEvents(); }, [limit]);

  const sortFn = SORT_OPTIONS.find(s => s.id === sortBy)?.fn || SORT_OPTIONS[0].fn;
  const allMarkets = events.flatMap(e =>
    (e.markets || []).map(m => ({ ...m, event_title: e.title }))
  ).filter(m => statusFilter === 'all' || m.status === statusFilter)
   .sort(sortFn);

  const statuses = [...new Set(events.flatMap(e => (e.markets || []).map(m => m.status)).filter(Boolean))];

  return (
    <div className="space-y-6">
      {selectedTicker && <MarketDetail ticker={selectedTicker} onClose={() => setSelectedTicker(null)} />}

      <div className="flex items-center justify-between flex-wrap gap-3">
        <h2 className="section-title">
          Markets ({allMarkets.length})
        </h2>
        <div className="flex items-center gap-2 flex-wrap">
          {/* Status filter pills */}
          <div className="flex items-center gap-0.5 bg-surface-2 border border-border rounded-lg p-0.5">
            <button onClick={() => setStatusFilter('all')}
              className={`px-2.5 py-1 text-[10px] font-semibold tracking-wide uppercase rounded-md transition-all ${
                statusFilter === 'all' ? 'bg-accent-green text-black' : 'text-text-muted hover:text-text-secondary'
              }`}>All</button>
            {statuses.map(s => (
              <button key={s} onClick={() => setStatusFilter(s)}
                className={`px-2.5 py-1 text-[10px] font-semibold tracking-wide uppercase rounded-md transition-all ${
                  statusFilter === s ? 'bg-accent-green text-black' : 'text-text-muted hover:text-text-secondary'
                }`}>{s}</button>
            ))}
          </div>
          {/* Sort */}
          <select value={sortBy} onChange={e => setSortBy(e.target.value)}
            className="input w-auto py-1.5 text-xs">
            {SORT_OPTIONS.map(s => (
              <option key={s.id} value={s.id}>{s.label}</option>
            ))}
          </select>
          <select value={limit} onChange={e => setLimit(+e.target.value)}
            className="input w-auto py-1.5 text-xs">
            <option value={10}>10 events</option>
            <option value={20}>20 events</option>
            <option value={50}>50 events</option>
          </select>
          <button onClick={loadEvents} disabled={loading}
            className="btn-secondary py-1.5">
            {loading ? '...' : 'Refresh'}
          </button>
        </div>
      </div>

      <div className="card overflow-hidden">
        {loading ? (
          <div className="p-8 text-center text-text-secondary text-xs">Loading...</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="table-header border-b border-border">
                  <th className="text-left table-cell">Ticker</th>
                  <th className="text-left table-cell">Market</th>
                  <th className="text-left table-cell">Event</th>
                  <th className="text-right table-cell">
                    <Tooltip text="Highest price a buyer will pay for YES shares (in cents)">Bid</Tooltip>
                  </th>
                  <th className="text-right table-cell">
                    <Tooltip text="Lowest price a seller will accept for YES shares (in cents)">Ask</Tooltip>
                  </th>
                  <th className="text-right table-cell">
                    <Tooltip text="Difference between Ask and Bid — lower means more liquid">Spread</Tooltip>
                  </th>
                  <th className="text-right table-cell">
                    <Tooltip text="Total number of contracts traded on this market">Vol</Tooltip>
                  </th>
                  <th className="text-right table-cell">
                    <Tooltip text="Open Interest — total outstanding contracts not yet settled">OI</Tooltip>
                  </th>
                  <th className="text-center table-cell">
                    <Tooltip text="Market status — open, closed, or settled">Status</Tooltip>
                  </th>
                  <th className="text-center table-cell"></th>
                </tr>
              </thead>
              <tbody>
                {allMarkets.map((m, i) => (
                  <tr key={`${m.ticker}-${i}`} className="table-row even:bg-surface-2/30">
                    <td className="table-cell font-mono text-white text-[10px]">{m.ticker}</td>
                    <td className="table-cell max-w-[180px] truncate" title={m.title}>{m.title}</td>
                    <td className="table-cell max-w-[140px] truncate text-text-secondary" title={m.event_title}>{m.event_title}</td>
                    <td className="table-cell text-right font-mono text-accent-green">{m.yes_bid}c</td>
                    <td className="table-cell text-right font-mono text-accent-red">{m.yes_ask}c</td>
                    <td className="table-cell text-right font-mono">{m.yes_ask - m.yes_bid}c</td>
                    <td className={`table-cell text-right font-mono ${volumeColor(m.volume)}`}>{m.volume?.toLocaleString()}</td>
                    <td className="table-cell text-right font-mono">{m.open_interest?.toLocaleString()}</td>
                    <td className="table-cell text-center">
                      <span className={`badge ${
                        m.status === 'open' ? 'badge-green' :
                        m.status === 'closed' ? 'badge-yellow' :
                        m.status === 'settled' ? 'badge-muted' : 'badge-muted'
                      }`}>{m.status || '--'}</span>
                    </td>
                    <td className="table-cell text-center">
                      <button onClick={() => setSelectedTicker(m.ticker)}
                        className="btn-ghost">
                        Analyze
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
