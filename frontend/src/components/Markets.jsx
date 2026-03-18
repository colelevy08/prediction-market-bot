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
    <div className="bg-card border border-accent-green/20 rounded-lg p-5 space-y-4">
      <div className="flex items-start justify-between">
        <div>
          <div className="text-[10px] uppercase tracking-widest text-accent-green font-mono">{market.ticker}</div>
          <div className="text-lg font-semibold mt-1">{market.title}</div>
        </div>
        <button onClick={onClose} className="text-text-secondary hover:text-white text-xs transition-colors">Close</button>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="bg-surface border border-border rounded-lg p-3 text-center">
          <div className="text-[10px] uppercase tracking-widest text-text-secondary">Model</div>
          <div className="text-2xl font-bold font-mono text-white">{(analysis.model_probability * 100).toFixed(1)}%</div>
        </div>
        <div className="bg-surface border border-border rounded-lg p-3 text-center">
          <div className="text-[10px] uppercase tracking-widest text-text-secondary">Market</div>
          <div className="text-2xl font-bold font-mono">{(analysis.market_probability * 100).toFixed(1)}%</div>
        </div>
        <div className="bg-surface border border-border rounded-lg p-3 text-center">
          <div className="text-[10px] uppercase tracking-widest text-text-secondary">Edge</div>
          <div className={`text-2xl font-bold font-mono ${analysis.edge > 0 ? 'text-accent-green' : 'text-accent-red'}`}>
            {analysis.edge > 0 ? '+' : ''}{(analysis.edge * 100).toFixed(1)}%
          </div>
        </div>
        <div className="bg-surface border border-border rounded-lg p-3 text-center">
          <div className="text-[10px] uppercase tracking-widest text-text-secondary">Spread</div>
          <div className="text-2xl font-bold font-mono">{market.yes_ask - market.yes_bid}c</div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className={`rounded-lg p-3 border ${analysis.entry_signal ? 'bg-accent-green/5 border-accent-green/20 glow-green' : 'bg-surface border-border'}`}>
          <div className="text-[10px] uppercase tracking-widest text-text-secondary mb-1">Entry: mkt &le; model &times; 0.5</div>
          <div className="font-mono text-xs">
            {(analysis.market_probability * 100).toFixed(0)}% &le; {(analysis.entry_threshold * 100).toFixed(0)}%
            <span className={`ml-2 font-bold ${analysis.entry_signal ? 'text-accent-green' : 'text-text-muted'}`}>
              {analysis.entry_signal ? 'BUY SIGNAL' : 'No entry'}
            </span>
          </div>
        </div>
        <div className={`rounded-lg p-3 border ${analysis.exit_signal ? 'bg-accent-red/5 border-accent-red/20 glow-red' : 'bg-surface border-border'}`}>
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
          { label: 'YES Bid', val: `${market.yes_bid}c`, color: 'text-accent-green' },
          { label: 'YES Ask', val: `${market.yes_ask}c`, color: 'text-accent-red' },
          { label: 'Volume', val: market.volume?.toLocaleString() },
          { label: 'OI', val: market.open_interest?.toLocaleString() },
        ].map((item, i) => (
          <div key={i} className="bg-surface border border-border rounded p-2 text-center">
            <div className="text-[9px] uppercase tracking-widest text-text-secondary">{item.label}</div>
            <div className={`font-mono ${item.color || 'text-white'}`}>{item.val}</div>
          </div>
        ))}
      </div>

      <details className="text-xs">
        <summary className="text-text-secondary cursor-pointer hover:text-white transition-colors">
          All {Object.keys(analysis.features || {}).length} features
        </summary>
        <div className="mt-2 grid grid-cols-2 md:grid-cols-4 gap-1 font-mono">
          {Object.entries(analysis.features || {}).map(([k, v]) => (
            <div key={k} className="bg-surface border border-border rounded px-2 py-1">
              <span className="text-text-secondary">{k}:</span> <span className="text-white">{v}</span>
            </div>
          ))}
        </div>
      </details>
    </div>
  );
}

export default function Markets() {
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedTicker, setSelectedTicker] = useState(null);
  const [limit, setLimit] = useState(20);

  const loadEvents = async () => {
    setLoading(true);
    try { const data = await api.getEvents(limit); setEvents(data.events || []); }
    catch (e) { console.error(e); }
    finally { setLoading(false); }
  };

  useEffect(() => { loadEvents(); }, [limit]);

  const allMarkets = events.flatMap(e =>
    (e.markets || []).map(m => ({ ...m, event_title: e.title }))
  ).sort((a, b) => b.volume - a.volume);

  return (
    <div className="space-y-6">
      {selectedTicker && <MarketDetail ticker={selectedTicker} onClose={() => setSelectedTicker(null)} />}

      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-widest text-text-secondary">
          Markets ({allMarkets.length})
        </h2>
        <div className="flex items-center gap-2">
          <select value={limit} onChange={e => setLimit(+e.target.value)}
            className="bg-card border border-border rounded-lg px-3 py-1.5 text-xs text-white">
            <option value={10}>10</option>
            <option value={20}>20</option>
            <option value={50}>50</option>
          </select>
          <button onClick={loadEvents} disabled={loading}
            className="px-4 py-1.5 bg-white text-black text-xs font-semibold rounded-lg hover:bg-gray-200 disabled:opacity-30 transition-all uppercase tracking-wide">
            {loading ? '...' : 'Refresh'}
          </button>
        </div>
      </div>

      <div className="bg-card border border-border rounded-lg overflow-hidden">
        {loading ? (
          <div className="p-8 text-center text-text-secondary text-xs">Loading...</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-text-secondary text-[10px] uppercase tracking-widest border-b border-border">
                  <th className="text-left px-4 py-2">Ticker</th>
                  <th className="text-left px-4 py-2">Market</th>
                  <th className="text-left px-4 py-2">Event</th>
                  <th className="text-right px-4 py-2">Bid</th>
                  <th className="text-right px-4 py-2">Ask</th>
                  <th className="text-right px-4 py-2">Spread</th>
                  <th className="text-right px-4 py-2">Vol</th>
                  <th className="text-right px-4 py-2">OI</th>
                  <th className="text-center px-4 py-2"></th>
                </tr>
              </thead>
              <tbody>
                {allMarkets.map((m, i) => (
                  <tr key={`${m.ticker}-${i}`} className="border-b border-border/50 hover:bg-surface/50">
                    <td className="px-4 py-2.5 font-mono text-white text-[10px]">{m.ticker}</td>
                    <td className="px-4 py-2.5 max-w-[180px] truncate" title={m.title}>{m.title}</td>
                    <td className="px-4 py-2.5 max-w-[140px] truncate text-text-secondary" title={m.event_title}>{m.event_title}</td>
                    <td className="px-4 py-2.5 text-right font-mono text-accent-green">{m.yes_bid}c</td>
                    <td className="px-4 py-2.5 text-right font-mono text-accent-red">{m.yes_ask}c</td>
                    <td className="px-4 py-2.5 text-right font-mono">{m.yes_ask - m.yes_bid}c</td>
                    <td className="px-4 py-2.5 text-right font-mono">{m.volume?.toLocaleString()}</td>
                    <td className="px-4 py-2.5 text-right font-mono">{m.open_interest?.toLocaleString()}</td>
                    <td className="px-4 py-2.5 text-center">
                      <button onClick={() => setSelectedTicker(m.ticker)}
                        className="px-3 py-1 bg-white/5 text-white text-[10px] font-semibold uppercase tracking-wide rounded hover:bg-white/10 transition-all">
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
