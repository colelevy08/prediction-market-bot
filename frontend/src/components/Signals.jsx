/**
 * Signals.jsx — Signal discovery and trade execution panel.
 */
import { useState, useEffect } from 'react';
import { api } from '../api';
import Tooltip from './Tooltip';

export default function Signals({ scanResult, onScan, scanning }) {
  const [signals, setSignals] = useState([]);
  const [exitSignals, setExitSignals] = useState([]);
  const [trading, setTrading] = useState({});
  const [filter, setFilter] = useState('all');

  useEffect(() => {
    if (scanResult) {
      setSignals(scanResult.signals || []);
      setExitSignals(scanResult.exit_signals_data || []);
    } else {
      api.getSignals().then(data => {
        setSignals(data.signals || []);
        setExitSignals(data.exit_signals || []);
      }).catch(() => {});
    }
  }, [scanResult]);

  const filteredSignals = signals.filter(s => {
    if (filter === 'ready') return s.risk_check?.allowed;
    if (filter === 'rf') return s.source === 'random_forest';
    if (filter === 'ai') return s.source === 'claude_ai';
    return true;
  });

  const handleTrade = async (signal) => {
    const key = signal.ticker;
    setTrading(prev => ({ ...prev, [key]: true }));
    try {
      const priceCents = Math.round(signal.market_probability * 100);
      await api.placeTrade(signal.ticker, signal.side, priceCents, 1);
      setSignals(prev => prev.map(s => s.ticker === key ? { ...s, _executed: true } : s));
    } catch (e) {
      alert(`Trade failed: ${e.message}`);
    } finally {
      setTrading(prev => ({ ...prev, [key]: false }));
    }
  };

  return (
    <div className="space-y-5 animate-fade-in">
      {/* Controls */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-2">
          <button onClick={() => onScan(false)} disabled={scanning} className="btn-primary">
            {scanning ? 'Scanning...' : 'RF Scan'}
          </button>
          <button onClick={() => onScan(true)} disabled={scanning} className="btn-secondary">
            {scanning ? 'Scanning...' : 'RF + AI'}
          </button>
        </div>

        {/* Filter pills */}
        <div className="flex items-center gap-0.5 bg-surface-2 border border-border rounded-lg p-0.5">
          {[['all', 'All'], ['ready', 'Ready'], ['rf', 'RF'], ['ai', 'AI']].map(([id, label]) => (
            <button
              key={id}
              onClick={() => setFilter(id)}
              className={`px-3 py-1.5 text-[10px] font-semibold tracking-wide uppercase rounded-md transition-all ${
                filter === id
                  ? 'bg-accent-green text-black'
                  : 'text-text-muted hover:text-text-secondary'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Entry Signals */}
      <div className="card overflow-hidden">
        <div className="px-5 py-3 border-b border-border flex items-center justify-between">
          <h3 className="section-title">
            Entry Signals ({filteredSignals.length})
            <span className="font-normal ml-2 normal-case tracking-normal text-text-muted">
              mkt ≤ model × 0.5
            </span>
          </h3>
        </div>

        {filteredSignals.length === 0 ? (
          <div className="p-10 text-center text-text-muted text-xs">
            No signals. Run a scan to analyze markets.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border">
                  <th className="table-header table-cell text-left">Ticker</th>
                  <th className="table-header table-cell text-left">Market</th>
                  <th className="table-header table-cell text-center">
                    <Tooltip text="YES or NO — which side the bot recommends buying">Side</Tooltip>
                  </th>
                  <th className="table-header table-cell text-center">
                    <Tooltip text="Signal source: RF = Random Forest model, AI = Claude analysis">Src</Tooltip>
                  </th>
                  <th className="table-header table-cell text-right">
                    <Tooltip text="Probability estimated by the model">Model</Tooltip>
                  </th>
                  <th className="table-header table-cell text-right">
                    <Tooltip text="Current implied probability from market price">Market</Tooltip>
                  </th>
                  <th className="table-header table-cell text-right">
                    <Tooltip text="Model probability minus market price — higher means more undervalued">Edge</Tooltip>
                  </th>
                  <th className="table-header table-cell text-right">
                    <Tooltip text="Model confidence in this prediction (70%+ required)">Conf</Tooltip>
                  </th>
                  <th className="table-header table-cell text-right">
                    <Tooltip text="Recommended position size based on Kelly Criterion">Size</Tooltip>
                  </th>
                  <th className="table-header table-cell text-center">
                    <Tooltip text="Risk check status — Ready means all risk limits pass">Status</Tooltip>
                  </th>
                  <th className="table-header table-cell text-center">Action</th>
                </tr>
              </thead>
              <tbody>
                {filteredSignals.map((sig, i) => {
                  const edgePct = (sig.edge * 100);
                  return (
                    <tr key={`${sig.ticker}-${i}`} className="table-row">
                      <td className="table-cell font-mono text-text-primary text-[11px]">{sig.ticker}</td>
                      <td className="table-cell max-w-[180px] truncate text-text-secondary" title={sig.market_title}>
                        {sig.market_title}
                      </td>
                      <td className="table-cell text-center">
                        <span className={`badge ${sig.side === 'yes' ? 'badge-green' : 'badge-red'}`}>{sig.side}</span>
                      </td>
                      <td className="table-cell text-center">
                        <span className={`badge ${sig.source === 'random_forest' ? 'badge-cyan' : 'badge-purple'}`}>
                          {sig.source === 'random_forest' ? 'RF' : 'AI'}
                        </span>
                      </td>
                      <td className="table-cell text-right font-mono text-accent-blue">{(sig.fair_probability * 100).toFixed(1)}%</td>
                      <td className="table-cell text-right font-mono text-text-secondary">{(sig.market_probability * 100).toFixed(1)}%</td>
                      <td className="table-cell text-right font-mono">
                        <span className={`badge ${
                          edgePct >= 15 ? 'badge-green' : edgePct >= 8 ? 'badge-yellow' : edgePct > 0 ? 'badge-muted' : 'badge-red'
                        }`}>
                          {sig.edge > 0 ? '+' : ''}{edgePct.toFixed(1)}%
                        </span>
                      </td>
                      <td className="table-cell text-right font-mono">
                        <span className={
                          sig.confidence >= 0.8 ? 'text-accent-green' :
                          sig.confidence >= 0.7 ? 'text-accent-yellow' :
                          'text-accent-red'
                        }>{(sig.confidence * 100).toFixed(0)}%</span>
                      </td>
                      <td className="table-cell text-right font-mono text-accent-cyan">${(sig.recommended_size_cents / 100).toFixed(2)}</td>
                      <td className="table-cell text-center">
                        {sig._executed ? <span className="badge badge-green">Done</span>
                          : sig.risk_check?.allowed ? <span className="badge badge-green">Ready</span>
                          : <span className="text-[10px] text-text-muted">{sig.risk_check?.reason?.split(' ').slice(0, 3).join(' ')}</span>
                        }
                      </td>
                      <td className="table-cell text-center">
                        {sig.risk_check?.allowed && !sig._executed && (
                          <button
                            onClick={() => handleTrade(sig)}
                            disabled={trading[sig.ticker]}
                            className="px-3 py-1 bg-accent-green/10 text-accent-green text-[10px] font-semibold uppercase tracking-wide rounded-md hover:bg-accent-green/20 disabled:opacity-30 transition-all"
                          >
                            {trading[sig.ticker] ? '...' : 'Trade'}
                          </button>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Exit Signals */}
      {exitSignals.length > 0 && (
        <div className="card overflow-hidden animate-slide-up">
          <div className="px-5 py-3 border-b border-border">
            <h3 className="text-xs font-semibold uppercase tracking-widest text-accent-red">
              Exit Signals ({exitSignals.length})
              <span className="font-normal ml-2 normal-case tracking-normal text-text-muted">
                mkt ≥ model × 0.9 or expiry &lt; 7d
              </span>
            </h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border">
                  <th className="table-header table-cell text-left">Ticker</th>
                  <th className="table-header table-cell text-left">Market</th>
                  <th className="table-header table-cell text-right">
                    <Tooltip text="Probability estimated by the model">Model</Tooltip>
                  </th>
                  <th className="table-header table-cell text-right">
                    <Tooltip text="Current market-implied probability">Market</Tooltip>
                  </th>
                  <th className="table-header table-cell text-left">Reason</th>
                </tr>
              </thead>
              <tbody>
                {exitSignals.map((sig, i) => (
                  <tr key={i} className="table-row">
                    <td className="table-cell font-mono text-accent-red">{sig.ticker}</td>
                    <td className="table-cell text-text-secondary">{sig.market_title}</td>
                    <td className="table-cell text-right font-mono">{(sig.fair_probability * 100).toFixed(1)}%</td>
                    <td className="table-cell text-right font-mono">{(sig.market_probability * 100).toFixed(1)}%</td>
                    <td className="table-cell text-accent-red/80">{sig.reasoning}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Reasoning */}
      {filteredSignals.length > 0 && (
        <div className="card p-5">
          <h3 className="section-title mb-3">Signal Reasoning</h3>
          <div className="space-y-1.5">
            {filteredSignals.slice(0, 5).map((sig, i) => (
              <div key={i} className="bg-surface-2 border border-border-subtle rounded-lg p-3 text-xs font-mono card-interactive">
                <span className="text-text-primary font-semibold">{sig.ticker}</span>
                <span className="text-text-secondary ml-2">{sig.reasoning}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
