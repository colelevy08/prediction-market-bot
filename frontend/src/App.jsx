import React, { useState, useEffect, useCallback } from 'react'
import {
  AreaChart, Area, BarChart, Bar, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, Cell, PieChart, Pie
} from 'recharts'

const BASELINE = 35300   // $353.00 — reset 2026-04-04
const REFRESH  = 5000    // 5s portfolio/trades/perf refresh
const FAST_REFRESH = 3000 // 3s signals/activity refresh (was 2s — 2s+5s = 120/min hits rate limit)

// ── Helpers ──────────────────────────────────────────────────────────────────
const ET_FMT_TIME = { timeZone: 'America/New_York', hour: '2-digit', minute: '2-digit', hour12: false }
const ET_FMT_FULL = { timeZone: 'America/New_York', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false }

// Convert any ISO timestamp (UTC or local) to ET display string
function toET(ts, fullDate = false) {
  if (!ts) return '--'
  try {
    return new Date(ts).toLocaleString('en-US', fullDate ? ET_FMT_FULL : ET_FMT_TIME)
  } catch { return ts.slice(11, 16) }
}

function dollars(cents) {
  if (cents == null) return '--'
  return '$' + (cents / 100).toFixed(2)
}
function dollarsShort(cents) {
  if (cents == null) return '--'
  const d = cents / 100
  return d >= 1000 ? '$' + (d / 1000).toFixed(1) + 'k' : '$' + d.toFixed(0)
}
function pctColor(pct) {
  if (pct == null) return 'text-slate-400'
  return pct >= 0 ? 'text-emerald-400' : 'text-red-400'
}
function statusDot(ok, pulse = false) {
  return ok
    ? <span className={`inline-block w-2 h-2 rounded-full bg-emerald-400 mr-1.5${pulse ? ' animate-pulse' : ''}`} />
    : <span className="inline-block w-2 h-2 rounded-full bg-red-400 mr-1.5" />
}
function coinFromTicker(ticker) {
  if (!ticker) return '?'
  if (ticker.includes('BTC'))  return 'BTC'
  if (ticker.includes('ETH'))  return 'ETH'
  if (ticker.includes('XRP'))  return 'XRP'
  if (ticker.includes('SOL'))  return 'SOL'
  if (ticker.includes('DOGE')) return 'DOGE'
  if (ticker.includes('BNB'))  return 'BNB'
  if (ticker.includes('HYPE')) return 'HYPE'
  return ticker.slice(2, 6)
}

// ── Sub-components ────────────────────────────────────────────────────────────
function StatCard({ label, value, sub, valueClass = 'text-white', accent }) {
  return (
    <div className="card flex flex-col gap-1 relative overflow-hidden">
      {accent && <div className="absolute top-0 left-0 right-0 h-0.5" style={{ background: accent }} />}
      <p className="text-xs text-slate-500 uppercase tracking-wider font-medium">{label}</p>
      <p className={`text-xl font-bold font-mono leading-tight ${valueClass}`}>{value}</p>
      {sub && <p className="text-xs text-slate-500 leading-tight">{sub}</p>}
    </div>
  )
}

function MiniGauge({ value, max = 100, color = '#34d399', label }) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100))
  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between text-xs">
        <span className="text-slate-500">{label}</span>
        <span className="text-white font-mono">{value?.toFixed(1)}%</span>
      </div>
      <div className="h-1.5 rounded-full bg-gray-800">
        <div className="h-full rounded-full transition-all duration-500" style={{ width: pct + '%', background: color }} />
      </div>
    </div>
  )
}

function EdgeBar({ value, max = 20 }) {
  const pct = Math.min(100, Math.max(0, ((value + max) / (max * 2)) * 100))
  const color = value >= 12 ? '#34d399' : value >= 0 ? '#fbbf24' : '#f87171'
  return (
    <div className="relative h-1.5 rounded-full bg-gray-800 w-24">
      <div className="absolute top-0 bottom-0 w-px bg-gray-600" style={{ left: '50%' }} />
      {value >= 0 ? (
        <div className="h-full rounded-full" style={{ marginLeft: '50%', width: (pct - 50) + '%', background: color }} />
      ) : (
        <div className="h-full rounded-full" style={{ marginLeft: pct + '%', width: (50 - pct) + '%', background: color }} />
      )}
    </div>
  )
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-xs shadow-xl">
      <p className="text-slate-400 mb-1">{label}</p>
      {d.total_portfolio_cents != null && <p className="text-emerald-400 font-mono">Total: {dollars(d.total_portfolio_cents)}</p>}
      {d.balance_cents != null && <p className="text-slate-300 font-mono">Cash: {dollars(d.balance_cents)}</p>}
      {d.portfolio_value_cents > 0 && <p className="text-blue-300 font-mono">Pos: {dollars(d.portfolio_value_cents)}</p>}
      {d.pct_from_base != null && (
        <p className={`font-mono ${pctColor(d.pct_from_base)}`}>
          {d.pct_from_base >= 0 ? '+' : ''}{d.pct_from_base.toFixed(2)}%
        </p>
      )}
    </div>
  )
}

function SectionHeader({ title, badge }) {
  return (
    <div className="flex items-center gap-2 mb-3">
      <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-widest">{title}</h2>
      {badge != null && (
        <span className="text-xs px-1.5 py-0.5 rounded bg-gray-800 text-slate-500 font-mono">{badge}</span>
      )}
    </div>
  )
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [portfolio,   setPortfolio]   = useState(null)
  const [status,      setStatus]      = useState(null)
  const [history,     setHistory]     = useState([])
  const [signals,     setSignals]     = useState([])
  const [nearMisses,  setNearMisses]  = useState([])
  const [lastScan,    setLastScan]    = useState(null)
  const [skips,       setSkips]       = useState([])
  const [trades,      setTrades]      = useState([])
  const [performance, setPerformance] = useState(null)
  const [lastUpd,     setLastUpd]     = useState(null)
  const [offline,     setOffline]     = useState(false)
  const [shadowW33,   setShadowW33]   = useState(null)

  // Fast fetch: signals + activity — every 3s
  // NOTE: keep stale state on 429 (rate-limit) — never wipe widgets with empty on throttle
  const fetchFast = useCallback(async () => {
    try {
      const [sigs, skipData] = await Promise.all([
        fetch('/api/signals').then(r => r.json()).catch(() => null),
        fetch('/api/monitor/skips?limit=60').then(r => r.json()).catch(() => null),
      ])
      // Only update if we got a real response (not a rate-limit {detail:...} or null)
      if (sigs && !sigs.detail) {
        setSignals(Array.isArray(sigs) ? sigs : (sigs.signals || []))
        setNearMisses(sigs?.near_misses || [])
        setLastScan(sigs?.last_scan || null)
      }
      if (skipData && !skipData.detail) {
        setSkips((skipData.skips || []).reverse())
      }
    } catch { /* silent — fast fetch failures don't toggle offline */ }
  }, [])

  // Slow fetch: portfolio / trades / perf / monitor history / shadow — every 5s
  const fetchAll = useCallback(async () => {
    try {
      const [port, stat, hist, tradeData, perfData, w33Data] = await Promise.all([
        fetch('/api/portfolio').then(r => r.json()),
        fetch('/api/status').then(r => r.json()),
        fetch('/api/monitor/history?limit=288').then(r => r.json()).catch(() => ({ history: [] })),
        fetch('/api/history/trades').then(r => r.json()).catch(() => []),
        fetch('/api/performance').then(r => r.json()).catch(() => null),
        fetch('/api/shadow/w33').then(r => r.json()).catch(() => null),
      ])
      setPortfolio(port)
      setStatus(stat)
      // Only update history if real response (not rate-limited)
      if (hist && !hist.detail) {
        setHistory((hist.history || []).map(row => ({
          ...row,
          label: row.iso ? row.iso.slice(11, 16) : '',
        })))
      }
      // Use performance trades (has live trades); fall back to history endpoint (paper only)
      const perfTrades = perfData?.trades || []
      const histTrades = Array.isArray(tradeData) ? tradeData : (tradeData.trades || [])
      setTrades(perfTrades.length > 0 ? perfTrades : histTrades)
      // Performance metrics live under perfData.metrics
      setPerformance(perfData?.metrics ?? null)
      if (w33Data && !w33Data.detail) setShadowW33(w33Data)
      setLastUpd(new Date())
      setOffline(false)
    } catch {
      setOffline(true)
    }
  }, [])

  useEffect(() => {
    fetchFast()
    fetchAll()
    const fastId = setInterval(fetchFast, FAST_REFRESH)
    const slowId = setInterval(fetchAll, REFRESH)
    return () => { clearInterval(fastId); clearInterval(slowId) }
  }, [fetchFast, fetchAll])

  // ── Derived values ──────────────────────────────────────────────────────────
  const cashCents   = portfolio?.balance_cents ?? null
  const posCents    = portfolio?.portfolio_value_cents ?? 0
  const totalCents  = cashCents != null ? cashCents + posCents : null
  const positions   = portfolio?.positions ?? []
  const pctFromBase = totalCents != null ? ((totalCents - BASELINE) / BASELINE * 100) : null

  const atOn = status?.auto_trade_enabled
  const asOn = status?.auto_scan_enabled
  const kcOn = status?.kalshi_connected

  // Chart domain
  const allTotals = history.map(r => r.total_portfolio_cents).filter(Boolean)
  const yMin = allTotals.length ? Math.min(...allTotals) * 0.97 : 0
  const yMax = allTotals.length ? Math.max(...allTotals) * 1.03 : 40000

  // Win/loss from trades
  const liveTrades = trades.filter(t => t.mode !== 'paper')
  const settledTrades = liveTrades.filter(t => t.pnl_cents != null || t.profit_cents != null || t.won !== undefined)
  const wins   = liveTrades.filter(t => t.won === true || (t.pnl_cents ?? t.profit_cents ?? 0) > 0).length
  const losses = liveTrades.filter(t => t.won === false || (t.pnl_cents ?? t.profit_cents ?? 0) < 0).length
  const winRate  = performance?.win_rate ?? (settledTrades.length ? wins / settledTrades.length : null)
  const totalPnl = performance?.total_pnl_cents ?? liveTrades.reduce((s, t) => s + (t.pnl_cents ?? t.profit_cents ?? 0), 0)
  const avgEdge  = performance?.avg_edge ?? performance?.avg_edge_cents ?? null

  // P&L sparkline (last 20 trades)
  const pnlData = liveTrades.slice(-20).map((t, i) => ({
    i,
    pnl: (t.pnl_cents ?? t.profit_cents ?? 0) / 100,
    coin: coinFromTicker(t.ticker || t.market_ticker || ''),
  }))

  // Win/loss pie
  const pieData = [
    { name: 'Win',  value: wins || 0,   fill: '#34d399' },
    { name: 'Loss', value: losses || 0, fill: '#f87171' },
  ]

  // Per-coin exposure from positions
  const coinExposure = {}
  for (const p of positions) {
    const coin = coinFromTicker(p.ticker || p.market_ticker || '')
    coinExposure[coin] = (coinExposure[coin] || 0) + (p.value_cents || p.avg_price_cents * (p.quantity || 0) || 0)
  }
  const coinExposureData = Object.entries(coinExposure).map(([coin, val]) => ({ coin, val: val / 100 }))

  // Recent skips parsed
  const parsedActivity = skips.map(line => {
    const isPlace = /PLACING/i.test(line)
    const isIoc   = /\bIOC\b/.test(line) && !isPlace
    // Log lines from server have no embedded timestamp — skip time extraction
    const ts = ''
    const tickerMatch = line.match(/KX\w+/)
    const ticker = tickerMatch ? tickerMatch[0] : ''
    const coin = coinFromTicker(ticker)

    let side = null, placePrice = null, count = null, edgePct = null, ce = null, kelly = null
    if (isPlace) {
      const sideM  = line.match(/PLACING\s+(YES|NO)/i)
      const priceM = line.match(/@\s*(\d+)c/)
      const countM = line.match(/x(\d+)/)
      const edgeM  = line.match(/edge=([+-][\d.]+)/)
      const kellyM = line.match(/kelly=([\d.]+)/)
      side       = sideM  ? sideM[1].toUpperCase()  : null
      placePrice = priceM ? parseInt(priceM[1])      : null
      count      = countM ? parseInt(countM[1])      : null
      edgePct    = edgeM  ? parseFloat(edgeM[1])     : null
      kelly      = kellyM ? parseFloat(kellyM[1])    : null
    } else if (isIoc) {
      const ceM = line.match(/ceil=(\d+)c/)
      ce = ceM ? parseInt(ceM[1]) : null
    } else {
      const ceM    = line.match(/ce=(-?\d+)¢/)
      ce = ceM ? parseInt(ceM[1]) : null
    }
    return { line, isPlace, isIoc, ts, ticker, coin, side, placePrice, count, edgePct, ce, kelly }
  })

  return (
    <div className="min-h-screen" style={{ background: '#080810', color: '#e2e8f0', fontFamily: "'Inter', system-ui, sans-serif" }}>

      {/* ── Header ── */}
      <header className="border-b border-gray-800/60 px-6 py-3 flex items-center justify-between sticky top-0 z-20 backdrop-blur-sm" style={{ background: 'rgba(8,8,16,0.92)' }}>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded bg-emerald-500/20 flex items-center justify-center">
              <span className="text-emerald-400 text-xs font-bold">P</span>
            </div>
            <h1 className="text-sm font-bold tracking-tight">
              PREDICTION<span className="text-emerald-400">BOT</span>
            </h1>
          </div>
          <span className="text-xs px-2 py-0.5 rounded-full bg-red-500/20 text-red-300 font-mono border border-red-800/50">● LIVE</span>
          {totalCents != null && (
            <span className="text-xs font-mono text-slate-400 hidden md:inline">
              {dollars(totalCents)} total
              <span className={`ml-2 ${pctColor(pctFromBase)}`}>
                ({pctFromBase != null ? (pctFromBase >= 0 ? '+' : '') + pctFromBase.toFixed(2) + '%' : '--'})
              </span>
            </span>
          )}
        </div>

        <div className="flex items-center gap-5 text-xs">
          <div className="flex items-center gap-3">
            <span className={`flex items-center gap-1 ${atOn ? 'text-emerald-400' : 'text-red-400'}`}>
              {statusDot(atOn, atOn)} Autotrade
            </span>
            <span className={`flex items-center gap-1 ${asOn ? 'text-emerald-400' : 'text-yellow-400'}`}>
              {statusDot(asOn)} Autoscan
            </span>
            <span className={`flex items-center gap-1 ${kcOn ? 'text-emerald-400' : 'text-red-400'}`}>
              {statusDot(kcOn)} Kalshi
            </span>
          </div>
          {offline && <span className="text-red-400 font-semibold animate-pulse">OFFLINE</span>}
          {lastUpd && <span className="text-slate-600 font-mono hidden lg:inline">{lastUpd.toLocaleTimeString('en-US', ET_FMT_TIME)} ET</span>}
        </div>
      </header>


      <main className="w-full max-w-[1800px] mx-auto px-6 py-5 space-y-5">

        {/* ── Row 1: Stat Cards ── */}
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-3">
          <StatCard label="Cash" value={dollars(cashCents)} sub="Kalshi balance" accent="#3b82f6" />
          <StatCard label="Positions" value={dollars(posCents)} sub={`${positions.length} open`} valueClass="text-blue-300" accent="#3b82f6" />
          <StatCard
            label="Total"
            value={dollars(totalCents)}
            sub="Cash + positions"
            valueClass="text-white"
            accent="#34d399"
          />
          <StatCard
            label="vs Baseline"
            value={pctFromBase != null ? (pctFromBase >= 0 ? '+' : '') + pctFromBase.toFixed(2) + '%' : '--'}
            sub={totalCents != null ? (totalCents >= BASELINE ? '+' : '') + dollars(totalCents - BASELINE) : '--'}
            valueClass={pctColor(pctFromBase)}
            accent={pctFromBase >= 0 ? '#34d399' : '#ef4444'}
          />
          <StatCard
            label="Win Rate"
            value={winRate != null ? (winRate * 100).toFixed(1) + '%' : '--'}
            sub={`${settledTrades.length} settled / ${liveTrades.length} total`}
            valueClass={winRate != null && winRate >= 0.5 ? 'text-emerald-400' : winRate != null ? 'text-red-400' : 'text-slate-400'}
            accent="#8b5cf6"
          />
          <StatCard
            label="Total P&L"
            value={settledTrades.length > 0 ? (totalPnl >= 0 ? '+' : '') + dollars(totalPnl) : '--'}
            sub={`${wins}W / ${losses}L`}
            valueClass={totalPnl > 0 ? 'text-emerald-400' : totalPnl < 0 ? 'text-red-400' : 'text-slate-400'}
            accent="#8b5cf6"
          />
          <StatCard
            label="Avg Edge"
            value={avgEdge != null ? (avgEdge >= 0 ? '+' : '') + avgEdge.toFixed(1) + '¢' : '--'}
            sub="Per fill"
            valueClass="text-yellow-400"
            accent="#f59e0b"
          />
          <StatCard
            label="Active Strategy"
            value="HIGH_CONV"
            sub="E15 / YES≥0.70 / NO≤0.35"
            valueClass="text-emerald-400"
            accent="#34d399"
          />
        </div>

        {/* ── Row 2: Portfolio Chart + Right Panel ── */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">

          {/* Portfolio chart — 2/3 width */}
          <div className="card lg:col-span-2">
            <div className="flex items-center justify-between mb-3">
              <SectionHeader title="Portfolio History (24h)" />
              <div className="flex items-center gap-4 text-xs text-slate-500">
                <span className="flex items-center gap-1.5"><span className="w-3 h-px bg-emerald-400 inline-block" />Total</span>
                <span className="flex items-center gap-1.5"><span className="w-3 h-px bg-blue-400 inline-block" />Cash</span>
                <span className="flex items-center gap-1.5"><span className="w-3 h-px bg-gray-600 inline-block border-dashed" />Baseline</span>
              </div>
            </div>
            {history.length === 0 ? (
              <div className="h-48 flex items-center justify-center text-slate-600 text-sm">Accumulating data every 5 min…</div>
            ) : (
              <ResponsiveContainer width="100%" height={220}>
                <AreaChart data={history} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="totalGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#34d399" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#34d399" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="cashGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor="#60a5fa" stopOpacity={0.15} />
                      <stop offset="95%" stopColor="#60a5fa" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1a2030" />
                  <XAxis dataKey="label" tick={{ fontSize: 9, fill: '#374151' }} interval="preserveStartEnd" />
                  <YAxis domain={[yMin, yMax]} tickFormatter={v => dollarsShort(v)} tick={{ fontSize: 9, fill: '#374151' }} width={46} />
                  <Tooltip content={<ChartTooltip />} />
                  <ReferenceLine y={BASELINE} stroke="#4b5563" strokeDasharray="4 4" strokeWidth={1} />
                  <Area type="monotone" dataKey="balance_cents"         stroke="#60a5fa" strokeWidth={1} fill="url(#cashGrad)"  dot={false} />
                  <Area type="monotone" dataKey="total_portfolio_cents" stroke="#34d399" strokeWidth={2} fill="url(#totalGrad)" dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </div>

          {/* Right panel — win rate + coin exposure */}
          <div className="flex flex-col gap-4">

            {/* Win rate donut */}
            <div className="card flex-1">
              <SectionHeader title="Win / Loss" badge={liveTrades.length} />
              <div className="flex items-center gap-4">
                {liveTrades.length > 0 ? (
                  <PieChart width={80} height={80}>
                    <Pie data={pieData} cx={35} cy={35} innerRadius={22} outerRadius={36} dataKey="value" strokeWidth={0}>
                      {pieData.map((e, i) => <Cell key={i} fill={e.fill} />)}
                    </Pie>
                  </PieChart>
                ) : (
                  <div className="w-20 h-20 rounded-full border-2 border-gray-700 flex items-center justify-center text-slate-600 text-xs">--</div>
                )}
                <div className="flex flex-col gap-2 flex-1">
                  <MiniGauge value={winRate != null ? winRate * 100 : 0} max={100} color="#34d399" label="Win Rate" />
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-emerald-400">{wins}W</span>
                    <span className="text-slate-500">/</span>
                    <span className="text-red-400">{losses}L</span>
                    <span className="text-slate-500">{liveTrades.length - wins - losses} open</span>
                  </div>
                  {performance?.sharpe_ratio != null && (
                    <div className="text-xs">
                      <span className="text-slate-500">Sharpe: </span>
                      <span className={`font-mono font-bold ${performance.sharpe_ratio >= 1 ? 'text-emerald-400' : performance.sharpe_ratio >= 0 ? 'text-yellow-400' : 'text-red-400'}`}>
                        {performance.sharpe_ratio.toFixed(2)}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Coin exposure */}
            <div className="card flex-1">
              <SectionHeader title="Coin Exposure" badge={positions.length + ' pos'} />
              {coinExposureData.length === 0 ? (
                <p className="text-slate-600 text-xs">No open positions</p>
              ) : (
                <div className="space-y-2">
                  {coinExposureData.map(({ coin, val }) => (
                    <div key={coin} className="flex items-center gap-2 text-xs">
                      <span className="text-slate-400 w-10 font-mono">{coin}</span>
                      <div className="flex-1 h-1.5 rounded-full bg-gray-800">
                        <div className="h-full rounded-full bg-blue-500" style={{ width: Math.min(100, (val / 50) * 100) + '%' }} />
                      </div>
                      <span className="text-slate-300 font-mono w-14 text-right">{dollars(val * 100)}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* ── Row 3: P&L Sparkline + Open Positions ── */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">

          {/* P&L per trade */}
          <div className="card">
            <SectionHeader title="P&L per Trade (last 20)" />
            {pnlData.length === 0 ? (
              <div className="h-32 flex items-center justify-center text-slate-600 text-sm">No settled trades yet</div>
            ) : (
              <ResponsiveContainer width="100%" height={130}>
                <BarChart data={pnlData} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1a2030" vertical={false} />
                  <XAxis dataKey="coin" tick={{ fontSize: 9, fill: '#374151' }} />
                  <YAxis tickFormatter={v => (v >= 0 ? '+' : '') + '$' + v.toFixed(0)} tick={{ fontSize: 9, fill: '#374151' }} width={40} />
                  <Tooltip
                    formatter={(v) => [(v >= 0 ? '+' : '') + '$' + v.toFixed(2), 'P&L']}
                    contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: '6px', fontSize: '11px' }}
                  />
                  <ReferenceLine y={0} stroke="#374151" />
                  <Bar dataKey="pnl" radius={[2, 2, 0, 0]}>
                    {pnlData.map((e, i) => <Cell key={i} fill={e.pnl >= 0 ? '#34d399' : '#f87171'} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>

          {/* Open Positions */}
          <div className="card">
            <SectionHeader title="Open Positions" badge={positions.length} />
            {positions.length === 0 ? (
              <p className="text-slate-600 text-sm">No open positions</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-slate-500 border-b border-gray-800">
                      <th className="text-left py-1.5 pr-3">Coin</th>
                      <th className="text-left py-1.5 pr-3">Side</th>
                      <th className="text-right py-1.5 pr-3">Qty</th>
                      <th className="text-right py-1.5 pr-3">Entry</th>
                      <th className="text-right py-1.5 pr-3">Value</th>
                      <th className="text-right py-1.5">Unr P&L</th>
                    </tr>
                  </thead>
                  <tbody>
                    {positions.map((p, i) => {
                      const qty   = p.quantity ?? p.contracts ?? 0
                      const entry = p.avg_price ?? p.entry_price ?? null
                      const curr  = p.current_price ?? p.last_price ?? null
                      let unr = null
                      if (entry != null && curr != null && qty) {
                        unr = p.side === 'YES' ? (curr - entry) * qty : (entry - curr) * qty
                      } else if (p.value_cents != null && entry != null && qty) {
                        unr = p.value_cents - entry * qty
                      }
                      return (
                        <tr key={i} className="border-b border-gray-800/40 hover:bg-gray-800/30">
                          <td className="py-1.5 pr-3 font-mono text-white text-xs">{coinFromTicker(p.ticker || p.market_ticker)}</td>
                          <td className={`py-1.5 pr-3 font-bold ${p.side === 'YES' ? 'text-emerald-400' : 'text-red-400'}`}>{p.side}</td>
                          <td className="py-1.5 pr-3 text-right font-mono text-slate-300">{qty}</td>
                          <td className="py-1.5 pr-3 text-right font-mono text-slate-400">{entry != null ? entry + '¢' : '--'}</td>
                          <td className="py-1.5 pr-3 text-right font-mono text-white">{p.value_cents != null ? dollars(p.value_cents) : '--'}</td>
                          <td className={`py-1.5 text-right font-mono font-bold ${unr == null ? 'text-slate-500' : unr >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                            {unr != null ? (unr >= 0 ? '+' : '') + dollars(unr) : '--'}
                          </td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>

        {/* ── Row 4: Live Activity + Signals ── */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">

          {/* Activity log — 2/3 */}
          <div className="card lg:col-span-2">
            <SectionHeader title="Live Activity" badge={parsedActivity.length} />
            <div className="space-y-px max-h-72 overflow-y-auto">
              {parsedActivity.length === 0 ? (
                <p className="text-slate-600 text-sm">Waiting for scan activity…</p>
              ) : parsedActivity.map((a, i) => (
                <div key={i} className={`flex items-center gap-2 text-xs font-mono px-2 py-1.5 rounded ${
                  a.isPlace ? 'bg-emerald-900/25 border border-emerald-900/40' :
                  a.isIoc   ? 'bg-blue-900/20' : 'hover:bg-gray-800/20'
                }`}>
                  <span className="text-slate-700 w-16 shrink-0 text-[10px]">{a.ts}</span>

                  {a.isPlace ? (
                    <span className={`font-bold w-10 shrink-0 ${a.side === 'YES' ? 'text-emerald-400' : 'text-red-400'}`}>
                      {a.side || 'PLACE'}
                    </span>
                  ) : a.isIoc ? (
                    <span className="text-blue-400 w-10 shrink-0">IOC</span>
                  ) : (
                    <span className="text-slate-700 w-10 shrink-0">SKIP</span>
                  )}

                  <span className="w-8 shrink-0 text-slate-500 text-[10px]">{a.coin}</span>
                  <span className={`truncate flex-1 text-[10px] ${a.isPlace ? 'text-slate-300' : 'text-slate-600'}`}>
                    {a.ticker || a.line.slice(0, 40)}
                  </span>

                  {a.isPlace && a.placePrice != null && (
                    <span className="text-white shrink-0">@{a.placePrice}¢</span>
                  )}
                  {a.isPlace && a.count != null && (
                    <span className="text-slate-400 shrink-0">×{a.count}</span>
                  )}
                  {a.isPlace && a.edgePct != null && (
                    <span className={`shrink-0 font-bold ${a.edgePct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                      {a.edgePct >= 0 ? '+' : ''}{(a.edgePct * 100).toFixed(1)}¢
                    </span>
                  )}
                  {!a.isPlace && a.ce != null && (
                    <span className={`shrink-0 ${a.ce >= 12 ? 'text-emerald-400' : a.ce >= 0 ? 'text-yellow-600' : 'text-red-900'}`}>
                      {a.ce >= 0 ? '+' : ''}{a.ce}¢
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>


          {/* Signals panel — 1/3 */}
          <div className="card">
            <div className="flex items-center justify-between mb-3">
              <SectionHeader title="Latest Signals" badge={signals.length > 0 ? signals.length : (nearMisses.length > 0 ? `${nearMisses.length} near` : 0)} />
              {lastScan && <span className="text-[10px] text-slate-600 font-mono">{toET(lastScan)} ET</span>}
            </div>
            {signals.length > 0 ? (
              <div className="space-y-2">
                {signals.slice(0, 10).map((s, i) => {
                  const edge = s.edge_cents ?? (s.fair_cents != null && s.ask_price != null ? s.fair_cents - s.ask_price : null)
                  const coin = coinFromTicker(s.ticker || s.market_ticker || '')
                  return (
                    <div key={i} className="flex items-center gap-2 text-xs">
                      <span className="text-slate-500 w-8 shrink-0 font-mono">{coin}</span>
                      <span className={`font-bold w-7 shrink-0 ${s.side === 'YES' ? 'text-emerald-400' : 'text-red-400'}`}>{s.side}</span>
                      <span className="text-slate-500 font-mono text-[10px] shrink-0">
                        {s.fair_cents != null ? s.fair_cents + '¢' : '--'}
                      </span>
                      <EdgeBar value={edge ?? 0} />
                      <span className={`font-mono font-bold shrink-0 text-[10px] ${edge != null && edge >= 9 ? 'text-emerald-400' : edge >= 0 ? 'text-yellow-500' : 'text-red-500'}`}>
                        {edge != null ? (edge >= 0 ? '+' : '') + edge + '¢' : '--'}
                      </span>
                    </div>
                  )
                })}
              </div>
            ) : nearMisses.length > 0 ? (
              <div className="space-y-2">
                <p className="text-[10px] text-slate-600 mb-1">Near misses (below gate)</p>
                {nearMisses.slice(0, 8).map((s, i) => {
                  const edgeCents = s.edge != null ? Math.round(s.edge * 100) : null
                  const fairCents = s.model_prob != null ? Math.round(s.model_prob * 100) : null
                  const coin = coinFromTicker(s.ticker || '')
                  const side = (s.side || '').toUpperCase()
                  return (
                    <div key={i} className="flex items-center gap-2 text-xs">
                      <span className="text-slate-600 w-8 shrink-0 font-mono">{coin}</span>
                      <span className={`font-bold w-7 shrink-0 ${side === 'YES' ? 'text-emerald-600' : 'text-red-600'}`}>{side || '--'}</span>
                      <span className="text-slate-600 font-mono text-[10px] shrink-0">
                        {fairCents != null ? fairCents + '¢' : '--'}
                      </span>
                      <EdgeBar value={edgeCents ?? 0} />
                      <span className={`font-mono font-bold shrink-0 text-[10px] ${edgeCents != null && edgeCents >= 0 ? 'text-yellow-700' : 'text-red-800'}`}>
                        {edgeCents != null ? (edgeCents >= 0 ? '+' : '') + edgeCents + '¢' : '--'}
                      </span>
                    </div>
                  )
                })}
              </div>
            ) : (
              <p className="text-slate-600 text-xs">
                No signals — {lastScan ? `last scan ${toET(lastScan)} ET` : 'waiting for scan…'}
              </p>
            )}
          </div>
        </div>

        {/* ── Row 5: Trade History ── */}
        <div className="card">
          <SectionHeader title="Trade History" badge={trades.length} />
          {trades.length === 0 ? (
            <p className="text-slate-600 text-sm">No trades yet</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-slate-500 border-b border-gray-800">
                    <th className="text-left py-2 pr-4">Time (ET)</th>
                    <th className="text-left py-2 pr-4">Coin</th>
                    <th className="text-left py-2 pr-4">Ticker</th>
                    <th className="text-left py-2 pr-4">Side</th>
                    <th className="text-right py-2 pr-4">Fill</th>
                    <th className="text-right py-2 pr-4">Qty</th>
                    <th className="text-right py-2 pr-4">P&L</th>
                    <th className="text-left py-2">Result</th>
                  </tr>
                </thead>
                <tbody>
                  {trades.slice(0, 30).map((t, i) => {
                    const ts = t.entry_time || t.created_time || t.timestamp || t.time || ''
                    const timeStr = toET(ts, true)
                    const pnl = t.pnl_cents ?? t.profit_cents ?? null
                    const won  = t.won === true  || (pnl != null ? pnl > 0 : t.result === 'win')
                    const lost = t.won === false || (pnl != null ? pnl < 0 : t.result === 'loss')
                    // entry_price in performance is 0–1 float; fill_price/avg_price already in cents
                    const rawFill = t.fill_price ?? t.avg_price ?? t.price ?? null
                    const fill = rawFill ?? (t.entry_price != null ? Math.round(t.entry_price * 100) : null)
                    const coin = coinFromTicker(t.ticker || t.market_ticker || '')
                    return (
                      <tr key={i} className={`border-b border-gray-800/40 hover:bg-gray-800/20 ${won ? 'bg-emerald-900/5' : lost ? 'bg-red-900/5' : ''}`}>
                        <td className="py-1.5 pr-4 font-mono text-slate-500 text-[10px]">{timeStr}</td>
                        <td className="py-1.5 pr-4 font-mono text-slate-400 text-[10px]">{coin}</td>
                        <td className="py-1.5 pr-4 font-mono text-slate-600 text-[10px] max-w-[120px] truncate">{t.ticker || t.market_ticker || '--'}</td>
                        <td className={`py-1.5 pr-4 font-bold ${(t.side || '').toUpperCase() === 'YES' ? 'text-emerald-400' : 'text-red-400'}`}>{(t.side || '--').toUpperCase()}</td>
                        <td className="py-1.5 pr-4 text-right font-mono text-slate-400">{fill != null ? fill + '¢' : '--'}</td>
                        <td className="py-1.5 pr-4 text-right font-mono text-slate-400">{t.quantity ?? t.contracts ?? '--'}</td>
                        <td className={`py-1.5 pr-4 text-right font-mono font-bold ${pnl == null ? 'text-slate-600' : pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                          {pnl != null ? (pnl >= 0 ? '+' : '') + dollars(pnl) : '--'}
                        </td>
                        <td className={`py-1.5 font-bold text-xs ${won ? 'text-emerald-400' : lost ? 'text-red-400' : 'text-slate-600'}`}>
                          {won ? '✓ WIN' : lost ? '✗ LOSS' : (t.status || t.result || 'open')}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* ── Row 6: Monitor Log + Config ── */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">

          {/* Monitor log table — 2/3 */}
          <div className="card lg:col-span-2">
            <SectionHeader title="Monitor Log" badge="last 20" />
            {history.length === 0 ? (
              <p className="text-slate-600 text-sm">No monitor events yet — accumulates every 5 min</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-xs font-mono">
                  <thead>
                    <tr className="text-slate-500 border-b border-gray-800">
                      <th className="text-left py-1.5 pr-4">Time</th>
                      <th className="text-right py-1.5 pr-4">Cash</th>
                      <th className="text-right py-1.5 pr-4">Pos</th>
                      <th className="text-right py-1.5 pr-4">Total</th>
                      <th className="text-right py-1.5 pr-4">vs Base</th>
                      <th className="text-left py-1.5">Note</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...history].reverse().slice(0, 20).map((row, i) => (
                      <tr key={i} className="border-b border-gray-800/20 hover:bg-gray-800/20">
                        <td className="py-1 pr-4 text-slate-500 text-[10px]">{row.iso ? toET(row.iso, true) : '--'}</td>
                        <td className="py-1 pr-4 text-right text-slate-400">{dollars(row.balance_cents)}</td>
                        <td className="py-1 pr-4 text-right text-blue-400">{dollars(row.portfolio_value_cents)}</td>
                        <td className="py-1 pr-4 text-right font-bold text-white">{dollars(row.total_portfolio_cents)}</td>
                        <td className={`py-1 pr-4 text-right ${pctColor(row.pct_from_base)}`}>
                          {row.pct_from_base != null ? (row.pct_from_base >= 0 ? '+' : '') + row.pct_from_base.toFixed(2) + '%' : '--'}
                        </td>
                        <td className="py-1 text-slate-600 text-[10px]">
                          {row.note || ''}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Config + status — 1/3 */}
          <div className="card">
            <SectionHeader title="Live Bot Config" />
            <div className="space-y-2 text-xs font-mono">
              {(() => {
                const lv = status?.live_trading || {}
                const wp  = lv.signal_wp  != null ? (lv.signal_wp  * 100).toFixed(0) + '%' : '--'
                const wm  = lv.signal_wm  != null ? (lv.signal_wm  * 100).toFixed(0) + '%' : '--'
                const wi  = lv.signal_wi  != null ? (lv.signal_wi  * 100).toFixed(0) + '%' : '--'
                const wf  = lv.signal_wf  != null ? (lv.signal_wf  * 100).toFixed(0) + '%' : '--'
                const edge = lv.min_edge_cents != null ? `E${lv.min_edge_cents} (≥${lv.min_edge_cents}¢)` : '--'
                const kelly = lv.kelly_base != null ? `K${(lv.kelly_base * 100).toFixed(0)} base` : '--'
                const last = lv.last_promoted_bot || '—'
                return [
                  ['Signal px', wp,   '#34d399'],
                  ['Signal mkt', wm,  '#34d399'],
                  ['Signal imb', wi,  '#34d399'],
                  ['Signal fund', wf, '#34d399'],
                  ['Edge',  edge,     '#fbbf24'],
                  ['Kelly', kelly,    '#60a5fa'],
                  ['15m budget', (lv.budget_15m_pct != null ? (lv.budget_15m_pct * 100).toFixed(0) + '%' : '--'), '#a3e635'],
                  ['Range budget', (lv.budget_range_pct != null ? (lv.budget_range_pct * 100).toFixed(0) + '%' : '--'), '#818cf8'],
                  ['AutoStop', 'DISABLED', '#6b7280'],
                  ['W33 regime', last.length > 20 ? last.slice(4, 20) + '…' : last, '#f97316'],
                ].map(([k, v, color]) => (
                  <div key={k} className="flex justify-between items-center py-1 border-b border-gray-800/50">
                    <span className="text-slate-500">{k}</span>
                    <span className="font-bold" style={{ color }}>{v}</span>
                  </div>
                ))
              })()}
            </div>

            <div className="mt-4 pt-3 border-t border-gray-800">
              <SectionHeader title="System" />
              <div className="space-y-1.5 text-xs font-mono">
                <div className="flex justify-between">
                  <span className="text-slate-500">Scan interval</span>
                  <span className="text-slate-300">{status?.scan_interval_seconds ?? '--'}s</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Uptime</span>
                  <span className="text-slate-300">{status?.uptime_seconds != null ? Math.floor(status.uptime_seconds / 60) + 'm' : '--'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Last scan</span>
                  <span className="text-slate-300">{status?.last_scan ? toET(status.last_scan) + ' ET' : '--'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-500">Baseline</span>
                  <span className="text-slate-300">$353.00</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* ── Row 7: W33 Shadow Tournament ── */}
        {shadowW33 && (() => {
          const bots = Object.values(shadowW33.bots || {})
          const withTrades = bots.filter(b => b.total_trades > 0)
          const sorted = [...withTrades].sort((a, b) => b.pnl_cents - a.pnl_cents)
          const top10 = sorted.slice(0, 10)
          const bottom3 = sorted.slice(-3)
          const lastPromoted = shadowW33.last_promoted_bot
          return (
            <div className="card">
              <div className="flex items-center justify-between mb-3">
                <SectionHeader title="W33 Shadow Tournament" badge={`${withTrades.length}/${bots.length} active`} />
                <div className="flex items-center gap-3">
                  {lastPromoted && (
                    <span className="text-[10px] text-orange-400 font-mono">▲ {lastPromoted.slice(4, 24)}</span>
                  )}
                  <span className="text-[10px] text-slate-600 font-mono">${(shadowW33.init_balance_cents / 100).toFixed(0)} each</span>
                </div>
              </div>
              {withTrades.length === 0 ? (
                <p className="text-slate-600 text-xs">No settled trades yet — accumulating…</p>
              ) : (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  <div>
                    <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1.5">Top 10</p>
                    <table className="w-full text-xs font-mono">
                      <thead>
                        <tr className="text-slate-500 border-b border-gray-800">
                          <th className="text-left py-1 pr-2">Bot</th>
                          <th className="text-right py-1 pr-2">T</th>
                          <th className="text-right py-1 pr-2">WR</th>
                          <th className="text-right py-1 pr-2">P&L</th>
                          <th className="text-right py-1">Bal</th>
                        </tr>
                      </thead>
                      <tbody>
                        {top10.map((b, i) => {
                          const wr = b.win_rate != null ? (b.win_rate * 100).toFixed(0) + '%' : '--'
                          const pnl = b.pnl_cents
                          const isLive = lastPromoted && b.label === lastPromoted
                          return (
                            <tr key={i} className={`border-b border-gray-800/30 ${isLive ? 'bg-orange-900/20' : 'hover:bg-gray-800/20'}`}>
                              <td className="py-1 pr-2 text-[10px] max-w-[140px] truncate">
                                {isLive && <span className="text-orange-400 mr-0.5">▲</span>}
                                <span className={isLive ? 'text-orange-300' : 'text-slate-300'}>{b.label?.slice(4, 22) || b.label}</span>
                              </td>
                              <td className="py-1 pr-2 text-right text-slate-400">{b.total_trades}</td>
                              <td className={`py-1 pr-2 text-right font-bold ${b.win_rate == null ? 'text-slate-600' : b.win_rate >= 0.55 ? 'text-emerald-400' : b.win_rate >= 0.45 ? 'text-yellow-400' : 'text-red-400'}`}>{wr}</td>
                              <td className={`py-1 pr-2 text-right font-bold ${pnl > 0 ? 'text-emerald-400' : pnl < 0 ? 'text-red-400' : 'text-slate-500'}`}>
                                {(pnl >= 0 ? '+' : '') + dollars(pnl)}
                              </td>
                              <td className="py-1 text-right text-slate-400 text-[10px]">{dollars(b.balance_cents)}</td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                  <div>
                    <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1.5">Bottom 3</p>
                    <table className="w-full text-xs font-mono">
                      <thead>
                        <tr className="text-slate-500 border-b border-gray-800">
                          <th className="text-left py-1 pr-2">Bot</th>
                          <th className="text-right py-1 pr-2">T</th>
                          <th className="text-right py-1 pr-2">WR</th>
                          <th className="text-right py-1">P&L</th>
                        </tr>
                      </thead>
                      <tbody>
                        {bottom3.map((b, i) => {
                          const wr = b.win_rate != null ? (b.win_rate * 100).toFixed(0) + '%' : '--'
                          const pnl = b.pnl_cents
                          return (
                            <tr key={i} className="border-b border-gray-800/30 hover:bg-gray-800/20">
                              <td className="py-1 pr-2 text-slate-400 text-[10px] max-w-[140px] truncate">{b.label?.slice(4, 22) || b.label}</td>
                              <td className="py-1 pr-2 text-right text-slate-400">{b.total_trades}</td>
                              <td className={`py-1 pr-2 text-right font-bold ${b.win_rate == null ? 'text-slate-600' : b.win_rate >= 0.55 ? 'text-emerald-400' : b.win_rate >= 0.45 ? 'text-yellow-400' : 'text-red-400'}`}>{wr}</td>
                              <td className={`py-1 text-right font-bold ${pnl > 0 ? 'text-emerald-400' : pnl < 0 ? 'text-red-400' : 'text-slate-500'}`}>
                                {(pnl >= 0 ? '+' : '') + dollars(pnl)}
                              </td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )
        })()}

      </main>

      <footer className="border-t border-gray-800/40 py-3 mt-2 text-center text-xs text-slate-700">
        PredictionBot v2 • W33 tournament 350 bots • 60% 15m / 40% range budget • signals 3s • portfolio 5s • baseline $353.00 • no floor
      </footer>
    </div>
  )
}
