#!/usr/bin/env bash
# shadow_report.sh — Live shadow trader reports. Runs every 60s.
# Usage: ./shadow_report.sh          # report every 60s
#        ./shadow_report.sh 30       # report every 30s

INTERVAL=${1:-60}

report() {
python3 -c "
import urllib.request, json, sys
from datetime import datetime

try:
    s = json.loads(urllib.request.urlopen('http://localhost:8000/api/shadow', timeout=5).read())
except Exception as e:
    print(f'[ERROR] Could not reach server: {e}')
    return

try:
    t_raw = json.loads(urllib.request.urlopen('http://localhost:8000/api/trades/shadow?limit=50', timeout=5).read())
    trades = t_raw if isinstance(t_raw, list) else t_raw.get('trades', [])
except:
    trades = []

bal    = s.get('balance_cents', 0)
start  = 24764
pnl    = bal - start
wins   = s.get('wins', 0)
losses = s.get('losses', 0)
total  = s.get('total_trades', 0)
wr     = s.get('win_rate')
open_p = s.get('open_positions', 0)
now    = datetime.now().strftime('%H:%M:%S')

print()
print(f'╔══════════════════════════════════════════════════════════╗')
print(f'║  SHADOW PAPER TRADER  ·  {now}                    ║')
print(f'╠══════════════════════════════════════════════════════════╣')
print(f'║  Balance:  \${bal/100:>8.2f}   P&L: \${pnl/100:>+8.2f}  ({pnl/start*100:>+5.1f}%)  ║')
if wr is not None:
    print(f'║  Win Rate: {wr*100:>7.1f}%   Trades: {total:>4d}  ({wins}W / {losses}L)          ║')
else:
    print(f'║  Win Rate:    --      Trades:    0  (no settled yet)       ║')
print(f'║  Open Positions: {open_p}                                       ║')
print(f'╠══════════════════════════════════════════════════════════╣')

positions = s.get('positions', [])
if positions:
    print(f'║  OPEN POSITIONS:                                          ║')
    for p in positions:
        tk = p['ticker'][-30:].ljust(30)
        cost = p.get('entry_price',0) * p.get('count',0)
        ep   = p.get('entry_price', 0)
        cnt  = p.get('count', 0)
        side = p.get('side','?').upper()
        edge = p.get('edge',0)*100
        print(f'║  {tk}  {side:3s} {cnt}×@{ep}¢ cost={cost:.0f}¢ edge={edge:+.1f}¢  ║')

settled = [t for t in trades if t.get('action') not in ('entry',)]
if settled:
    print(f'╠══════════════════════════════════════════════════════════╣')
    print(f'║  RECENT SETTLED (last {min(5,len(settled))}):                               ║')
    for t in settled[-5:]:
        tk   = t.get('ticker','')[-28:].ljust(28)
        res  = 'WIN ' if t.get('won') else 'LOSS'
        pnlc = t.get('pnl_cents', 0)
        print(f'║  {tk}  {res}  {pnlc/100:>+6.2f}                    ║')

print(f'╚══════════════════════════════════════════════════════════╝')
"
}

echo "Shadow report every ${INTERVAL}s — Ctrl+C to stop"
echo "Frontend: http://localhost:5173 → Paper tab"
echo ""

while true; do
    report
    sleep "$INTERVAL"
done
