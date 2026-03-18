const API = (import.meta.env.VITE_API_URL || '') + '/api';

async function request(path, options = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export const api = {
  getStatus: () => request('/status'),
  getPortfolio: () => request('/portfolio'),
  getPositions: () => request('/positions'),
  getEvents: (limit = 20) => request(`/events?limit=${limit}`),
  getMarket: (ticker) => request(`/market/${ticker}`),
  getSignals: () => request('/signals'),
  getOrders: () => request('/orders'),
  getPerformance: () => request('/performance'),
  getFeatureImportance: () => request('/model/features'),
  getArbitrage: () => request('/arbitrage'),

  runScan: (maxEvents = 20, useAi = false) =>
    request('/scan', {
      method: 'POST',
      body: JSON.stringify({ max_events: maxEvents, use_ai: useAi }),
    }),

  placeTrade: (ticker, side, priceCents, count) =>
    request('/trade', {
      method: 'POST',
      body: JSON.stringify({ ticker, side, price_cents: priceCents, count }),
    }),

  cancelOrder: (orderId) =>
    request(`/order/${orderId}`, { method: 'DELETE' }),

  updateConfig: (updates) =>
    request('/config', {
      method: 'PATCH',
      body: JSON.stringify(updates),
    }),

  // Backtest
  runBacktest: (params = {}) =>
    request('/backtest', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  runSweep: (params = {}) =>
    request('/backtest/sweep', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  // Paper Trading
  getPaperState: () => request('/paper'),
  configurePaper: (balanceCents = 10000) =>
    request('/paper/configure', {
      method: 'POST',
      body: JSON.stringify({ balance_cents: balanceCents }),
    }),
  paperScan: () => request('/paper/scan', { method: 'POST' }),
  paperTrain: () => request('/paper/train', { method: 'POST' }),

  // Auto-scan / Auto-trade
  toggleAutoScan: (enabled, intervalSeconds = 60) =>
    request('/autoscan', {
      method: 'POST',
      body: JSON.stringify({ enabled, interval_seconds: intervalSeconds }),
    }),
  toggleAutoTrade: (enabled) =>
    request('/autotrade', {
      method: 'POST',
      body: JSON.stringify({ enabled }),
    }),
  getAutoScanStatus: () => request('/autoscan/status'),
};
