/**
 * api.js - Centralized API client for the PredictionBot frontend.
 *
 * Provides a single `api` export containing named methods for every backend
 * endpoint. All requests go through a shared `request()` helper that prepends
 * the base URL, sets JSON headers, and normalizes error handling.
 *
 * Key feature groups:
 *  - Status & config: getStatus, updateConfig.
 *  - Market data: getEvents, getMarket, getSignals, getArbitrage.
 *  - Portfolio & orders: getPortfolio, getPositions, getOrders, placeTrade,
 *    cancelOrder.
 *  - Scanning & automation: runScan, toggleAutoScan, toggleAutoTrade,
 *    getAutoScanStatus.
 *  - Paper trading: getPaperState, configurePaper, addPaperFunds, paperScan,
 *    paperTrain.
 *  - Backtesting: runBacktest, runSweep.
 *  - Performance & analytics: getPerformance, getFeatureImportance,
 *    getPerformanceByCategory, getPortfolioHeatmap.
 *  - Trade logs & notes: getShadowTradeLog, getLiveTradeLog,
 *    updateTradeNotes, exportTradesCsv.
 *  - Risk & retraining: resetDailyPnl, getRetrainSchedule,
 *    updateRetrainSchedule, retrainNow.
 *  - Notifications & webhooks: testNotifications, getNotificationConfig,
 *    triggerWebhook.
 *
 * Connects to:
 *  - The FastAPI backend at bot.server (default same-origin, overridable via
 *    the VITE_API_URL environment variable).
 *  - Consumed by App.jsx and individual page components throughout the
 *    frontend.
 */
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
  addPaperFunds: (amountCents) =>
    request('/paper/add-funds', {
      method: 'POST',
      body: JSON.stringify({ amount_cents: amountCents }),
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

  // Notifications
  testNotifications: () => request('/notifications/test', { method: 'POST' }),
  getNotificationConfig: () => request('/notifications/config'),

  // CSV Export
  exportTradesCsv: (source = 'paper') => {
    const url = `${API}/export/trades?source=${source}`;
    window.open(url, '_blank');
  },

  // Category P&L & Heatmap
  getPerformanceByCategory: (source = 'paper') =>
    request(`/performance/by-category?source=${source}`),
  getPortfolioHeatmap: () => request('/portfolio/heatmap'),

  // Trade Notes
  updateTradeNotes: (tradeIndex, notes, source = 'paper') =>
    request(`/trade/${tradeIndex}/notes?source=${source}`, {
      method: 'PATCH',
      body: JSON.stringify({ notes }),
    }),

  // Trade Logs (separate shadow vs live)
  getShadowTradeLog: (limit = 200) => request(`/trades/shadow?limit=${limit}`),
  getLiveTradeLog: (limit = 200) => request(`/trades/live?limit=${limit}`),

  // Risk Management
  resetDailyPnl: () => request('/risk/reset-daily', { method: 'POST' }),

  // Retrain Schedule
  getRetrainSchedule: () => request('/retrain/schedule'),
  updateRetrainSchedule: (days, hour) =>
    request('/retrain/schedule', {
      method: 'POST',
      body: JSON.stringify({ days, hour }),
    }),
  retrainNow: () => request('/retrain/now', { method: 'POST' }),

  // Webhook
  triggerWebhook: (action, params = {}) =>
    request('/webhook', {
      method: 'POST',
      body: JSON.stringify({ action, params }),
    }),
};
