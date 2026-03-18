/**
 * Tooltip.jsx — Reusable tooltip wrapper used throughout the app.
 *
 * Wraps any children in a hover-triggered tooltip showing explanatory text.
 * Used to explain cryptic column headers like OI, MAE, MFE, Sharpe, etc.
 */
export default function Tooltip({ text, children }) {
  if (!text) return children;
  return (
    <span className="tooltip-wrapper">
      {children}
      <span className="tooltip-text">{text}</span>
    </span>
  );
}
