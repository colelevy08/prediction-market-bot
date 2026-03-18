/**
 * main.jsx - Entry point for the PredictionBot React application.
 *
 * Bootstraps the app by mounting the root <App /> component into the DOM
 * element with id "root". Wraps the tree in React.StrictMode to surface
 * potential problems during development (double-invoked effects, deprecated
 * API warnings, etc.).
 *
 * Connects to:
 *  - ./App.jsx (root component containing all routing and layout).
 *  - ./index.css (global Tailwind/CSS styles).
 */
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
