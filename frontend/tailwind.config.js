/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        surface: 'var(--color-surface)',
        card: 'var(--color-card)',
        border: 'var(--color-border)',
        accent: 'var(--color-accent)',
        'accent-green': '#00ff87',
        'accent-red': '#ff3b3b',
        'accent-yellow': '#ffb800',
        'accent-blue': '#3b82f6',
        'accent-purple': '#a855f7',
        'accent-cyan': '#22d3ee',
        'text-primary': 'var(--color-text-primary)',
        'text-secondary': 'var(--color-text-secondary)',
        'text-muted': 'var(--color-text-muted)',
      },
    },
  },
  plugins: [],
}
