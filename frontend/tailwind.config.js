/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        surface: '#000000',
        card: '#0a0a0a',
        border: '#1a1a1a',
        accent: '#ffffff',
        'accent-green': '#00ff87',
        'accent-red': '#ff3b3b',
        'accent-yellow': '#ffb800',
        'text-primary': '#ffffff',
        'text-secondary': '#666666',
        'text-muted': '#444444',
      },
    },
  },
  plugins: [],
}
