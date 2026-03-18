/** @type {import('tailwindcss').Config} */

function withOpacity(variable) {
  return ({ opacityValue }) => {
    if (opacityValue !== undefined) {
      return `rgba(var(${variable}), ${opacityValue})`;
    }
    return `rgb(var(${variable}))`;
  };
}

export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        surface: withOpacity('--color-surface'),
        'surface-2': withOpacity('--color-surface-2'),
        card: withOpacity('--color-card'),
        'card-hover': withOpacity('--color-card-hover'),
        border: withOpacity('--color-border'),
        'border-subtle': withOpacity('--color-border-subtle'),
        accent: withOpacity('--color-accent'),
        'accent-green': withOpacity('--color-green'),
        'accent-red': withOpacity('--color-red'),
        'accent-yellow': withOpacity('--color-yellow'),
        'accent-blue': withOpacity('--color-blue'),
        'accent-purple': withOpacity('--color-purple'),
        'accent-cyan': withOpacity('--color-cyan'),
        'accent-orange': withOpacity('--color-orange'),
        'text-primary': withOpacity('--color-text-primary'),
        'text-secondary': withOpacity('--color-text-secondary'),
        'text-muted': withOpacity('--color-text-muted'),
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
        mono: ['JetBrains Mono', 'Menlo', 'Monaco', 'monospace'],
      },
      borderRadius: {
        xl: '12px',
        '2xl': '16px',
      },
      boxShadow: {
        glow: '0 0 20px rgba(52, 211, 153, 0.1)',
        'glow-green': '0 0 20px rgba(52, 211, 153, 0.15)',
        'glow-red': '0 0 20px rgba(248, 113, 113, 0.15)',
        'glow-blue': '0 0 20px rgba(96, 165, 250, 0.15)',
        card: '0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.06)',
        'card-hover': '0 4px 12px rgba(0,0,0,0.15)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-in': 'fadeIn 0.3s ease-out',
        'slide-up': 'slideUp 0.3s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
}
