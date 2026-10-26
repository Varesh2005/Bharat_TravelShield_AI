/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        navy: {
          800: '#0c2464',
          850: '#0a1e54',
          900: '#081b4b', // Wireframe dark navy bar
          950: '#051130',
        },
        skybtn: {
          400: '#38bdf8',
          500: '#00a8e8', // Wireframe bright cyan/blue search button
          600: '#0284c7',
        },
        hazard: {
          low: '#10b981',
          moderate: '#f59e0b',
          elevated: '#f97316',
          high: '#ef4444',
          severe: '#991b1b',
          unsupported: '#64748b',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
