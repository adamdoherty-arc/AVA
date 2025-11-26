/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                background: "#0a0e1a",
                surface: "#111827",
                "surface-elevated": "#1f2937",
                primary: "#3b82f6",
                secondary: "#8b5cf6",
                accent: "#10b981",
                warning: "#f59e0b",
                danger: "#ef4444",
                text: "#f8fafc",
                muted: "#94a3b8",
                border: "#1e293b",
            },
            fontFamily: {
                sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
            },
            boxShadow: {
                'glow-sm': '0 0 15px -3px rgba(59, 130, 246, 0.3)',
                'glow': '0 0 25px -5px rgba(59, 130, 246, 0.4)',
                'glow-lg': '0 0 40px -5px rgba(59, 130, 246, 0.5)',
                'glow-success': '0 0 25px -5px rgba(16, 185, 129, 0.4)',
                'glow-danger': '0 0 25px -5px rgba(239, 68, 68, 0.4)',
            },
            animation: {
                'fade-in': 'fade-in 0.4s ease-out forwards',
                'slide-in': 'slide-in-right 0.4s ease-out forwards',
                'scale-in': 'scale-in 0.3s ease-out forwards',
                'float': 'float 3s ease-in-out infinite',
                'pulse-slow': 'pulse 3s ease-in-out infinite',
                'spin-slow': 'spin 3s linear infinite',
            },
            keyframes: {
                'fade-in': {
                    '0%': { opacity: '0', transform: 'translateY(10px)' },
                    '100%': { opacity: '1', transform: 'translateY(0)' },
                },
                'slide-in-right': {
                    '0%': { opacity: '0', transform: 'translateX(20px)' },
                    '100%': { opacity: '1', transform: 'translateX(0)' },
                },
                'scale-in': {
                    '0%': { opacity: '0', transform: 'scale(0.95)' },
                    '100%': { opacity: '1', transform: 'scale(1)' },
                },
                'float': {
                    '0%, 100%': { transform: 'translateY(0)' },
                    '50%': { transform: 'translateY(-10px)' },
                },
            },
        },
    },
    plugins: [],
}
