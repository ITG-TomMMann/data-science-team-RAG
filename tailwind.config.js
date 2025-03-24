/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        'itg-pink': '#FF1493',
        'itg-pink-dark': '#CC1077',
        'chat-gray': '#F3F4F6',
      },
    },
  },
  plugins: [],
}
