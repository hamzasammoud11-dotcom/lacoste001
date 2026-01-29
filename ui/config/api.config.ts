export const API_CONFIG = {
  baseUrl: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  name: 'BioFlow',
  description:
    'AI-Powered Drug Discovery Platform for molecular embedding and binding prediction.',
  version: '2.0.0',
  author: 'BioFlow Team',
  url: process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000',
} as const;
