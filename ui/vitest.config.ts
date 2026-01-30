import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'node',
    globals: true,
    testTimeout: 30000,
    include: ['__tests__/**/*.test.ts'],
    // Allow longer timeouts for API integration tests
    hookTimeout: 30000,
  },
});
