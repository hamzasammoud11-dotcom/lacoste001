import { defineConfig, globalIgnores } from 'eslint/config';
import nextVitals from 'eslint-config-next/core-web-vitals';
import nextTs from 'eslint-config-next/typescript';
import reactHooks from 'eslint-plugin-react-hooks';
import simpleImportSort from 'eslint-plugin-simple-import-sort';
import unusedImports from 'eslint-plugin-unused-imports';

export default defineConfig([
  ...nextVitals,
  ...nextTs,

  {
    plugins: {
      'react-hooks': reactHooks,
      'unused-imports': unusedImports,
      'simple-import-sort': simpleImportSort,
    },

    rules: {
      'react-hooks/rules-of-hooks': 'error',
      'react-hooks/exhaustive-deps': 'warn',

      'unused-imports/no-unused-imports': 'warn',
      'unused-imports/no-unused-vars': [
        'warn',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          caughtErrorsIgnorePattern: '^_',
        },
      ],

      '@typescript-eslint/no-unused-vars': 'off',

      'no-console': ['warn', { allow: ['info', 'warn', 'error'] }],

      'simple-import-sort/imports': 'warn',
      'simple-import-sort/exports': 'warn',

    },
  },

  globalIgnores([
    '.next/**',
    'out/**',
    'build/**',
    'dist/**',
    'coverage/**',
    'node_modules/**',
    'next-env.d.ts',
  ]),
]);
