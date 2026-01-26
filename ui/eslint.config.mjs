// eslint.config.ts
import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";
import reactHooks from "eslint-plugin-react-hooks";
import simpleImportSort from "eslint-plugin-simple-import-sort";
import unusedImports from "eslint-plugin-unused-imports";

export default defineConfig([
  ...nextVitals,
  ...nextTs,

  {
    plugins: {
      "react-hooks": reactHooks,
      "unused-imports": unusedImports,
      "simple-import-sort": simpleImportSort,
    },

    rules: {
      // React hooks (Next includes some of this, but this keeps it explicit)
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "warn",

      // Prefer removing unused imports entirely
      "unused-imports/no-unused-imports": "warn",
      "unused-imports/no-unused-vars": [
        "warn",
        {
          argsIgnorePattern: "^_",
          varsIgnorePattern: "^_",
          caughtErrorsIgnorePattern: "^_",
        },
      ],

      // If unused-imports handles it, avoid double-reporting from TS ESLint
      "@typescript-eslint/no-unused-vars": "off",

      // Practical console policy
      "no-console": ["warn", { allow: ["info", "warn", "error"] }],

      // Clean, deterministic import ordering
      "simple-import-sort/imports": "warn",
      "simple-import-sort/exports": "warn",

      // NOTE: Tailwind rules disabled - plugin not compatible with Tailwind v4
    },
  },

  // Ignore generated/build output (and override eslint-config-next defaults explicitly)
  globalIgnores([
    ".next/**",
    "out/**",
    "build/**",
    "dist/**",
    "coverage/**",
    "node_modules/**",
    "next-env.d.ts",
  ]),
]);
