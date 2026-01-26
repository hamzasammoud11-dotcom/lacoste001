// eslint.config.ts
import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

import reactHooks from "eslint-plugin-react-hooks";
import unusedImports from "eslint-plugin-unused-imports";
import simpleImportSort from "eslint-plugin-simple-import-sort";
import tailwindcss from "eslint-plugin-tailwindcss";

export default defineConfig([
  ...nextVitals,
  ...nextTs,

  {
    plugins: {
      "react-hooks": reactHooks,
      "unused-imports": unusedImports,
      "simple-import-sort": simpleImportSort,
      tailwindcss,
    },

    settings: {
      // Helps eslint-plugin-tailwindcss understand cn()/cva() patterns (common with shadcn)
      tailwindcss: {
        callees: ["cn", "cva"],
        // If you still have a config file, keep it here; otherwise it’s harmless.
        config: "./tailwind.config.ts",
      },
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

      // Tailwind + shadcn friendly defaults
      "tailwindcss/classnames-order": "warn",
      "tailwindcss/no-contradicting-classname": "error",
      // shadcn often uses design-token classes (e.g. bg-background, text-foreground)
      "tailwindcss/no-custom-classname": "off",
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
