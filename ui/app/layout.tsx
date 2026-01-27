import type React from "react";

import { Inter, Space_Mono } from "next/font/google";
import type { Metadata, Viewport } from "next";

import "./globals.css";
import { ProjectStateProvider } from "@/hooks/use-project-state";
import { cn } from "@/lib/utils";

import type { WebApplication, WithContext } from "schema-dts";
import { API_CONFIG } from "@/config/api.config";

import { Toaster } from "@/components/ui/sonner";
import { ThemeProvider } from "next-themes";

const INTER = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const SPACE_MONO = Space_Mono({
  subsets: ["latin"],
  variable: "--font-space-mono",
  display: "swap",
  weight: ["400", "700"],
});

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
  userScalable: true,
  viewportFit: "cover",
  interactiveWidget: "resizes-content",
  colorScheme: "light dark",
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "oklch(1 0 0)" },
    { media: "(prefers-color-scheme: dark)", color: "oklch(0.1 0.02 265)" },
  ],
};

export const metadata: Metadata = {
  title: {
    default: "BioFlow",
    template: "%s | BioFlow",
  },
  description: "AI-Powered Drug Discovery Platform for molecular embedding and binding prediction.",
  keywords: ["Drug Discovery", "AI", "Bioinformatics", "Machine Learning", "Molecular Dynamics"],
  authors: [{ name: "BioFlow Team" }],
  creator: "BioFlow",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const jsonLd: WithContext<WebApplication> = {
    "@context": "https://schema.org",
    "@type": "WebApplication",
    name: API_CONFIG.name,
    description: API_CONFIG.description,
    applicationCategory: "ScienceApplication",
    operatingSystem: "Web",
    offers: {
      "@type": "Offer",
      price: "0",
      priceCurrency: "USD",
    },
    author: {
      "@type": "Organization",
      name: API_CONFIG.author,
    },
  };

  const safeJsonLd = JSON.stringify(jsonLd).replace(/</g, "\\u003c");

  return (
    <html lang="fr" className="scroll-smooth" suppressHydrationWarning>
      <head>
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: safeJsonLd }}
        />
      </head>
      <body
        className={cn(
          INTER.variable,
          SPACE_MONO.variable,
          "min-h-screen bg-background font-sans text-foreground antialiased"
        )}
      >
        <ProjectStateProvider>
          <ThemeProvider
            attribute="class"
            defaultTheme="system"
            enableSystem
            disableTransitionOnChange
            storageKey="bisoness-theme"
          >
            {children}
            <Toaster />
          </ThemeProvider>
        </ProjectStateProvider>
      </body>
    </html>
  );
}
