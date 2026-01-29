import './globals.css';

import type { Metadata, Viewport } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import { ThemeProvider } from 'next-themes';

import {
  SidebarInset,
  SidebarProvider,
} from '@/components/animate-ui/components/radix/sidebar';
import { AppSidebar } from '@/components/sidebar';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const viewport: Viewport = {
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: 'white' },
    { media: '(prefers-color-scheme: dark)', color: '#0C0E14' },
  ],
  width: 'device-width',
  initialScale: 1,
};

export const metadata: Metadata = {
  title: {
    default: 'BioFlow',
    template: '%s | BioFlow',
  },
  description:
    'AI-Powered Drug Discovery Platform for molecular embedding and binding prediction.',
  keywords: [
    'Drug Discovery',
    'AI',
    'Bioinformatics',
    'Machine Learning',
    'Molecular Dynamics',
  ],
  authors: [{ name: 'BioFlow Team' }],
  creator: 'BioFlow',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          <SidebarProvider>
            <AppSidebar />
            <SidebarInset>
              <main className="bg-background flex-1 overflow-y-auto p-8">
                <div className="mx-auto max-w-7xl">{children}</div>
              </main>
            </SidebarInset>
          </SidebarProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
