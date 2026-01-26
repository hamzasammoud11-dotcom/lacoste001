'use client';

import { usePathname } from 'next/navigation';

// Workflow has its own layout that bypasses the main sidebar
// This gives Langflow the full viewport for proper UX
export default function WorkflowLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="fixed inset-0 z-50 bg-background">
      {children}
    </div>
  );
}
