import type { ReactNode } from 'react';
import {
  SidebarInset,
  SidebarProvider,
} from "@/components/animate-ui/components/radix/sidebar";
import { AppSidebar } from "@/components/sidebar";

export default function DashboardLayout({ children }: { children: ReactNode }) {
  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        <main className="flex-1 overflow-y-auto bg-background p-8">
          <div className="mx-auto max-w-7xl">
            {children}
          </div>
        </main>
      </SidebarInset>
    </SidebarProvider>
  );
}
