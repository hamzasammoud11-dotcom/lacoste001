// ui/components/sidebar.tsx
"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Home, Microscope, Dna, BarChart2, Settings, Terminal } from "lucide-react"

import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"

const sidebarItems = [
  {
    title: "Home",
    href: "/",
    icon: Home,
  },
  {
    title: "Discovery",
    href: "/discovery",
    icon: Microscope,
  },
  {
    title: "Explorer",
    href: "/explorer",
    icon: Dna,
  },
  {
    title: "Data",
    href: "/data",
    icon: BarChart2,
  },
  {
    title: "Settings",
    href: "/settings",
    icon: Settings,
  },
]

export function Sidebar() {
  const pathname = usePathname()

  return (
    <div className="flex h-screen w-[250px] flex-col border-r bg-card pb-4 pt-6">
      <div className="px-6 mb-8 flex items-center gap-2">
        <div className="h-8 w-8 rounded-lg bg-primary/20 flex items-center justify-center text-primary">
            <Dna className="h-5 w-5" />
        </div>
        <div className="font-bold text-xl tracking-tight">
          Bio<span className="text-primary">Flow</span>
        </div>
      </div>
      
      <div className="px-4 py-2">
        <div className="text-xs font-semibold text-muted-foreground mb-4 px-2 uppercase tracking-wider">
          Navigation
        </div>
        <nav className="space-y-1">
          {sidebarItems.map((item) => (
            <Link
              key={item.href}
              href={item.href}
            >
              <Button
                variant={pathname === item.href ? "secondary" : "ghost"}
                className={cn(
                  "w-full justify-start gap-3",
                  pathname === item.href && "bg-secondary font-medium"
                )}
              >
                <item.icon className="h-4 w-4" />
                {item.title}
              </Button>
            </Link>
          ))}
        </nav>
      </div>

      <div className="mt-auto px-4">
        <div className="rounded-lg border bg-muted/50 p-4">
            <div className="flex items-center gap-2 mb-2">
                <Terminal className="h-4 w-4 text-muted-foreground" />
                <span className="text-xs font-medium">Status</span>
            </div>
            <div className="flex items-center gap-2">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                </span>
                <span className="text-xs text-muted-foreground">System Online</span>
            </div>
        </div>
      </div>
    </div>
  )
}
