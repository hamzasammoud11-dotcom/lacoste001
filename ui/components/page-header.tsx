// ui/components/page-header.tsx
import { cn } from "@/lib/utils"

interface PageHeaderProps {
  title: string
  subtitle?: string
  icon?: React.ReactNode
  className?: string
}

export function PageHeader({ title, subtitle, icon, className }: PageHeaderProps) {
  return (
    <div className={cn("mb-8 space-y-2", className)}>
        <h1 className="flex items-center gap-3 text-3xl font-bold tracking-tight">
            {icon && <span className="text-primary">{icon}</span>}
            {title}
        </h1>
        {subtitle && (
            <p className="text-lg text-muted-foreground">
                {subtitle}
            </p>
        )}
    </div>
  )
}

export function SectionHeader({ title, icon, action }: { title: string, icon?: React.ReactNode, action?: React.ReactNode }) {
    return (
        <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold flex items-center gap-2">
                {icon}
                {title}
            </h2>
            {action}
        </div>
    )
}
