// ui/components/sidebar.tsx
'use client';

import {
  BadgeCheck,
  BarChart2,
  Bell,
  ChevronRight,
  ChevronsUpDown,
  CreditCard,
  Dna,
  FlaskConical,
  Home,
  LogOut,
  Microscope,
  Settings,
  Sparkles,
  Terminal,
  User,
} from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import * as React from 'react';

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/animate-ui/components/radix/dropdown-menu';
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSub,
  SidebarMenuSubButton,
  SidebarMenuSubItem,
  SidebarRail,
} from '@/components/animate-ui/components/radix/sidebar';
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/animate-ui/primitives/radix/collapsible';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { ThemeToggle } from '@/components/theme-toggle';
import { useIsMobile } from '@/hooks/use-mobile';

const navMain = [
  {
    title: 'Home',
    url: '/dashboard',
    icon: Home,
    isActive: true,
  },
  {
    title: 'Visualization',
    url: '/dashboard/molecules-2d',
    icon: FlaskConical,
    items: [
      {
        title: 'Molecules 2D',
        url: '/dashboard/molecules-2d',
      },
      {
        title: 'Molecules 3D',
        url: '/dashboard/molecules-3d',
      },
      {
        title: 'Proteins 3D',
        url: '/dashboard/proteins-3d',
      },
    ],
  },
  {
    title: 'Discovery',
    url: '/dashboard/discovery',
    icon: Microscope,
    items: [
      {
        title: 'Drug Discovery',
        url: '/dashboard/discovery',
      },
      {
        title: 'Molecule Search',
        url: '/dashboard/discovery#search',
      },
    ],
  },
  {
    title: 'Explorer',
    url: '/dashboard/explorer',
    icon: Dna,
    items: [
      {
        title: 'Embeddings',
        url: '/dashboard/explorer',
      },
      {
        title: 'Predictions',
        url: '/dashboard/explorer#predictions',
      },
    ],
  },
  {
    title: 'Data',
    url: '/dashboard/data',
    icon: BarChart2,
    items: [
      {
        title: 'Datasets',
        url: '/dashboard/data',
      },
      {
        title: 'Analytics',
        url: '/dashboard/data#analytics',
      },
    ],
  },
  {
    title: 'Settings',
    url: '/dashboard/settings',
    icon: Settings,
    items: [
      {
        title: 'General',
        url: '/dashboard/settings',
      },
      {
        title: 'Models',
        url: '/dashboard/settings#models',
      },
    ],
  },
];

const userData = {
  name: 'BioFlow User',
  email: 'user@bioflow.ai',
  avatar: '',
};

export function AppSidebar() {
  const isMobile = useIsMobile();
  const pathname = usePathname();

  return (
    <Sidebar collapsible="icon">
      <SidebarHeader>
        {/* App Header */}
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" asChild>
              <Link href="/dashboard">
                <div className="flex aspect-square size-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
                  <Dna className="size-4" />
                </div>
                <div className="grid flex-1 text-left text-sm leading-tight">
                  <span className="truncate font-semibold">BioFlow</span>
                  <span className="truncate text-xs">AI Drug Discovery</span>
                </div>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
        {/* Theme Toggle */}
        <div className="group-data-[collapsible=icon]:hidden px-2">
          <ThemeToggle />
        </div>
        {/* App Header */}
      </SidebarHeader>

      <SidebarContent>
        {/* Nav Main */}
        <SidebarGroup>
          <SidebarGroupLabel>Platform</SidebarGroupLabel>
          <SidebarMenu>
            {navMain.map((item) => {
              const isActive = pathname === item.url || pathname?.startsWith(item.url + '/');

              if (!item.items || item.items.length === 0) {
                return (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton asChild isActive={isActive} tooltip={item.title}>
                      <Link href={item.url}>
                        {item.icon && <item.icon />}
                        <span>{item.title}</span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              }

              return (
                <Collapsible
                  key={item.title}
                  asChild
                  defaultOpen={isActive}
                  className="group/collapsible"
                >
                  <SidebarMenuItem>
                    <CollapsibleTrigger asChild>
                      <SidebarMenuButton tooltip={item.title} isActive={isActive}>
                        {item.icon && <item.icon />}
                        <span>{item.title}</span>
                        <ChevronRight className="ml-auto transition-transform duration-300 group-data-[state=open]/collapsible:rotate-90" />
                      </SidebarMenuButton>
                    </CollapsibleTrigger>
                    <CollapsibleContent>
                      <SidebarMenuSub>
                        {item.items?.map((subItem) => (
                          <SidebarMenuSubItem key={subItem.title}>
                            <SidebarMenuSubButton asChild isActive={pathname === subItem.url}>
                              <Link href={subItem.url}>
                                <span>{subItem.title}</span>
                              </Link>
                            </SidebarMenuSubButton>
                          </SidebarMenuSubItem>
                        ))}
                      </SidebarMenuSub>
                    </CollapsibleContent>
                  </SidebarMenuItem>
                </Collapsible>
              );
            })}
          </SidebarMenu>
        </SidebarGroup>
        {/* Nav Main */}

        {/* Status Section */}
        <SidebarGroup className="group-data-[collapsible=icon]:hidden mt-auto">
          <SidebarGroupLabel>System Status</SidebarGroupLabel>
          <div className="px-3 py-2">
            <div className="rounded-lg border bg-muted/50 p-3">
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
        </SidebarGroup>
        {/* Status Section */}
      </SidebarContent>

      <SidebarFooter>
        {/* Nav User */}
        <SidebarMenu>
          <SidebarMenuItem>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <SidebarMenuButton
                  size="lg"
                  className="data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground"
                >
                  <Avatar className="h-8 w-8 rounded-lg">
                    <AvatarImage src={userData.avatar} alt={userData.name} />
                    <AvatarFallback className="rounded-lg bg-primary text-primary-foreground">
                      <User className="h-4 w-4" />
                    </AvatarFallback>
                  </Avatar>
                  <div className="grid flex-1 text-left text-sm leading-tight">
                    <span className="truncate font-semibold">{userData.name}</span>
                    <span className="truncate text-xs">{userData.email}</span>
                  </div>
                  <ChevronsUpDown className="ml-auto size-4" />
                </SidebarMenuButton>
              </DropdownMenuTrigger>
              <DropdownMenuContent
                className="w-[--radix-dropdown-menu-trigger-width] min-w-56 rounded-lg"
                side={isMobile ? 'bottom' : 'right'}
                align="end"
                sideOffset={4}
              >
                <DropdownMenuLabel className="p-0 font-normal">
                  <div className="flex items-center gap-2 px-1 py-1.5 text-left text-sm">
                    <Avatar className="h-8 w-8 rounded-lg">
                      <AvatarImage src={userData.avatar} alt={userData.name} />
                      <AvatarFallback className="rounded-lg bg-primary text-primary-foreground">
                        <User className="h-4 w-4" />
                      </AvatarFallback>
                    </Avatar>
                    <div className="grid flex-1 text-left text-sm leading-tight">
                      <span className="truncate font-semibold">{userData.name}</span>
                      <span className="truncate text-xs">{userData.email}</span>
                    </div>
                  </div>
                </DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuGroup>
                  <DropdownMenuItem>
                    <Sparkles />
                    Upgrade to Pro
                  </DropdownMenuItem>
                </DropdownMenuGroup>
                <DropdownMenuSeparator />
                <DropdownMenuGroup>
                  <DropdownMenuItem>
                    <BadgeCheck />
                    Account
                  </DropdownMenuItem>
                  <DropdownMenuItem>
                    <CreditCard />
                    Billing
                  </DropdownMenuItem>
                  <DropdownMenuItem>
                    <Bell />
                    Notifications
                  </DropdownMenuItem>
                </DropdownMenuGroup>
                <DropdownMenuSeparator />
                <DropdownMenuItem>
                  <LogOut />
                  Log out
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </SidebarMenuItem>
        </SidebarMenu>
        {/* Nav User */}
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  );
}
