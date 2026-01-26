"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  MessageSquare,
  Database,
  Settings,
  ChevronLeft,
  ChevronRight,
  Network,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useAppStore } from "@/store/app-store";

const navItems = [
  { name: "Assistant", href: "/assistant", icon: MessageSquare },
  { name: "Data", href: "/data", icon: Database },
  { name: "Knowledge Graph", href: "/graph", icon: Network },
  { name: "Settings", href: "/settings", icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();
  const { sidebarCollapsed, toggleSidebar, mobileMenuOpen, setMobileMenuOpen } = useAppStore();

  // Close mobile menu on route change
  React.useEffect(() => {
    setMobileMenuOpen(false);
  }, [pathname, setMobileMenuOpen]);

  return (
    <>
      {/* Mobile overlay */}
      {mobileMenuOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 lg:hidden"
          onClick={() => setMobileMenuOpen(false)}
        />
      )}

      <aside
        className={cn(
          "fixed left-0 top-0 z-50 flex h-screen flex-col border-r border-[var(--stroke-1)] bg-[var(--bg-1)] transition-all duration-300",
          // Desktop: show based on collapsed state
          "hidden lg:flex",
          sidebarCollapsed ? "lg:w-[72px]" : "lg:w-[260px]",
          // Mobile: show as overlay when menu is open
          mobileMenuOpen && "flex w-[280px]"
        )}
      >
      {/* Header */}
      <div className="flex h-14 items-center justify-between gap-3 border-b border-[var(--stroke-1)] px-4">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[var(--violet)]">
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <circle cx="12" cy="12" r="3" stroke="white" strokeWidth="2" />
              <circle cx="12" cy="4" r="2" fill="white" />
              <circle cx="12" cy="20" r="2" fill="white" />
              <circle cx="4" cy="12" r="2" fill="white" />
              <circle cx="20" cy="12" r="2" fill="white" />
              <path d="M12 6V9M12 15V18M6 12H9M15 12H18" stroke="white" strokeWidth="1.5" />
            </svg>
          </div>
          {(!sidebarCollapsed || mobileMenuOpen) && (
            <span className="font-semibold text-[var(--text-primary)]">
              BioMed GraphRAG
            </span>
          )}
        </div>
        {/* Mobile close button */}
        {mobileMenuOpen && (
          <button
            onClick={() => setMobileMenuOpen(false)}
            className="lg:hidden p-1 rounded-lg text-[var(--text-tertiary)] hover:bg-[var(--bg-2)] hover:text-[var(--text-primary)]"
          >
            <X className="h-5 w-5" />
          </button>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto px-2 py-2">
        <ul className="space-y-1">
          {navItems.map((item) => {
            const isActive = pathname === item.href || pathname.startsWith(item.href + "/");
            const showText = !sidebarCollapsed || mobileMenuOpen;
            return (
              <li key={item.name}>
                <Link
                  href={item.href}
                  className={cn(
                    "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                    isActive
                      ? "bg-[var(--bg-2)] text-[var(--text-primary)]"
                      : "text-[var(--text-secondary)] hover:bg-[var(--bg-2)] hover:text-[var(--text-primary)]",
                    !showText && "justify-center px-2"
                  )}
                  title={!showText ? item.name : undefined}
                >
                  <item.icon className="h-5 w-5 shrink-0" />
                  {showText && <span>{item.name}</span>}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      {/* Footer - hide on mobile */}
      <div className="hidden lg:block border-t border-[var(--stroke-1)] p-2">
        <button
          onClick={toggleSidebar}
          className="flex w-full items-center justify-center gap-3 rounded-lg px-3 py-2 text-sm text-[var(--text-tertiary)] hover:bg-[var(--bg-2)] hover:text-[var(--text-primary)]"
        >
          {sidebarCollapsed ? (
            <ChevronRight className="h-5 w-5" />
          ) : (
            <>
              <ChevronLeft className="h-5 w-5" />
              <span>Collapse</span>
            </>
          )}
        </button>
      </div>
    </aside>
    </>
  );
}
