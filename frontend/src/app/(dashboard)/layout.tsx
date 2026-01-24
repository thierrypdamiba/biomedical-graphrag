"use client";

import { Sidebar } from "@/components/layout/sidebar";
import { TopBar } from "@/components/layout/topbar";
import { useAppStore } from "@/store/app-store";
import { cn } from "@/lib/utils";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const { sidebarCollapsed } = useAppStore();

  return (
    <div className="min-h-screen bg-[var(--bg-0)]">
      <Sidebar />
      <TopBar />
      <main
        className={cn(
          "pt-14 transition-all duration-300",
          // Mobile: no margin
          "ml-0",
          // Desktop: account for sidebar
          sidebarCollapsed ? "lg:ml-[72px]" : "lg:ml-[260px]"
        )}
      >
        <div className="p-4 md:p-6">{children}</div>
      </main>
    </div>
  );
}
