"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { Moon, Sun, Monitor, Search, X, Menu } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useAppStore } from "@/store/app-store";
import { cn } from "@/lib/utils";
import { useMounted } from "@/hooks/use-mounted";

export function TopBar() {
  const { theme, setTheme, sidebarCollapsed, setPendingSearch, toggleMobileMenu } = useAppStore();
  const mounted = useMounted();
  const router = useRouter();
  const [searchOpen, setSearchOpen] = React.useState(false);
  const [searchQuery, setSearchQuery] = React.useState("");
  const inputRef = React.useRef<HTMLInputElement>(null);

  const ThemeIcon = theme === "dark" ? Moon : theme === "light" ? Sun : Monitor;

  // Handle keyboard shortcut
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setSearchOpen(true);
      }
      if (e.key === "Escape") {
        setSearchOpen(false);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  // Focus input when search opens
  React.useEffect(() => {
    if (searchOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [searchOpen]);

  const handleSearch = () => {
    if (searchQuery.trim()) {
      setPendingSearch(searchQuery.trim());
      setSearchOpen(false);
      setSearchQuery("");
      router.push("/assistant");
    }
  };

  return (
    <header
      className={cn(
        "fixed top-0 right-0 z-30 flex h-14 items-center justify-between border-b border-[var(--stroke-1)] bg-[var(--bg-0)] px-4 md:px-6 transition-all duration-300",
        // Mobile: full width
        "left-0",
        // Desktop: account for sidebar
        sidebarCollapsed ? "lg:left-[72px]" : "lg:left-[260px]"
      )}
    >
      {/* Left: Mobile menu + Title */}
      <div className="flex items-center gap-3">
        {/* Mobile hamburger menu */}
        <Button
          variant="ghost"
          size="icon"
          className="lg:hidden"
          onClick={toggleMobileMenu}
        >
          <Menu className="h-5 w-5" />
        </Button>
        <span className="font-semibold text-[var(--text-primary)] text-sm md:text-base truncate">
          <span className="hidden sm:inline">Biomedical Research Assistant</span>
          <span className="sm:hidden">BioMed GraphRAG</span>
        </span>
      </div>

      {/* Center: Search */}
      <div className="hidden lg:flex">
        {searchOpen ? (
          <div className="flex items-center gap-2 rounded-lg border border-[var(--stroke-2)] bg-[var(--bg-1)] px-3 py-1.5" style={{ outline: 'none' }}>
            <Search className="h-4 w-4 text-[var(--text-tertiary)]" />
            <input
              ref={inputRef}
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  handleSearch();
                }
              }}
              placeholder="Search papers, genes, diseases..."
              className="w-64 bg-transparent text-sm text-[var(--text-primary)] placeholder:text-[var(--text-tertiary)] outline-none focus:outline-none focus:ring-0"
            />
            <button
              onClick={() => setSearchOpen(false)}
              className="text-[var(--text-tertiary)] hover:text-[var(--text-secondary)]"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        ) : (
          <button
            onClick={() => setSearchOpen(true)}
            className="flex items-center gap-2 rounded-lg border border-[var(--stroke-1)] bg-[var(--bg-1)] px-3 py-1.5 text-sm text-[var(--text-tertiary)] hover:border-[var(--stroke-2)] hover:text-[var(--text-secondary)]"
          >
            <Search className="h-4 w-4" />
            <span>Search papers, genes, diseases...</span>
            <kbd className="ml-4 rounded bg-[var(--bg-2)] px-1.5 py-0.5 text-xs">
              ⌘K
            </kbd>
          </button>
        )}
      </div>

      {/* Right: Actions */}
      <div className="flex items-center gap-2">
        {/* Theme Toggle */}
        {mounted ? (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon">
                <ThemeIcon className="h-5 w-5" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={() => setTheme("light")}>
                <Sun className="mr-2 h-4 w-4" />
                Light
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setTheme("dark")}>
                <Moon className="mr-2 h-4 w-4" />
                Dark
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setTheme("system")}>
                <Monitor className="mr-2 h-4 w-4" />
                System
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        ) : (
          <Button variant="ghost" size="icon">
            <Monitor className="h-5 w-5" />
          </Button>
        )}
      </div>
    </header>
  );
}
