import { create } from "zustand";
import { persist } from "zustand/middleware";

export type Theme = "light" | "dark" | "system";
export type VectorMode = "graphrag" | "dense" | "sparse" | "hybrid";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  metadata?: {
    collection?: string;
    filter?: string;
    vectorMode?: VectorMode;
    latency?: number;
    request?: string;
    results?: SearchResult[];
  };
}

interface SearchResult {
  id: string;
  score: number;
  payload: Record<string, unknown>;
  vector?: number[];
}

interface Collection {
  name: string;
  pointsCount: number;
  vectorsConfig: {
    size: number;
    distance: "Cosine" | "Dot" | "Euclid";
  };
  payloadFields: string[];
}

interface ConsoleRequest {
  id: string;
  method: string;
  path: string;
  body?: string;
  response?: string;
  status?: number;
  timestamp: Date;
  latency?: number;
}

interface AppState {
  // Theme
  theme: Theme;
  setTheme: (theme: Theme) => void;

  // Sidebar
  sidebarCollapsed: boolean;
  toggleSidebar: () => void;

  // Assistant
  messages: Message[];
  addMessage: (message: Omit<Message, "id" | "timestamp">) => void;
  clearMessages: () => void;

  // Current query settings
  vectorMode: VectorMode;
  setVectorMode: (mode: VectorMode) => void;
  topK: number;
  setTopK: (k: number) => void;
  activeCollection: string | null;
  setActiveCollection: (name: string | null) => void;

  // Collections
  collections: Collection[];
  setCollections: (collections: Collection[]) => void;

  // Console
  consoleHistory: ConsoleRequest[];
  addConsoleRequest: (request: Omit<ConsoleRequest, "id" | "timestamp">) => void;
  clearConsoleHistory: () => void;

  // Trace drawer
  traceDrawerOpen: boolean;
  toggleTraceDrawer: () => void;

  // Artifacts pane
  artifactsPaneOpen: boolean;
  toggleArtifactsPane: () => void;
  activeArtifactTab: "results" | "point" | "code" | "trace";
  setActiveArtifactTab: (tab: "results" | "point" | "code" | "trace") => void;

  // Selected point
  selectedPoint: SearchResult | null;
  setSelectedPoint: (point: SearchResult | null) => void;

  // Global search (from top bar)
  pendingSearch: string | null;
  setPendingSearch: (query: string | null) => void;

  // Mobile menu
  mobileMenuOpen: boolean;
  setMobileMenuOpen: (open: boolean) => void;
  toggleMobileMenu: () => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      // Theme
      theme: "dark",
      setTheme: (theme) => set({ theme }),

      // Sidebar
      sidebarCollapsed: false,
      toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),

      // Assistant
      messages: [],
      addMessage: (message) =>
        set((state) => ({
          messages: [
            ...state.messages,
            {
              ...message,
              id: crypto.randomUUID(),
              timestamp: new Date(),
            },
          ],
        })),
      clearMessages: () => set({ messages: [] }),

      // Query settings
      vectorMode: "graphrag",
      setVectorMode: (mode) => set({ vectorMode: mode }),
      topK: 10,
      setTopK: (k) => set({ topK: k }),
      activeCollection: null,
      setActiveCollection: (name) => set({ activeCollection: name }),

      // Collections
      collections: [],
      setCollections: (collections) => set({ collections }),

      // Console
      consoleHistory: [],
      addConsoleRequest: (request) =>
        set((state) => ({
          consoleHistory: [
            {
              ...request,
              id: crypto.randomUUID(),
              timestamp: new Date(),
            },
            ...state.consoleHistory,
          ],
        })),
      clearConsoleHistory: () => set({ consoleHistory: [] }),

      // Trace drawer
      traceDrawerOpen: false,
      toggleTraceDrawer: () => set((state) => ({ traceDrawerOpen: !state.traceDrawerOpen })),

      // Artifacts pane
      artifactsPaneOpen: true,
      toggleArtifactsPane: () => set((state) => ({ artifactsPaneOpen: !state.artifactsPaneOpen })),
      activeArtifactTab: "results",
      setActiveArtifactTab: (tab) => set({ activeArtifactTab: tab }),

      // Selected point
      selectedPoint: null,
      setSelectedPoint: (point) => set({ selectedPoint: point }),

      // Global search
      pendingSearch: null,
      setPendingSearch: (query) => set({ pendingSearch: query }),

      // Mobile menu
      mobileMenuOpen: false,
      setMobileMenuOpen: (open) => set({ mobileMenuOpen: open }),
      toggleMobileMenu: () => set((state) => ({ mobileMenuOpen: !state.mobileMenuOpen })),
    }),
    {
      name: "qdrant-demo-storage",
      partialize: (state) => ({
        theme: state.theme,
        sidebarCollapsed: state.sidebarCollapsed,
        vectorMode: state.vectorMode,
        topK: state.topK,
      }),
    }
  )
);
