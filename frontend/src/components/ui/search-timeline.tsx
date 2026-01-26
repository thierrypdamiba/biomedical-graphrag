"use client";

import { cn } from "@/lib/utils";

interface TraceStep {
  id: string;
  name: string;
  status: "pending" | "running" | "complete" | "error";
  startTime?: number;
  endTime?: number;
  duration?: number;
  details?: Record<string, unknown>;
}

interface SearchTimelineProps {
  steps: TraceStep[];
  totalTime?: number;
  isVisible: boolean;
}

export function SearchTimeline({ steps, totalTime, isVisible }: SearchTimelineProps) {
  if (!isVisible || steps.length === 0) return null;

  const getStatusIcon = (status: TraceStep["status"]) => {
    switch (status) {
      case "pending":
        return (
          <div className="w-4 h-4 rounded-full border-2 border-zinc-600" />
        );
      case "running":
        return (
          <div className="w-4 h-4 rounded-full border-2 border-blue-500 border-t-transparent animate-spin" />
        );
      case "complete":
        return (
          <div className="w-4 h-4 rounded-full bg-green-500 flex items-center justify-center">
            <svg className="w-2.5 h-2.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
            </svg>
          </div>
        );
      case "error":
        return (
          <div className="w-4 h-4 rounded-full bg-red-500 flex items-center justify-center">
            <svg className="w-2.5 h-2.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </div>
        );
    }
  };

  const getStatusColor = (status: TraceStep["status"]) => {
    switch (status) {
      case "pending": return "text-zinc-500";
      case "running": return "text-blue-400";
      case "complete": return "text-green-400";
      case "error": return "text-red-400";
    }
  };

  // Calculate progress bar width based on timing
  const maxTime = Math.max(...steps.filter(s => s.endTime).map(s => s.endTime || 0), totalTime || 0);

  return (
    <div className="bg-zinc-900/50 border border-zinc-800 rounded-lg p-4 mt-4">
      <div className="flex items-center justify-between mb-3">
        <h4 className="text-sm font-medium text-zinc-300">Search Timeline</h4>
        {totalTime && (
          <span className="text-xs text-zinc-500">
            Total: {totalTime}ms
          </span>
        )}
      </div>

      <div className="space-y-3">
        {steps.map((step, index) => (
          <div key={step.id} className="relative">
            {/* Connector line */}
            {index < steps.length - 1 && (
              <div 
                className={cn(
                  "absolute left-[7px] top-5 w-0.5 h-6",
                  step.status === "complete" ? "bg-green-500/30" : "bg-zinc-700"
                )}
              />
            )}

            <div className="flex items-start gap-3">
              {/* Status icon */}
              <div className="flex-shrink-0 mt-0.5">
                {getStatusIcon(step.status)}
              </div>

              {/* Step info */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <span className={cn("text-sm font-medium", getStatusColor(step.status))}>
                    {step.name}
                  </span>
                  {step.duration !== undefined && (
                    <span className="text-xs text-zinc-500 ml-2">
                      {step.duration}ms
                    </span>
                  )}
                </div>

                {/* Progress bar */}
                {maxTime > 0 && step.startTime !== undefined && (
                  <div className="mt-1.5 h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                    <div
                      className={cn(
                        "h-full rounded-full transition-all duration-300",
                        step.status === "running" && "animate-pulse",
                        step.status === "complete" && "bg-green-500",
                        step.status === "running" && "bg-blue-500",
                        step.status === "error" && "bg-red-500",
                        step.status === "pending" && "bg-zinc-700"
                      )}
                      style={{
                        marginLeft: `${(step.startTime / maxTime) * 100}%`,
                        width: step.duration 
                          ? `${(step.duration / maxTime) * 100}%`
                          : step.status === "running" 
                            ? "20%" 
                            : "0%",
                      }}
                    />
                  </div>
                )}

                {/* Details */}
                {step.details && Object.keys(step.details).length > 0 && (
                  <div className="mt-1 text-xs text-zinc-600">
                    {Object.entries(step.details).map(([key, value]) => (
                      <span key={key} className="mr-3">
                        {key}: {String(value)}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
