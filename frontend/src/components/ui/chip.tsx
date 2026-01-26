"use client";

import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const chipVariants = cva(
  "inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium transition-colors",
  {
    variants: {
      variant: {
        default: "bg-[rgba(255,255,255,0.06)] text-[var(--text-secondary)]",
        violet: "bg-[rgba(133,71,255,0.18)] text-[var(--violet)]",
        blue: "bg-[rgba(47,111,240,0.18)] text-[var(--blue)]",
        teal: "bg-[rgba(3,133,133,0.18)] text-[var(--teal)]",
        amaranth: "bg-[rgba(220,36,76,0.18)] text-[var(--amaranth)]",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

export interface ChipProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof chipVariants> {}

const Chip = React.forwardRef<HTMLSpanElement, ChipProps>(
  ({ className, variant, ...props }, ref) => {
    return (
      <span
        ref={ref}
        className={cn(chipVariants({ variant, className }))}
        {...props}
      />
    );
  }
);
Chip.displayName = "Chip";

export { Chip, chipVariants };
