"use client";

import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--blue)] disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 [&_svg]:shrink-0",
  {
    variants: {
      variant: {
        primary:
          "bg-[var(--amaranth)] text-white hover:bg-[#c41f43] active:bg-[#b01c3c]",
        secondary:
          "border border-[var(--stroke-1)] bg-transparent text-[var(--text-primary)] hover:bg-[var(--bg-2)] hover:border-[var(--stroke-2)]",
        ghost:
          "bg-transparent text-[var(--text-primary)] hover:bg-[var(--bg-2)]",
        violet:
          "bg-[var(--violet)] text-white hover:bg-[#7339e6] active:bg-[#662ed9]",
        teal:
          "bg-[var(--teal)] text-white hover:bg-[#027474] active:bg-[#026363]",
        destructive:
          "bg-[var(--amaranth)] text-white hover:bg-[#c41f43]",
      },
      size: {
        default: "h-9 px-4 py-2 text-sm rounded-[10px]",
        sm: "h-8 px-3 text-sm rounded-[8px]",
        lg: "h-10 px-6 text-base rounded-[10px]",
        icon: "h-9 w-9 rounded-[10px]",
      },
    },
    defaultVariants: {
      variant: "primary",
      size: "default",
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
Button.displayName = "Button";

export { Button, buttonVariants };
