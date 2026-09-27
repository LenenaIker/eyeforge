import clsx from "clsx";
import type { ButtonHTMLAttributes, HTMLAttributes, ReactNode } from "react";
import { Link } from "react-router-dom";

export const cx = clsx;

type Variant = "primary" | "secondary" | "ghost" | "danger";
const VARIANTS: Record<Variant, string> = {
  primary: "bg-brand-600 text-white shadow-sm hover:bg-brand-700 disabled:bg-brand-200",
  secondary: "bg-white text-ink border border-line shadow-sm hover:bg-stone-50 disabled:text-stone-400",
  ghost: "text-muted hover:bg-stone-100 hover:text-ink disabled:text-stone-300",
  danger: "bg-white text-red-600 border border-red-200 hover:bg-red-50",
};

export function Button({ variant = "primary", size = "md", className, ...rest }:
  ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: "sm" | "md" | "lg" }) {
  return (
    <button
      className={cx(
        "inline-flex items-center justify-center gap-2 rounded-[var(--radius-control)] font-medium transition-colors",
        "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-500 disabled:cursor-not-allowed",
        size === "sm" && "h-9 px-3 text-sm", size === "md" && "h-11 px-4 text-[15px]",
        size === "lg" && "h-12 px-6 text-base",
        VARIANTS[variant], className)}
      {...rest}
    />
  );
}

export function ButtonLink({ to, children, variant = "primary", className }:
  { to: string; children: ReactNode; variant?: Variant; className?: string }) {
  return (
    <Link to={to} className={cx(
      "inline-flex h-12 items-center justify-center gap-2 rounded-[var(--radius-control)] px-6 font-medium transition-colors",
      VARIANTS[variant], className)}>
      {children}
    </Link>
  );
}

export function Card({ className, ...rest }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cx("rounded-[var(--radius-card)] border border-line bg-white shadow-sm", className)} {...rest} />;
}

type Tone = "neutral" | "good" | "bad" | "brand" | "ember";
const TONES: Record<Tone, string> = {
  neutral: "bg-stone-100 text-muted",
  good: "bg-green-50 text-green-700",
  bad: "bg-red-50 text-red-700",
  brand: "bg-brand-50 text-brand-700",
  ember: "bg-ember-50 text-ember-700",
};

export function Badge({ tone = "neutral", children, className }: { tone?: Tone; children: ReactNode; className?: string }) {
  return <span className={cx("inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium", TONES[tone],
                             className)}>{children}</span>;
}

export function Logo({ className }: { className?: string }) {
  return (
    <Link to="/" className={cx("flex items-center gap-2 font-semibold tracking-tight", className)}>
      <img src="/logo.svg" alt="" className="h-7 w-7" />
      <span className="text-lg">EyeForge</span>
    </Link>
  );
}

export function Spinner({ className }: { className?: string }) {
  return <span className={cx("inline-flex gap-1", className)} aria-label="working">
    {[0, 1, 2].map((i) => (
      <span key={i} className="h-1.5 w-1.5 animate-bounce rounded-full bg-current"
            style={{ animationDelay: `${i * 120}ms` }} />
    ))}
  </span>;
}

export function Notice({ tone = "ember", children }: { tone?: "ember" | "bad"; children: ReactNode }) {
  return <div className={cx("rounded-[var(--radius-control)] px-4 py-3 text-sm",
                            tone === "bad" ? "bg-red-50 text-red-800" : "bg-ember-50 text-ember-700")}>{children}</div>;
}
