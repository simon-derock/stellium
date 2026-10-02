// Flat rule-bordered menu ported from Lunarbit: drives VIEW, STYLE and THEME.
import { useEffect, useRef, useState, type CSSProperties } from "react";

export function Menu({
  tag,
  value,
  options,
  onChange,
  align = "start",
  width = "16rem",
  keepOpenOnSelect = false,
  wheelExplore = false,
}: {
  tag: string;
  value: string;
  options: { id: string; name: string; hint?: string; swatches?: string[] }[];
  onChange: (id: string) => void;
  align?: "start" | "end";
  width?: string;
  keepOpenOnSelect?: boolean;
  wheelExplore?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [popoverStyle, setPopoverStyle] = useState<CSSProperties | undefined>();
  const box = useRef<HTMLDivElement>(null);
  const active = options.find((o) => o.id === value) ?? options[0]!;

  const positionPopover = () => {
    const anchor = box.current?.querySelector<HTMLButtonElement>(":scope > button");
    if (!anchor || window.innerWidth > 560) {
      setPopoverStyle(undefined);
      return;
    }
    const rect = anchor.getBoundingClientRect();
    const visual = window.visualViewport;
    const viewportWidth = visual?.width ?? window.innerWidth;
    const viewportHeight = visual?.height ?? window.innerHeight;
    const width = Math.min(192, viewportWidth - 24);
    const margin = 12;
    const left = Math.max(margin, Math.min(rect.right - width, viewportWidth - width - margin));
    const estimatedHeight = Math.min(360, Math.max(96, options.length * 52));
    const below = rect.bottom + 6;
    const top = below + estimatedHeight <= viewportHeight - margin
      ? below
      : Math.max(margin, rect.top - estimatedHeight - 6);
    setPopoverStyle({
      left: `${Math.round(left)}px`,
      right: "auto",
      top: `${Math.round(top)}px`,
      width: `${Math.round(width)}px`,
    });
  };

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: PointerEvent) => {
      if (!box.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", onDoc);
    positionPopover();
    const visual = window.visualViewport;
    window.addEventListener("resize", positionPopover, { passive: true });
    visual?.addEventListener("resize", positionPopover, { passive: true });
    visual?.addEventListener("scroll", positionPopover, { passive: true });
    return () => {
      document.removeEventListener("pointerdown", onDoc);
      window.removeEventListener("resize", positionPopover);
      visual?.removeEventListener("resize", positionPopover);
      visual?.removeEventListener("scroll", positionPopover);
    };
  }, [open, options.length]);

  return (
    <div ref={box} className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-haspopup="menu"
        onWheel={(event) => {
          if (!wheelExplore || options.length < 2) return;
          event.preventDefault();
          const currentIndex = Math.max(0, options.findIndex((option) => option.id === value));
          const direction = event.deltaY > 0 ? 1 : -1;
          const nextIndex = (currentIndex + direction + options.length) % options.length;
          onChange(options[nextIndex]!.id);
        }}
        data-wheel-explore={wheelExplore ? "true" : undefined}
        className="plate flex h-9 min-w-[9.5rem] items-center gap-3 px-3 text-left transition-colors hover:border-foreground/40"
      >
        <span className="tag">{tag}</span>
        <span className="flex-1 truncate text-[11px] text-foreground">{active.name}</span>
        {active.swatches && (
          <span className="flex gap-[2px]">
            {active.swatches.slice(0, 4).map((c, i) => (
              <span key={i} className="h-2.5 w-[3px]" style={{ background: c }} />
            ))}
          </span>
        )}
        <span className="tag">{open ? "—" : "+"}</span>
      </button>
      {open && (
        <div
          className={`menu-popover menu-popover-${align} absolute z-50 mt-[-1px] max-h-[22rem] overflow-y-auto no-scrollbar`}
          style={popoverStyle ?? { width, [align === "end" ? "right" : "left"]: 0 }}
        >
          {options.map((o) => (
            <button
              key={o.id}
              onClick={() => {
                onChange(o.id);
                if (!keepOpenOnSelect) setOpen(false);
              }}
              role="menuitem"
              className={`flex w-full items-start gap-2 border-b border-border px-3 py-2.5 text-left transition-colors last:border-b-0 hover:bg-foreground/5 ${
                o.id === value ? "bg-foreground/[0.07]" : ""
              }`}
            >
              <span className="tag w-3 pt-[2px]">{o.id === value ? "▪" : ""}</span>
              <span className="flex-1">
                <span className="block text-[11px] text-foreground">{o.name}</span>
                {o.hint && <span className="mt-[3px] block text-[10px] text-muted-foreground">{o.hint}</span>}
              </span>
              {o.swatches && (
                <span className="flex gap-[2px] pt-[3px]">
                  {o.swatches.slice(0, 6).map((c, i) => (
                    <span key={i} className="h-3 w-[3px]" style={{ background: c }} />
                  ))}
                </span>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
