// Menu ported from Lunarbit (wheel-to-cycle, popover placement), restyled as a fixed-width
// segment so the control bar never changes size with the selected label.
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
        data-open={open ? "true" : undefined}
        className="control"
        title={`${tag}: ${active.name} (scroll to cycle)`}
      >
        <span className="control-tag">{tag}</span>
        <span className="control-value">{active.name}</span>
        {active.swatches && (
          <span className="control-swatches" aria-hidden="true">
            {active.swatches.slice(0, 5).map((c, i) => (
              <span key={i} style={{ background: c }} />
            ))}
          </span>
        )}
        <svg className="control-chevron" viewBox="0 0 12 12" aria-hidden="true">
          <path d="M3 4.5 6 7.5 9 4.5" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
      {open && (
        <div
          role="menu"
          className={`menu-popover menu-popover-${align} no-scrollbar`}
          style={popoverStyle ?? { width, [align === "end" ? "right" : "left"]: 0 }}
        >
          <div className="menu-heading">
            <span>{tag}</span>
            <span>{options.length} options · scroll the button to cycle</span>
          </div>
          {options.map((o) => (
            <button
              key={o.id}
              onClick={() => {
                onChange(o.id);
                if (!keepOpenOnSelect) setOpen(false);
              }}
              role="menuitemradio"
              aria-checked={o.id === value}
              className="menu-item"
            >
              <span className="menu-item-mark" aria-hidden="true" />
              <span className="min-w-0 flex-1">
                <span className="menu-item-name">{o.name}</span>
                {o.hint && <span className="menu-item-hint">{o.hint}</span>}
              </span>
              {o.swatches && (
                <span className="menu-item-swatches" aria-hidden="true">
                  {o.swatches.slice(0, 6).map((c, i) => (
                    <span key={i} style={{ background: c }} />
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
