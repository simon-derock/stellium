// A floating pill bar whose hover highlight glides between items, settling on the active one.
import { useCallback, useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";

export function Pill({ children, label, activeKey }: { children: ReactNode; label: string; activeKey?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [glide, setGlide] = useState({ x: 0, y: 0, w: 0, h: 0, on: false });
  // The first placement (and re-placements after layout shifts) jump; only pointer moves glide.
  const [animate, setAnimate] = useState(false);

  const moveTo = useCallback((target: Element | null) => {
    const box = ref.current;
    if (!box || !(target instanceof HTMLElement)) {
      setGlide((g) => ({ ...g, on: false }));
      return;
    }
    const inner = box.getBoundingClientRect();
    const rect = target.getBoundingClientRect();
    setGlide({
      x: rect.left - inner.left - box.clientLeft,
      y: rect.top - inner.top - box.clientTop,
      w: rect.width,
      h: rect.height,
      on: true,
    });
  }, []);

  const settle = useCallback(() => {
    moveTo(ref.current?.querySelector(`[data-glide="${activeKey ?? ""}"]`) ?? null);
  }, [activeKey, moveTo]);

  useLayoutEffect(settle, [settle]);
  // Item widths change when the web fonts arrive and when the bar reflows; measure again then.
  useEffect(() => {
    const box = ref.current;
    if (!box) return;
    void document.fonts?.ready.then(settle);
    const observer = new ResizeObserver(settle);
    observer.observe(box);
    box.querySelectorAll("[data-glide]").forEach((item) => observer.observe(item));
    window.addEventListener("resize", settle, { passive: true });
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", settle);
    };
  }, [settle]);

  return (
    <div
      ref={ref}
      role="group"
      aria-label={label}
      className="pill"
      onMouseOver={(event) => {
        setAnimate(true);
        moveTo((event.target as Element).closest("[data-glide]"));
      }}
      onMouseLeave={settle}
    >
      <span
        className="glide"
        aria-hidden="true"
        style={{
          width: glide.w,
          height: glide.h,
          transform: `translate3d(${glide.x}px, ${glide.y}px, 0)`,
          opacity: glide.on ? 1 : 0,
          transition: animate ? undefined : "none",
        }}
      />
      {children}
    </div>
  );
}
