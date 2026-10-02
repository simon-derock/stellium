// A document-style page over the dimmed graph: contents on the left, prose on the right.
// Scrolling is inertia-smoothed (Lenis) and layered for depth: the page title drifts slower than
// the text and fades, and sections rise in as they arrive. The graph behind stays still.
import Lenis from "lenis";
import { useEffect, useRef, useState, type ReactNode } from "react";

interface Props {
  sections: { id: string; title: string }[];
  children: ReactNode;
}

export function LongPage({ sections, children }: Props) {
  const scroller = useRef<HTMLElement>(null);
  const content = useRef<HTMLDivElement>(null);
  const lenis = useRef<Lenis | null>(null);
  const [current, setCurrent] = useState(sections[0]?.id ?? "");

  useEffect(() => {
    const wrapper = scroller.current;
    const inner = content.current;
    if (!wrapper || !inner) return;
    // Title drift: written straight onto the title block. A custom property on the scroller
    // would be inherited by the whole page and restyle all of it on every frame.
    const titles = Array.from(inner.querySelectorAll<HTMLElement>(":scope > h1, :scope > .lede"));
    let last = -1;
    const publish = (top: number) => {
      const y = Math.min(top, 300);
      if (y === last) return;
      last = y;
      for (const title of titles) {
        title.style.transform = `translate3d(0, ${(y * 0.32).toFixed(1)}px, 0)`;
        title.style.opacity = String(Math.max(0, 1 - y / 240));
      }
    };
    // Reduced motion: native scrolling, and the title stays put.
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const smooth = new Lenis({ wrapper, content: inner, lerp: 0.085, smoothWheel: true });
    lenis.current = smooth;
    smooth.on("scroll", ({ scroll }: { scroll: number }) => publish(scroll));
    let frame = requestAnimationFrame(function tick(time) {
      smooth.raf(time);
      frame = requestAnimationFrame(tick);
    });
    return () => {
      cancelAnimationFrame(frame);
      smooth.destroy();
      lenis.current = null;
    };
  }, []);

  // Scroll reveal: number the blocks in each section for the stagger, show whatever is already in
  // view at once, and reveal the rest as it enters. Armed only after the observer exists.
  useEffect(() => {
    const root = scroller.current;
    if (!root || !("IntersectionObserver" in window)) return;
    const sections = Array.from(root.querySelectorAll<HTMLElement>(".reveal"));
    for (const section of sections) {
      Array.from(section.children).forEach((child, index) => {
        (child as HTMLElement).style.setProperty("--i", String(Math.min(index, 8)));
      });
    }
    const reveal = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          (entry.target as HTMLElement).dataset.shown = "true";
          reveal.unobserve(entry.target);
        }
      },
      { root, rootMargin: "0px 0px -8% 0px", threshold: 0.02 },
    );
    const view = root.getBoundingClientRect();
    for (const section of sections) {
      const box = section.getBoundingClientRect();
      if (box.top < view.bottom && box.bottom > view.top) section.dataset.shown = "true";
      else reveal.observe(section);
    }
    root.dataset.reveal = "on";
    return () => reveal.disconnect();
  }, [children]);

  useEffect(() => {
    const root = scroller.current;
    if (!root) return;
    const spy = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible[0]) setCurrent(visible[0].target.id);
      },
      { root, rootMargin: "0px 0px -65% 0px" },
    );
    root.querySelectorAll("section[id]").forEach((el) => spy.observe(el));
    return () => spy.disconnect();
  }, [children]);

  return (
    <section className="page pointer-events-auto flex h-full w-full overflow-hidden">
      <nav className="page-nav hidden w-56 flex-none overflow-y-auto border-r border-[color:var(--hair)] px-3 py-6 lg:block">
        <div className="px-3 pb-2 text-[12px] font-semibold text-[color:var(--faint)]">On this page</div>
        {sections.map((section) => (
          <a
            key={section.id}
            href={`#${section.id}`}
            aria-current={current === section.id ? "true" : undefined}
            onClick={(event) => {
              event.preventDefault();
              const target = document.getElementById(section.id);
              if (!target) return;
              if (lenis.current) lenis.current.scrollTo(target, { offset: -16, duration: 1.1 });
              else target.scrollIntoView({ behavior: "smooth", block: "start" });
            }}
          >
            {section.title}
          </a>
        ))}
      </nav>
      <article ref={scroller} className="prose page-scroll no-scrollbar min-w-0 flex-1 overflow-y-auto px-6 py-8 md:px-12">
        <div ref={content} className="mx-auto max-w-[60rem]">
          {children}
        </div>
      </article>
    </section>
  );
}

export function Section({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <section id={id} className="reveal">
      <h2>{title}</h2>
      {children}
    </section>
  );
}
