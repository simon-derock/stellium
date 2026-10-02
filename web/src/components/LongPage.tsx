// A document-style page over the dimmed graph: contents on the left, prose on the right.
// Scrolling is inertia-smoothed (Lenis) and layered for depth: the page title drifts slower than
// the text and fades, sections rise in (CSS scroll-driven), and the scroll position is reported
// so the graph behind can move on its own, deeper layer.
import Lenis from "lenis";
import { useEffect, useRef, useState, type ReactNode } from "react";

interface Props {
  sections: { id: string; title: string }[];
  children: ReactNode;
  onScroll?: (top: number) => void;
}

export function LongPage({ sections, children, onScroll }: Props) {
  const scroller = useRef<HTMLElement>(null);
  const content = useRef<HTMLDivElement>(null);
  const lenis = useRef<Lenis | null>(null);
  const report = useRef(onScroll);
  report.current = onScroll;
  const [current, setCurrent] = useState(sections[0]?.id ?? "");

  useEffect(() => {
    const wrapper = scroller.current;
    const inner = content.current;
    if (!wrapper || !inner) return;
    const publish = (top: number) => {
      wrapper.style.setProperty("--scroll", top.toFixed(1));
      report.current?.(top);
    };
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      const onNative = () => publish(wrapper.scrollTop);
      wrapper.addEventListener("scroll", onNative, { passive: true });
      return () => wrapper.removeEventListener("scroll", onNative);
    }
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
