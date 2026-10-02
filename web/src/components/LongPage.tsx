// A document-style page over the dimmed graph: contents on the left, prose on the right. Sections
// rise into place as they scroll in (pure CSS), the contents track the section in view, and the
// scroll position is reported so the canvas behind can drift for depth.
import { useEffect, useRef, useState, type ReactNode } from "react";

interface Props {
  sections: { id: string; title: string }[];
  children: ReactNode;
  onScroll?: (top: number) => void;
}

export function LongPage({ sections, children, onScroll }: Props) {
  const scroller = useRef<HTMLElement>(null);
  const [current, setCurrent] = useState(sections[0]?.id ?? "");

  useEffect(() => {
    const root = scroller.current;
    if (!root) return;
    const spy = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
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
              document.getElementById(section.id)?.scrollIntoView({ behavior: "smooth", block: "start" });
            }}
          >
            {section.title}
          </a>
        ))}
      </nav>
      <article
        ref={scroller}
        className="prose no-scrollbar min-w-0 flex-1 overflow-y-auto scroll-smooth px-6 py-8 md:px-12"
        onScroll={(event) => onScroll?.(event.currentTarget.scrollTop)}
      >
        <div className="mx-auto max-w-[60rem]">{children}</div>
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
