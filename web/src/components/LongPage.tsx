// A document-style page over the dimmed graph: contents on the left, readable prose on the right.
import type { ReactNode } from "react";

export function LongPage({ sections, children }: { sections: { id: string; title: string }[]; children: ReactNode }) {
  return (
    <section className="page pointer-events-auto flex h-full w-full overflow-hidden">
      <nav className="page-nav hidden w-56 flex-none overflow-y-auto border-r border-[color:var(--hair)] p-4 lg:block">
        <div className="tag px-2.5 pb-2">Contents</div>
        {sections.map((section) => (
          <a
            key={section.id}
            href={`#${section.id}`}
            onClick={(event) => {
              event.preventDefault();
              document.getElementById(section.id)?.scrollIntoView({ behavior: "smooth", block: "start" });
            }}
          >
            {section.title}
          </a>
        ))}
      </nav>
      <article className="prose no-scrollbar min-w-0 flex-1 overflow-y-auto px-6 py-7 md:px-10">{children}</article>
    </section>
  );
}

export function Section({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <section id={id}>
      <h2 id={`${id}-title`}>{title}</h2>
      {children}
    </section>
  );
}
