// The STELLIUM lockup: a stellium is a cluster of bodies in one sign, drawn here as three stars on
// an orbit with the brightest in the accent colour, set beside a wide-tracked display serif.
export function Wordmark() {
  return (
    <a className="wordmark" href="/" aria-label="STELLIUM home">
      <svg className="wordmark-glyph" viewBox="0 0 32 32" aria-hidden="true">
        <ellipse cx="16" cy="17" rx="13.5" ry="6.2" transform="rotate(-18 16 17)" fill="none"
          stroke="currentColor" strokeOpacity="0.32" strokeWidth="0.6" />
        <path d="M6.2 21.4 L15.6 13.2 L25.4 15.6" fill="none" stroke="currentColor"
          strokeOpacity="0.55" strokeWidth="0.7" strokeLinecap="round" />
        <circle cx="6.2" cy="21.4" r="1.5" fill="currentColor" />
        <circle cx="25.4" cy="15.6" r="1.8" fill="currentColor" />
        <path d="M15.6 6.6 Q16.1 12.7 22.2 13.2 Q16.1 13.7 15.6 19.8 Q15.1 13.7 9 13.2 Q15.1 12.7 15.6 6.6 Z"
          fill="var(--accent)" />
      </svg>
      <span>
        <span className="wordmark-type block">STELLIUM</span>
        <span className="wordmark-sub block">Agentic GraphRAG · TigerGraph</span>
      </span>
    </a>
  );
}
