// The STELLIUM lockup. A stellium is three or more bodies gathered in one sign: the mark draws
// the sign as a ring and the gathering as three bodies close together on a tilted orbit, the
// brightest one a four-point star. Pure vector, so it stays sharp at any pixel density.
export function Wordmark() {
  return (
    <a className="wordmark" href="/" aria-label="STELLIUM home">
      <svg className="wordmark-mark" viewBox="0 0 48 48" aria-hidden="true" shapeRendering="geometricPrecision">
        <circle cx="24" cy="24" r="21.5" fill="none" stroke="currentColor" strokeOpacity="0.42" strokeWidth="1.1" />
        <circle cx="24" cy="24" r="1.1" fill="currentColor" fillOpacity="0.8" />
        <g className="wordmark-orbit">
          <ellipse cx="24" cy="24" rx="21.5" ry="7.4" transform="rotate(-28 24 24)" fill="none"
            stroke="currentColor" strokeOpacity="0.85" strokeWidth="1.15" />
          <circle cx="25.22" cy="15.22" r="2.1" fill="currentColor" />
          <circle cx="42.52" cy="13.28" r="2.7" fill="currentColor" />
          <path className="wordmark-star"
            d="M36.31 5.67 C36.81 10.17 38.21 11.57 42.71 12.07 C38.21 12.57 36.81 13.97 36.31 18.47 C35.81 13.97 34.41 12.57 29.91 12.07 C34.41 11.57 35.81 10.17 36.31 5.67 Z" />
        </g>
      </svg>
      <span className="wordmark-text">
        <span className="wordmark-name">Stellium</span>
        <span className="wordmark-sub">Agentic GraphRAG on TigerGraph</span>
      </span>
    </a>
  );
}
