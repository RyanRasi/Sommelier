import { useState, useRef, useEffect, useCallback } from "react";

const API_URL = "http://localhost:8000";

const SUGGESTIONS = [
  "wine for medium rare steak",
  "fruity wine",
  "dry wine from France",
  "Italian red under $25",
  "something sweet for dessert",
  "recommend something interesting",
];

const RANKS = ["I", "II", "III"];

// ─────────────────────────────────────────────
// Skeleton loader
// ─────────────────────────────────────────────
function SkeletonCard() {
  return (
    <div className="skel-card">
      <div className="skeleton" style={{ width: "58%", height: 20, marginBottom: 14 }} />
      <div className="skeleton" style={{ width: "38%", height: 9 }} />
      <div className="skeleton" style={{ width: "32%", height: 9, marginBottom: 18 }} />
      <div className="skeleton" style={{ width: "100%", height: 9 }} />
      <div className="skeleton" style={{ width: "100%", height: 9 }} />
      <div className="skeleton" style={{ width: "68%", height: 9 }} />
    </div>
  );
}

// ─────────────────────────────────────────────
// Desktop card — description scrolls
// ─────────────────────────────────────────────
function DesktopCard({ rec, index }) {
  const whyRef = useRef(null);
  const [overflowing, setOverflowing] = useState(false);

  useEffect(() => {
    const el = whyRef.current;
    if (el) setOverflowing(el.scrollHeight > el.clientHeight + 4);
  }, [rec]);

  // Hide fade once user scrolls down
  const handleScroll = (e) => {
    if (e.target.scrollTop > 4) setOverflowing(false);
  };

  return (
    <div className="wine-card">
      <div className="card-top">
        <span className="rank">No. {RANKS[index] ?? index + 1}</span>
        <h2 className="card-title">{rec.title}</h2>
      </div>
      <div className="divider" />
      <div className="why-wrap">
        <p className="why" ref={whyRef} onScroll={handleScroll}>{rec.why}</p>
        {overflowing && <div className="why-fade" />}
      </div>
      <div className="detail-row">
        <div className="detail">
          <div className="detail-label">Food Pairing</div>
          <div className="detail-text">{rec.food_pairing}</div>
        </div>
        <div className="detail">
          <div className="detail-label">Serving Tip</div>
          <div className="detail-text">{rec.serving_tip}</div>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────
// Swipeable card stack — touch + mouse drag
// ─────────────────────────────────────────────
function SwipeStack({ results }) {
  const [active, setActive] = useState(0);
  const [dragX, setDragX] = useState(0);
  const [dragging, setDragging] = useState(false);
  const startX = useRef(0);
  const mouseDown = useRef(false);
  const containerRef = useRef(null);

  // ── Touch ──
  const onTouchStart = useCallback((e) => {
    startX.current = e.touches[0].clientX;
    setDragging(true);
  }, []);

  const onTouchMove = useCallback((e) => {
    setDragX(e.touches[0].clientX - startX.current);
  }, []);

  const onTouchEnd = useCallback(() => {
    commitDrag();
  }, [dragX, active, results.length]);

  // ── Mouse ──
  const onMouseDown = useCallback((e) => {
    mouseDown.current = true;
    startX.current = e.clientX;
    setDragging(true);
    e.preventDefault(); // prevent text selection while dragging
  }, []);

  const onMouseMove = useCallback((e) => {
    if (!mouseDown.current) return;
    setDragX(e.clientX - startX.current);
  }, []);

  const onMouseUp = useCallback(() => {
    if (!mouseDown.current) return;
    mouseDown.current = false;
    commitDrag();
  }, [dragX, active, results.length]);

  const onMouseLeave = useCallback(() => {
    if (!mouseDown.current) return;
    mouseDown.current = false;
    commitDrag();
  }, [dragX, active, results.length]);

  // ── Shared snap logic ──
  const commitDrag = useCallback(() => {
    if (dragX < -50 && active < results.length - 1) setActive((a) => a + 1);
    else if (dragX > 50 && active > 0) setActive((a) => a - 1);
    setDragX(0);
    setDragging(false);
  }, [dragX, active, results.length]);

  // Re-bind commitDrag to touch/mouse handlers that captured it in closure
  // by using a ref so they always call the latest version
  const commitRef = useRef(commitDrag);
  useEffect(() => { commitRef.current = commitDrag; }, [commitDrag]);

  const getCardStyle = (index) => {
    const cw = containerRef.current?.offsetWidth || 320;
    const offset = index - active;
    const drag = dragX / cw;
    const eff = offset - drag;
    const tx = eff * 105;
    const rot = eff * 5;
    const scale = Math.max(0.86, 1 - Math.abs(eff) * 0.07);
    const opacity = Math.abs(eff) > 1.6 ? 0 : 1;

    return {
      left: "50%",
      top: 0,
      transform: `translateX(calc(-50% + ${tx}%)) rotate(${rot}deg) scale(${scale})`,
      zIndex: 20 - Math.round(Math.abs(offset) * 2),
      opacity,
      cursor: dragging ? "grabbing" : "grab",
      transition: dragging
        ? "none"
        : "transform 0.32s cubic-bezier(.25,.46,.45,.94), opacity 0.32s ease",
    };
  };

  return (
    <div className="swipe-stack">
      <p className="stack-hint">← drag or swipe to explore →</p>
      <div
        ref={containerRef}
        className="stack-wrap"
        onTouchStart={onTouchStart}
        onTouchMove={onTouchMove}
        onTouchEnd={onTouchEnd}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
        onMouseLeave={onMouseLeave}
      >
        {results.map((rec, i) => (
          <div key={i} className="swipe-card" style={getCardStyle(i)}>
            <div className="card-top">
              <span className="rank">No. {RANKS[i] ?? i + 1}</span>
              <h2 className="card-title">{rec.title}</h2>
            </div>
            <div className="divider" />
            <p className="why swipe-why">{rec.why}</p>
            <div className="detail-row">
              <div className="detail">
                <div className="detail-label">Food Pairing</div>
                <div className="detail-text">{rec.food_pairing}</div>
              </div>
              <div className="detail">
                <div className="detail-label">Serving Tip</div>
                <div className="detail-text">{rec.serving_tip}</div>
              </div>
            </div>
          </div>
        ))}
      </div>
      <div className="dots">
        {results.map((_, i) => (
          <div
            key={i}
            className={`dot${i === active ? " dot-on" : ""}`}
            onClick={() => setActive(i)}
          />
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────
// Main App
// ─────────────────────────────────────────────
export default function App() {
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState("idle");
  const [results, setResults] = useState([]);
  const [errorMsg, setErrorMsg] = useState("");
  const [apiReady, setApiReady] = useState(true);

  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then((r) => r.json())
      .then((d) => setApiReady(d.status === "healthy"))
      .catch(() => setApiReady(false));
  }, []);

  const submit = async (overrideQuery) => {
    const q = (overrideQuery ?? query).trim();
    if (!q) return;
    setQuery(q);
    setStatus("loading");
    setResults([]);
    setErrorMsg("");

    try {
      const res = await fetch(`${API_URL}/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error ?? "Something went wrong.");
      }
      const data = await res.json();
      setResults(data.recommendations);
      setStatus("done");
    } catch (e) {
      setErrorMsg(e.message);
      setStatus("error");
    }
  };

  const ResultsContent = ({ mobile }) => (
    <>
      {status === "loading" && (
        <>
          <p className="results-label">Consulting the cellar&hellip;</p>
          {!mobile && [0, 1, 2].map((i) => <SkeletonCard key={i} />)}
        </>
      )}
      {status === "error" && (
        <div className="error-box">
          <p className="error-title">Something went wrong</p>
          <p className="error-detail">{errorMsg}</p>
        </div>
      )}
      {status === "done" && results.length > 0 && (
        <>
          <p className="results-label">3 selections for &ldquo;{query}&rdquo;</p>
          {mobile
            ? <SwipeStack results={results} />
            : results.map((rec, i) => <DesktopCard key={i} rec={rec} index={i} />)
          }
        </>
      )}
      {status === "idle" && (
        <div className="empty">
          <div className="empty-icon">🍷</div>
          <p className="empty-text">Your recommendations will appear here</p>
        </div>
      )}
    </>
  );

  return (
    <>
      <style>{CSS}</style>
      <div className="root">

        <header className="hero">
          <p className="eyebrow">AI-Powered Wine Discovery</p>
          <h1 className="title"><em>Sommelier</em></h1>
          <p className="subtitle">
            Every great meal deserves the right bottle — what&apos;s on your table tonight?
          </p>
          <div className="hero-rule" />
        </header>

        <div className="search-wrap">
          {!apiReady && (
            <p className="api-warning">
              ⚠ API offline — make sure the server is running on port 8000
            </p>
          )}
          <div className={`input-row${!apiReady ? " disabled" : ""}`}>
            <span className="input-icon">◈</span>
            <input
              className="input"
              placeholder="e.g. dry wine from France, wine for steak..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && submit()}
              disabled={status === "loading" || !apiReady}
            />
            <button
              className="btn"
              onClick={() => submit()}
              disabled={status === "loading" || !query.trim() || !apiReady}
            >
              {status === "loading" ? "···" : "Recommend"}
            </button>
          </div>
          <div className="suggestions">
            {SUGGESTIONS.slice(0, 5).map((s) => (
              <button key={s} className="chip" onClick={() => submit(s)}>{s}</button>
            ))}
          </div>
        </div>

        <div className="results desktop-results">
          <ResultsContent mobile={false} />
        </div>

        <div className="results mobile-results">
          <ResultsContent mobile={true} />
        </div>

      </div>
    </>
  );
}

// ─────────────────────────────────────────────
// Styles
// ─────────────────────────────────────────────
const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=DM+Sans:wght@300;400;500&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: #f7f0e8;
    color: #1a0a0e;
    font-family: 'DM Sans', sans-serif;
    min-height: 100vh;
  }

  .root { padding-bottom: 80px; }

  /* ── Hero ── */
  .hero { text-align: center; padding: 72px 24px 52px; }

  .eyebrow {
    font-size: 11px; font-weight: 400; letter-spacing: 4px;
    text-transform: uppercase; color: #b07060; margin-bottom: 16px;
  }

  .title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 64px; font-weight: 300; line-height: 1.05;
    color: #1a0a0e; letter-spacing: -0.5px;
  }
  .title em { font-style: italic; color: #8a1f35; }

  .subtitle {
    margin-top: 14px; font-size: 15px; font-weight: 300;
    color: #8a6a60; letter-spacing: 0.2px;
  }

  .hero-rule {
    width: 60px; height: 1px; background: #c4a090; margin: 28px auto 0;
  }

  /* ── Search ── */
  .search-wrap { max-width: 640px; margin: 0 auto; padding: 0 24px; }

  .api-warning {
    text-align: center; font-size: 13px; color: #b07040;
    margin-bottom: 12px; font-weight: 300;
  }

  .input-row {
    display: flex; background: #fff; border: 1px solid #e0cfc5;
    border-radius: 4px; overflow: hidden; transition: border-color 0.2s;
    box-shadow: 0 2px 12px rgba(100,40,30,0.06);
  }
  .input-row:focus-within { border-color: #8a1f35; }
  .input-row.disabled { opacity: 0.5; }

  .input-icon {
    display: flex; align-items: center; padding: 0 16px;
    color: #c4a090; font-size: 18px; flex-shrink: 0; user-select: none;
  }

  .input {
    flex: 1; background: transparent; border: none; outline: none;
    padding: 18px 8px; font-family: 'DM Sans', sans-serif;
    font-size: 15px; font-weight: 300; color: #1a0a0e; caret-color: #8a1f35;
  }
  .input::placeholder { color: #c4b0a8; }

  .btn {
    padding: 0 28px; background: #8a1f35; border: none; color: #fff;
    font-family: 'DM Sans', sans-serif; font-size: 12px; font-weight: 500;
    letter-spacing: 2px; text-transform: uppercase; cursor: pointer;
    transition: background 0.2s; flex-shrink: 0;
  }
  .btn:hover { background: #a82540; }
  .btn:active { background: #6e1828; }
  .btn:disabled { background: #ddd0cc; color: #b8a8a4; cursor: not-allowed; }

  .suggestions {
    display: flex; flex-wrap: wrap; gap: 8px;
    margin-top: 16px; justify-content: center;
  }

  .chip {
    font-size: 12px; font-weight: 300; color: #9a7060;
    border: 1px solid #e0cfc5; border-radius: 20px; padding: 5px 14px;
    cursor: pointer; background: transparent; transition: all 0.15s;
    font-family: 'DM Sans', sans-serif;
  }
  .chip:hover { border-color: #8a1f35; color: #8a1f35; background: #fdf5f0; }

  /* ── Results layout ── */
  .results { max-width: 700px; margin: 52px auto 0; padding: 0 24px; }

  .results-label {
    font-family: 'Cormorant Garamond', serif; font-size: 13px;
    font-weight: 400; letter-spacing: 3px; text-transform: uppercase;
    color: #c4a090; margin-bottom: 28px; text-align: center;
  }

  /* Desktop shown by default, mobile hidden */
  .desktop-results { display: block; }
  .mobile-results  { display: none;  }

  @media (max-width: 600px) {
    .desktop-results { display: none;  }
    .mobile-results  { display: block; }
    .title { font-size: 40px; }
    .btn   { padding: 0 16px; font-size: 11px; }
  }

  /* ── Desktop wine card ── */
  .wine-card {
    background: #fff; border: 1px solid #ecddd5; border-radius: 3px;
    padding: 28px 32px; margin-bottom: 14px; position: relative;
    overflow: hidden; animation: fadeUp 0.5s ease both;
    box-shadow: 0 2px 8px rgba(100,40,30,0.04); transition: border-color 0.2s;
  }
  .wine-card:nth-child(2) { animation-delay: 0.1s; }
  .wine-card:nth-child(3) { animation-delay: 0.2s; }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .wine-card::before {
    content: ''; position: absolute; top: 0; left: 0;
    width: 3px; height: 100%; background: #8a1f35;
    opacity: 0; transition: opacity 0.2s;
  }
  .wine-card:hover::before { opacity: 1; }
  .wine-card:hover { border-color: #dcc8bc; }

  /* ── Shared card internals ── */
  .card-top {
    display: flex; align-items: flex-start; gap: 14px; margin-bottom: 12px;
  }

  .rank {
    font-family: 'Cormorant Garamond', serif; font-size: 11px;
    font-weight: 400; letter-spacing: 3px; color: #8a1f35;
    text-transform: uppercase; flex-shrink: 0; padding-top: 4px;
  }

  .card-title {
    font-family: 'Cormorant Garamond', serif; font-size: 20px;
    font-weight: 400; color: #1a0a0e; line-height: 1.25;
  }

  .divider { height: 1px; background: #f0e4dc; margin: 0 0 12px; }

  /* ── Scrollable description (desktop) ── */
  .why-wrap { position: relative; margin-bottom: 14px; }

  .why {
    font-size: 13px; font-weight: 300; color: #5a3a30;
    line-height: 1.8; max-height: 88px;
    overflow-y: scroll;      /* always show scrollbar so user knows it scrolls */
    padding-right: 10px;     /* room for scrollbar */
    scroll-behavior: smooth;
  }

  /* Custom scrollbar — thin and on-brand */
  .why::-webkit-scrollbar       { width: 4px; }
  .why::-webkit-scrollbar-track { background: #f5ede6; border-radius: 2px; }
  .why::-webkit-scrollbar-thumb { background: #c4a090; border-radius: 2px; }
  .why::-webkit-scrollbar-thumb:hover { background: #8a1f35; }

  /* Fade hints there's more to read — disappears once scrolled */
  .why-fade {
    position: absolute; bottom: 0; left: 0;
    right: 14px;  /* don't overlap the scrollbar */
    height: 28px;
    background: linear-gradient(transparent, #fff);
    pointer-events: none;
  }

  /* ── Detail chips ── */
  .detail-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }

  .detail {
    background: #fdf7f3; border: 1px solid #ede0d8;
    border-radius: 2px; padding: 10px 12px;
  }
  .detail-label {
    font-size: 9px; font-weight: 500; letter-spacing: 2px;
    text-transform: uppercase; color: #c4a090; margin-bottom: 4px;
  }
  .detail-text { font-size: 12px; font-weight: 300; color: #6a4a40; line-height: 1.5; }

  /* ── Mobile swipe stack ── */
  .swipe-stack { width: 100%; }

  .stack-hint {
    text-align: center; font-size: 11px; font-weight: 300;
    color: #c4a090; letter-spacing: 1px; margin-bottom: 14px;
  }

  .stack-wrap {
    position: relative;
    height: 430px;
    overflow: visible;
    touch-action: pan-y;
    user-select: none;     /* prevent text selection while dragging */
  }

  .swipe-card {
    position: absolute; width: 82%; left: 50%;
    background: #fff; border: 1px solid #ecddd5; border-radius: 4px;
    padding: 22px 20px 18px;
    box-shadow: 0 4px 24px rgba(100,40,30,0.12);
    will-change: transform;
  }

  /* Description inside swipe card — full text, no truncation */
  .swipe-why {
    font-size: 13px; font-weight: 300; color: #5a3a30;
    line-height: 1.8; margin-bottom: 14px;
    max-height: none;  /* show all text on the swipe card */
    overflow: visible;
    padding-right: 0;
  }

  /* Dot indicators */
  .dots { display: flex; justify-content: center; gap: 7px; margin-top: 18px; }

  .dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #e0cfc5; transition: all 0.2s; cursor: pointer;
  }
  .dot-on { background: #8a1f35; width: 18px; border-radius: 3px; }

  /* ── Skeleton ── */
  .skel-card {
    background: #fff; border: 1px solid #ecddd5; border-radius: 3px;
    padding: 24px 28px; margin-bottom: 12px;
  }
  .skeleton {
    display: block; border-radius: 2px; margin-bottom: 9px;
    animation: shimmer 1.4s infinite;
    background: linear-gradient(90deg, #f5ede6 25%, #ecddd5 50%, #f5ede6 75%);
    background-size: 200% 100%;
  }
  @keyframes shimmer {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
  }

  /* ── Error ── */
  .error-box {
    background: #fff5f5; border: 1px solid #f0d0d0;
    border-radius: 3px; padding: 24px; text-align: center;
  }
  .error-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 18px; color: #a03030; margin-bottom: 8px;
  }
  .error-detail { font-size: 13px; font-weight: 300; color: #8a5050; }

  /* ── Empty state ── */
  .empty { text-align: center; padding: 80px 24px; }
  .empty-icon { font-size: 36px; margin-bottom: 14px; opacity: 0.25; }
  .empty-text {
    font-family: 'Cormorant Garamond', serif; font-size: 18px;
    font-weight: 300; font-style: italic; color: #c4a090;
  }
`;
