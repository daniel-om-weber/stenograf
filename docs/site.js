// Shared by every language version of the page. It holds no visible text:
// labels and captions are read from data attributes in the HTML.
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const calm = matchMedia("(prefers-reduced-motion: reduce)").matches;

  // Tabs. Without this script every panel stays visible under its own name.
  $$('[role="tablist"]').forEach((list) => {
    const tabs = $$('[role="tab"]', list);
    const select = (tab, focus) => {
      tabs.forEach((t) => {
        const on = t === tab;
        t.setAttribute("aria-selected", on);
        t.tabIndex = on ? 0 : -1;
        document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
      });
      if (focus) tab.focus();
    };
    tabs.forEach((t, i) => {
      t.addEventListener("click", () => select(t));
      t.addEventListener("keydown", (e) => {
        if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
        e.preventDefault();
        select(tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length], true);
      });
    });
    let first = tabs[0];
    if ("os" in list.dataset) {
      const p = ((navigator.userAgentData && navigator.userAgentData.platform) || navigator.platform || "").toLowerCase();
      const android = navigator.userAgent.toLowerCase().includes("android");
      const os = p.includes("win") ? "windows" : p.includes("linux") && !android ? "linux" : "macos";
      first = tabs.find((t) => t.dataset.os === os) || first;
    }
    select(first);
  });

  // Copy buttons.
  $$(".copy").forEach((b) => b.addEventListener("click", async () => {
    const el = document.getElementById(b.dataset.copy);
    b.dataset.label = b.dataset.label || b.textContent;
    let done = false;
    try { await navigator.clipboard.writeText(el.textContent.trim()); done = true; }
    catch {
      const range = document.createRange();
      range.selectNodeContents(el);
      const sel = getSelection();
      sel.removeAllRanges(); sel.addRange(range);
      try { done = document.execCommand("copy"); } catch {}
    }
    b.textContent = done ? b.dataset.copied : b.dataset.selected;
    setTimeout(() => { b.textContent = b.dataset.label; }, 1800);
  }));

  // The window in the hero: live captions, then the finished transcript.
  const app = $("#demo");
  const rows = $$(".caps li", app);
  const buttons = $$(".seg button");
  const note = $("#demo-note"), phase = $("#demo-phase"), clock = $("#demo-clock");
  rows.forEach((r) => { r.dataset.live = $(".who", r).textContent; r.dataset.text = $("p", r).textContent; });

  let seconds = 12 * 60 + 36, ticker = null, timers = [];
  const show = () => { clock.textContent = Math.floor(seconds / 60) + ":" + String(seconds % 60).padStart(2, "0"); };
  const settle = () => {
    timers.forEach(clearTimeout); timers = [];
    rows.forEach((r) => { r.style.visibility = ""; $("p", r).textContent = r.dataset.text; });
    seconds = Math.max(seconds, 12 * 60 + 36); show();  // never earlier than the last line's stamp
  };
  const setState = (state) => {
    app.dataset.state = state;
    buttons.forEach((b) => b.setAttribute("aria-pressed", b.dataset.state === state));
    rows.forEach((r) => { $(".who", r).textContent = state === "live" ? r.dataset.live : r.dataset.final; });
    phase.textContent = state === "live" ? "LIVE" : "DONE";
    note.textContent = note.dataset[state];
    clearInterval(ticker);
    if (state === "done") settle();
    else if (!calm) ticker = setInterval(() => { if (seconds < 14 * 60) { seconds += 1; show(); } }, 1000);
  };
  buttons.forEach((b) => b.addEventListener("click", () => setState(b.dataset.state)));

  // The last two lines arrive word by word, the newest words still provisional.
  const span = (cls, text) => { const el = document.createElement("span"); el.className = cls; el.textContent = text; return el; };
  const type = (row, at) => {
    const p = $("p", row), words = row.dataset.text.split(" ");
    timers.push(setTimeout(() => { p.textContent = ""; p.append(span("rest", row.dataset.text)); row.style.visibility = ""; }, at));
    words.forEach((_, i) => timers.push(setTimeout(() => {
      if (i === words.length - 1) { p.textContent = row.dataset.text; return; }
      const cut = Math.max(0, i - 1);
      p.textContent = words.slice(0, cut).join(" ") + (cut ? " " : "");
      p.append(span("tail", words.slice(cut, i + 1).join(" ")), span("rest", " " + words.slice(i + 1).join(" ")));
    }, at + 150 + i * 130)));
    return at + 150 + words.length * 130;
  };
  if (!calm) {
    seconds = 12 * 60 + 22; show();
    rows.forEach((r) => { if (r.hasAttribute("data-late")) { r.style.visibility = "hidden"; r.removeAttribute("data-late"); } });
    const next = type(rows[3], 1200);
    type(rows[4], next + 1400);
  }
  setState("live");

  // The language link is an explicit choice; the start page honours it from then on.
  $$("a[data-lang]").forEach((a) => a.addEventListener("click", () => {
    try { localStorage.setItem("lang", a.dataset.lang); } catch {}
  }));
})();
