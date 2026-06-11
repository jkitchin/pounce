// POUNCE docs version selector + context banner.
//
// This is the canonical selector. The docs build (scripts/build-versioned-docs.sh)
// copies this file into every built book — including archived tag builds whose
// own source predates it — and injects a
//
//     <script defer src="<path_to_root>versions.js"
//             id="pounce-versions" data-p2r="<path_to_root>"></script>
//
// tag into every page, where <path_to_root> is mdBook's per-page relative path
// back to that version's root. We read that prefix from our own <script> tag's
// data-p2r, so this file never hardcodes the site base (`/pounce/`) and works
// served at any path or domain.
//
// Layout it expects on the deployed site (see versions.json):
//   <siteRoot>/            -> stable (latest release)
//   <siteRoot>/dev/        -> main
//   <siteRoot>/vX.Y.Z/     -> archived releases
//   <siteRoot>/versions.json (copied into every book root)
(function () {
  var SELF_ID = "pounce-versions";

  function self() {
    return document.getElementById(SELF_ID) || document.currentScript;
  }

  // Absolute URL of *this version's* root directory, from our injected data-p2r.
  function versionRoot() {
    var el = self();
    var p2r = (el && el.getAttribute("data-p2r")) || "";
    // p2r is "" at a version's root page, "../" one level deep, etc.
    // `|| "."` turns "" into the current directory rather than the page itself.
    return new URL(p2r || ".", window.location.href);
  }

  function resolve(manifest) {
    var vr = versionRoot();
    var vrPath = vr.pathname; // e.g. /pounce/dev/ , /pounce/ , /pounce/v0.4.0/
    var versions = (manifest && manifest.versions) || [];

    // Longest non-empty path first so v0.4.0 doesn't shadow a hypothetical
    // nested path, and so "" (stable at root) is only chosen as a last resort.
    var byLen = versions
      .filter(function (v) { return v.path; })
      .sort(function (a, b) { return b.path.length - a.path.length; });

    var current = null;
    var siteRoot = vr;
    for (var i = 0; i < byLen.length; i++) {
      var seg = "/" + byLen[i].path + "/";
      if (vrPath.length >= seg.length && vrPath.slice(-seg.length) === seg) {
        current = byLen[i];
        siteRoot = new URL(vrPath.slice(0, vrPath.length - (byLen[i].path.length + 1)), vr);
        break;
      }
    }
    if (!current) {
      current = versions.filter(function (v) { return v.path === ""; })[0] || null;
      siteRoot = vr; // we are at the stable root
    }

    // Current page path within this version (e.g. "options.html",
    // "schema/solve-report-v1.html", or "" on the index).
    var relPage = window.location.pathname;
    if (relPage.indexOf(vrPath) === 0) relPage = relPage.slice(vrPath.length);

    return { versions: versions, current: current, siteRoot: siteRoot, relPage: relPage };
  }

  function targetUrlFor(entry, ctx) {
    var root = new URL(entry.path ? entry.path + "/" : "", ctx.siteRoot);
    return new URL(ctx.relPage || "index.html", root);
  }

  // Navigate to the same page in another version, falling back to that
  // version's index when the page does not exist there (e.g. a page added
  // after an older release).
  function goTo(entry, ctx) {
    var target = targetUrlFor(entry, ctx);
    var fallback = new URL(entry.path ? entry.path + "/" : "", ctx.siteRoot);
    if (!ctx.relPage) { window.location.assign(target.href); return; }
    try {
      fetch(target.href, { method: "HEAD" })
        .then(function (r) {
          window.location.assign(r && r.ok ? target.href : fallback.href);
        })
        .catch(function () { window.location.assign(fallback.href); });
    } catch (e) {
      window.location.assign(target.href);
    }
  }

  function buildSelector(ctx) {
    var bar = document.querySelector(".right-buttons");
    if (!bar || document.getElementById("pounce-version-select")) return;

    var wrap = document.createElement("div");
    wrap.className = "pounce-version-wrap";

    var sel = document.createElement("select");
    sel.id = "pounce-version-select";
    sel.className = "pounce-version-select";
    sel.setAttribute("aria-label", "Documentation version");
    sel.title = "Documentation version";

    ctx.versions.forEach(function (v) {
      var opt = document.createElement("option");
      opt.value = v.id;
      opt.textContent = v.label;
      if (ctx.current && v.id === ctx.current.id) opt.selected = true;
      sel.appendChild(opt);
    });

    sel.addEventListener("change", function () {
      var chosen = ctx.versions.filter(function (v) { return v.id === sel.value; })[0];
      if (chosen) goTo(chosen, ctx);
    });

    wrap.appendChild(sel);
    // Put the version selector at the front of the right-side buttons.
    bar.insertBefore(wrap, bar.firstChild);
  }

  function buildBanner(ctx) {
    if (!ctx.current || ctx.current.kind === "stable") return;
    if (document.getElementById("pounce-version-banner")) return;

    var main = document.querySelector("main") || document.querySelector(".content");
    if (!main) return;

    var stable = ctx.versions.filter(function (v) { return v.kind === "stable"; })[0];
    var stableHref = stable ? targetUrlFor(stable, ctx).href : ctx.siteRoot.href;
    var stableLabel = stable ? stable.id : "the latest release";

    var msg;
    if (ctx.current.kind === "dev") {
      msg = "You are reading the in-development docs (unreleased). ";
    } else {
      msg = "You are reading docs for " + ctx.current.id + ", an older release. ";
    }

    var banner = document.createElement("div");
    banner.id = "pounce-version-banner";
    banner.className = "pounce-version-banner pounce-version-banner-" + ctx.current.kind;
    var span = document.createElement("span");
    span.textContent = msg;
    var a = document.createElement("a");
    a.href = stableHref;
    a.textContent = "Go to the current release (" + stableLabel + ").";
    banner.appendChild(span);
    banner.appendChild(a);

    main.insertBefore(banner, main.firstChild);
  }

  function init() {
    var manifestUrl;
    try {
      manifestUrl = new URL("versions.json", versionRoot());
    } catch (e) {
      return; // can't resolve our own location; do nothing
    }
    fetch(manifestUrl.href, { cache: "no-cache" })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (manifest) {
        if (!manifest) return;
        var ctx = resolve(manifest);
        if (!ctx.versions.length) return;
        buildSelector(ctx);
        buildBanner(ctx);
      })
      .catch(function () { /* offline / first deploy: render nothing */ });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
