

(function () {
  function markInheritedToc() {
    const tag = document.getElementById("api-inherited-anchors");
    if (!tag) return;

    let anchors = [];
    try { anchors = JSON.parse(tag.textContent || "[]"); } catch { return; }
    if (!anchors.length) return;

    const root = document.querySelector(".md-sidebar--secondary");
    const toc = root?.querySelector('ul[data-md-component="toc"]');
    if (!toc) return;

    for (const a of anchors) {
      const link = toc.querySelector(`a.md-nav__link[href="#${CSS.escape(a)}"]`);
      if (link) link.classList.add("api-inherited-toc");
    }
  }

  function applyTocToggles() {
    const root = document.querySelector(".md-sidebar--secondary");
    const toc = root?.querySelector('ul[data-md-component="toc"]');
    if (!toc) return;

    toc.querySelectorAll("li.md-nav__item").forEach((li) => {
      const link = li.querySelector(":scope > a.md-nav__link");
      const childNav = li.querySelector(":scope > nav.md-nav");
      if (!link || !childNav) return;

      // avoid double-adding
      if (link.querySelector(".toc-expander")) return;

      let open = false;
      childNav.style.display = open ? "" : "none";

      const exp = document.createElement("span");
      exp.className = "toc-expander";
      exp.textContent = open ? "[-]" : "[+]";
      exp.style.float = "right";
      link.appendChild(exp);

      link.addEventListener("click", (e) => {
        if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return; // allow modified-click navigation
        e.preventDefault();
        open = !open;
        childNav.style.display = open ? "" : "none";
        exp.textContent = open ? "[-]" : "[+]";

        // Re-apply inherited marking in case DOM was rebuilt
        markInheritedToc();
      });
    });
  }

  function init() {
    applyTocToggles();
    markInheritedToc();
  }

  if (window.document$) {
    window.document$.subscribe(init);
  } else {
    document.addEventListener("DOMContentLoaded", init);
  }
})();