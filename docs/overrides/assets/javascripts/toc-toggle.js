(function () {
  function applyToggles() {
    const toc = document.querySelector(".md-sidebar--secondary nav.md-nav");
    if (!toc) return;

    const items = toc.querySelectorAll("li.md-nav__item");
    items.forEach((li) => {
      const link = li.querySelector(":scope > a.md-nav__link");
      const childNav = li.querySelector(":scope > nav.md-nav");
      if (!link || !childNav) return;

      const childList = childNav.querySelector(":scope > ul.md-nav__list");
      if (!childList) return;

      // Avoid double injection (Material instant navigation)
      if (link.querySelector(".toc-expander")) return;

      // Default collapsed
      let open = false;
      childNav.style.display = open ? "" : "none";

      const exp = document.createElement("span");
      exp.className = "toc-expander";
      exp.textContent = open ? "[-]" : "[+]";

      // Put expander after the link text
      link.append(exp);

      exp.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        open = !open;
        childNav.style.display = open ? "" : "none";
        exp.textContent = open ? "[-]" : "[+]";
      });
    });
  }

  if (window.document$) {
    window.document$.subscribe(applyToggles);
  } else {
    document.addEventListener("DOMContentLoaded", applyToggles);
  }
})();