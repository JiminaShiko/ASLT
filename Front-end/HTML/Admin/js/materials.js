(() => {
    "use strict";

    /* ==========================================================
       CONFIG: change these to match your backend
       ========================================================== */
    const DEMO_MODE = true;                       // set to false to use the real API
    const MATERIALS_ENDPOINT = "/api/admin/materials";
    const PROFILE_ENDPOINT = "/api/admin/profile";
    const SIDEBAR_URL = "/admin-dashboard/admin_sidebar.html";
    const SIDEBAR_ALIASES = ["sign_material.html", "materials.html"]; // pages that highlight "Sign Materials"
    const CATEGORIES = ["Alphabets", "Words", "Numbers"];
    const PAGE_SIZE = 8;
    const MAX_FILE_BYTES = 25 * 1024 * 1024;
    const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp", "video/mp4", "video/webm"];
    /* Expected item shape:
       { id, title, category, description, status: "published"|"draft",
         media_type: "image"|"video", media_url, updated_at (ISO string) }  */

    const byId = (id) => document.getElementById(id);
    const app = byId("app");
    const state = { items: [], filters: { q: "", category: "", status: "" }, page: 1, loading: true, error: "", editing: null, deleting: null };

    const ICON = {
      edit: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 20h4L19 9l-4-4L4 16v4Z"/><path d="m13.5 6.5 4 4"/></svg>',
      trash: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/></svg>',
      play: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>',
      hand: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 13V6a1.5 1.5 0 0 1 3 0v5m0-6.5a1.5 1.5 0 0 1 3 0V11m0-4.5a1.5 1.5 0 0 1 3 0V13m0-3.5a1.5 1.5 0 0 1 3 0V15a7 7 0 0 1-7 7h-1a7 7 0 0 1-5.6-2.8L3.5 15.5a1.5 1.5 0 0 1 2.3-1.9L8 16"/></svg>',
      alert: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3 2 20h20L12 3ZM12 10v4M12 17.5v.01"/></svg>',
      search: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/></svg>'
    };

    /* ---------------- Demo data ---------------- */
    const daysAgo = (n) => new Date(Date.now() - n * 86400000).toISOString();
    let demoItems = [
      ["Letter A", "Alphabets", "Closed fist with the thumb resting against the side of the index finger.", "published", "image", 2],
      ["Letter B", "Alphabets", "Flat hand, fingers together, thumb folded across the palm.", "published", "image", 3],
      ["Letter C", "Alphabets", "Curve the hand into a C shape, thumb opposite the fingers.", "draft", "image", 5],
      ["Number 5", "Numbers", "Open hand, all five fingers spread.", "published", "image", 6],
      ["Hello", "Words", "Salute-like motion moving away from the forehead.", "published", "video", 7],
      ["Thank you", "Words", "Flat hand from the chin moving forward and down.", "published", "video", 9]
    ].map((r, i) => ({ id: i + 1, title: r[0], category: r[1], description: r[2], status: r[3], media_type: r[4], media_url: "", updated_at: daysAgo(r[5]) }));
    let demoNextId = demoItems.length + 1;
    const delay = (ms) => new Promise((r) => setTimeout(r, ms));

    /* ---------------- Utilities ---------------- */
    function esc(value) {
      return String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
    }

    function safeUrl(value) {
      if (!value) return null;
      try {
        const url = new URL(value, window.location.origin);
        return url.protocol === "https:" || url.origin === window.location.origin ? url.href : null;
      } catch { return null; }
    }

    function formatDate(iso) {
      const date = new Date(iso);
      if (Number.isNaN(date.getTime())) return "—";
      return date.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
    }

    function initialsFor(name) {
      const parts = (name || "Administrator").trim().split(/\s+/).filter(Boolean);
      return parts.length > 1 ? `${parts[0][0]}${parts[parts.length - 1][0]}`.toUpperCase() : (name || "Administrator").slice(0, 2).toUpperCase();
    }

    let toastTimer;
    function toast(message, type) {
      const el = byId("toast");
      el.textContent = message;
      el.classList.toggle("is-error", type === "error");
      el.classList.add("is-visible");
      clearTimeout(toastTimer);
      toastTimer = setTimeout(() => el.classList.remove("is-visible"), 3500);
    }

    function showMessage(id, message) {
      const el = byId(id);
      el.textContent = message;
      el.classList.toggle("is-visible", Boolean(message));
    }

    async function apiRequest(url, options = {}) {
      const opts = { credentials: "same-origin", ...options, headers: { ...(options.headers || {}) } };
      if (opts.body && !(opts.body instanceof FormData)) opts.headers["Content-Type"] = "application/json";
      const response = await fetch(url, opts);
      let result;
      try { result = await response.json(); } catch { throw new Error("The server returned an invalid response."); }
      if (response.status === 401) {
        window.location.assign("/login");
        throw new Error("Your session has expired. Please sign in again.");
      }
      if (!response.ok || result.success === false) throw new Error(result.message || result.detail || "The request could not be completed.");
      return result;
    }

    /* ---------------- Data layer ---------------- */
    const data = {
      async list() {
        if (DEMO_MODE) { await delay(450); return demoItems.map((i) => ({ ...i })); }
        const result = await apiRequest(MATERIALS_ENDPOINT);
        return result.items || [];
      },
      async save(payload, file, id) {
        if (DEMO_MODE) {
          await delay(500);
          const previewUrl = file ? URL.createObjectURL(file) : null;
          const mediaType = file ? (file.type.startsWith("video/") ? "video" : "image") : null;
          if (id) {
            const item = demoItems.find((i) => i.id === id);
            Object.assign(item, payload, { updated_at: new Date().toISOString() });
            if (file) { item.media_type = mediaType; item.media_url = mediaType === "image" ? previewUrl : ""; }
          } else {
            demoItems.push({ id: demoNextId++, ...payload, media_type: mediaType || "image", media_url: mediaType === "image" ? previewUrl : "", updated_at: new Date().toISOString() });
          }
          return { message: id ? "Material updated." : "Material added." };
        }
        const form = new FormData();
        Object.entries(payload).forEach(([k, v]) => form.append(k, v));
        if (file) form.append("media", file);
        return apiRequest(id ? `${MATERIALS_ENDPOINT}/${id}` : MATERIALS_ENDPOINT, { method: id ? "PUT" : "POST", body: form });
      },
      async remove(id) {
        if (DEMO_MODE) { await delay(400); demoItems = demoItems.filter((i) => i.id !== id); return { message: "Material deleted." }; }
        return apiRequest(`${MATERIALS_ENDPOINT}/${id}`, { method: "DELETE" });
      }
    };

    /* ---------------- Rendering ---------------- */
    function filtered() {
      const { q, category, status } = state.filters;
      const needle = q.trim().toLowerCase();
      return state.items
        .filter((i) => (!category || normalizeCategory(i.category) === category) && (!status || i.status === status)
          && (!needle || `${i.title} ${i.description || ""}`.toLowerCase().includes(needle)))
        .sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at));
    }

    function normalizeCategory(category) {
      if (category === "Alphabet") return "Alphabets";
      return CATEGORIES.includes(category) ? category : "";
    }

    function thumbHtml(item) {
      const url = item.media_type === "image" ? safeUrl(item.media_url) : null;
      const badge = item.media_type === "video" ? `<span class="ms-type-badge" title="Video">${ICON.play}</span>` : "";
      const inner = url ? `<img src="${esc(url)}" alt="" loading="lazy">` : esc((item.title || "?").trim().charAt(0).toUpperCase());
      return `<div class="ms-thumb" aria-hidden="true">${inner}${badge}</div>`;
    }

    function rowHtml(item) {
      const published = item.status === "published";
      const category = normalizeCategory(item.category);
      return `<tr data-id="${esc(item.id)}">
        <td><div class="ms-item">${thumbHtml(item)}<div>
          <p class="ms-item-title">${esc(item.title)}</p>
          ${item.description ? `<p class="ms-item-desc">${esc(item.description)}</p>` : ""}
        </div></div></td>
        <td class="col-category ms-muted">${esc(category || item.category)}</td>
        <td><span class="ms-badge ${published ? "is-published" : "is-draft"}">${published ? "Published" : "Draft"}</span></td>
        <td class="col-updated ms-muted">${esc(formatDate(item.updated_at))}</td>
        <td class="col-actions">
          <button class="ms-icon-button" type="button" data-action="edit" aria-label="Edit ${esc(item.title)}" title="Edit">${ICON.edit}</button>
          <button class="ms-icon-button is-danger" type="button" data-action="delete" aria-label="Delete ${esc(item.title)}" title="Delete">${ICON.trash}</button>
        </td></tr>`;
    }

    function stateHtml(kind, title, text, buttonLabel, action) {
      const icon = kind === "error" ? ICON.alert : kind === "filtered" ? ICON.search : ICON.hand;
      return `<div class="ms-state ${kind === "error" ? "is-error" : ""}">
        <div class="ms-state-icon">${icon}</div><h2>${esc(title)}</h2><p>${esc(text)}</p>
        ${buttonLabel ? `<button class="ms-button ms-button-secondary" type="button" data-state-action="${action}">${esc(buttonLabel)}</button>` : ""}
      </div>`;
    }

    function render() {
      const body = byId("materialsBody");
      const stateBox = byId("materialsState");
      const pagination = byId("pagination");
      const count = byId("resultCount");
      pagination.hidden = true;

      if (state.loading) {
        body.innerHTML = Array.from({ length: 5 }, () =>
          `<tr class="ms-skeleton"><td><div style="width:60%"></div></td><td class="col-category"><div style="width:70%"></div></td><td><div style="width:60px"></div></td><td class="col-updated"><div style="width:80%"></div></td><td></td></tr>`).join("");
        stateBox.innerHTML = "";
        count.textContent = "Loading…";
        return;
      }

      if (state.error) {
        body.innerHTML = "";
        stateBox.innerHTML = stateHtml("error", "Couldn't load materials", state.error, "Try again", "retry");
        count.textContent = "";
        return;
      }

      const rows = filtered();
      const published = state.items.filter((i) => i.status === "published").length;
      count.textContent = state.items.length ? `${published} published, ${state.items.length - published} draft` : "";

      if (!rows.length) {
        body.innerHTML = "";
        stateBox.innerHTML = state.items.length
          ? stateHtml("filtered", "No materials match", "Try a different search or clear the filters.", "Clear filters", "clear")
          : stateHtml("empty", "No materials yet", "Add your first sign image or video so learners can start studying.", "Add material", "add");
        return;
      }

      const pages = Math.ceil(rows.length / PAGE_SIZE);
      state.page = Math.min(state.page, pages);
      const start = (state.page - 1) * PAGE_SIZE;
      body.innerHTML = rows.slice(start, start + PAGE_SIZE).map(rowHtml).join("");
      stateBox.innerHTML = "";

      if (pages > 1) {
        pagination.hidden = false;
        byId("pageInfo").textContent = `Showing ${start + 1}–${Math.min(start + PAGE_SIZE, rows.length)} of ${rows.length}`;
        let buttons = `<button class="ms-page-button" type="button" data-page="${state.page - 1}" ${state.page === 1 ? "disabled" : ""}>Previous</button>`;
        for (let p = 1; p <= pages; p += 1) {
          buttons += `<button class="ms-page-button ${p === state.page ? "is-current" : ""}" type="button" data-page="${p}" ${p === state.page ? 'aria-current="page"' : ""}>${p}</button>`;
        }
        buttons += `<button class="ms-page-button" type="button" data-page="${state.page + 1}" ${state.page === pages ? "disabled" : ""}>Next</button>`;
        byId("pageButtons").innerHTML = buttons;
      }
    }

    async function loadMaterials() {
      state.loading = true; state.error = ""; render();
      try { state.items = await data.list(); }
      catch (error) { state.error = error.message; }
      state.loading = false; render();
    }

    /* ---------------- Filters ---------------- */
    function setFilter(key, value) { state.filters[key] = value; state.page = 1; render(); }

    window.searchMaterials = (query) => {
      byId("filterSearch").value = query;
      setFilter("q", query);
    };

    function fillCategorySelects() {
      byId("filterCategory").innerHTML = `<option value="">All categories</option>` + CATEGORIES.map((c) => `<option value="${esc(c)}">${esc(c)}</option>`).join("");
      byId("materialCategory").innerHTML = `<option value="" disabled selected>Choose a category</option>` + CATEGORIES.map((c) => `<option value="${esc(c)}">${esc(c)}</option>`).join("");
    }

    function syncCategoryMode(category) {
      document.querySelectorAll(".ms-category-button").forEach((button) => {
        const active = button.dataset.category === category;
        button.classList.toggle("active", active);
        button.setAttribute("aria-selected", String(active));
      });
    }

    document.querySelectorAll(".ms-category-button").forEach((button) => {
      button.addEventListener("click", () => {
        const category = button.dataset.category || "";
        byId("filterCategory").value = category;
        syncCategoryMode(category);
        setFilter("category", category);
      });
    });

    byId("filterCategory").addEventListener("change", (event) => {
      syncCategoryMode(event.target.value);
    });

    byId("filterSearch").addEventListener("input", (e) => { byId("search").value = e.target.value; setFilter("q", e.target.value); });
    byId("filterCategory").addEventListener("change", (e) => setFilter("category", e.target.value));
    byId("filterStatus").addEventListener("change", (e) => setFilter("status", e.target.value));

    function clearFilters() {
      state.filters = { q: "", category: "", status: "" };
      byId("filterSearch").value = ""; byId("search").value = "";
      byId("filterCategory").value = ""; byId("filterStatus").value = "";
      syncCategoryMode("");
      state.page = 1; render();
    }

    /* ---------------- Table + state interactions ---------------- */
    byId("materialsBody").addEventListener("click", (event) => {
      const button = event.target.closest("[data-action]");
      if (!button) return;
      const id = button.closest("tr").dataset.id;
      const item = state.items.find((i) => String(i.id) === id);
      if (!item) return;
      if (button.dataset.action === "edit") openMaterialDialog(item);
      else openDeleteDialog(item);
    });

    byId("materialsState").addEventListener("click", (event) => {
      const button = event.target.closest("[data-state-action]");
      if (!button) return;
      const action = button.dataset.stateAction;
      if (action === "retry") loadMaterials();
      else if (action === "clear") clearFilters();
      else if (action === "add") openMaterialDialog(null);
    });

    byId("pageButtons").addEventListener("click", (event) => {
      const button = event.target.closest("[data-page]");
      if (!button || button.disabled) return;
      state.page = Number(button.dataset.page);
      render();
      byId("materialsBody").closest(".ms-panel").scrollIntoView({ behavior: "smooth", block: "start" });
    });

    /* ---------------- Add / edit dialog ---------------- */
    const materialDialog = byId("materialDialog");
    const fileInput = byId("materialFile");
    let chosenFile = null;
    let previewUrl = null;

    function resetPreview(item) {
      if (previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl = null; }
      const preview = byId("materialPreview");
      const url = item && item.media_type === "image" ? safeUrl(item.media_url) : null;
      preview.innerHTML = url ? `<img src="${esc(url)}" alt="">` : esc(item ? (item.title || "?").charAt(0).toUpperCase() : "?");
      byId("materialFileName").textContent = item && item.media_url ? "Current file kept" : item ? "Current file kept" : "No file chosen";
    }

    function openMaterialDialog(item) {
      state.editing = item;
      chosenFile = null;
      fileInput.value = "";
      showMessage("materialMessage", "");
      byId("materialDialogTitle").textContent = item ? "Edit material" : "Add material";
      byId("materialDialogDesc").textContent = item ? "Update the details or replace the file." : "Upload a sign image or video and choose where it appears.";
      byId("materialSave").textContent = item ? "Save changes" : "Save material";
      byId("materialTitle").value = item ? item.title : "";
      byId("materialCategory").value = item ? normalizeCategory(item.category) : "";
      byId("materialStatus").value = item ? item.status : "draft";
      byId("materialDescription").value = item ? item.description || "" : "";
      resetPreview(item);
      materialDialog.showModal();
      byId("materialTitle").focus();
    }

    function closeMaterialDialog() {
      if (materialDialog.open) materialDialog.close();
      if (previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl = null; }
    }

    byId("addMaterial").addEventListener("click", () => openMaterialDialog(null));
    byId("materialCancel").addEventListener("click", closeMaterialDialog);

    fileInput.addEventListener("change", () => {
      const file = fileInput.files[0];
      if (!file) return;
      if (!ALLOWED_TYPES.includes(file.type) || file.size > MAX_FILE_BYTES) {
        fileInput.value = "";
        showMessage("materialMessage", "Choose a JPG, PNG, WEBP, MP4 or WEBM file smaller than 25 MB.");
        return;
      }
      showMessage("materialMessage", "");
      chosenFile = file;
      byId("materialFileName").textContent = file.name;
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      previewUrl = file.type.startsWith("image/") ? URL.createObjectURL(file) : null;
      const preview = byId("materialPreview");
      preview.innerHTML = previewUrl ? `<img src="${esc(previewUrl)}" alt="">` : ICON.play.replace("<svg", '<svg style="width:22px;height:22px;fill:currentColor;stroke:none"');
    });

    byId("materialForm").addEventListener("submit", async (event) => {
      event.preventDefault();
      const title = byId("materialTitle").value.trim();
      const category = byId("materialCategory").value;
      if (!title) return showMessage("materialMessage", "Enter a title for this material.");
      if (!category) return showMessage("materialMessage", "Choose a category.");
      if (!state.editing && !chosenFile) return showMessage("materialMessage", "Choose an image or video to upload.");

      const button = byId("materialSave");
      const label = button.textContent;
      button.disabled = true; button.textContent = "Saving…";
      showMessage("materialMessage", "");
      try {
        const result = await data.save({
          title, category,
          description: byId("materialDescription").value.trim(),
          status: byId("materialStatus").value
        }, chosenFile, state.editing ? state.editing.id : null);
        closeMaterialDialog();
        toast(result.message || "Material saved.");
        await loadMaterials();
      } catch (error) {
        showMessage("materialMessage", error.message);
      } finally {
        button.disabled = false; button.textContent = label;
      }
    });

    /* ---------------- Delete dialog ---------------- */
    const deleteDialog = byId("deleteDialog");

    function openDeleteDialog(item) {
      state.deleting = item;
      showMessage("deleteMessage", "");
      byId("deleteDialogDesc").textContent = `"${item.title}" will be removed for all learners. This can't be undone.`;
      deleteDialog.showModal();
      byId("deleteCancel").focus();
    }

    byId("deleteCancel").addEventListener("click", () => deleteDialog.close());

    byId("deleteForm").addEventListener("submit", async (event) => {
      event.preventDefault();
      if (!state.deleting) return;
      const button = byId("deleteConfirm");
      const label = button.textContent;
      button.disabled = true; button.textContent = "Deleting…";
      try {
        const result = await data.remove(state.deleting.id);
        deleteDialog.close();
        toast(result.message || "Material deleted.");
        await loadMaterials();
      } catch (error) {
        showMessage("deleteMessage", error.message);
      } finally {
        button.disabled = false; button.textContent = label;
      }
    });

    // Close dialogs when clicking the backdrop
    [materialDialog, deleteDialog].forEach((dialog) => {
      dialog.addEventListener("mousedown", (event) => { if (event.target === dialog) dialog.close(); });
    });

    const THEME_ICONS = {
      sun: '<svg class="theme-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.42 1.42M17.65 17.65l1.42 1.42M2 12h2M20 12h2M4.93 19.07l1.42-1.42M17.65 6.35l1.42-1.42"/></svg>',
      moon: '<svg class="theme-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5Z"/></svg>'
    };

    /* ---------------- Shell: theme, sidebar, profile ---------------- */
    function applyTheme(theme) {
      const selected = theme === "dark" ? "dark" : "light";
      document.documentElement.dataset.theme = selected;
      const control = byId("theme-toggle");
      if (!control) return;
      const label = selected === "dark" ? "Switch to light mode" : "Switch to dark mode";
      control.setAttribute("aria-label", label);
      control.title = label;
      control.innerHTML = THEME_ICONS[selected === "dark" ? "sun" : "moon"];
    }

    window.toggleTheme = () => {
      const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      applyTheme(theme);
      try { localStorage.setItem("signrest-admin-theme", theme); } catch { /* storage unavailable */ }
    };

    window.collapseSidebar = () => {
      if (window.matchMedia("(max-width: 800px)").matches) app.classList.toggle("mobile-open");
      else app.classList.toggle("collapsed");
    };
    window.closeMobile = () => app.classList.remove("mobile-open");
    window.logout = async () => {
      try {
        const result = await apiRequest("/api/logout", { method: "POST" });
        window.location.assign(result.redirect_url || "/login");
      } catch (error) { toast(error.message, "error"); }
    };

    function setAvatar(element, initials, photoUrl) {
      if (!element) return;
      element.replaceChildren();
      const url = safeUrl(photoUrl);
      if (!url) { element.textContent = initials; return; }
      const image = document.createElement("img");
      image.src = url;
      image.alt = "Administrator profile";
      image.addEventListener("error", () => { element.textContent = initials; }, { once: true });
      element.append(image);
    }

    let adminProfile = null;
    function renderAdmin() {
      if (!adminProfile) return;
      const name = adminProfile.full_name || adminProfile.username || "Administrator";
      const initials = initialsFor(name);
      const nameEl = byId("admin-name");
      if (nameEl) nameEl.textContent = name;
      setAvatar(byId("admin-initials"), initials, adminProfile.profile_photo);
      setAvatar(byId("top-initials"), initials, adminProfile.profile_photo);
    }

    async function loadSidebar() {
      const host = byId("adminSidebarHost");
      let markup = "";
      try {
        const response = await fetch(SIDEBAR_URL, { credentials: "same-origin" });
        if (response.ok) markup = await response.text();
      } catch { /* use the built-in menu below */ }
      if (markup) host.innerHTML = markup;
      else host.replaceChildren(byId("fallbackSidebar").content.cloneNode(true));
      const currentFile = window.location.pathname.split("/").pop();
      const onThisPage = SIDEBAR_ALIASES.includes(currentFile);
      host.querySelectorAll(".nav-item[href]").forEach((link) => {
        const linkFile = new URL(link.href, window.location.origin).pathname.split("/").pop();
        const active = onThisPage ? SIDEBAR_ALIASES.includes(linkFile) : new URL(link.href, window.location.origin).pathname === window.location.pathname;
        link.classList.toggle("active", active);
        if (active) link.setAttribute("aria-current", "page"); else link.removeAttribute("aria-current");
      });
      renderAdmin();
    }

    async function loadAdmin() {
      if (DEMO_MODE) return;
      try {
        const result = await apiRequest(PROFILE_ENDPOINT);
        adminProfile = result.item || null;
        renderAdmin();
      } catch { /* the page still works without the profile */ }
    }

    /* ---------------- Init ---------------- */
    let savedTheme = "light";
    try { savedTheme = localStorage.getItem("signrest-admin-theme") || "light"; } catch { savedTheme = "light"; }
    applyTheme(savedTheme);
    fillCategorySelects();
    loadSidebar().catch((error) => toast(error.message, "error"));
    loadAdmin();
    loadMaterials();
  })();
