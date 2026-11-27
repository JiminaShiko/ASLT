/*
 * Signrest admin: Users & Sessions
 * Location in the project: Front-end/js/user_management.js  (served at /js/user_management.js)
 *
 * Talks to Backend/admin/user_management.py:
 *   GET  /api/admin/users, /api/admin/users/<id>
 *   POST /api/admin/users/<id>/(ban|deactivate|reactivate)
 *   GET  /api/admin/sessions, /sessions/stats, /sessions/history
 *   POST /api/admin/sessions/<id>/revoke, /api/admin/users/<id>/sessions/revoke-all
 *
 * No sample data: everything comes from the API. Nothing sensitive is stored in the browser
 * (localStorage is used only for the light/dark theme choice).
 */
(() => {
  "use strict";

  const API = "/api/admin";
  const SIDEBAR_URL = "/admin-dashboard/admin_sidebar.html";
  const PAGE_SIZE = 12;
  const SESSION_PAGE_SIZE = 9;
  const HISTORY_PAGE_SIZE = 15;

  const $ = (id) => document.getElementById(id);

  /* ====================================================================
     Small DOM helpers (user data is always inserted as text, never as HTML)
     ==================================================================== */
  function el(tag, props = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
      if (value === null || value === undefined || value === false) continue;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value === true ? "" : value);
    }
    children.flat().forEach((child) => {
      if (child === null || child === undefined || child === false) return;
      node.append(child.nodeType ? child : document.createTextNode(String(child)));
    });
    return node;
  }

  const ICONS = {
    dots: '<circle cx="12" cy="5" r="1.6" fill="currentColor" stroke="none"/><circle cx="12" cy="12" r="1.6" fill="currentColor" stroke="none"/><circle cx="12" cy="19" r="1.6" fill="currentColor" stroke="none"/>',
    ban: '<circle cx="12" cy="12" r="9"/><path d="m5.6 5.6 12.8 12.8"/>',
    pause: '<circle cx="12" cy="12" r="9"/><path d="M10 9v6M14 9v6"/>',
    restore: '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/>',
    user: '<circle cx="12" cy="8" r="4"/><path d="M4 21v-1a6 6 0 0 1 6-6h4a6 6 0 0 1 6 6v1"/>',
    users: '<path d="M16 21v-2c0-4-12-4-12 0v2M10 11a4 4 0 1 0 0-8 4 4 0 0 0 0 8ZM17 4c4 0 4 6 0 6M20 21v-3c0-2-1-3-3-3"/>',
    shield: '<path d="m12 2 9 4v6c0 5-9 10-9 10S3 17 3 12V6Z"/><path d="m8.5 12 2.5 2.5 4.5-5"/>',
    activity: '<path d="M3 12h4l3-8 4 16 3-8h4"/>',
    desktop: '<rect x="3" y="4" width="18" height="12" rx="2"/><path d="M8 20h8M12 16v4"/>',
    mobile: '<rect x="7" y="2" width="10" height="20" rx="2"/><path d="M11 18h2"/>',
    tablet: '<rect x="4" y="3" width="16" height="18" rx="2"/><path d="M11 17h2"/>',
    other: '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c3 3 3 15 0 18M12 3c-3 3-3 15 0 18"/>',
    login: '<path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4M10 17l5-5-5-5M15 12H3"/>',
    logout: '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9"/>',
    clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    lock: '<rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>',
    revoke: '<circle cx="12" cy="12" r="9"/><path d="m9 9 6 6M15 9l-6 6"/>',
    alert: '<path d="M12 3 2 20h20L12 3ZM12 10v4M12 17.5v.01"/>',
    search: '<circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/>',
    translate: '<path d="M3 5h12M9 2v3M5 8c1 5 5 8 9 9M12 5c0 6-4 10-9 12M14 21l4-11 4 11M16 17h4"/>',
    list: '<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/>',
  };

  function svg(name, className) {
    const template = document.createElement("template");
    template.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true"${className ? ` class="${className}"` : ""}>${ICONS[name] || ICONS.other}</svg>`;
    return template.content.firstChild;
  }

  function safeUrl(value) {
    if (!value) return null;
    try {
      const url = new URL(value, location.origin);
      return url.protocol === "https:" || url.origin === location.origin ? url.href : null;
    } catch {
      return null;
    }
  }

  function initialsOf(name) {
    const parts = String(name || "?").split(/[\s._-]+/).filter(Boolean);
    const text = parts.length > 1 ? parts[0][0] + parts[1][0] : (parts[0] || "?").slice(0, 2);
    return text.toUpperCase();
  }

  function avatarContent(user) {
    const url = safeUrl(user.profile_photo);
    const initials = initialsOf(user.display_name);
    if (!url) return document.createTextNode(initials);
    const img = el("img", { src: url, alt: "", loading: "lazy" });
    img.addEventListener("error", () => img.replaceWith(document.createTextNode(initials)), { once: true });
    return img;
  }

  function fmtDateTime(iso) {
    if (!iso) return null;
    const date = new Date(iso);
    return Number.isNaN(date.getTime()) ? null : date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
  }

  function timeAgo(iso) {
    if (!iso) return "never";
    const seconds = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
    if (Number.isNaN(seconds)) return "unknown";
    if (seconds < 60) return "just now";
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes} min ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours} hr ago`;
    const days = Math.floor(hours / 24);
    return days < 30 ? `${days} d ago` : fmtDateTime(iso) || "unknown";
  }

  const handle = (user) => (user.username ? `@${user.username}` : user.display_name);

  let toastTimer;
  function toast(message, type) {
    const node = $("toast");
    node.textContent = message;
    node.classList.toggle("is-error", type === "error");
    node.classList.add("is-visible");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => node.classList.remove("is-visible"), 4000);
  }

  function debounce(fn, wait = 300) {
    let timer;
    return (...args) => {
      clearTimeout(timer);
      timer = setTimeout(() => fn(...args), wait);
    };
  }

  /* ====================================================================
     API
     ==================================================================== */
  async function api(path, { method = "GET", params, body } = {}) {
    const url = new URL(API + path, location.origin);
    Object.entries(params || {}).forEach(([key, value]) => {
      if (value !== "" && value !== null && value !== undefined) url.searchParams.set(key, value);
    });

    const headers = { Accept: "application/json" };
    const options = { method, headers, credentials: "same-origin", cache: "no-store" };
    if (method !== "GET") {
      headers["Content-Type"] = "application/json";
      headers["X-Signrest-Request"] = "admin";
      const csrf = window.SIGNREST_API_CONFIG && window.SIGNREST_API_CONFIG.csrfToken;
      if (csrf) headers["X-CSRF-Token"] = csrf;
      options.body = JSON.stringify(body || {});
    }

    let response;
    try {
      response = await fetch(url, options);
    } catch {
      throw new Error("Can’t reach the server. Check your connection and try again.");
    }
    let data = null;
    try {
      data = await response.json();
    } catch {
      /* non-JSON response handled below */
    }
    if (response.status === 401) {
      location.assign("/login");
      throw new Error("Your session has ended. Please sign in again.");
    }
    if (!response.ok || !data || data.success === false) {
      throw new Error((data && (data.message || data.detail)) || "The request could not be completed.");
    }
    return data;
  }

  /* ====================================================================
     Shared UI: state blocks, floating menu, confirmation dialog
     ==================================================================== */
  function stateBlock(kind, title, text, actionLabel, onAction) {
    const icon = kind === "error" ? "alert" : kind === "filtered" ? "search" : "users";
    return el(
      "div",
      { class: "um-state" + (kind === "error" ? " is-error" : "") },
      el("div", { class: "um-state-icon" }, svg(icon)),
      el("h2", { text: title }),
      el("p", { text }),
      actionLabel ? el("button", { class: "um-btn um-btn-secondary", type: "button", onclick: onAction, text: actionLabel }) : null
    );
  }

  const menuEl = $("menu");
  let menuAnchor = null;

  function closeMenu(restoreFocus) {
    if (menuEl.hidden) return;
    menuEl.hidden = true;
    menuEl.replaceChildren();
    if (menuAnchor) {
      menuAnchor.setAttribute("aria-expanded", "false");
      if (restoreFocus) menuAnchor.focus();
    }
    menuAnchor = null;
  }

  /** items: [{label, icon, danger, run}] */
  function toggleMenu(anchor, items) {
    if (menuAnchor === anchor) {
      closeMenu(true);
      return;
    }
    closeMenu();
    menuEl.replaceChildren(
      ...items.map((item) =>
        el(
          "button",
          {
            type: "button",
            role: "menuitem",
            class: item.danger ? "is-danger" : "",
            onclick: () => {
              closeMenu();
              item.run();
            },
          },
          svg(item.icon),
          item.label
        )
      )
    );
    menuEl.hidden = false;
    const rect = anchor.getBoundingClientRect();
    const width = menuEl.offsetWidth;
    const height = menuEl.offsetHeight;
    const left = Math.max(8, Math.min(rect.right - width, window.innerWidth - width - 8));
    let top = rect.bottom + 6;
    if (top + height > window.innerHeight - 8) top = Math.max(8, rect.top - height - 6);
    menuEl.style.left = `${left}px`;
    menuEl.style.top = `${top}px`;
    anchor.setAttribute("aria-expanded", "true");
    menuAnchor = anchor;
    menuEl.querySelector("button").focus();
  }

  document.addEventListener("click", (event) => {
    if (menuEl.hidden) return;
    if (menuEl.contains(event.target) || (menuAnchor && menuAnchor.contains(event.target))) return;
    closeMenu();
  });
  document.addEventListener("keydown", (event) => {
    if (menuEl.hidden) return;
    if (event.key === "Escape") {
      event.preventDefault();
      closeMenu(true);
    } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const buttons = [...menuEl.querySelectorAll("button")];
      const index = buttons.indexOf(document.activeElement);
      const next = event.key === "ArrowDown" ? (index + 1) % buttons.length : (index - 1 + buttons.length) % buttons.length;
      buttons[next].focus();
    } else if (event.key === "Tab") {
      closeMenu();
    }
  });
  window.addEventListener("resize", () => closeMenu());
  window.addEventListener("scroll", () => closeMenu(), true);

  const confirmDialog = $("confirm");
  let confirmRun = null;

  /** run(reason) must throw on failure; the dialog stays open and shows the message. */
  function openConfirm({ title, text, confirmLabel, danger, withReason }, run) {
    $("confirmTitle").textContent = title;
    $("confirmText").textContent = text;
    $("confirmOk").textContent = confirmLabel;
    $("confirmOk").className = "um-btn " + (danger ? "um-btn-danger" : "um-btn-primary");
    $("confirmReasonWrap").hidden = !withReason;
    $("confirmReason").value = "";
    $("confirmError").classList.remove("is-visible");
    confirmRun = run;
    confirmDialog.showModal();
    $("confirmCancel").focus();
  }

  $("confirmCancel").addEventListener("click", () => confirmDialog.close());
  confirmDialog.addEventListener("mousedown", (event) => {
    if (event.target === confirmDialog) confirmDialog.close();
  });
  $("confirmForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!confirmRun) return;
    const button = $("confirmOk");
    const label = button.textContent;
    button.disabled = true;
    button.textContent = "Working…";
    $("confirmError").classList.remove("is-visible");
    try {
      await confirmRun($("confirmReason").value.trim());
      confirmDialog.close();
    } catch (error) {
      $("confirmError").textContent = error.message;
      $("confirmError").classList.add("is-visible");
    } finally {
      button.disabled = false;
      button.textContent = label;
    }
  });

  /* ====================================================================
     ACCOUNTS: profile gallery
     ==================================================================== */
  const STATUS_LABEL = { active: "Active", inactive: "Inactive", banned: "Banned", pending: "Pending" };
  const PRESENCE_LABEL = { online: "Online", away: "Away", offline: "Offline" };
  const PILLS = [["all", "All users"], ["active", "Active"], ["inactive", "Inactive"], ["banned", "Banned"], ["pending", "Pending"]];

  const users = { q: "", status: "all", sort: "newest", page: 1, total: 0, items: [], counts: null, loading: false, error: "", req: 0 };
  let sessionsStale = false;

  const ACTIONS = {
    ban: {
      label: "Ban user", icon: "ban", danger: true, withReason: true, confirmLabel: "Ban user",
      title: (u) => `Ban ${handle(u)}?`,
      text: () => "They’ll be signed out everywhere and won’t be able to log in until an administrator reactivates the account. Their data is kept.",
    },
    deactivate: {
      label: "Deactivate user", icon: "pause", danger: false, withReason: true, confirmLabel: "Deactivate",
      title: (u) => `Deactivate ${handle(u)}?`,
      text: () => "This temporarily disables the account and signs the user out everywhere. You can reactivate it at any time.",
    },
    reactivate: {
      label: "Reactivate user", icon: "restore", danger: false, withReason: false, confirmLabel: "Reactivate",
      title: (u) => `Reactivate ${handle(u)}?`,
      text: () => "The user will be able to log in again.",
    },
  };

  function actionsFor(user) {
    if (user.role === "admin") return []; // the backend also refuses these
    const order = { active: ["ban", "deactivate"], inactive: ["reactivate", "ban"], banned: ["reactivate"], pending: ["ban"] };
    return (order[user.status] || []).map((name) => ({ ...ACTIONS[name], run: () => confirmAccountAction(user, name) }));
  }

  function confirmAccountAction(user, name) {
    const action = ACTIONS[name];
    openConfirm(
      { title: action.title(user), text: action.text(user), confirmLabel: action.confirmLabel, danger: action.danger, withReason: action.withReason },
      async (reason) => {
        const result = await api(`/users/${user.user_id}/${name}`, { method: "POST", body: reason ? { reason } : {} });
        applyStatusChange(user, result.item.status);
        sessionsStale = true;
        toast(result.message || "Account updated.");
      }
    );
  }

  /** Update the gallery immediately after a successful backend change. */
  function applyStatusChange(user, newStatus) {
    const old = user.status;
    user.status = newStatus;
    if (users.counts) {
      users.counts[old] = Math.max(0, (users.counts[old] || 0) - 1);
      users.counts[newStatus] = (users.counts[newStatus] || 0) + 1;
    }
    const node = $("gallery").querySelector(`[data-user-id="${user.user_id}"]`);
    if (users.status !== "all" && users.status !== newStatus) {
      users.items = users.items.filter((item) => item.user_id !== user.user_id);
      users.total = Math.max(0, users.total - 1);
      if (node) node.remove();
    } else if (node) {
      node.replaceWith(userTile(user));
    }
    renderPills();
    renderGalleryState();
    if (drawerUserId === user.user_id && drawer.open) openDetails(user);
  }

  function frame(user, small) {
    const art = document.createElement("template");
    art.innerHTML = '<svg class="um-art" viewBox="0 0 200 200" aria-hidden="true"><use href="#um-frame-art"/></svg>';
    return el("div", { class: "um-frame" + (small ? " is-small" : "") }, art.content.firstChild, el("div", { class: "um-photo" }, avatarContent(user)));
  }

  function userTile(user) {
    const actions = actionsFor(user);
    const menuButton = actions.length
      ? el("button", { class: "um-icon-btn um-tile-menu", type: "button", "aria-haspopup": "menu", "aria-expanded": "false", "aria-label": `Account actions for ${user.display_name}` }, svg("dots"))
      : null;
    if (menuButton) menuButton.addEventListener("click", () => toggleMenu(menuButton, actionsFor(user)));

    return el(
      "article",
      { class: `um-tile is-${user.status}`, "data-user-id": user.user_id },
      menuButton,
      frame(user, false),
      el("h3", { class: "um-name", text: user.display_name, title: user.display_name }),
      el("p", { class: "um-username", text: user.username ? `@${user.username}` : user.email || "", title: user.email || "" }),
      el("div", { class: "um-tile-meta" },
        el("span", { class: `um-status is-${user.status}`, text: STATUS_LABEL[user.status] || user.status }),
        el("span", { class: `um-role is-${user.role}`, text: user.role === "admin" ? "Admin" : "User" })
      ),
      el("span", { class: `um-live is-${user.presence || "offline"}`, text: PRESENCE_LABEL[user.presence] || PRESENCE_LABEL.offline }),
      el("button", { class: "um-btn um-btn-secondary", type: "button", onclick: (event) => openDetails(user, event.currentTarget), text: "View Details" })
    );
  }

  function renderPills() {
    const wrap = $("userPills");
    wrap.replaceChildren(
      ...PILLS.map(([value, label]) =>
        el(
          "button",
          { class: "um-pill", type: "button", "aria-pressed": String(users.status === value), onclick: () => { users.status = value; loadUsers(); } },
          label,
          users.counts ? el("small", { text: String(users.counts[value] ?? 0) }) : null
        )
      )
    );
  }

  function renderGalleryState() {
    const box = $("galleryState");
    const more = $("galleryMore");
    box.replaceChildren();
    more.hidden = true;
    if (users.loading && !users.items.length) return;
    if (users.error && !users.items.length) {
      box.append(stateBlock("error", "Couldn’t load users", users.error, "Try again", () => loadUsers()));
      return;
    }
    if (!users.items.length) {
      const filtered = users.q || users.status !== "all";
      box.append(
        filtered
          ? stateBlock("filtered", "No users match", "Try a different search or status filter.", "Clear filters", clearUserFilters)
          : stateBlock("empty", "No users yet", "Accounts appear here as soon as people sign up.")
      );
      return;
    }
    if (users.error) box.append(el("p", { class: "um-note", style: "text-align:center", text: users.error }));
    more.hidden = users.items.length >= users.total;
    $("loadMore").disabled = users.loading;
    $("loadMore").textContent = users.loading ? "Loading…" : "Load more";
  }

  function renderGallery() {
    const gallery = $("gallery");
    if (users.loading && !users.items.length) {
      gallery.replaceChildren(...Array.from({ length: 6 }, () => el("div", { class: "um-skeleton" })));
    } else {
      gallery.replaceChildren(...users.items.map(userTile));
    }
    renderGalleryState();
  }

  async function loadUsers({ append = false, silent = false } = {}) {
    const mine = ++users.req;
    closeMenu();
    if (append) users.page += 1;
    else if (!silent) {
      users.page = 1;
      users.items = [];
    }
    users.loading = true;
    users.error = "";
    if (append) renderGalleryState();
    else if (!silent) {
      renderGallery();
      renderPills();
    }
    // A silent refresh keeps the tiles on screen while presence/reordering changes.
    const page = silent ? 1 : users.page;
    const size = silent ? Math.min(PAGE_SIZE * Math.max(users.page, 1), 48) : PAGE_SIZE;
    try {
      const data = await api("/users", { params: { q: users.q, status: users.status, sort: users.sort, page, page_size: size } });
      if (mine !== users.req) return;
      users.items = append ? users.items.concat(data.items) : data.items;
      users.total = data.total;
      users.counts = data.counts;
    } catch (error) {
      if (mine !== users.req) return;
      users.error = error.message;
      if (append) users.page -= 1;
    }
    users.loading = false;
    renderGallery();
    renderPills();
  }

  /** Manual refresh: the fallback for a missed or slow realtime event. */
  function wireRefresh(buttonId, run) {
    const button = $(buttonId);
    if (!button) return;
    button.addEventListener("click", async () => {
      if (button.disabled) return;
      button.disabled = true;
      button.classList.add("is-busy");
      try {
        await run();
      } finally {
        button.disabled = false;
        button.classList.remove("is-busy");
      }
    });
  }

  wireRefresh("refreshAccounts", async () => {
    accountsStale = false;
    users.error = "";
    await loadUsers({ silent: true });
    if (users.error) toast(users.error, "error");
  });

  function clearUserFilters() {
    users.q = "";
    users.status = "all";
    $("userSearch").value = "";
    $("search").value = "";
    loadUsers();
  }

  $("loadMore").addEventListener("click", () => loadUsers({ append: true }));
  $("userSort").addEventListener("change", (event) => {
    users.sort = event.target.value;
    loadUsers();
  });
  const searchUsers = debounce((value) => {
    users.q = value.trim();
    loadUsers();
  });
  $("userSearch").addEventListener("input", (event) => {
    $("search").value = event.target.value;
    searchUsers(event.target.value);
  });

  /* ====================================================================
     ACCOUNTS: details drawer (read-only)
     ==================================================================== */
  const drawer = $("drawer");
  let drawerUserId = null;

  const EVENT_INFO = {
    registered: { label: "Account registered", icon: "user", tone: "" },
    login: { label: "Successful login", icon: "login", tone: "is-good" },
    failed_login: { label: "Failed login", icon: "lock", tone: "is-bad" },
    logout: { label: "Signed out", icon: "logout", tone: "" },
    expired: { label: "Session expired", icon: "clock", tone: "is-warn" },
    revoked: { label: "Session revoked by an administrator", icon: "revoke", tone: "is-bad" },
    sessions_revoked: { label: "All sessions revoked", icon: "revoke", tone: "is-bad" },
    banned: { label: "Account banned", icon: "ban", tone: "is-bad" },
    deactivated: { label: "Account deactivated", icon: "pause", tone: "is-warn" },
    reactivated: { label: "Account reactivated", icon: "restore", tone: "is-good" },
  };

  function row(label, value, missingText = "Not recorded") {
    const missing = value === null || value === undefined || value === "";
    return el("div", {}, el("dt", { text: label }), el("dd", { class: missing ? "is-missing" : "", text: missing ? missingText : String(value) }));
  }

  function section(title, icon, ...content) {
    return el("section", { class: "um-section" }, el("h3", {}, svg(icon), title), ...content);
  }

  function renderDetails(item) {
    const loginDevice = item.sessions.last_login
      ? [item.sessions.last_login.browser, item.sessions.last_login.os].filter(Boolean).join(" · ") + ` (${item.sessions.last_login.device_type})`
      : null;
    const security = item.security;

    const overview = el(
      "div",
      { class: "um-overview" },
      frame(item, true),
      el("h3", { class: "um-name", text: item.display_name }),
      el("p", { class: "um-username", text: item.username ? `@${item.username}` : "No username yet" }),
      el("p", { class: "um-username", text: item.email || "" }),
      el("span", { class: `um-status is-${item.status}`, text: STATUS_LABEL[item.status] || item.status }),
      el("div", { class: "um-chips" }, el("span", { class: "um-chip", text: item.email_verified ? "Email verified" : "Email not verified" }), el("span", { class: "um-chip", text: item.role === "admin" ? "Administrator" : "User" }))
    );

    const account = section(
      "Account information", "user",
      el("dl", { class: "um-dl" },
        row("User ID", item.user_id), row("Username", item.username, "Not set yet"), row("Email", item.email),
        row("Registered", fmtDateTime(item.created_at)), row("Status", STATUS_LABEL[item.status] || item.status), row("Last updated", fmtDateTime(item.updated_at)))
    );

    const sessionsInfo = section(
      "Session and login activity", "shield",
      el("dl", { class: "um-dl" },
        row("Active sessions", item.sessions.active_count),
        row("Last login", fmtDateTime(item.sessions.last_login && item.sessions.last_login.at), "No logins recorded"),
        row("Last login device", loginDevice, "No logins recorded"),
        row("Last login IP", item.sessions.last_login && item.sessions.last_login.ip_address, "Not recorded"),
        row("Failed attempts", security.failed_login_attempts),
        row("Account lock", security.locked_until ? `Locked until ${fmtDateTime(security.locked_until)}` : "Not locked")),
      item.sessions.active_count && item.role !== "admin"
        ? el("p", { class: "um-note" }, "Manage devices from the ", el("a", { href: "#sessions", onclick: () => { drawer.close(); showTab("sessions"); }, text: "Sessions tab" }), ".")
        : null
    );

    const learning = item.learning;
    const learningInfo = section(
      "Learning activity", "translate",
      el("dl", { class: "um-dl" },
        row("Translations", learning.translations_total),
        row("Sign to text", learning.sign_to_text), row("Text to sign", learning.text_to_sign),
        row("Last translation", fmtDateTime(learning.last_translation_at), "None yet"),
        row("Feedback sent", learning.feedback_count)),
      el("p", { class: "um-note", text: "Lesson and quiz progress isn’t stored in the database yet, so it isn’t shown." })
    );

    const timeline = item.activity.length
      ? el("ul", { class: "um-timeline" },
          ...item.activity.map((event) => {
            const info = EVENT_INFO[event.type] || { label: event.type, icon: "list", tone: "" };
            return el("li", {},
              el("span", { class: `um-dot ${info.tone}` }, svg(info.icon)),
              el("div", { class: "um-event" }, el("strong", { text: info.label }), el("span", { text: [fmtDateTime(event.at), event.detail].filter(Boolean).join(" · ") })));
          }))
      : el("p", { class: "um-note", text: "No activity recorded yet." });
    const activity = section("Recent account activity", "activity", timeline);

    $("drawerBody").replaceChildren(overview, account, sessionsInfo, learningInfo, activity);
  }

  async function openDetails(user, trigger) {
    closeMenu();
    drawerUserId = user.user_id;
    $("drawerTitle").textContent = "User details";
    $("drawerBody").replaceChildren(el("p", { class: "um-note", role: "status", text: "Loading details…" }));
    if (!drawer.open) drawer.showModal();
    try {
      const data = await api(`/users/${user.user_id}`);
      if (drawerUserId !== user.user_id) return;
      renderDetails(data.item);
    } catch (error) {
      if (drawerUserId !== user.user_id) return;
      $("drawerBody").replaceChildren(stateBlock("error", "Couldn’t load details", error.message, "Try again", () => openDetails(user, trigger)));
    }
  }

  $("drawerClose").addEventListener("click", () => drawer.close());
  drawer.addEventListener("mousedown", (event) => {
    if (event.target === drawer) drawer.close();
  });
  drawer.addEventListener("close", () => { drawerUserId = null; });

  /* ====================================================================
     SESSIONS
     ==================================================================== */
  const sessions = { status: "active", q: "", page: 1, total: 0, items: [], loading: false, error: "", req: 0 };
  const hist = { event: "all", q: "", from: "", to: "", page: 1, total: 0, items: [], loading: false, error: "", req: 0 };
  let sessionsLoaded = false;

  const DEVICE_COLORS = { desktop: "var(--um-accent)", mobile: "#c7a35f", tablet: "#8aa896", other: "#b7bbad" };
  const DEVICE_LABELS = { desktop: "Desktop / laptop", mobile: "Smartphone", tablet: "Tablet", other: "Other" };
  const SESSION_LABEL = { online: "Online", away: "Away", offline: "Offline", active: "Active", expired: "Expired", revoked: "Revoked", signed_out: "Signed out" };

  /* ---- statistics + device overview ---- */
  function renderStats(data) {
    const t = data ? data.totals : null;
    const devices = (data && data.devices) || {};
    // Live device view only: ended sessions live in Login history instead.
    const cards = [
      ["Signed-in sessions", t && t.active, "activity", ""],
      ["People signed in", t && t.unique_active_users, "users", ""],
      ["Desktop / laptop", devices.desktop, "desktop", ""],
      ["Smartphones", devices.mobile, "mobile", ""],
      ["Tablets", devices.tablet, "tablet", ""],
    ];
    $("sessionStats").replaceChildren(
      ...cards.map(([label, value, icon, tone]) =>
        el("div", { class: "um-stat" },
          el("span", { class: `um-stat-icon ${tone}` }, svg(icon)),
          el("div", {}, el("span", { class: "um-stat-value", text: value === null || value === undefined ? "—" : new Intl.NumberFormat().format(value) }), el("span", { class: "um-stat-label", text: label })))
      )
    );
  }

  function renderDevices(data) {
    const box = $("deviceOverview");
    if (!data) {
      box.replaceChildren(el("p", { class: "um-note", text: "Loading…" }));
      return;
    }
    const total = Object.values(data.devices).reduce((sum, n) => sum + n, 0);
    if (!total) {
      box.replaceChildren(    el("p", { class: "um-note", text: "No signed-in sessions right now." }));
      return;
    }
    const kinds = Object.keys(DEVICE_LABELS).filter((kind) => data.devices[kind] > 0);
    const left = el("div", {},
      el("div", { class: "um-bar", role: "img", "aria-label": "Share of active sessions by device type" },
        ...kinds.map((kind) => el("span", { style: `width:${(data.devices[kind] / total) * 100}%;background:${DEVICE_COLORS[kind]}`, title: `${DEVICE_LABELS[kind]}: ${data.devices[kind]}` }))),
      el("ul", { class: "um-legend" },
        ...kinds.map((kind) => el("li", {}, el("i", { style: `background:${DEVICE_COLORS[kind]}` }), DEVICE_LABELS[kind], el("b", { text: `${data.devices[kind]} (${Math.round((data.devices[kind] / total) * 100)}%)` })))));
    const right = el("div", {},
      ...data.browsers.map((browser) =>
        el("div", { class: "um-browser" },
          el("div", {}, el("span", { text: browser.name }), el("b", { text: String(browser.count) })),
          el("div", { class: "um-bar" }, el("span", { style: `width:${(browser.count / total) * 100}%` })))));
    box.replaceChildren(left, right);
  }

  async function loadSessionStats({ silent = false } = {}) {
    if (!silent) {
      renderStats(null);
      renderDevices(null);
    }
    try {
      const data = await api("/sessions/stats");
      renderStats(data);
      renderDevices(data);
    } catch (error) {
      $("sessionStats").replaceChildren(stateBlock("error", "Couldn’t load statistics", error.message, "Try again", loadSessionStats));
      $("deviceOverview").replaceChildren();
    }
  }

  /* ---- session monitor ---- */
  function sessionCard(item) {
    const live = ["online", "away", "offline"].includes(item.status);
    const ended = !live;
    const items = [];
    // Administrator accounts keep their sessions: only user accounts can be revoked here.
    if (live && !item.is_current && item.user.role !== "admin") {
      items.push({ label: "Revoke session", icon: "revoke", danger: true, run: () => confirmRevoke(item) });
      items.push({ label: `Revoke all for ${handle(item.user)}`, icon: "ban", danger: true, run: () => confirmRevokeAll(item.user) });
    }
    const menuButton = items.length
      ? el("button", { class: "um-icon-btn", type: "button", "aria-haspopup": "menu", "aria-expanded": "false", "aria-label": `Session actions for ${item.user.display_name}` }, svg("dots"))
      : null;
    if (menuButton) menuButton.addEventListener("click", () => toggleMenu(menuButton, items));

    const where = [`${item.browser} · ${item.os}`, item.ip_address].filter(Boolean).join(" · ");
    return el("article", { class: "um-session" + (ended ? " is-ended" : ""), "data-session-id": item.session_id },
      el("span", { class: "um-device", title: DEVICE_LABELS[item.device_type] || "Device" }, svg(item.device_type in ICONS ? item.device_type : "other")),
      el("div", { style: "min-width:0" },
        el("div", { class: "um-session-user" }, el("span", { class: "um-mini-wrap" }, el("span", { class: "um-mini" }, avatarContent(item.user)), el("i", { class: `um-presence is-${item.status}`, title: SESSION_LABEL[item.status] || item.status })), el("strong", { text: item.user.display_name }), item.user.username ? el("span", { text: `@${item.user.username}` }) : null),
        el("p", { class: "um-meta" },
          el("span", { text: where }),
          el("span", { text: `Signed in ${fmtDateTime(item.login_at) || "—"}` }),
          el("span", { text: ended && item.revoked_at ? `Ended ${fmtDateTime(item.revoked_at)}` : `Last active ${timeAgo(item.last_active_at)}` }))),
      el("div", { class: "um-session-side" },
        menuButton,
        el("span", { class: `um-badge is-${item.status}`, text: SESSION_LABEL[item.status] || item.status }),
        item.is_current ? el("span", { class: "um-chip-current", text: "This device" }) : null));
  }

  function renderSessions() {
    const list = $("sessionList");
    const box = $("sessionState");
    const pager = $("sessionPager");
    box.replaceChildren();
    pager.hidden = true;
    $("sessionCount").textContent = sessions.loading && !sessions.total
      ? "Loading sessions…"
      : sessions.total
        ? `${sessions.total} signed-in ${sessions.total === 1 ? "session" : "sessions"} · online → away → offline, then your account, admins, users`
        : "Nobody is signed in right now.";

    if (sessions.loading && !sessions.items.length) {
      list.replaceChildren(...Array.from({ length: 3 }, () => el("div", { class: "um-skeleton", style: "height:112px;border-radius:12px" })));
      return;
    }
    list.replaceChildren(...sessions.items.map(sessionCard));
    if (sessions.error && !sessions.items.length) {
      box.append(stateBlock("error", "Couldn’t load sessions", sessions.error, "Try again", () => loadSessions()));
    } else if (!sessions.items.length) {
      box.append(stateBlock("empty", "No sessions", sessions.q ? "Nothing matches this search." : "Nobody is signed in right now."));
    } else {
      const start = (sessions.page - 1) * SESSION_PAGE_SIZE;
      $("sessionInfo").textContent = `Showing ${start + 1}–${start + sessions.items.length} of ${sessions.total}`;
      $("sessionPrev").disabled = sessions.page <= 1;
      $("sessionNext").disabled = start + sessions.items.length >= sessions.total;
      pager.hidden = false;
    }
  }

  async function loadSessions({ silent = false } = {}) {
    const mine = ++sessions.req;
    closeMenu();
    sessions.loading = true;
    sessions.error = "";
    if (!silent) renderSessions();
    try {
      const data = await api("/sessions", { params: { status: sessions.status, q: sessions.q, page: sessions.page, page_size: SESSION_PAGE_SIZE } });
      if (mine !== sessions.req) return;
      sessions.items = data.items;
      sessions.total = data.total;
      // Sessions can end while a page is open: step back instead of showing a blank page.
      if (!data.items.length && data.total && sessions.page > 1) {
        sessions.page = Math.max(1, Math.ceil(data.total / SESSION_PAGE_SIZE));
        sessions.loading = false;
        loadSessions({ silent: true });
        return;
      }
    } catch (error) {
      if (mine !== sessions.req) return;
      sessions.error = error.message;
      if (sessions.page > 1) {
        sessions.page -= 1;
        sessions.loading = false;
        loadSessions({ silent: true });
        return;
      }
    }
    sessions.loading = false;
    renderSessions();
  }

  function refreshSessionData() {
    loadSessionStats();
    loadSessions();
    loadHistory();
  }

  // Refresh while the tab is open so Online/Away/Offline stays true. Skipped while a
  // menu, dialog or a deeper page of results is in use.
  const LIVE_REFRESH_MS = 30000;
  setInterval(() => {
    if (document.hidden) return;
    if (!menuEl.hidden || confirmDialog.open || drawer.open) return;
    // The gallery keeps the same 30 s safety net, so a missed event never leaves
    // the two tabs disagreeing about who is online.
    if (!$("view-accounts").hidden && !users.loading) loadUsers({ silent: true });
    if ($("view-sessions").hidden || sessions.loading) return;
    loadSessionStats({ silent: true });
    loadSessions({ silent: true });
  }, LIVE_REFRESH_MS);

  /* ---- realtime (Pusher Channels; the 30 s poll above stays as the fallback) ---- */
  let realtimeClient = null;
  let accountsStale = false;

  // One event covers both tabs: "account_*" reasons also refresh the gallery.
  const realtimeChanged = debounce((payload) => {
    const busy = !menuEl.hidden || confirmDialog.open || drawer.open;

    // Sign-ins and sign-outs change who is online, so the gallery reorders too.
    if (document.hidden || $("view-accounts").hidden || busy) accountsStale = true;
    else loadUsers({ silent: true });

    if (document.hidden || $("view-sessions").hidden) {
      sessionsStale = true; // refreshed when the admin comes back to the tab
      return;
    }
    if (busy) {
      sessionsStale = true; // don't yank the UI out from under a click
      return;
    }
    loadSessionStats({ silent: true });
    loadSessions({ silent: true });
  }, 150);

  async function startRealtime() {
    if (!window.Pusher) return;
    let config;
    try {
      config = await api("/realtime/config");
    } catch {
      return; // polling still keeps the monitor current
    }
    if (!config || !config.enabled) return;
    try {
      realtimeClient = new window.Pusher(config.key, {
        cluster: config.cluster,
        forceTLS: true,
        channelAuthorization: { endpoint: config.authEndpoint, transport: "ajax" },
      });
      realtimeClient.subscribe(config.channel).bind(config.event, realtimeChanged);
    } catch {
      realtimeClient = null;
    }
  }

  function confirmRevoke(item) {
    openConfirm(
      { title: "Revoke this session?", text: `${item.user.display_name} will be signed out on ${item.browser} · ${item.os} straight away and will need to log in again.`, confirmLabel: "Revoke session", danger: true },
      async () => {
        const result = await api(`/sessions/${item.session_id}/revoke`, { method: "POST" });
        toast(result.message || "Session revoked.");
        refreshSessionData();
      }
    );
  }

  function confirmRevokeAll(user) {
    openConfirm(
      { title: `Revoke all sessions for ${handle(user)}?`, text: "Every device this user is signed in on will be signed out immediately. The account itself stays active.", confirmLabel: "Revoke all", danger: true },
      async () => {
        const result = await api(`/users/${user.user_id}/sessions/revoke-all`, { method: "POST" });
        toast(result.message || "Sessions revoked.");
        refreshSessionData();
      }
    );
  }

  $("sessionPrev").addEventListener("click", () => {
    if (sessions.page <= 1) return;
    sessions.page -= 1;
    sessions.items = [];
    loadSessions();
  });
  $("sessionNext").addEventListener("click", () => {
    if ((sessions.page - 1) * SESSION_PAGE_SIZE + sessions.items.length >= sessions.total) return;
    sessions.page += 1;
    sessions.items = [];
    loadSessions();
  });
  const searchSessions = debounce((value) => {
    sessions.q = value.trim();
    sessions.page = 1;
    sessions.items = [];
    loadSessions();
  });
  $("sessionSearch").addEventListener("input", (event) => {
    $("search").value = event.target.value;
    searchSessions(event.target.value);
  });

  /* ---- history (read-only) ---- */
  const HISTORY_INFO = {
    login: { label: "Successful login", icon: "login", tone: "is-good" },
    logout: { label: "Signed out", icon: "logout", tone: "" },
    expired: { label: "Session expired", icon: "clock", tone: "is-warn" },
    revoked: { label: "Session revoked", icon: "revoke", tone: "is-bad" },
    failed_login: { label: "Failed login", icon: "lock", tone: "is-bad" },
  };

  function renderHistory() {
    const list = $("historyList");
    const box = $("historyState");
    const pager = $("historyPager");
    box.replaceChildren();
    pager.hidden = true;

    if (hist.loading) {
      list.replaceChildren(el("li", {}, el("span", {}), el("span", { class: "um-note", role: "status", text: "Loading history…" })));
      return;
    }
    list.replaceChildren(
      ...hist.items.map((event) => {
        const info = HISTORY_INFO[event.event] || { label: event.event, icon: "list", tone: "" };
        return el("li", {},
          el("span", { class: `um-dot ${info.tone}` }, svg(info.icon)),
          el("div", { class: "um-event" },
            el("strong", { text: `${info.label} · ${event.user.display_name}` }),
            el("span", { text: [event.device, event.ip_address].filter(Boolean).join(" · ") })),
          el("time", { datetime: event.at || "", title: fmtDateTime(event.at) || "", text: fmtDateTime(event.at) || "—" }));
      })
    );
    if (hist.error) {
      box.append(stateBlock("error", "Couldn’t load history", hist.error, "Try again", () => loadHistory()));
    } else if (!hist.items.length) {
      box.append(stateBlock("empty", "No events found", "Nothing matches these filters yet."));
    } else {
      const start = (hist.page - 1) * HISTORY_PAGE_SIZE;
      $("historyInfo").textContent = `Showing ${start + 1}–${start + hist.items.length} of ${hist.total}`;
      $("historyPrev").disabled = hist.page <= 1;
      $("historyNext").disabled = start + hist.items.length >= hist.total;
      pager.hidden = false;
    }
  }

  async function loadHistory() {
    const mine = ++hist.req;
    hist.loading = true;
    hist.error = "";
    renderHistory();
    try {
      const data = await api("/sessions/history", {
        params: { event: hist.event, q: hist.q, from: hist.from, to: hist.to, page: hist.page, page_size: HISTORY_PAGE_SIZE },
      });
      if (mine !== hist.req) return;
      hist.items = data.items;
      hist.total = data.total;
    } catch (error) {
      if (mine !== hist.req) return;
      hist.items = [];
      hist.error = error.message;
    }
    hist.loading = false;
    renderHistory();
  }

  function filterHistory(patch) {
    Object.assign(hist, patch, { page: 1 });
    loadHistory();
  }
  $("historyEvent").addEventListener("change", (event) => filterHistory({ event: event.target.value }));

  /* ---- date range: no future days, and start/end can’t cross each other ---- */
  const pad2 = (value) => String(value).padStart(2, "0");
  const toInputDate = (date) => `${date.getFullYear()}-${pad2(date.getMonth() + 1)}-${pad2(date.getDate())}`;
  const HISTORY_TODAY = toInputDate(new Date());

  /** changed: "from" | "to" | undefined (just re-apply limits). */
  function syncHistoryDates(changed) {
    const from = $("historyFrom");
    const to = $("historyTo");
    if (changed === "from" && from.value && to.value && to.value < from.value) to.value = from.value;
    if (changed === "to" && from.value && to.value && to.value < from.value) from.value = to.value;
    // Picking a start date restricts the end date to that day or later, and the
    // other way round; neither calendar offers days after today.
    to.max = HISTORY_TODAY;
    from.max = to.value && to.value < HISTORY_TODAY ? to.value : HISTORY_TODAY;
    to.min = from.value || "";
  }
  syncHistoryDates();

  $("historyFrom").addEventListener("change", (event) => {
    syncHistoryDates("from");
    filterHistory({ from: event.target.value, to: $("historyTo").value });
  });
  $("historyTo").addEventListener("change", (event) => {
    syncHistoryDates("to");
    filterHistory({ from: $("historyFrom").value, to: event.target.value });
  });
  const searchHistory = debounce((value) => filterHistory({ q: value.trim() }));
  $("historySearch").addEventListener("input", (event) => searchHistory(event.target.value));
  $("historyPrev").addEventListener("click", () => { hist.page -= 1; loadHistory(); });
  $("historyNext").addEventListener("click", () => { hist.page += 1; loadHistory(); });

  wireRefresh("refreshSessions", async () => {
    sessionsStale = false;
    sessions.error = "";
    await Promise.all([loadSessionStats({ silent: true }), loadSessions({ silent: true }), loadHistory()]);
    if (sessions.error) toast(sessions.error, "error");
  });

  /* ====================================================================
     Tabs
     ==================================================================== */
  function showTab(name) {
    const tab = name === "sessions" ? "sessions" : "accounts";
    closeMenu();
    ["accounts", "sessions"].forEach((id) => {
      $(`tab-${id}`).setAttribute("aria-selected", String(id === tab));
      $(`view-${id}`).hidden = id !== tab;
    });
    $("search").placeholder = tab === "sessions" ? "Search sessions..." : "Search users...";
    $("search").value = tab === "sessions" ? sessions.q : users.q;
    window.history.replaceState(null, "", `#${tab}`);
    if (tab === "sessions" && (!sessionsLoaded || sessionsStale)) {
      sessionsLoaded = true;
      sessionsStale = false;
      refreshSessionData();
    }
    if (tab === "accounts" && accountsStale) {
      accountsStale = false;
      loadUsers({ silent: true });
    }
  }
  $("tab-accounts").addEventListener("click", () => showTab("accounts"));
  $("tab-sessions").addEventListener("click", () => showTab("sessions"));
  window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));

  // Header search follows the visible tab.
  $("search").addEventListener("input", (event) => {
    const value = event.target.value;
    if ($("view-sessions").hidden) {
      $("userSearch").value = value;
      searchUsers(value);
    } else {
      $("sessionSearch").value = value;
      searchSessions(value);
    }
  });

  const THEME_ICONS = {
    sun: '<svg class="theme-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.42 1.42M17.65 17.65l1.42 1.42M2 12h2M20 12h2M4.93 19.07l1.42-1.42M17.65 6.35l1.42-1.42"/></svg>',
    moon: '<svg class="theme-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5Z"/></svg>'
  };

  /* ====================================================================
     Shell: theme, sidebar, administrator profile
     ==================================================================== */
  function applyTheme(theme) {
    const selected = theme === "dark" ? "dark" : "light";
    document.documentElement.dataset.theme = selected;
    const control = $("theme-toggle");
    if (!control) return;
    const label = selected === "dark" ? "Switch to light mode" : "Switch to dark mode";
    control.setAttribute("aria-label", label);
    control.title = label;
    control.innerHTML = THEME_ICONS[selected === "dark" ? "sun" : "moon"];
  }

  window.toggleTheme = () => {
    const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    applyTheme(theme);
    try { localStorage.setItem("signrest_theme", theme); } catch { /* storage unavailable */ }
  };
  window.collapseSidebar = () => {
    const app = $("app");
    if (window.matchMedia("(max-width: 800px)").matches) {
      app.classList.toggle("mobile-open");
      return;
    }
    const collapsed = app.classList.toggle("collapsed");
    try { localStorage.setItem("signrest-admin-sidebar", collapsed ? "collapsed" : "expanded"); } catch { /* storage unavailable */ }
  };
  window.closeMobile = () => $("app").classList.remove("mobile-open");
  window.logout = async () => {
    try {
      const response = await fetch("/api/logout", { method: "POST", credentials: "same-origin" });
      const data = await response.json().catch(() => ({}));
      location.assign(data.redirect_url || "/login");
    } catch {
      toast("Couldn’t log out. Please try again.", "error");
    }
  };

  let adminProfile = null;
  function setAvatar(node, name, photo) {
    if (!node) return;
    node.replaceChildren(avatarContent({ display_name: name, profile_photo: photo }));
  }
  function renderAdmin() {
    if (!adminProfile) return;
    const name = adminProfile.username || "Administrator";
    const nameNode = $("admin-name");
    if (nameNode) nameNode.textContent = name;
    setAvatar($("admin-initials"), name, adminProfile.profile_photo);
    setAvatar($("top-initials"), name, adminProfile.profile_photo);
  }

  async function loadSidebar() {
    const host = $("adminSidebarHost");
    if (!host) return;
    // The server puts the sidebar in the page itself; only fetch when it is missing.
    if (!host.querySelector("#admin-sidebar")) {
      const response = await fetch(SIDEBAR_URL, { credentials: "same-origin" });
      if (!response.ok) throw new Error("Admin navigation could not be loaded.");
      const template = document.createElement("template");
      template.innerHTML = (await response.text()).trim();
      host.replaceChildren(template.content);
    }
    host.querySelectorAll(".nav-item[href]").forEach((link) => {
      const active = new URL(link.href, location.origin).pathname === location.pathname;
      link.classList.toggle("active", active);
      if (active) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
    renderAdmin();
  }

  async function loadAdminProfile() {
    try {
      const data = await api("/profile");
      adminProfile = data.item || null;
      renderAdmin();
    } catch {
      /* the page works without the header avatar */
    }
  }

  /* ====================================================================
     Start
     ==================================================================== */
  applyTheme(document.documentElement.dataset.theme);
  loadSidebar().catch((error) => toast(error.message, "error"));
  loadAdminProfile();
  renderPills();
  loadUsers();
  startRealtime();
  if (location.hash === "#sessions") showTab("sessions");
})();
