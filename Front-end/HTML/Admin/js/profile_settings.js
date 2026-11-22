(() => {
  const profileEndpoint = "/api/admin/profile";
  const byId = (id) => document.getElementById(id);
  const app = byId("app");
  let originalProfile = null;
  let selectedPhoto = null;

  function showMessage(elementId, message, type) {
    const element = byId(elementId);
    if (!element) return;

    element.textContent = message;
    element.classList.remove("is-success", "is-error");
    element.classList.add("is-visible", type === "error" ? "is-error" : "is-success");
  }

  function initialsFor(name, username) {
    const value = (name || username || "Administrator").trim();
    const parts = value.split(/\s+/).filter(Boolean);
    return parts.length > 1
      ? `${parts[0][0]}${parts[parts.length - 1][0]}`.toUpperCase()
      : value.slice(0, 2).toUpperCase();
  }

  function safePhotoUrl(value) {
    if (!value) return null;
    try {
      const url = new URL(value, window.location.origin);
      if (url.protocol === "https:" || url.origin === window.location.origin) {
        return url.href;
      }
    } catch {
      return null;
    }
    return null;
  }

  function setAvatar(element, initials, photoUrl) {
    if (!element) return;
    element.replaceChildren();

    const safeUrl = safePhotoUrl(photoUrl);
    if (!safeUrl) {
      element.textContent = initials;
      return;
    }

    const image = document.createElement("img");
    image.src = safeUrl;
    image.alt = "Administrator profile";
    image.addEventListener("error", () => {
      element.textContent = initials;
    }, { once: true });
    element.append(image);
  }

  function renderProfile(profile) {
    const fullName = profile.full_name || "";
    const username = profile.username || "";
    const initials = initialsFor(fullName, username);

    byId("profileSettingsFullName").value = fullName;
    byId("profileSettingsUsername").value = username;
    byId("profileSettingsEmail").value = profile.email || "";
    byId("profileSettingsRole").value = profile.role || "Administrator";
    byId("profileSettingsSummaryName").textContent = fullName || username || "Administrator";
    byId("profileSettingsSummaryEmail").textContent = profile.email || "";
    setAvatar(byId("profileSettingsSummaryAvatar"), initials, profile.profile_photo);
    setAvatar(byId("profileSettingsEditAvatar"), initials, profile.profile_photo);

    const sidebarName = byId("admin-name");
    if (sidebarName) sidebarName.textContent = fullName || username || "Administrator";
    setAvatar(byId("admin-initials"), initials, profile.profile_photo);
    setAvatar(byId("top-initials"), initials, profile.profile_photo);
  }

  function applyTheme(theme) {
    const selected = theme === "dark" ? "dark" : "light";
    document.documentElement.dataset.theme = selected;
    const control = byId("theme-toggle");
    if (!control) return;

    const label = selected === "dark" ? "Switch to light mode" : "Switch to dark mode";
    control.setAttribute("aria-label", label);
    control.title = label;
    const image = selected === "dark" ? "sun_bgremove.png" : "moon_bgremove.png";
    control.innerHTML = `<img class="theme-icon" src="/assets/images/${image}" alt="" width="20" height="20">`;
  }

  window.toggleTheme = () => {
    const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    applyTheme(theme);
    try {
      localStorage.setItem("signrest-admin-theme", theme);
    } catch {
      // Theme selection still works for this page when storage is unavailable.
    }
  };

  window.searchProfile = (query) => {
    const normalizedQuery = query.trim().toLowerCase();
    document.querySelectorAll(".profile-settings-card").forEach((card) => {
      card.hidden = normalizedQuery !== "" && !card.textContent.toLowerCase().includes(normalizedQuery);
    });
  };

  async function apiRequest(url, options) {
    const requestOptions = {
      credentials: "same-origin",
      ...options,
      headers: { ...(options && options.headers) }
    };

    if (requestOptions.body && !(requestOptions.body instanceof FormData)) {
      requestOptions.headers["Content-Type"] = "application/json";
    }

    const response = await fetch(url, requestOptions);
    let result;
    try {
      result = await response.json();
    } catch {
      throw new Error("The server returned an invalid response.");
    }

    if (response.status === 401) {
      window.location.assign("/login");
      throw new Error("Your session has expired. Please sign in again.");
    }
    if (!response.ok || result.success === false) {
      throw new Error(result.message || result.detail || "The request could not be completed.");
    }
    return result;
  }

  async function loadProfile() {
    const result = await apiRequest(profileEndpoint);
    if (!result.item) throw new Error("The server did not return your profile.");

    originalProfile = { ...result.item };
    renderProfile(originalProfile);
  }

  async function loadSidebar() {
    const host = byId("adminSidebarHost");
    const response = await fetch("/admin-dashboard/admin_sidebar.html", {
      credentials: "same-origin"
    });
    if (!response.ok) throw new Error("Admin navigation could not be loaded.");

    host.innerHTML = await response.text();
    host.querySelectorAll(".nav-item[href]").forEach((link) => {
      const active = new URL(link.href, window.location.origin).pathname === window.location.pathname;
      link.classList.toggle("active", active);
      if (active) link.setAttribute("aria-current", "page");
      else link.removeAttribute("aria-current");
    });
    if (originalProfile) renderProfile(originalProfile);
  }

  window.collapseSidebar = () => {
    if (!app) return;
    if (window.matchMedia("(max-width: 800px)").matches) {
      app.classList.toggle("mobile-open");
    } else {
      app.classList.toggle("collapsed");
    }
  };
  window.closeMobile = () => app && app.classList.remove("mobile-open");
  window.logout = async () => {
    try {
      const result = await apiRequest("/api/logout", { method: "POST" });
      window.location.assign(result.redirect_url || "/login");
    } catch (error) {
      showMessage("profileSettingsAccountMessage", error.message, "error");
    }
  };

  function setButtonLoading(button, loading) {
    if (!button) return;
    if (loading) {
      button.dataset.originalText = button.textContent;
      button.textContent = "Please wait...";
      button.disabled = true;
    } else {
      button.textContent = button.dataset.originalText || button.textContent;
      button.disabled = false;
    }
  }

  function clearPasswordFields() {
    byId("profileSettingsPasswordForm").reset();
    updatePasswordStrength();
  }

  function passwordRules(password) {
    return {
      length: password.length >= 8,
      upper: /[A-Z]/.test(password),
      number: /\d/.test(password),
      special: /[^A-Za-z0-9]/.test(password)
    };
  }

  function updatePasswordStrength() {
    const password = byId("profileSettingsNewPassword").value;
    const rules = passwordRules(password);
    const strength = Object.values(rules).filter(Boolean).length;
    for (let index = 1; index <= 4; index += 1) {
      byId(`profileSettingsStrength${index}`).classList.toggle("is-active", index <= strength);
    }

    const labels = ["Not entered", "Weak", "Fair", "Good", "Strong"];
    byId("profileSettingsStrengthText").textContent = `Password strength: ${labels[strength]}`;
    byId("profileSettingsRequirements").querySelectorAll("[data-rule]").forEach((rule) => {
      rule.classList.toggle("is-met", Boolean(rules[rule.dataset.rule]));
    });
  }

  function previewPhoto(file) {
    const reader = new FileReader();
    reader.addEventListener("load", () => {
      if (typeof reader.result !== "string") return;
      [byId("profileSettingsSummaryAvatar"), byId("profileSettingsEditAvatar")].forEach((element) => {
        element.replaceChildren();
        const image = document.createElement("img");
        image.src = reader.result;
        image.alt = "Selected profile picture preview";
        element.append(image);
      });
    });
    reader.readAsDataURL(file);
  }

  byId("profileSettingsAccountForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    if (!form.reportValidity()) return;

    const fullName = byId("profileSettingsFullName").value.trim();
    const username = byId("profileSettingsUsername").value.trim();
    if (!fullName || !username) {
      showMessage("profileSettingsAccountMessage", "Full name and username are required.", "error");
      return;
    }
    if (!/^[A-Za-z0-9_.-]+$/.test(username)) {
      showMessage("profileSettingsAccountMessage", "Username can only contain letters, numbers, dots, underscores, and hyphens.", "error");
      return;
    }

    const button = form.querySelector('button[type="submit"]');
    setButtonLoading(button, true);
    try {
      const result = await apiRequest(profileEndpoint, {
        method: "PUT",
        body: JSON.stringify({ full_name: fullName, username })
      });
      originalProfile = { ...originalProfile, ...(result.item || {}), full_name: fullName, username };
      renderProfile(originalProfile);
      showMessage("profileSettingsAccountMessage", result.message || "Profile information saved.", "success");
    } catch (error) {
      showMessage("profileSettingsAccountMessage", error.message, "error");
    } finally {
      setButtonLoading(button, false);
    }
  });

  byId("profileSettingsCancel").addEventListener("click", () => {
    if (originalProfile) renderProfile(originalProfile);
    byId("profileSettingsAvatarInput").value = "";
    selectedPhoto = null;
    showMessage("profileSettingsAccountMessage", "Unsaved profile changes were canceled.", "success");
  });

  byId("profileSettingsAvatarInput").addEventListener("change", async (event) => {
    const input = event.currentTarget;
    const file = input.files[0];
    if (!file) return;

    const allowedTypes = ["image/jpeg", "image/png", "image/webp"];
    if ((file.type && !allowedTypes.includes(file.type)) || file.size > 5 * 1024 * 1024) {
      input.value = "";
      showMessage("profileSettingsAccountMessage", "Choose a JPG, PNG, or WEBP image smaller than 5 MB.", "error");
      return;
    }

    selectedPhoto = file;
    previewPhoto(file);
    const formData = new FormData();
    formData.append("profile_picture", selectedPhoto);
    input.disabled = true;
    try {
      const result = await apiRequest(`${profileEndpoint}/upload-picture`, {
        method: "POST",
        body: formData
      });
      originalProfile = { ...originalProfile, profile_photo: result.profile_photo };
      renderProfile(originalProfile);
      showMessage("profileSettingsAccountMessage", result.message || "Profile picture updated.", "success");
    } catch (error) {
      if (originalProfile) renderProfile(originalProfile);
      input.value = "";
      showMessage("profileSettingsAccountMessage", error.message, "error");
    } finally {
      selectedPhoto = null;
      input.disabled = false;
    }
  });

  document.querySelectorAll("[data-toggle-password]").forEach((button) => {
    button.addEventListener("click", () => {
      const input = byId(button.dataset.togglePassword);
      if (!input) return;
      const showing = input.type === "password";
      input.type = showing ? "text" : "password";
      button.textContent = showing ? "Hide" : "Show";
    });
  });

  byId("profileSettingsNewPassword").addEventListener("input", updatePasswordStrength);
  byId("profileSettingsPasswordForm").addEventListener("reset", () => {
    window.setTimeout(updatePasswordStrength, 0);
  });

  byId("profileSettingsPasswordForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    if (!form.reportValidity()) return;

    const currentPassword = byId("profileSettingsCurrentPassword").value;
    const newPassword = byId("profileSettingsNewPassword").value;
    const confirmPassword = byId("profileSettingsConfirmPassword").value;
    const rules = passwordRules(newPassword);
    if (!Object.values(rules).every(Boolean)) {
      showMessage("profileSettingsPasswordMessage", "Meet all listed password requirements.", "error");
      return;
    }
    if (newPassword !== confirmPassword) {
      showMessage("profileSettingsPasswordMessage", "New password and confirmation do not match.", "error");
      return;
    }

    const button = form.querySelector('button[type="submit"]');
    setButtonLoading(button, true);
    try {
      const result = await apiRequest(`${profileEndpoint}/change-password`, {
        method: "POST",
        body: JSON.stringify({ current_password: currentPassword, new_password: newPassword, confirm_password: confirmPassword })
      });
      clearPasswordFields();
      showMessage("profileSettingsPasswordMessage", result.message || "Password updated.", "success");
    } catch (error) {
      showMessage("profileSettingsPasswordMessage", error.message, "error");
    } finally {
      setButtonLoading(button, false);
    }
  });

  let savedTheme = "light";
  try {
    savedTheme = localStorage.getItem("signrest-admin-theme") || "light";
  } catch {
    savedTheme = "light";
  }
  applyTheme(savedTheme);

  loadSidebar().catch((error) => {
    showMessage("profileSettingsAccountMessage", error.message, "error");
  });
  loadProfile().catch((error) => {
    showMessage("profileSettingsAccountMessage", error.message, "error");
  });
})();
