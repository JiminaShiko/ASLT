/**
 * Signrest Admin Portal - Profile Settings Script
 * File: js/profile_settings.js
 * Matches exact IDs, classes, and inline handlers in profile_setting.html
 */

(function () {
  'use strict';

  // Configuration & Constants
  const MAX_AVATAR_BYTES = 5 * 1024 * 1024; // 5 MB limit
  const ALLOWED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
  const USERNAME_MAX_LENGTH = 100;

  // SVG Icons for Theme Toggle & Shared Controls
  const ICONS = {
    moon: '<svg class="theme-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5Z"/></svg>',
    sun: '<svg class="theme-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.42 1.42M17.65 17.65l1.42 1.42M2 12h2M20 12h2M4.93 19.07l1.42-1.42M17.65 6.35l1.42-1.42"/></svg>',
    eye: '<svg class="password-eye" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>',
    eyeOff: '<svg class="password-eye" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10.6 5.1A9.9 9.9 0 0 1 12 5c6.4 0 10 7 10 7a18.4 18.4 0 0 1-2.6 3.6"/><path d="M6.9 6.9A18.4 18.4 0 0 0 2 12s3.6 7 10 7a9.9 9.9 0 0 0 4.1-.9"/><path d="M9.9 9.9a3 3 0 0 0 4.2 4.2"/><path d="M2 2l20 20"/></svg>'
  };

  // Frontend-only snapshot state aligned with users table (username & email)
  const profileState = {
    username: '',
    email: '',
    avatarUrl: null,
    pendingAvatarUrl: null
  };

  async function adminProfileRequest(url, options) {
    const response = await fetch(url, {
      ...options,
      credentials: 'same-origin',
      headers: { Accept: 'application/json', ...(options.headers || {}) }
    });
    const result = await response.json();
    if (!response.ok || result.success === false) {
      throw new Error(result.message || 'The profile update could not be completed.');
    }
    return result;
  }

  /**
   * Computes 1-2 uppercase initials from a username.
   */
  function getInitials(name) {
    const cleaned = (name || '').trim();
    if (!cleaned) return 'A';
    const parts = cleaned.split(/[\s._-]+/).filter(Boolean);
    if (parts.length <= 1) {
      return cleaned.slice(0, 2).toUpperCase();
    }
    return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
  }

  /**
   * Displays status messages inside existing `.profile-settings-message` containers.
   */
  function showFormMessage(element, text, type) {
    if (!element) return;
    element.textContent = text;
    element.classList.remove('is-success', 'is-error', 'is-visible');
    if (text) {
      element.classList.add(type === 'success' ? 'is-success' : 'is-error', 'is-visible');
    }
  }

  /**
   * Clears any visible message inside a message container.
   */
  function clearFormMessage(element) {
    if (!element) return;
    element.textContent = '';
    element.classList.remove('is-success', 'is-error', 'is-visible');
  }

  /**
   * Renders either an <img> preview or fallback initials across all avatar targets:
   * - #profileSettingsSummaryAvatar
   * - #profileSettingsEditAvatar
   * - #top-initials (header avatar)
   */
  function renderAvatars(imageUrl, name) {
    const initials = getInitials(name);
    const avatarTargets = [
      document.getElementById('profileSettingsSummaryAvatar'),
      document.getElementById('profileSettingsEditAvatar'),
      document.getElementById('top-initials')
    ];

    avatarTargets.forEach((container) => {
      if (!container) return;
      container.innerHTML = '';

      if (imageUrl) {
        const img = document.createElement('img');
        img.src = imageUrl;
        img.alt = (name || 'Administrator') + ' avatar';
        container.appendChild(img);
      } else {
        container.textContent = initials;
      }
    });
  }

  /**
   * Updates the left summary card and header initials from current profile state.
   */
  function syncProfileSummary(state) {
    const summaryName = document.getElementById('profileSettingsSummaryName');
    const summaryEmail = document.getElementById('profileSettingsSummaryEmail');

    const displayName = state.username.trim() || 'Administrator';

    if (summaryName) {
      summaryName.textContent = displayName;
    }

    if (summaryEmail) {
      summaryEmail.textContent = state.email && state.email.trim() ? state.email.trim() : '';
    }

    renderAvatars(state.avatarUrl, displayName);
  }

  /**
   * Initializes Account Information form, file upload validation, and Cancel reset.
   */
  function initAccountSection() {
    const accountForm = document.getElementById('profileSettingsAccountForm');
    const avatarInput = document.getElementById('profileSettingsAvatarInput');
    const usernameInput = document.getElementById('profileSettingsUsername');
    const emailInput = document.getElementById('profileSettingsEmail');
    const cancelBtn = document.getElementById('profileSettingsCancel');
    const messageEl = document.getElementById('profileSettingsAccountMessage');

    if (!accountForm) return;

    // Seed initial state from existing DOM values
    const initialSummaryName = document.getElementById('profileSettingsSummaryName');
    profileState.username = usernameInput ? usernameInput.value.trim() : '';
    profileState.email = emailInput ? emailInput.value.trim() : '';

    if (!profileState.username && initialSummaryName && initialSummaryName.textContent.trim() !== 'Administrator') {
      profileState.username = initialSummaryName.textContent.trim();
      if (usernameInput) usernameInput.value = profileState.username;
    }

    syncProfileSummary(profileState);

    // 1. Profile Picture Upload & Validation (JPG, PNG, WEBP up to 5 MB)
    if (avatarInput) {
      avatarInput.addEventListener('change', function () {
        clearFormMessage(messageEl);
        const file = avatarInput.files && avatarInput.files[0];
        if (!file) return;

        if (!ALLOWED_IMAGE_TYPES.includes(file.type)) {
          avatarInput.value = '';
          showFormMessage(
            messageEl,
            'Invalid file format. Please select a JPG, JPEG, PNG, or WEBP image.',
            'error'
          );
          return;
        }

        if (file.size > MAX_AVATAR_BYTES) {
          avatarInput.value = '';
          showFormMessage(
            messageEl,
            'Image exceeds the 5 MB limit. Please choose a smaller file.',
            'error'
          );
          return;
        }

        const reader = new FileReader();
        reader.onload = function (event) {
          const previewUrl = event.target && event.target.result;
          if (typeof previewUrl === 'string') {
            profileState.pendingAvatarUrl = previewUrl;
            const currentName = (usernameInput && usernameInput.value) || profileState.username;
            renderAvatars(previewUrl, currentName);
            showFormMessage(
              messageEl,
              'Photo preview updated. Click "Save changes" to apply.',
              'success'
            );
          }
        };
        reader.onerror = function () {
          showFormMessage(messageEl, 'Unable to read the selected image file.', 'error');
        };
        reader.readAsDataURL(file);
      });
    }

    // 2. Save account changes to the admin profile API
    accountForm.addEventListener('submit', async function (event) {
      event.preventDefault();
      clearFormMessage(messageEl);

      const username = usernameInput ? usernameInput.value.trim() : '';
      const email = emailInput ? emailInput.value.trim() : '';

      if (!username) {
        showFormMessage(messageEl, 'Please enter a name.', 'error');
        if (usernameInput) usernameInput.focus();
        return;
      }

      if (username.length > USERNAME_MAX_LENGTH) {
        showFormMessage(
          messageEl,
          `Name must contain 1–${USERNAME_MAX_LENGTH} characters.`,
          'error'
        );
        if (usernameInput) usernameInput.focus();
        return;
      }

      // Validate email if editable (not readonly) and non-empty
      if (emailInput && !emailInput.readOnly && email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(email)) {
          showFormMessage(messageEl, 'Please enter a valid email address.', 'error');
          emailInput.focus();
          return;
        }
      }

      const selectedImage = avatarInput && avatarInput.files ? avatarInput.files[0] : null;
      const submitBtn = accountForm.querySelector('button[type="submit"]');
      if (submitBtn) submitBtn.disabled = true;

      try {
        if (username !== profileState.username) {
          await adminProfileRequest('/api/admin/profile', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username })
          });
          profileState.username = username;
        }

        if (selectedImage) {
          const formData = new FormData();
          formData.append('profile_picture', selectedImage);
          const result = await adminProfileRequest('/api/admin/profile/upload-picture', {
            method: 'POST',
            body: formData
          });
          profileState.avatarUrl = result.profile_photo;
          profileState.pendingAvatarUrl = null;
          avatarInput.value = '';
        }

        syncProfileSummary(profileState);
        showFormMessage(messageEl, 'Your account information has been updated.', 'success');
      } catch (error) {
        syncProfileSummary(profileState);
        showFormMessage(
          messageEl,
          error.message || 'Unable to save account information. Please try again.',
          'error'
        );
      } finally {
        if (submitBtn) submitBtn.disabled = false;
      }
    });

    // 3. Cancel Button — Restore Original Snapshot Values
    if (cancelBtn) {
      cancelBtn.addEventListener('click', function () {
        if (usernameInput) usernameInput.value = profileState.username;
        if (emailInput && !emailInput.readOnly) emailInput.value = profileState.email;
        if (avatarInput) avatarInput.value = '';

        profileState.pendingAvatarUrl = null;
        syncProfileSummary(profileState);
        clearFormMessage(messageEl);
      });
    }
  }

  /**
   * Evaluates password against the 4 checklist rules in `#profileSettingsRequirements`.
   */
  function evaluatePasswordRules(password) {
    const value = password || '';
    return {
      length: value.length >= 8,
      upper: /[A-Z]/.test(value),
      number: /[0-9]/.test(value),
      special: /[^A-Za-z0-9\s]/.test(value)
    };
  }

  /**
   * Updates strength bars (#profileSettingsStrength1..4), strength label,
   * and `.is-met` states on `.profile-settings-requirement` elements.
   */
  function updatePasswordStrengthUI(password) {
    const bars = [
      document.getElementById('profileSettingsStrength1'),
      document.getElementById('profileSettingsStrength2'),
      document.getElementById('profileSettingsStrength3'),
      document.getElementById('profileSettingsStrength4')
    ];
    const strengthText = document.getElementById('profileSettingsStrengthText');
    const reqContainer = document.getElementById('profileSettingsRequirements');

    const rules = evaluatePasswordRules(password);
    const metCount = Object.values(rules).filter(Boolean).length;

    // Update checklist items
    if (reqContainer) {
      const ruleElements = reqContainer.querySelectorAll('.profile-settings-requirement');
      ruleElements.forEach((el) => {
        const ruleKey = el.getAttribute('data-rule');
        if (ruleKey && Object.prototype.hasOwnProperty.call(rules, ruleKey)) {
          el.classList.toggle('is-met', Boolean(rules[ruleKey]));
        }
      });
    }

    // Update 4 strength bars
    const activeBars = password ? metCount : 0;
    bars.forEach((bar, index) => {
      if (!bar) return;
      bar.classList.toggle('is-active', index < activeBars);
    });

    // Update strength text label
    if (strengthText) {
      if (!password) {
        strengthText.textContent = 'Password strength: Not entered';
      } else if (metCount <= 1) {
        strengthText.textContent = 'Password strength: Weak';
      } else if (metCount === 2) {
        strengthText.textContent = 'Password strength: Fair';
      } else if (metCount === 3) {
        strengthText.textContent = 'Password strength: Good';
      } else {
        strengthText.textContent = 'Password strength: Strong';
      }
    }

    return rules;
  }

  /**
   * Initializes Password & Security form, Show/Hide toggles, strength meter,
   * validation, and backend API hook preparation.
   */
  function initPasswordSection() {
    const passwordForm = document.getElementById('profileSettingsPasswordForm');
    const currentInput = document.getElementById('profileSettingsCurrentPassword');
    const newInput = document.getElementById('profileSettingsNewPassword');
    const confirmInput = document.getElementById('profileSettingsConfirmPassword');
    const messageEl = document.getElementById('profileSettingsPasswordMessage');

    if (!passwordForm) return;

    // 1. Show / Hide Password Toggle Buttons
    const toggleButtons = passwordForm.querySelectorAll('[data-toggle-password]');
    toggleButtons.forEach((btn) => {
      btn.addEventListener('click', function () {
        const targetId = btn.getAttribute('data-toggle-password');
        const targetInput = targetId ? document.getElementById(targetId) : null;
        if (!targetInput) return;

        const isHidden = targetInput.type === 'password';
        targetInput.type = isHidden ? 'text' : 'password';
        btn.innerHTML = isHidden ? ICONS.eyeOff : ICONS.eye;
        btn.setAttribute('aria-label', isHidden ? 'Hide password' : 'Show password');
        btn.setAttribute('title', isHidden ? 'Hide password' : 'Show password');
        btn.setAttribute('aria-pressed', String(isHidden));
      });
    });

    // 2. Live Password Strength & Checklist Listener
    if (newInput) {
      newInput.addEventListener('input', function () {
        updatePasswordStrengthUI(newInput.value);
      });
    }

    // 3. Clear Fields (Reset) Handler
    passwordForm.addEventListener('reset', function () {
      // Allow native form reset to clear inputs first
      setTimeout(function () {
        updatePasswordStrengthUI('');
        clearFormMessage(messageEl);

        // Reset all password inputs back to masked state
        toggleButtons.forEach((btn) => {
          const targetId = btn.getAttribute('data-toggle-password');
          const targetInput = targetId ? document.getElementById(targetId) : null;
          if (targetInput) targetInput.type = 'password';
          btn.innerHTML = ICONS.eye;
          btn.setAttribute('aria-label', 'Show password');
          btn.setAttribute('title', 'Show password');
          btn.setAttribute('aria-pressed', 'false');
        });
      }, 0);
    });

    // 4. Password Form Validation & Backend-Ready Submission Handler
    passwordForm.addEventListener('submit', async function (event) {
      event.preventDefault();
      clearFormMessage(messageEl);

      const currentPassword = currentInput ? currentInput.value : '';
      const newPassword = newInput ? newInput.value : '';
      const confirmPassword = confirmInput ? confirmInput.value : '';

      // Validate all fields are completed
      if (!currentPassword || !newPassword || !confirmPassword) {
        showFormMessage(messageEl, 'Please fill in all password fields.', 'error');
        if (!currentPassword && currentInput) currentInput.focus();
        else if (!newPassword && newInput) newInput.focus();
        else if (confirmInput) confirmInput.focus();
        return;
      }

      // Validate all 4 password requirements
      const rules = evaluatePasswordRules(newPassword);
      const allRulesMet = rules.length && rules.upper && rules.number && rules.special;
      if (!allRulesMet) {
        showFormMessage(
          messageEl,
          'New password must meet all four security requirements listed below.',
          'error'
        );
        if (newInput) newInput.focus();
        return;
      }

      // Ensure new password differs from current password
      if (currentPassword === newPassword) {
        showFormMessage(
          messageEl,
          'New password must be different from your current password.',
          'error'
        );
        if (newInput) newInput.focus();
        return;
      }

      // Confirm new password and confirmation match
      if (newPassword !== confirmPassword) {
        showFormMessage(
          messageEl,
          'New password and confirmation password do not match.',
          'error'
        );
        if (confirmInput) confirmInput.focus();
        return;
      }

      const submitBtn = passwordForm.querySelector('button[type="submit"]');
      if (submitBtn) submitBtn.disabled = true;

      try {
        await adminProfileRequest('/api/admin/profile/change-password', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            current_password: currentPassword,
            new_password: newPassword,
            confirm_password: confirmPassword
          })
        });
        passwordForm.reset();
        // The reset handler clears the message area on a 0ms timer, so confirm after it runs.
        setTimeout(function () {
          showFormMessage(messageEl, 'Your password has been updated.', 'success');
        }, 0);
      } catch (err) {
        const errorMsg = (err && err.message) ? err.message : 'Unable to update password. Please try again.';
        showFormMessage(messageEl, errorMsg, 'error');
      } finally {
        if (submitBtn) submitBtn.disabled = false;
      }
    });
  }

  /**
   * Dashboard Shell Controls (matches inline onclick/oninput in profile_setting.html):
   * - collapseSidebar()
   * - toggleTheme()
   * - searchProfile(query)
   * Only defines global functions if not already provided by another shared script.
   */
  function initDashboardControls() {
    const themeToggleBtn = document.getElementById('theme-toggle');

    function syncThemeButtonIcon() {
      if (!themeToggleBtn) return;
      const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
      themeToggleBtn.innerHTML = isDark ? ICONS.sun : ICONS.moon;
      const nextLabel = isDark ? 'Switch to light mode' : 'Switch to dark mode';
      themeToggleBtn.setAttribute('aria-label', nextLabel);
      themeToggleBtn.setAttribute('title', nextLabel);
    }

    if (typeof window.closeMobile !== 'function') {
      window.closeMobile = function () {
        const app = document.getElementById('app');
        if (app) app.classList.remove('mobile-open');
      };
    }

    if (typeof window.logout !== 'function') {
      window.logout = async function () {
        try {
          await fetch('/api/logout', {
            method: 'POST',
            credentials: 'same-origin',
            headers: { Accept: 'application/json' }
          });
        } catch (error) {
          // Still leave the protected admin page if logout cannot reach the server.
        }
        window.location.href = '/login';
      };
    }

    // Apply saved theme preference (non-sensitive UI preference only)
    let savedTheme = 'light';
    try {
      savedTheme = localStorage.getItem('signrest-admin-theme') || 'light';
    } catch (error) {
      console.warn('Unable to read the saved admin theme preference.', error);
    }
    if (savedTheme === 'dark' || savedTheme === 'light') {
      document.documentElement.setAttribute('data-theme', savedTheme);
    }
    syncThemeButtonIcon();

    if (typeof window.toggleTheme !== 'function') {
      window.toggleTheme = function () {
        const root = document.documentElement;
        const isDark = root.getAttribute('data-theme') === 'dark';
        const nextTheme = isDark ? 'light' : 'dark';
        root.setAttribute('data-theme', nextTheme);
        try {
          localStorage.setItem('signrest-admin-theme', nextTheme);
        } catch (error) {
          console.warn('Unable to save the admin theme preference.', error);
        }
        syncThemeButtonIcon();
      };
    }

    if (typeof window.collapseSidebar !== 'function') {
      window.collapseSidebar = function () {
        const app = document.getElementById('app');
        if (!app) return;
        if (window.innerWidth <= 800) {
          app.classList.toggle('mobile-open');
          return;
        }
        const collapsed = app.classList.toggle('collapsed');
        try {
          localStorage.setItem('signrest-admin-sidebar', collapsed ? 'collapsed' : 'expanded');
        } catch (error) {
          console.warn('Unable to save the sidebar preference.', error);
        }
      };
    }

    if (typeof window.searchProfile !== 'function') {
      window.searchProfile = function (query) {
        const term = (query || '').trim().toLowerCase();
        const cards = document.querySelectorAll('.profile-settings-card');
        cards.forEach((card) => {
          if (!term) {
            card.style.display = '';
            return;
          }
          const text = card.textContent.toLowerCase();
          card.style.display = text.includes(term) ? '' : 'none';
        });
      };
    }
  }

  /**
   * Loads the signed-in administrator's record and fills the summary card,
   * the account form and the sidebar identity. Runs even when the page is
   * embedded in the shell frame, where there is no sidebar slot of its own.
   */
  async function loadAdminProfile() {
    try {
      const result = await adminProfileRequest('/api/admin/profile', { cache: 'no-store' });
      const profile = (result && result.item) || {};
      const username = typeof profile.username === 'string' ? profile.username.trim() : '';
      const email = typeof profile.email === 'string' ? profile.email.trim() : '';

      const sidebarInitials = document.getElementById('admin-initials');
      const sidebarName = document.getElementById('admin-name');
      if (username && sidebarName) sidebarName.textContent = username;
      if (username && sidebarInitials) sidebarInitials.textContent = getInitials(username);

      const usernameInput = document.getElementById('profileSettingsUsername');
      const emailInput = document.getElementById('profileSettingsEmail');
      if (usernameInput) usernameInput.value = username;
      if (emailInput) emailInput.value = email;

      profileState.username = username;
      profileState.email = email;
      profileState.avatarUrl = typeof profile.profile_photo === 'string'
        ? profile.profile_photo
        : null;
      syncProfileSummary(profileState);
    } catch (error) {
      console.error('Unable to load the administrator profile.', error);
      showFormMessage(
        document.getElementById('profileSettingsAccountMessage'),
        (error && error.message) || 'Unable to load the administrator profile.',
        'error'
      );
    }
  }

  async function initAdminSidebar() {
    const host = document.getElementById('adminSidebarHost');
    if (!host) return;

    try {
      // The server puts the sidebar in the page itself; only fetch when it is missing.
      if (!host.querySelector('#admin-sidebar')) {
        const response = await fetch('admin_sidebar.html', {
          credentials: 'same-origin',
          cache: 'no-store'
        });
        if (!response.ok) {
          throw new Error('Unable to load the admin sidebar.');
        }

        const html = await response.text();
        const template = document.createElement('template');
        template.innerHTML = html.trim();
        host.replaceChildren(template.content);
      }

      const profileLink = host.querySelector('a[href="/admin-dashboard/profile.html"], a[href="/admin-dashboard/profile_setting.html"]');
      if (profileLink) {
        profileLink.classList.add('active');
        profileLink.setAttribute('aria-current', 'page');
      }

      const overviewLink = host.querySelector('a[href="/admin-dashboard/"]');
      if (overviewLink) overviewLink.classList.remove('active');

    } catch (error) {
      host.textContent = '';
      console.error(error);
    }
  }

  // Initialize everything once DOM is ready
  document.addEventListener('DOMContentLoaded', async function () {
    initDashboardControls();
    initAccountSection();
    initPasswordSection();
    await initAdminSidebar();
    await loadAdminProfile();
  });
})();