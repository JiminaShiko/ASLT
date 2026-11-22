
document.addEventListener("DOMContentLoaded", () => {
    // ==========================================
    // CONFIGURATION
    // ==========================================

    const API = {
        profile: "/api/admin/profile",
        uploadPicture: "/api/admin/profile/upload-picture",
        changePassword: "/api/admin/profile/change-password"
    };

    const MAX_IMAGE_SIZE = 5 * 1024 * 1024;

    let originalProfile = {};
    let selectedImage = null;
    let originalPhotoUrl = "";

    // ==========================================
    // ELEMENTS
    // ==========================================

    const fullNameInput = document.getElementById("fullName");
    const usernameInput = document.getElementById("username");
    const emailInput = document.getElementById("email");

    const profileImageInput = document.getElementById("profileImage");
    const profileImagePreview = document.getElementById("profileImagePreview");

    const currentPasswordInput = document.getElementById("currentPassword");
    const newPasswordInput = document.getElementById("newPassword");
    const confirmPasswordInput = document.getElementById("confirmPassword");

    const saveProfileButton = document.getElementById("saveChanges");
    const cancelButton = document.getElementById("cancelChanges");
    const updatePasswordButton = document.getElementById("updatePassword");
    const clearPasswordButton = document.getElementById("clearPasswordFields");

    // ==========================================
    // HELPERS
    // ==========================================

    async function apiRequest(url, options = {}) {
        const response = await fetch(url, {
            credentials: "same-origin",
            ...options,
            headers: {
                ...(options.body instanceof FormData
                    ? {}
                    : { "Content-Type": "application/json" }),
                ...options.headers
            }
        });

        let result;

        try {
            result = await response.json();
        } catch {
            throw new Error("Unexpected server response. Please try again.");
        }

        if (response.status === 401) {
            showMessage("Your session has expired. Please log in again.", "error");
            setTimeout(() => {
                window.location.href = "/login";
            }, 1500);

            throw new Error("Authentication required.");
        }

        if (!response.ok || result.success === false) {
            throw new Error(result.message || "Something went wrong.");
        }

        return result;
    }

    function showMessage(message, type = "success") {
        let container = document.getElementById("profileMessage");

        if (!container) {
            container = document.createElement("div");
            container.id = "profileMessage";
            container.style.cssText = `
                position: fixed;
                top: 25px;
                right: 25px;
                z-index: 9999;
                max-width: 350px;
                padding: 14px 20px;
                border-radius: 10px;
                font-size: 14px;
                font-weight: 500;
                box-shadow: 0 4px 15px rgba(0,0,0,0.12);
            `;
            document.body.appendChild(container);
        }

        container.textContent = message;

        container.style.background =
            type === "error" ? "#FDE8E7" : "#E5F4E8";

        container.style.color =
            type === "error" ? "#A83232" : "#27633A";

        container.style.display = "block";

        clearTimeout(container.hideTimeout);

        container.hideTimeout = setTimeout(() => {
            container.style.display = "none";
        }, 4000);
    }

    function setLoading(button, loading, originalText) {
        if (!button) return;

        button.disabled = loading;

        if (loading) {
            button.dataset.originalText = button.textContent;
            button.textContent = "Please wait...";
        } else {
            button.textContent =
                button.dataset.originalText || originalText;
        }
    }

    // ==========================================
    // LOAD ADMIN PROFILE
    // ==========================================

    async function loadAdminProfile() {
        try {
            const result = await apiRequest(API.profile);

            const user = result.item;

            if (!user) {
                throw new Error("Profile information was not returned.");
            }

            originalProfile = {
                full_name: user.full_name || "",
                username: user.username || "",
                email: user.email || "",
                profile_photo: user.profile_photo || ""
            };

            originalPhotoUrl = originalProfile.profile_photo;

            fullNameInput.value = originalProfile.full_name;
            usernameInput.value = originalProfile.username;

            if (emailInput) {
                emailInput.value = originalProfile.email;
            }

            updateProfileSummary(
                originalProfile.full_name,
                originalProfile.username,
                originalProfile.email,
                originalPhotoUrl
            );

        } catch (error) {
            console.error("Profile loading error:", error);
            showMessage(error.message, "error");
        }
    }

    // ==========================================
    // UPDATE PROFILE SUMMARY
    // ==========================================

    function updateProfileSummary(fullName, username, email, photoUrl) {
        const nameElements = [
            document.getElementById("displayName"),
            document.getElementById("profileName"),
            document.getElementById("adminName")
        ];

        nameElements.forEach(element => {
            if (element) {
                element.textContent = fullName || username || "Administrator";
            }
        });

        const emailElements = [
            document.getElementById("displayEmail"),
            document.getElementById("profileEmail"),
            document.getElementById("adminEmail")
        ];

        emailElements.forEach(element => {
            if (element) {
                element.textContent = email || "";
            }
        });

        if (profileImagePreview && photoUrl) {
            profileImagePreview.src = photoUrl;
        }
    }

    // ==========================================
    // LIVE PROFILE SUMMARY
    // ==========================================

    function updateLiveSummary() {
        const fullName = fullNameInput?.value.trim() || "";
        const username = usernameInput?.value.trim() || "";

        const name = fullName || username || "Administrator";

        const nameElements = [
            document.getElementById("displayName"),
            document.getElementById("profileName"),
            document.getElementById("adminName")
        ];

        nameElements.forEach(element => {
            if (element) element.textContent = name;
        });
    }

    if (fullNameInput) {
        fullNameInput.addEventListener("input", updateLiveSummary);
    }

    if (usernameInput) {
        usernameInput.addEventListener("input", updateLiveSummary);
    }

    // ==========================================
    // SAVE PROFILE INFORMATION
    // ==========================================

    if (saveProfileButton) {
        saveProfileButton.addEventListener("click", async event => {
            event.preventDefault();

            const fullName = fullNameInput.value.trim();
            const username = usernameInput.value.trim();

            if (!fullName || !username) {
                showMessage("Please complete all required fields.", "error");
                return;
            }

            if (fullName.length > 100 || username.length > 100) {
                showMessage("Full name and username must be 100 characters or fewer.", "error");
                return;
            }

            if (!/^[A-Za-z0-9_.-]+$/.test(username)) {
                showMessage(
                    "Username can only contain letters, numbers, dots, underscores, and hyphens.",
                    "error"
                );
                return;
            }

            setLoading(saveProfileButton, true);

            try {
                const result = await apiRequest(API.profile, {
                    method: "PUT",
                    body: JSON.stringify({
                        full_name: fullName,
                        username: username
                    })
                });

                originalProfile.full_name = fullName;
                originalProfile.username = username;

                updateLiveSummary();

                showMessage(
                    result.message || "Profile updated successfully."
                );

            } catch (error) {
                showMessage(error.message, "error");

            } finally {
                setLoading(saveProfileButton, false, "Save Changes");
            }
        });
    }

    // ==========================================
    // CANCEL PROFILE CHANGES
    // ==========================================

    if (cancelButton) {
        cancelButton.addEventListener("click", event => {
            event.preventDefault();

            fullNameInput.value = originalProfile.full_name || "";
            usernameInput.value = originalProfile.username || "";

            if (emailInput) {
                emailInput.value = originalProfile.email || "";
            }

            selectedImage = null;

            if (profileImageInput) {
                profileImageInput.value = "";
            }

            if (profileImagePreview) {
                profileImagePreview.src = originalPhotoUrl || "";
            }

            updateLiveSummary();

            showMessage("Unsaved changes have been canceled.");
        });
    }

    // ==========================================
    // PROFILE PICTURE PREVIEW
    // ==========================================

    if (profileImageInput) {
        profileImageInput.addEventListener("change", event => {
            const file = event.target.files[0];

            if (!file) return;

            const allowedTypes = [
                "image/jpeg",
                "image/png",
                "image/webp"
            ];

            if (!allowedTypes.includes(file.type)) {
                showMessage(
                    "Only JPG, PNG, and WEBP images are allowed.",
                    "error"
                );

                profileImageInput.value = "";
                selectedImage = null;
                return;
            }

            if (file.size > MAX_IMAGE_SIZE) {
                showMessage(
                    "Profile picture must be smaller than 5 MB.",
                    "error"
                );

                profileImageInput.value = "";
                selectedImage = null;
                return;
            }

            selectedImage = file;

            const reader = new FileReader();

            reader.onload = readerEvent => {
                if (profileImagePreview) {
                    profileImagePreview.src = readerEvent.target.result;
                }
            };

            reader.readAsDataURL(file);

            uploadProfilePicture();
        });
    }

    // ==========================================
    // UPLOAD PROFILE PICTURE
    // ==========================================

    async function uploadProfilePicture() {
        if (!selectedImage) return;

        const formData = new FormData();

        formData.append("profile_picture", selectedImage);

        if (profileImageInput) {
            profileImageInput.disabled = true;
        }

        try {
            const result = await apiRequest(API.uploadPicture, {
                method: "POST",
                body: formData
            });

            originalPhotoUrl = result.profile_photo;
            originalProfile.profile_photo = result.profile_photo;

            if (profileImagePreview) {
                profileImagePreview.src = result.profile_photo;
            }

            selectedImage = null;

            showMessage(
                result.message || "Profile picture updated successfully."
            );

        } catch (error) {
            showMessage(error.message, "error");

            if (profileImagePreview) {
                profileImagePreview.src = originalPhotoUrl || "";
            }

            if (profileImageInput) {
                profileImageInput.value = "";
            }

            selectedImage = null;

        } finally {
            if (profileImageInput) {
                profileImageInput.disabled = false;
            }
        }
    }

    // ==========================================
    // SHOW / HIDE PASSWORD
    // ==========================================

    document.querySelectorAll(".toggle-password").forEach(button => {
        button.addEventListener("click", () => {
            const targetId = button.dataset.target;
            const passwordInput = document.getElementById(targetId);

            if (!passwordInput) return;

            const isPassword =
                passwordInput.type === "password";

            passwordInput.type = isPassword ? "text" : "password";

            const eyeIcon = button.querySelector("i");

            if (eyeIcon) {
                eyeIcon.classList.toggle("fa-eye", !isPassword);
                eyeIcon.classList.toggle("fa-eye-slash", isPassword);
            }
        });
    });

    // ==========================================
    // PASSWORD STRENGTH
    // ==========================================

    if (newPasswordInput) {
        newPasswordInput.addEventListener("input", () => {
            const password = newPasswordInput.value;

            const requirements = {
                length: password.length >= 8,
                uppercase: /[A-Z]/.test(password),
                number: /\d/.test(password),
                special: /[^A-Za-z0-9]/.test(password)
            };

            const strength = Object.values(requirements)
                .filter(Boolean).length;

            const bars = document.querySelectorAll(".strength-bar");

            bars.forEach((bar, index) => {
                bar.classList.toggle("active", index < strength);
            });

            const strengthText =
                document.getElementById("passwordStrengthText");

            if (strengthText) {
                const labels = [
                    "Very Weak",
                    "Weak",
                    "Fair",
                    "Good",
                    "Strong"
                ];

                strengthText.textContent =
                    password.length === 0
                        ? "Password strength"
                        : labels[strength];
            }

            updatePasswordRequirements(requirements);
        });
    }

    function updatePasswordRequirements(requirements) {
        const requirementElements = {
            length: document.getElementById("reqLength"),
            uppercase: document.getElementById("reqUppercase"),
            number: document.getElementById("reqNumber"),
            special: document.getElementById("reqSpecial")
        };

        Object.entries(requirements).forEach(([key, passed]) => {
            const element = requirementElements[key];

            if (element) {
                element.classList.toggle("valid", passed);
            }
        });
    }

    // ==========================================
    // CLEAR PASSWORD FIELDS
    // ==========================================

    function clearPasswordFields() {
        if (currentPasswordInput) currentPasswordInput.value = "";
        if (newPasswordInput) newPasswordInput.value = "";
        if (confirmPasswordInput) confirmPasswordInput.value = "";

        if (newPasswordInput) {
            newPasswordInput.dispatchEvent(new Event("input"));
        }
    }

    if (clearPasswordButton) {
        clearPasswordButton.addEventListener("click", event => {
            event.preventDefault();
            clearPasswordFields();
        });
    }

    // ==========================================
    // CHANGE PASSWORD
    // ==========================================

    if (updatePasswordButton) {
        updatePasswordButton.addEventListener("click", async event => {
            event.preventDefault();

            const currentPassword = currentPasswordInput?.value || "";
            const newPassword = newPasswordInput?.value || "";
            const confirmPassword = confirmPasswordInput?.value || "";

            if (!currentPassword || !newPassword || !confirmPassword) {
                showMessage("Please complete all password fields.", "error");
                return;
            }

            if (newPassword.length < 8) {
                showMessage(
                    "Password must contain at least 8 characters.",
                    "error"
                );
                return;
            }

            if (!/[A-Z]/.test(newPassword)) {
                showMessage(
                    "Password must contain at least one uppercase letter.",
                    "error"
                );
                return;
            }

            if (!/\d/.test(newPassword)) {
                showMessage(
                    "Password must contain at least one number.",
                    "error"
                );
                return;
            }

            if (!/[^A-Za-z0-9]/.test(newPassword)) {
                showMessage(
                    "Password must contain at least one special character.",
                    "error"
                );
                return;
            }

            if (newPassword !== confirmPassword) {
                showMessage(
                    "New password and confirmation do not match.",
                    "error"
                );
                return;
            }

            setLoading(updatePasswordButton, true);

            try {
                const result = await apiRequest(API.changePassword, {
                    method: "POST",
                    body: JSON.stringify({
                        current_password: currentPassword,
                        new_password: newPassword,
                        confirm_password: confirmPassword
                    })
                });

                clearPasswordFields();

                showMessage(
                    result.message || "Password updated successfully."
                );

            } catch (error) {
                showMessage(error.message, "error");

            } finally {
                setLoading(
                    updatePasswordButton,
                    false,
                    "Update Password"
                );
            }
        });
    }

    // ==========================================
    // INITIALIZE
    // ==========================================

    loadAdminProfile();
});
