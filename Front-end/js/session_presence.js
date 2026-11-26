/*
 * Signrest session presence.
 *
 * A closing tab tells the server this device is no longer in use, so the admin session
 * monitor shows it as Offline almost immediately. This never ends the login: the marker
 * is cleared by the next request from the same cookie, so "Remember me" still resumes
 * the session when the browser is reopened.
 */
(() => {
  "use strict";
  if (!navigator.sendBeacon) return;

  window.addEventListener("pagehide", () => {
    try {
      navigator.sendBeacon("/api/session/offline", new Blob([""], { type: "text/plain" }));
    } catch {
      /* presence is best effort; inactivity still marks the session offline */
    }
  });
})();
