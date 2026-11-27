/*
 * Signs this user out the moment an administrator bans or deactivates the account.
 *
 * The server ends every session in the database and publishes on the user's own
 * private channel; without this the tab would keep showing until its next request
 * happened to hit the session guard. Pages work as before when Pusher is not
 * configured (the guard still redirects on the next request).
 */
(function () {
  "use strict";

  var PUSHER_SRC = "https://js.pusher.com/8.3.0/pusher.min.js";
  var leaving = false;

  function leave() {
    if (leaving) return;
    leaving = true;
    // The sessions are already ended server-side, so this only clears the cookie.
    fetch("/api/logout", { method: "POST", credentials: "same-origin" })
      .catch(function () { /* the guard clears the cookie on the next request */ })
      .then(function () { window.location.replace("/login"); });
  }

  function loadPusher() {
    return new Promise(function (resolve, reject) {
      if (window.Pusher) return resolve();
      var script = document.createElement("script");
      script.src = PUSHER_SRC;
      script.async = true;
      script.onload = function () { resolve(); };
      script.onerror = function () { reject(new Error("Pusher could not be loaded")); };
      document.head.appendChild(script);
    });
  }

  async function start() {
    var config;
    try {
      var response = await fetch("/api/realtime/me", { credentials: "same-origin", cache: "no-store" });
      if (!response.ok) return;
      config = await response.json();
    } catch (error) {
      return; // signed out or offline: nothing to listen for
    }
    if (!config || !config.enabled || !config.channel) return;
    try {
      await loadPusher();
    } catch (error) {
      return;
    }
    var client = new window.Pusher(config.key, {
      cluster: config.cluster,
      forceTLS: true,
      channelAuthorization: { endpoint: config.authEndpoint, transport: "ajax" },
    });
    client.subscribe(config.channel).bind(config.event, leave);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
