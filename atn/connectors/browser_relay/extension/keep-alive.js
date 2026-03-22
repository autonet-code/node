/**
 * Content script that keeps the service worker alive via a persistent port.
 * Injected into all http/https pages. Reconnects every 4.5 minutes
 * (before Chrome's 5-minute port timeout).
 */

let port;

function connectPort() {
  try {
    port = chrome.runtime.connect({ name: "keepAlive" });
    port.onDisconnect.addListener(() => {
      // Reconnect after a short delay
      setTimeout(connectPort, 1000);
    });
  } catch (e) {
    // Extension context invalidated -- stop trying
  }
}

connectPort();

// Chrome closes ports after 5 minutes of inactivity.
// Send a ping every 4 minutes to keep it alive.
setInterval(() => {
  try {
    if (port) {
      port.postMessage({ type: "ping" });
    } else {
      connectPort();
    }
  } catch (e) {
    connectPort();
  }
}, 240000);
