/**
 * Autonet -- Chrome Extension Service Worker
 *
 * Connects to a local Autonet daemon via WebSocket.
 * Translates CDP protocol commands into chrome.debugger API calls,
 * allowing AI agents to control the real browser profile.
 *
 * Keep-alive: A content script (keep-alive.js) maintains a persistent port
 * connection that prevents Chrome from suspending this service worker.
 */

const RELAY_URL = "ws://127.0.0.1:9222/extension";

let ws = null;
let reconnectTimer = null;

// sessionId -> tabId
const sessionToTab = new Map();
// tabId -> sessionId
const tabToSession = new Map();

let autoAttach = false;

const TAB_MARKER = "\uD83D\uDD34 "; // red circle prefix for agent-controlled tabs

// Methods that are "virtual" -- handled at browser level, not by chrome.debugger
const VIRTUAL_SESSION_METHODS = new Set([
  "Target.setAutoAttach",
  "Target.setDiscoverTargets",
  "Target.setRemoteLocations",
  "Target.getTargets",
  "Target.attachToTarget",
  "Target.detachFromTarget",
  "Target.createTarget",
  "Target.closeTarget",
]);

// -- Keep-alive port from content scripts -------------------------------------
// Content scripts connect via chrome.runtime.connect() which prevents
// the MV3 service worker from being suspended.

const keepAlivePorts = new Set();

chrome.runtime.onConnect.addListener((port) => {
  if (port.name === "keepAlive") {
    keepAlivePorts.add(port);
    port.onDisconnect.addListener(() => {
      keepAlivePorts.delete(port);
    });
    port.onMessage.addListener((msg) => {
      // Just acknowledge pings -- the port being open is what matters
    });
  }
});

// -- WebSocket connection -----------------------------------------------------

function connect() {
  if (ws && ws.readyState === WebSocket.OPEN) return;

  try {
    ws = new WebSocket(RELAY_URL);

    ws.onopen = () => {
      console.log("[Autonet] Connected to daemon");
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
    };

    ws.onmessage = async (event) => {
      try {
        const msg = JSON.parse(event.data);
        await handleMessage(msg);
      } catch (e) {
        console.error("[Autonet] Error handling message:", e);
      }
    };

    ws.onclose = () => {
      console.log("[Autonet] Disconnected from daemon");
      ws = null;
      scheduleReconnect();
    };

    ws.onerror = () => {
      // onclose will fire after this
      ws = null;
    };
  } catch (e) {
    console.error("[Autonet] Connection failed:", e);
    scheduleReconnect();
  }
}

function scheduleReconnect() {
  if (reconnectTimer) return;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    connect();
  }, 3000);
}

function send(obj) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj));
  }
}

// -- Training capture config management ---------------------------------------

let trainingCaptureEnabled = false;

// Listen for messages from training-capture.js content script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "training_frame" && trainingCaptureEnabled) {
    // Forward training frame to relay
    send({
      type: "training_frame",
      data: message.data,
      tabId: sender.tab?.id,
    });
  }
  return false; // sync response
});

// -- Message dispatch ---------------------------------------------------------

async function handleMessage(msg) {
  switch (msg.type) {
    case "cdp_command":
      return handleCDPCommand(msg);
    case "get_tabs":
      return handleGetTabs(msg);
    case "close_tab":
      return handleCloseTab(msg);
    case "new_tab":
      return handleNewTab(msg);
    case "training_config":
      return handleTrainingConfig(msg);
  }
}

async function handleTrainingConfig(msg) {
  const { enabled, excludePatterns, captureIntervalMs } = msg;

  trainingCaptureEnabled = enabled ?? false;

  // Push config to all content scripts via storage
  const captureConfig = {
    enabled: trainingCaptureEnabled,
    excludePatterns: excludePatterns || [],
    captureIntervalMs: captureIntervalMs || 5000,
  };

  try {
    await chrome.storage.local.set({ trainingCapture: captureConfig });
  } catch (e) {
    console.warn("[Autonet] Failed to save training config:", e.message);
  }

  send({ type: "training_config_response", id: msg.id, result: captureConfig });
}

// -- CDP command handling -----------------------------------------------------

async function handleCDPCommand(msg) {
  const { id, method, params = {}, sessionId } = msg;

  try {
    if (sessionId) {
      // Session-scoped command
      if (VIRTUAL_SESSION_METHODS.has(method)) {
        send({ type: "cdp_response", id, result: {}, sessionId });
        return;
      }

      const tabId = sessionToTab.get(sessionId);
      if (!tabId) {
        send({ type: "cdp_response", id, error: { message: `Unknown session: ${sessionId}`, code: -32000 }, sessionId });
        return;
      }
      const result = await chrome.debugger.sendCommand({ tabId }, method, params);
      send({ type: "cdp_response", id, result: result || {}, sessionId });
    } else {
      // Browser-level command
      await handleBrowserCommand(id, method, params);
    }
  } catch (e) {
    send({ type: "cdp_response", id, error: { message: e.message || String(e), code: -32000 }, sessionId });
  }
}

async function handleBrowserCommand(id, method, params) {
  switch (method) {
    case "Target.getTargets": {
      const tabs = await chrome.tabs.query({});
      const targetInfos = tabs
        .filter((t) => {
          const url = t.url || "";
          return !url.startsWith("chrome://") && !url.startsWith("chrome-extension://");
        })
        .map((t) => ({
          targetId: String(t.id),
          type: "page",
          title: t.title || "",
          url: t.url || "",
          attached: tabToSession.has(t.id),
          browserContextId: "default",
        }));
      send({ type: "cdp_response", id, result: { targetInfos } });
      break;
    }

    case "Target.setAutoAttach": {
      autoAttach = params.autoAttach ?? false;
      send({ type: "cdp_response", id, result: {} });
      break;
    }

    case "Target.attachToTarget": {
      const tabId = parseInt(params.targetId, 10);

      // Check tab URL -- skip chrome:// and extension pages
      try {
        const checkTab = await chrome.tabs.get(tabId);
        const checkUrl = checkTab.url || checkTab.pendingUrl || "";
        if (checkUrl.startsWith("chrome://") || checkUrl.startsWith("chrome-extension://")) {
          send({ type: "cdp_response", id, error: { message: `Cannot attach to ${checkUrl}`, code: -32000 } });
          break;
        }
      } catch (_) {}

      // Check if already attached -- reuse existing session
      if (tabToSession.has(tabId)) {
        const existingSid = tabToSession.get(tabId);
        send({ type: "cdp_response", id, result: { sessionId: existingSid } });
        // Re-emit attachedToTarget event so SessionManager picks it up
        const tab = await chrome.tabs.get(tabId);
        send({
          type: "cdp_event",
          method: "Target.attachedToTarget",
          params: {
            sessionId: existingSid,
            targetInfo: {
              targetId: String(tabId),
              type: "page",
              title: tab.title || "",
              url: tab.url || "",
            },
            waitingForDebugger: false,
          },
        });
        break;
      }

      const sid = `session-${tabId}-${Date.now()}`;

      await chrome.debugger.attach({ tabId }, "1.3");

      sessionToTab.set(sid, tabId);
      tabToSession.set(tabId, sid);

      // Mark the tab title with red circle
      markTab(tabId);

      send({ type: "cdp_response", id, result: { sessionId: sid } });

      // Notify client of attach
      const tab = await chrome.tabs.get(tabId);
      send({
        type: "cdp_event",
        method: "Target.attachedToTarget",
        params: {
          sessionId: sid,
          targetInfo: {
            targetId: String(tabId),
            type: "page",
            title: tab.title || "",
            url: tab.url || "",
          },
          waitingForDebugger: false,
        },
      });
      break;
    }

    case "Target.detachFromTarget": {
      const sid = params.sessionId;
      const tabId = sessionToTab.get(sid);
      if (tabId) {
        // Remove marker before detaching
        await unmarkTab(tabId);
        try { await chrome.debugger.detach({ tabId }); } catch (_) {}
        sessionToTab.delete(sid);
        tabToSession.delete(tabId);
      }
      send({ type: "cdp_response", id, result: {} });
      break;
    }

    case "Target.createTarget": {
      const tab = await chrome.tabs.create({ url: params.url || "about:blank" });
      const newTargetId = String(tab.id);
      send({ type: "cdp_response", id, result: { targetId: newTargetId } });

      // Auto-attach if enabled (Browser Use expects this)
      if (autoAttach) {
        try {
          await autoAttachTab(tab.id);
        } catch (e) {
          console.warn("[Autonet] Auto-attach failed for new target:", e);
        }
      }
      break;
    }

    case "Target.closeTarget": {
      const tabId = parseInt(params.targetId, 10);
      await chrome.tabs.remove(tabId);
      send({ type: "cdp_response", id, result: { success: true } });
      break;
    }

    case "Browser.getVersion": {
      send({
        type: "cdp_response",
        id,
        result: {
          protocolVersion: "1.3",
          product: "Chrome (via Autonet)",
          userAgent: navigator.userAgent,
          jsVersion: "",
        },
      });
      break;
    }

    case "Target.setDiscoverTargets":
    case "Target.setRemoteLocations": {
      send({ type: "cdp_response", id, result: {} });
      break;
    }

    default: {
      // Try forwarding as a session-less command to the first attached tab
      const firstEntry = sessionToTab.entries().next().value;
      if (firstEntry) {
        const [, tabId] = firstEntry;
        try {
          const result = await chrome.debugger.sendCommand({ tabId }, method, params);
          send({ type: "cdp_response", id, result: result || {} });
          return;
        } catch (_) {}
      }
      send({ type: "cdp_response", id, error: { message: `Unhandled: ${method}`, code: -32601 } });
    }
  }
}

// -- HTTP-endpoint helpers (for /json/list, /json/new, etc.) -------------------

async function handleGetTabs(msg) {
  const tabs = await chrome.tabs.query({});
  send({
    type: "tabs_response",
    id: msg.id,
    result: tabs
      .filter((t) => {
        const url = t.url || "";
        return !url.startsWith("chrome://") && !url.startsWith("chrome-extension://");
      })
      .map((t) => ({
        id: String(t.id),
        type: "page",
        title: t.title || "Untitled",
        url: t.url || "about:blank",
        webSocketDebuggerUrl: `ws://127.0.0.1:9222/devtools/page/${t.id}`,
      })),
  });
}

async function handleCloseTab(msg) {
  try {
    await chrome.tabs.remove(parseInt(msg.tabId, 10));
    send({ type: "close_tab_response", id: msg.id, success: true });
  } catch (e) {
    send({ type: "close_tab_response", id: msg.id, success: false, error: e.message });
  }
}

async function handleNewTab(msg) {
  const tab = await chrome.tabs.create({ url: msg.url || "about:blank" });
  send({ type: "new_tab_response", id: msg.id, tabId: String(tab.id) });
}

// -- Debugger events -> relay -------------------------------------------------

chrome.debugger.onEvent.addListener((source, method, params) => {
  const sid = tabToSession.get(source.tabId);
  if (sid) {
    send({ type: "cdp_event", method, params: params || {}, sessionId: sid });
  }
});

chrome.debugger.onDetach.addListener((source, reason) => {
  const sid = tabToSession.get(source.tabId);
  if (sid) {
    sessionToTab.delete(sid);
    tabToSession.delete(source.tabId);
    send({ type: "cdp_event", method: "Target.detachedFromTarget", params: { sessionId: sid, reason } });
  }
});

// -- Tab title marker ---------------------------------------------------------
// Marks agent-controlled tabs with a red circle prefix. Uses a MutationObserver
// so SPAs that change document.title don't lose the marker.

async function markTab(tabId) {
  try {
    await chrome.debugger.sendCommand({ tabId }, "Runtime.evaluate", {
      expression: `(function() {
        var M = ${JSON.stringify(TAB_MARKER)};
        if (!document.title.startsWith(M)) document.title = M + document.title;
        if (!window.__agentMarkerObserver) {
          var titleEl = document.querySelector('title');
          if (titleEl) {
            window.__agentMarkerObserver = new MutationObserver(function() {
              if (!document.title.startsWith(M)) document.title = M + document.title;
            });
            window.__agentMarkerObserver.observe(titleEl, { childList: true, characterData: true, subtree: true });
          }
        }
      })()`,
    });
  } catch (e) {
    console.warn("[Autonet] markTab failed:", e.message);
  }
}

async function unmarkTab(tabId) {
  try {
    await chrome.debugger.sendCommand({ tabId }, "Runtime.evaluate", {
      expression: `(function() {
        var M = ${JSON.stringify(TAB_MARKER)};
        if (window.__agentMarkerObserver) {
          window.__agentMarkerObserver.disconnect();
          delete window.__agentMarkerObserver;
        }
        if (document.title.startsWith(M)) document.title = document.title.slice(M.length);
      })()`,
    });
  } catch (_) {} // detach may race -- ignore errors
}

// -- Auto-attach helper -------------------------------------------------------

async function autoAttachTab(tabId) {
  if (tabToSession.has(tabId)) return; // already attached

  const sid = `session-${tabId}-${Date.now()}`;
  await chrome.debugger.attach({ tabId }, "1.3");
  sessionToTab.set(sid, tabId);
  tabToSession.set(tabId, sid);

  // Mark the tab title with red circle
  markTab(tabId);

  let tab;
  try { tab = await chrome.tabs.get(tabId); } catch (_) { tab = {}; }

  send({
    type: "cdp_event",
    method: "Target.attachedToTarget",
    params: {
      sessionId: sid,
      targetInfo: {
        targetId: String(tabId),
        type: "page",
        title: tab.title || "",
        url: tab.url || "about:blank",
      },
      waitingForDebugger: false,
    },
  });
}

// -- Tab lifecycle -> Target events -------------------------------------------

chrome.tabs.onCreated.addListener(async (tab) => {
  const url = tab.url || tab.pendingUrl || "";
  if (url.startsWith("chrome://") || url.startsWith("chrome-extension://")) return;

  send({
    type: "cdp_event",
    method: "Target.targetCreated",
    params: { targetInfo: { targetId: String(tab.id), type: "page", title: tab.title || "", url: url || "about:blank" } },
  });

  if (autoAttach) {
    try {
      await autoAttachTab(tab.id);
    } catch (e) {
      console.warn("[Autonet] Auto-attach on tab created failed:", e);
    }
  }
});

chrome.tabs.onRemoved.addListener((tabId) => {
  const sid = tabToSession.get(tabId);
  if (sid) {
    sessionToTab.delete(sid);
    tabToSession.delete(tabId);
  }
  send({ type: "cdp_event", method: "Target.targetDestroyed", params: { targetId: String(tabId) } });
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  send({
    type: "cdp_event",
    method: "Target.targetInfoChanged",
    params: {
      targetInfo: { targetId: String(tabId), type: "page", title: tab.title || "", url: tab.url || "", attached: tabToSession.has(tabId) },
    },
  });

  // Re-inject marker after page navigations (the old document is gone)
  if (changeInfo.status === "complete" && tabToSession.has(tabId)) {
    markTab(tabId);
  }
});

// -- Boot ---------------------------------------------------------------------

connect();
