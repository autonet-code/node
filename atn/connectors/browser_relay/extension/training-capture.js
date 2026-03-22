/**
 * Training Capture Content Script
 *
 * Extracts text content from web pages for Autonet training data.
 * Runs on all http/https pages. Captures on:
 *   - Initial page load (after DOM ready)
 *   - Significant DOM mutations (throttled)
 *   - SPA navigation (URL changes)
 *
 * Sends structured text + metadata to the service worker, which
 * forwards it through the relay to any connected training subscriber.
 *
 * Privacy: Respects an exclude-list of URL patterns. Does NOT capture
 * from pages matching patterns in the configured exclude list.
 */

(function () {
  "use strict";

  // -- Config (overridable via storage) -----------------------------------------

  const DEFAULT_CONFIG = {
    enabled: false, // Off by default -- user must opt in
    captureIntervalMs: 5000, // Min ms between captures (throttle)
    minTextLength: 50, // Skip pages with less text than this
    maxTextLength: 50000, // Truncate text beyond this
    excludePatterns: [], // URL substrings to skip
  };

  let config = { ...DEFAULT_CONFIG };
  let lastCaptureTime = 0;
  let lastTextHash = 0;
  let lastUrl = location.href;
  let mutationTimer = null;

  // -- Load config from storage -------------------------------------------------

  function loadConfig() {
    try {
      chrome.storage?.local?.get("trainingCapture", (result) => {
        if (result && result.trainingCapture) {
          config = { ...DEFAULT_CONFIG, ...result.trainingCapture };
        }
      });
    } catch (_) {
      // storage API may not be available
    }
  }

  // Listen for config updates
  try {
    chrome.storage?.onChanged?.addListener((changes, area) => {
      if (area === "local" && changes.trainingCapture) {
        config = { ...DEFAULT_CONFIG, ...changes.trainingCapture.newValue };
      }
    });
  } catch (_) {}

  // -- URL exclusion check ------------------------------------------------------

  function shouldExclude() {
    const url = location.href.toLowerCase();

    // Always skip non-content pages
    if (
      url.startsWith("chrome://") ||
      url.startsWith("chrome-extension://") ||
      url.startsWith("about:") ||
      url.startsWith("data:")
    ) {
      return true;
    }

    // Check user-configured exclude patterns
    for (const pattern of config.excludePatterns) {
      if (url.includes(pattern.toLowerCase())) {
        return true;
      }
    }

    return false;
  }

  // -- Text extraction ----------------------------------------------------------

  function extractText() {
    // Get visible text content
    const text = document.body?.innerText || "";

    if (text.length < config.minTextLength) {
      return null;
    }

    // Truncate if too long
    const trimmed =
      text.length > config.maxTextLength
        ? text.slice(0, config.maxTextLength)
        : text;

    return trimmed;
  }

  function extractMetadata() {
    return {
      url: location.href,
      title: document.title || "",
      lang: document.documentElement?.lang || "",
      timestamp: Date.now() / 1000,
    };
  }

  // -- Simple string hash for dedup ---------------------------------------------

  function simpleHash(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const ch = str.charCodeAt(i);
      hash = ((hash << 5) - hash + ch) | 0;
    }
    return hash;
  }

  // -- Capture and send ---------------------------------------------------------

  function captureAndSend(trigger) {
    if (!config.enabled) return;
    if (shouldExclude()) return;

    // Throttle
    const now = Date.now();
    if (now - lastCaptureTime < config.captureIntervalMs) return;

    const text = extractText();
    if (!text) return;

    // Dedup: skip if text hasn't changed
    const hash = simpleHash(text);
    if (hash === lastTextHash && location.href === lastUrl) return;

    lastTextHash = hash;
    lastUrl = location.href;
    lastCaptureTime = now;

    const metadata = extractMetadata();

    try {
      chrome.runtime.sendMessage({
        type: "training_frame",
        data: {
          text: text,
          metadata: metadata,
          trigger: trigger,
        },
      });
    } catch (_) {
      // Extension context invalidated
    }
  }

  // -- DOM mutation observer (throttled) ----------------------------------------

  function onMutations() {
    if (mutationTimer) return; // already scheduled
    mutationTimer = setTimeout(() => {
      mutationTimer = null;
      captureAndSend("mutation");
    }, config.captureIntervalMs);
  }

  function startObserver() {
    if (!document.body) return;

    const observer = new MutationObserver(onMutations);
    observer.observe(document.body, {
      childList: true,
      subtree: true,
      characterData: true,
    });
  }

  // -- SPA navigation detection -------------------------------------------------

  function watchNavigation() {
    // Detect URL changes (pushState/replaceState in SPAs)
    let currentUrl = location.href;

    const check = () => {
      if (location.href !== currentUrl) {
        currentUrl = location.href;
        // Small delay to let new content render
        setTimeout(() => captureAndSend("navigation"), 1000);
      }
    };

    // Poll for URL changes (catches pushState, replaceState, hash changes)
    setInterval(check, 2000);

    // Also listen for popstate
    window.addEventListener("popstate", () => {
      setTimeout(() => captureAndSend("navigation"), 1000);
    });
  }

  // -- Init ---------------------------------------------------------------------

  loadConfig();

  // Capture on initial load
  if (document.readyState === "complete") {
    captureAndSend("load");
  } else {
    window.addEventListener("load", () => captureAndSend("load"));
  }

  startObserver();
  watchNavigation();
})();
