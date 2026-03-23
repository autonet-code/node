async function checkStatus() {
  const el = document.getElementById("status");
  const statusText = document.getElementById("status-text");
  const info = document.getElementById("info");

  try {
    const resp = await fetch("http://127.0.0.1:9222/json/version");
    if (resp.ok) {
      const data = await resp.json();
      statusText.textContent = "Connected to daemon";
      el.className = "status connected";
      info.textContent = data.extensionConnected
        ? "Extension WebSocket active"
        : "Extension WebSocket connecting...";
    } else {
      statusText.textContent = "Daemon not responding";
      el.className = "status disconnected";
      info.textContent = "Start the daemon: pip install autonet-daemon && atn-daemon";
    }
  } catch (e) {
    statusText.textContent = "Daemon not running";
    el.className = "status disconnected";
    info.textContent = "Start the daemon: pip install autonet-daemon && atn-daemon";
  }
}

checkStatus();
setInterval(checkStatus, 3000);
