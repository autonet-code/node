async function checkStatus() {
  const el = document.getElementById("status");
  const info = document.getElementById("info");

  try {
    const resp = await fetch("http://127.0.0.1:9222/json/version");
    if (resp.ok) {
      const data = await resp.json();
      el.textContent = "Connected to relay";
      el.className = "status connected";
      info.textContent = `Extension WS: ${data.extensionConnected ? "yes" : "waiting..."}`;
    } else {
      el.textContent = "Relay not responding";
      el.className = "status disconnected";
      info.textContent = "Start relay: python relay.py";
    }
  } catch (e) {
    el.textContent = "Relay not running";
    el.className = "status disconnected";
    info.textContent = "Start relay: python relay.py";
  }
}

checkStatus();
setInterval(checkStatus, 3000);
