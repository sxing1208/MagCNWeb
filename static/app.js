const WS_URL = `ws://${location.host}/ws`;
const tbl     = document.getElementById("gridTable");
const ledWrap = document.getElementById("ledGrid");
const slider  = document.getElementById("slider");
const output  = document.getElementById("sliderValue");
const jsonLog = document.getElementById("jsonLog"); // NEW: JSON log display
let max  = 500;

// Create LED grid
for (let i = 0; i < 25; ++i) {
  const d = document.createElement("div");
  d.className = "led";
  ledWrap.appendChild(d);
}
const leds = [...ledWrap.children];

// Turn 21 sensor values into a 5×5 matrix with corners filled
function processStream(stream) {
  const W = 1.414, D = 3.828;
  const data = stream.map((v) => v);
  const g = Array.from({ length: 5 }, () => Array(5).fill(0));

  for (let i = 0; i < 3; i++) g[0][i + 1] = data[i];
  for (let i = 0; i < 5; i++) g[1][i] = data[i + 3];
  for (let i = 0; i < 5; i++) g[2][i] = data[i + 8];
  for (let i = 0; i < 5; i++) g[3][i] = data[i + 13];
  for (let i = 0; i < 3; i++) g[4][i + 1] = data[i + 18];

  g[0][0] = (g[1][0] * W + g[0][1] * W + g[1][1]) / D;
  g[4][4] = (g[4][3] * W + g[3][4] * W + g[3][3]) / D;
  g[4][0] = (g[3][0] * W + g[4][1] * W + g[3][1]) / D;
  g[0][4] = (g[0][3] * W + g[1][4] * W + g[1][3]) / D;

  return g;
}

// Render numeric table
function updateTable(matrix) {
  tbl.innerHTML = matrix
    .map(row => `<tr>${row.map(v => `<td>${v.toFixed(0)}</td>`).join("")}</tr>`)
    .join("");
}

// Update LED colors
function updateLeds(matrix) {
  const flat = matrix.flat();
  flat.forEach((val, i) => {
    const p = Math.min(val / max, 1);
    leds[i].style.background =
      `color-mix(in srgb, var(--led-off) ${100 - p * 100}%, var(--accent))`;
  });
}

// WebSocket setup
const ws = new WebSocket(WS_URL);
ws.addEventListener("open",  () => console.log("[WS] open"));
ws.addEventListener("close", () => console.log("[WS] closed"));

ws.addEventListener("message", ev => {
  try {
    const data = JSON.parse(ev.data);

    // Always log the raw JSON to the debug box
    jsonLog.textContent = JSON.stringify(data, null, 2);

    // Handle by pid / type
    if (data.pid === "device" && typeof data.value === "string") {
      // Process device numeric frame
      const values = data.value.split(",").map(Number);

      // If needed, reduce 64 to 21 values for LED processing
      const reduced = values.slice(0, 21);
      const grid = processStream(reduced);
      updateTable(grid);
      updateLeds(grid);
    }
    else if (data.pid === "central") {
      switch (data.type) {
        case "msg":
          console.log("[Central Message]", data.message);
          break;

        case "login_success":
          console.log("[Login] Success");
          // Optionally update UI to show logged-in state
          break;

        case "login_fail":
          console.warn("[Login] Failed:", data.message);
          // Optionally show login error to user
          break;

        case "prediction":
          // Update vitals display
          document.getElementById("hrText").textContent =
            `HR: ${Math.round(data.HR)} bpm`;
          document.getElementById("efText").textContent =
            `EF: ${(data.EF * 100).toFixed(1)} %`;
          // Could also show diagnosis somewhere
          console.log("[Prediction]", data.diagnosis, `Conf: ${data.confidence}`);
          break;

        default:
          console.warn("[Central] Unhandled message type:", data.type);
      }
    }
    else {
      console.warn("[WS] Unknown message format", data);
    }

  } catch (err) {
    console.warn("Invalid WS message", ev.data);
  }
});


// Slider for sensitivity
slider.addEventListener("input", () => {
  output.textContent = slider.value;
  max = slider.value * 170;
});
