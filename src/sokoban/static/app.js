"use strict";

const $ = (id) => document.getElementById(id);
const svgNS = "http://www.w3.org/2000/svg";
const directionKeys = {
  ArrowUp: "up",
  ArrowLeft: "left",
  ArrowDown: "down",
  ArrowRight: "right",
  w: "up",
  a: "left",
  s: "down",
  d: "right",
};
const pad = Array.from(document.querySelectorAll("[data-move]"));
const storeKey = "sokoban.campaign.v1";
let saved;
try {
  saved = JSON.parse(localStorage.getItem(storeKey) || "{}") || {};
} catch {
  saved = {};
}

let levels = [];
let level = 1;
let histories = saved.histories && typeof saved.histories === "object" ? saved.histories : {};
let finished = new Set(Array.isArray(saved.finished) ? saved.finished : []);
let actions = [];
let state = null;
let busy = false;
let playing = false;
let version = 0;
let controller;

function persist() {
  histories[level] = actions;
  try {
    localStorage.setItem(
        storeKey, JSON.stringify({level, histories, finished: [...finished]}));
  } catch { /* The game still runs if browser storage is unavailable. */
  }
}

function message(value, alert = false) {
  $("status").textContent = value;
  $("status").dataset.tone = alert ? "alert" : "normal";
}

function drawBoard() {
  if (!state) return;
  const canvas = $("board");
  const {width, height} = canvas.getBoundingClientRect();
  if (!width || !height) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const ctx = canvas.getContext("2d");
  const board = state.board;
  const cols = Math.max(...board.map((row) => row.length));
  const tile = 40;
  const stride = 34;
  const boardWidth = cols * tile;
  const boardHeight = board.length * stride;
  const scale =
      Math.min((width - 40) / (boardWidth + 6), (height - 38) / (boardHeight + 16), 2.1);
  ctx.setTransform(
      ratio * scale, 0, 0, ratio * scale, ratio * (width - scale * boardWidth) / 2,
      ratio * ((height - scale * boardHeight) / 2 + 3 * scale));
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  function rect(x, y, w, h, color) {
    ctx.fillStyle = color;
    ctx.fillRect(x, y, w, h);
  }
  function round(x, y, w, h, r, fill, stroke) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
    if (stroke) {
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }
  function line(points, color, size) {
    ctx.beginPath();
    ctx.moveTo(...points[0]);
    for (const p of points.slice(1)) ctx.lineTo(...p);
    ctx.strokeStyle = color;
    ctx.lineWidth = size;
    ctx.stroke();
  }
  function ellipse(x, y, rx, ry, color) {
    ctx.beginPath();
    ctx.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
  }
  // Draw the floor first so raised faces remain visible over adjacent tiles.
  round(2.5, 5, boardWidth + 1, boardHeight + 3, 4, "#48593c29");
  round(-1, -3, boardWidth + 2, boardHeight + 8, 3, "#385142", "#304c3d");
  for (let y = 0; y < board.length; y++) {
    for (let x = 0; x < board[y].length; x++) {
      const px = x * tile, py = y * stride;
      rect(px, py, tile, stride, "#c1b79a");
      rect(px + 0.6, py + 0.6, tile - 1.2, stride - 1.2, (x + y) % 2 ? "#eddfbd" : "#e8dab8");
      rect(px + 1, py + 1, tile - 2, 1, "#fff4d6");
      rect(px + 6 + ((x * 7 + y * 3) % 17), py + 23, 2, 1, "#c4ba9c");
    }
  }
  const worker = [
    "     kkkkkk    ",
    "   kkrrrrrrkk  ",
    "  krrrllllrrrk ",
    "  krrrrrrrrrrk ",
    " kkrrrrrrrrrrkk",
    " kcccccccccccck",
    "   khhhhhhkkk  ",
    "  khffhffhffhk ",
    "  khffhffhffhk ",
    "   khffffffhk  ",
    "    khhffhhk   ",
    "  kkwwggggwwkk ",
    " khhwwggggwwhhk",
    " khhwggggggwhhk",
    "  kkkggyyggkkk ",
    "    kggkkggk   ",
    "   kbbk  kbbk  ",
    "   kkkk  kkkk  ",
  ];
  const colors = {
    k: "#334438",
    r: "#dd8549",
    l: "#f5aa60",
    c: "#a45938",
    h: "#d7a073",
    f: "#f2bc89",
    w: "#e5d9ac",
    g: "#5b8c7a",
    y: "#edbd6d",
    b: "#596553"
  };
  for (let y = 0; y < board.length; y++) {
    for (let x = 0; x < board[y].length; x++) {
      const cell = board[y][x], px = x * tile, py = y * stride;
      ctx.save();
      ctx.translate(px, py);
      if (cell === "#") {
        const tint = ["#8ba28b", "#91a58d", "#879e86"][(x * 5 + y * 7) % 3];
        round(0.1, -4, 39.8, 38, 1.5, "#54715e", "#445f50");
        round(0.5, -6, 39, 32.8, 1.5, tint, "#536e5c");
        line([[2, -4.5], [37, -4.5]], "#afbea2", 1.1);
        line([[2, 24.8], [37, 24.8]], "#7d957d", 1);
        line([[2, 27.7], [37, 27.7]], "#728b71", 1.3);
        if ((x * 3 + y) % 4 === 0) {
          line([[23, -5], [22, 0], [18, 2], [20, 5]], "#748f79", 0.9);
          rect(7, 13, 2, 1, "#a1b098");
        }
      } else {
        if (".*+".includes(cell)) {
          ellipse(20, 17.5, 9.7, 8.3, "#b86b40");
          ellipse(20, 17.5, 7.1, 5.9, "#eddfbd");
          ellipse(20, 17.5, 3, 2.6, "#b86b40");
        }
        if (cell === "$" || cell === "*") {
          const done = cell === "*";
          const edge = done ? "#47694b" : "#8c572f";
          const slat = done ? "#a6bd83" : "#efb568";
          const front = done ? "#7d9a68" : "#ce8c4b";
          ellipse(21.5, 26.5, 16.8, 5, "#74694d38");
          round(4.3, -0.5, 31.4, 29.7, 1.7, front, edge);
          round(4.3, -4.8, 31.4, 27.7, 1.5, slat, edge);
          round(7.5, -2, 25, 22.3, .8, front, edge);
          for (const i of [15.2, 24.8]) line([[i, -1], [i, 19.3]], edge, .8);
          line([[8.6, 18.2], [31, -.3]], edge, 5.3);
          line([[8.6, 18.2], [31, -.3]], slat, 3.1);
          for (const i of [6.4, 33.6])
            for (const j of [-2.5, 20.8]) ellipse(i, j, .8, .8, edge);
          line([[5.8, 24.6], [34, 24.6]], slat, .8);
          if (done) {
            ellipse(20, 10, 6.2, 6.2, edge);
            line([[16.7, 10], [19, 12.5], [23.5, 7.3]], "#f6f4d8", 1.5);
          }
        }
        if (cell === "@" || cell === "+") {
          ellipse(20, 27, 12, 4, "#6d6d5240");
          const pixel = 1.8;
          worker.forEach((row, j) => [...row].forEach((c, i) => {
            if (colors[c])
              rect(6.5 + i * pixel, -4 + j * pixel, pixel + .03, pixel + .03, colors[c]);
          }));
        }
      }
      ctx.restore();
    }
  }
}

function refresh() {
  const item = levels.find((entry) => entry.id === level);
  $("level-number").textContent = String(level).padStart(2, "0");
  $("level-title").textContent = item?.title ?? "载入中";
  $("prev").disabled = !item || level === levels[0].id;
  $("next").disabled = !item || level === levels.at(-1).id;
  $("finished").textContent = String(finished.size).padStart(2, "0");
  $("total").textContent = levels.length || 30;
  $("progress").max = levels.length || 30;
  $("progress").value = finished.size;
  $("undo").disabled = !state || actions.length === 0;
  $("reset").disabled = !state || (actions.length === 0 && !playing);
  $("auto").disabled = !state || state.won;
  $("auto").hidden = !!state?.won && level !== levels.at(-1)?.id;
  $("advance").hidden = !$("auto").hidden;
  $("auto").querySelector("use").setAttribute("href", playing ? "#icon-stop" : "#icon-play");
  $("auto").querySelector("span").textContent = playing ? "停止演示" : "机器演示";
  for (const key of pad) key.disabled = !state || state.won;
  if (!state) return;
  const cells = state.board.join("");
  const packed = [...cells].filter((c) => c === "*").length;
  const total = [...cells].filter((c) => c === "*" || c === "$").length;
  $("moves").textContent = state.moves;
  $("packed").replaceChildren(document.createTextNode(String(packed) + " "));
  const suffix = document.createElement("small");
  suffix.textContent = "/ " + total;
  $("packed").append(suffix);
  $("board").setAttribute(
      "aria-label",
      item.title + "，已移动 " + state.moves + " 步，" + total + " 个木箱中 " + packed +
          " 个归位");
  drawBoard();
}

function cancel() {
  version++;
  controller?.abort();
  busy = false;
  playing = false;
}

async function request(path, command) {
  controller = new AbortController();
  const reply = await fetch("/api/game", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({level, actions: path, command}),
    signal: controller.signal,
  });
  if (!reply.ok) throw new Error("游戏请求失败");
  return reply.json();
}

function adopt(newState, path) {
  state = newState;
  actions = path;
  if (state.won) {
    finished.add(level);
    playing = false;
    message("本关已完成");
  } else {
    message(playing ? "机器演示中" : "进行中");
  }
  persist();
  refresh();
}

function fail(error, token) {
  if (error.name === "AbortError" || token !== version) return;
  cancel();
  message("连接失败，请重试", true);
  refresh();
}

async function selectLevel(id) {
  cancel();
  const token = version;
  level = id;
  state = null;
  actions = [];
  const ctx = $("board").getContext("2d");
  ctx.clearRect(-10000, -10000, 20000, 20000);
  refresh();
  message("载入中");
  const path = Array.isArray(histories[level]) ?
      histories[level].filter((dir) => ["up", "down", "left", "right"].includes(dir)) :
      [];
  try {
    const next = await request(path, "state");
    if (version !== token) return;
    if (next.moves === path.length) {
      adopt(next, path);
    } else {
      const initial = await request([], "state");
      if (version === token) adopt(initial, []);
    }
  } catch (error) {
    fail(error, token);
  }
}

async function update(path) {
  const token = version;
  busy = true;
  try {
    const next = await request(path, "state");
    if (token !== version) return false;
    const moved = next.moves === path.length;
    adopt(next, moved ? path : actions);
    if (!moved) message("前方受阻", true);
    return moved;
  } catch (error) {
    fail(error, token);
    return false;
  } finally {
    if (token === version) busy = false;
  }
}

function move(dir) {
  if (!state || state.won || (busy && !playing)) return;
  if (playing) cancel();
  const key = pad.find((el) => el.dataset.move === dir);
  key.classList.add("pressed");
  setTimeout(() => key.classList.remove("pressed"), 120);
  void update([...actions, dir]);
}

function rewind(restart) {
  if (!state) return;
  cancel();
  refresh();
  void update(restart ? [] : actions.slice(0, -1));
}

async function autoPlay() {
  if (!state || state.won) return;
  if (playing) {
    cancel();
    message("演示已停止");
    refresh();
    return;
  }
  if (busy) return;
  cancel();
  const token = version;
  playing = true;
  refresh();
  message("正在寻找解法");
  try {
    const result = await request(actions, "solve");
    if (version !== token) return;
    if (!Array.isArray(result.solution)) {
      playing = false;
      refresh();
      message("当前局面无解，可撤销或重来", true);
      return;
    }
    for (const dir of result.solution) {
      await new Promise((resolve) => setTimeout(resolve, 240));
      if (version !== token || !playing) return;
      if (!await update([...actions, dir])) break;
    }
    if (version === token) {
      playing = false;
      refresh();
    }
  } catch (error) {
    fail(error, token);
  }
}

function levelMenu() {
  const list = $("level-list");
  list.replaceChildren();
  for (const item of levels) {
    const btn = document.createElement("button");
    btn.className = "level-choice";
    const num = document.createElement("span");
    num.textContent = String(item.id).padStart(2, "0");
    btn.append(num, document.createTextNode(item.title));
    btn.setAttribute("aria-current", String(item.id === level));
    if (finished.has(item.id)) {
      const svg = document.createElementNS(svgNS, "svg");
      svg.setAttribute("class", "ico");
      svg.setAttribute("aria-hidden", "true");
      const use = document.createElementNS(svgNS, "use");
      use.setAttribute("href", "#icon-check");
      svg.append(use);
      btn.append(svg);
      btn.setAttribute("aria-label", item.id + " " + item.title + "，已通关");
    }
    btn.addEventListener("click", () => {
      $("levels-dialog").close();
      void selectLevel(item.id);
    });
    list.append(btn);
  }
  $("levels-dialog").showModal();
}

function adjacent(offset) {
  const index = levels.findIndex((item) => item.id === level);
  if (levels[index + offset]) void selectLevel(levels[index + offset].id);
}

$("prev").onclick = () => adjacent(-1);
$("next").onclick = $("advance").onclick = () => adjacent(1);
$("auto").onclick = autoPlay;
$("undo").onclick = () => rewind(false);
$("reset").onclick = () => rewind(true);
$("show-levels").onclick = levelMenu;
$("show-help").onclick = () => $("help-dialog").showModal();
for (const btn of pad) btn.onclick = () => move(btn.dataset.move);
for (const dialog of document.querySelectorAll("dialog")) {
  dialog.querySelector(".close").onclick = () => dialog.close();
  dialog.addEventListener("click", (e) => {
    if (e.target === dialog) {
      const bounds = dialog.getBoundingClientRect();
      if (e.clientX < bounds.left || e.clientX > bounds.right || e.clientY < bounds.top ||
          e.clientY > bounds.bottom)
        dialog.close();
    }
  });
}
window.addEventListener("keydown", (event) => {
  if (document.querySelector("dialog[open]") || event.metaKey || event.ctrlKey || event.altKey)
    return;
  const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
  if (directionKeys[key]) {
    event.preventDefault();
    move(directionKeys[key]);
  } else if (key === "z" || key === "r") {
    event.preventDefault();
    if (!event.repeat) rewind(key === "r");
  } else if (key === "Escape" && playing) {
    event.preventDefault();
    void autoPlay();
  }
});
new ResizeObserver(drawBoard).observe($("stage"));

async function start() {
  refresh();
  try {
    const reply = await fetch("/api/levels");
    if (!reply.ok) throw new Error("关卡载入失败");
    const body = await reply.json();
    levels = body.levels;
    finished = new Set(levels.map((item) => item.id).filter((id) => finished.has(id)));
    await selectLevel(levels.find((item) => item.id === saved.level)?.id ?? levels[0].id);
  } catch (error) {
    fail(error, version);
  }
}
void start();
