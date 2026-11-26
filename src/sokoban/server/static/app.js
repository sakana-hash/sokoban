import {drawBoard} from "./board.js";

const $ = (id) => document.getElementById(id);
const canvas = $("board");
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

function redrawBoard() {
  if (state) drawBoard(canvas, state.board, window.devicePixelRatio || 1);
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
  canvas.setAttribute(
      "aria-label",
      item.title + "，已移动 " + state.moves + " 步，" + total + " 个木箱中 " + packed +
          " 个归位");
  redrawBoard();
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
  const ctx = canvas.getContext("2d");
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
new ResizeObserver(redrawBoard).observe($("stage"));

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
