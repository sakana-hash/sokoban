export function drawBoard(canvas, board, pixelRatio) {
  const {width, height} = canvas.getBoundingClientRect();
  if (!width || !height) return;
  canvas.width = Math.round(width * pixelRatio);
  canvas.height = Math.round(height * pixelRatio);
  const ctx = canvas.getContext("2d");
  const cols = Math.max(...board.map((row) => row.length));
  const tile = 40;
  const stride = 34;
  const boardWidth = cols * tile;
  const boardHeight = board.length * stride;
  const scale =
      Math.min((width - 40) / (boardWidth + 6), (height - 38) / (boardHeight + 16), 2.1);
  ctx.setTransform(
      pixelRatio * scale, 0, 0, pixelRatio * scale, pixelRatio * (width - scale * boardWidth) / 2,
      pixelRatio * ((height - scale * boardHeight) / 2 + 3 * scale));
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
