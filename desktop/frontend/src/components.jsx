import React, { useEffect, useRef } from "react";

export const Icon = ({
  d,
  size = 16,
  stroke = "currentColor",
  fill = "none",
  sw = 1.5,
  vb = "0 0 24 24",
}) => (
  <svg
    width={size}
    height={size}
    viewBox={vb}
    fill={fill}
    stroke={stroke}
    strokeWidth={sw}
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d={d} />
  </svg>
);

export const Icons = {
  train: "M13 10V3L4 14h7v7l9-11h-7z",
  infer: "M2 13.5V7a2 2 0 012-2h4l2-3h4l2 3h4a2 2 0 012 2v9a2 2 0 01-2 2H4a2 2 0 01-2-2v-1.5",
  dataset: "M4 7h16M4 12h16M4 17h10",
  models:
    "M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z",
  settings:
    "M12 15a3 3 0 100-6 3 3 0 000 6zm0 0v2m0-8V7M4.22 4.22l1.42 1.42m12.72 12.72l1.42 1.42M2 12h2m16 0h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42",
  play: "M5 3l14 9-14 9V3z",
  stop: "M18 18H6V6h12z",
  pause: "M10 4H6v16h4V4zm8 0h-4v16h4V4z",
  upload: "M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12",
  folder: "M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z",
  check: "M20 6L9 17l-5-5",
  trash: "M3 6h18M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6M9 6V4h6v2",
};

export function LineChart({ data, color, label, w = 200, h = 60 }) {
  if (!data || data.length < 2) return null;
  const min = Math.min(...data),
    max = Math.max(...data);
  const range = max - min || 1;
  const pts = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * w;
      const y = h - ((v - min) / range) * (h - 8) - 4;
      return `${x},${y}`;
    })
    .join(" ");
  const last = data[data.length - 1];
  return (
    <svg width={w} height={h} style={{ overflow: "visible" }}>
      <defs>
        <linearGradient id={`g-${label}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.25" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polyline
        points={pts}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
      <circle cx={w} cy={h - ((last - min) / range) * (h - 8) - 4} r="3" fill={color} />
    </svg>
  );
}

export function Gauge({ value, label, color }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ color: "var(--text2)", fontSize: 11, fontFamily: "var(--mono)" }}>
          {label}
        </span>
        <span
          style={{ color, fontSize: 11, fontFamily: "var(--mono)", fontWeight: 600 }}
        >
          {value}%
        </span>
      </div>
      <div
        style={{
          height: 4,
          background: "var(--surface3)",
          borderRadius: 2,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${value}%`,
            background: `linear-gradient(90deg, ${color}99, ${color})`,
            borderRadius: 2,
            transition: "width 0.6s ease",
          }}
        />
      </div>
    </div>
  );
}

export function Terminal({ logs }) {
  const ref = useRef();
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [logs]);
  return (
    <div
      ref={ref}
      style={{
        fontFamily: "var(--mono)",
        fontSize: 11.5,
        lineHeight: 1.7,
        overflowY: "auto",
        flex: 1,
        padding: "10px 14px",
        color: "#a0a0c0",
      }}
    >
      {logs.map((l, i) => (
        <div
          key={i}
          style={{
            color:
              l.type === "err"
                ? "var(--red)"
                : l.type === "ok"
                ? "var(--green)"
                : l.type === "hi"
                ? "var(--accent2)"
                : "#9090b8",
          }}
        >
          <span style={{ color: "var(--text3)", userSelect: "none" }}>{l.time} </span>
          {l.msg}
        </div>
      ))}
      <div
        style={{
          display: "inline-block",
          width: 8,
          height: 13,
          background: "var(--accent)",
          animation: "blink 1s step-end infinite",
          verticalAlign: "middle",
          marginLeft: 2,
        }}
      />
    </div>
  );
}

export function MetricCard({ title, value, sub, color, chart }) {
  return (
    <div
      style={{
        background: "var(--surface2)",
        border: "1px solid var(--border)",
        borderRadius: 8,
        padding: "12px 14px",
        display: "flex",
        flexDirection: "column",
        gap: 6,
        minWidth: 0,
      }}
    >
      <span
        style={{
          color: "var(--text3)",
          fontSize: 11,
          textTransform: "uppercase",
          letterSpacing: "0.07em",
        }}
      >
        {title}
      </span>
      <div
        style={{
          display: "flex",
          alignItems: "flex-end",
          justifyContent: "space-between",
          gap: 8,
        }}
      >
        <div>
          <span
            style={{
              fontSize: 22,
              fontWeight: 700,
              color,
              fontFamily: "var(--mono)",
            }}
          >
            {value}
          </span>
          {sub && (
            <span style={{ fontSize: 11, color: "var(--text3)", marginLeft: 6 }}>{sub}</span>
          )}
        </div>
        {chart && <div style={{ opacity: 0.9 }}>{chart}</div>}
      </div>
    </div>
  );
}

export const SEG_COLORS_FILL = [
  "#8b5cf699",
  "#06b6d499",
  "#22c55e99",
  "#f4743b99",
  "#eab30899",
];

export function DetectionCanvas({
  imageUrl,
  results,
  task,
  segResults,
  poseKpts,
  poseBones,
}) {
  const canvasRef = useRef();
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width,
      H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const draw = (img) => {
      if (img) ctx.drawImage(img, 0, 0, W, H);
      else {
        ctx.fillStyle = "#1a1a28";
        ctx.fillRect(0, 0, W, H);
        for (let i = 0; i < W + H; i += 12) {
          ctx.strokeStyle = "#21212f";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(i, 0);
          ctx.lineTo(0, i);
          ctx.stroke();
        }
        ctx.fillStyle = "#5a5a7a";
        ctx.font = "13px monospace";
        ctx.textAlign = "center";
        ctx.fillText("upload image to preview", W / 2, H / 2);
        return;
      }
      if (task === "detect") {
        results.forEach((r) => {
          const x = r.x * W,
            y = r.y * H,
            w = r.w * W,
            h = r.h * H;
          ctx.strokeStyle = r.color;
          ctx.lineWidth = 2.5;
          ctx.strokeRect(x, y, w, h);
          const txt = r.label + " " + (r.conf * 100).toFixed(0) + "%";
          const tw = ctx.measureText(txt).width;
          ctx.fillStyle = r.color + "dd";
          ctx.fillRect(x, y - 20, tw + 12, 20);
          ctx.fillStyle = "white";
          ctx.font = "bold 11px Inter,sans-serif";
          ctx.textAlign = "left";
          ctx.fillText(txt, x + 6, y - 5);
        });
      } else if (task === "segment") {
        const shapes = [
          [[0.08, 0.1], [0.3, 0.1], [0.3, 0.85], [0.08, 0.85]],
          [[0.0, 0.55], [1.0, 0.55], [1.0, 1.0], [0.0, 1.0]],
          [[0.6, 0.0], [1.0, 0.0], [1.0, 0.5], [0.6, 0.3]],
          [[0.38, 0.42], [0.73, 0.42], [0.73, 0.8], [0.38, 0.8]],
          [[0.0, 0.0], [1.0, 0.0], [1.0, 0.55], [0.0, 0.55]],
        ];
        segResults.forEach((s, i) => {
          const poly = shapes[i];
          ctx.beginPath();
          poly.forEach(([px, py], j) =>
            j === 0 ? ctx.moveTo(px * W, py * H) : ctx.lineTo(px * W, py * H),
          );
          ctx.closePath();
          ctx.fillStyle = SEG_COLORS_FILL[i];
          ctx.fill();
        });
      } else if (task === "pose") {
        poseBones.forEach(([a, b]) => {
          const p1 = poseKpts[a],
            p2 = poseKpts[b];
          ctx.strokeStyle = "#06b6d4cc";
          ctx.lineWidth = 2.5;
          ctx.beginPath();
          ctx.moveTo((p1.x / 100) * W, (p1.y / 100) * H);
          ctx.lineTo((p2.x / 100) * W, (p2.y / 100) * H);
          ctx.stroke();
        });
        poseKpts.forEach((p) => {
          ctx.beginPath();
          ctx.arc((p.x / 100) * W, (p.y / 100) * H, p.r + 1, 0, Math.PI * 2);
          ctx.fillStyle = "#8b5cf6";
          ctx.fill();
          ctx.strokeStyle = "white";
          ctx.lineWidth = 1.5;
          ctx.stroke();
        });
      }
    };
    if (!imageUrl) {
      draw(null);
      return;
    }
    const img = new Image();
    img.onload = () => draw(img);
    img.src = imageUrl;
  }, [imageUrl, results, task, segResults, poseKpts, poseBones]);
  return (
    <canvas
      ref={canvasRef}
      width={640}
      height={400}
      style={{
        width: "100%",
        height: "100%",
        objectFit: "contain",
        borderRadius: 8,
        display: "block",
      }}
    />
  );
}
