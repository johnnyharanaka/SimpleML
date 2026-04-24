import React, { useState } from "react";

const MODEL_LIST = [
  {
    name: "best_model.pt",
    arch: "ResNet-50",
    epoch: 12,
    loss: "0.82",
    acc: "81.4%",
    size: "98 MB",
    ts: "2024-04-22 14:44",
    best: true,
  },
  {
    name: "checkpoint_e10.pt",
    arch: "ResNet-50",
    epoch: 10,
    loss: "0.91",
    acc: "78.2%",
    size: "98 MB",
    ts: "2024-04-22 14:38",
    best: false,
  },
  {
    name: "checkpoint_e05.pt",
    arch: "ResNet-50",
    epoch: 5,
    loss: "1.45",
    acc: "58.0%",
    size: "98 MB",
    ts: "2024-04-22 14:22",
    best: false,
  },
  {
    name: "checkpoint_e01.pt",
    arch: "ResNet-50",
    epoch: 1,
    loss: "2.10",
    acc: "31.0%",
    size: "98 MB",
    ts: "2024-04-22 14:10",
    best: false,
  },
];

export default function ModelsTab() {
  const [sel, setSel] = useState(0);
  const m = MODEL_LIST[sel];
  return (
    <div style={{ display: "flex", height: "100%", gap: 14, padding: 16 }}>
      <div style={{ flex: "0 0 340px", display: "flex", flexDirection: "column", gap: 8 }}>
        {MODEL_LIST.map((m, i) => (
          <div
            key={i}
            onClick={() => setSel(i)}
            style={{
              background: "var(--surface2)",
              border: `1px solid ${sel === i ? "var(--accent)" : "var(--border)"}`,
              borderRadius: 8,
              padding: "12px 14px",
              cursor: "pointer",
              transition: "all 0.15s",
              position: "relative",
              overflow: "hidden",
            }}
          >
            {m.best && (
              <div
                style={{
                  position: "absolute",
                  top: 8,
                  right: 8,
                  background: "var(--accent-dim)",
                  color: "var(--accent)",
                  fontSize: 10,
                  padding: "1px 7px",
                  borderRadius: 10,
                  fontFamily: "var(--mono)",
                }}
              >
                best
              </div>
            )}
            <div
              style={{
                color: "var(--text)",
                fontFamily: "var(--mono)",
                fontSize: 13,
                fontWeight: 500,
              }}
            >
              {m.name}
            </div>
            <div style={{ color: "var(--text3)", fontSize: 11, marginTop: 3 }}>
              {m.arch} · epoch {m.epoch} · {m.size}
            </div>
            <div style={{ display: "flex", gap: 12, marginTop: 6 }}>
              <span
                style={{ color: "var(--accent)", fontFamily: "var(--mono)", fontSize: 12 }}
              >
                loss {m.loss}
              </span>
              <span
                style={{ color: "var(--accent2)", fontFamily: "var(--mono)", fontSize: 12 }}
              >
                acc {m.acc}
              </span>
            </div>
          </div>
        ))}
      </div>
      <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 10 }}>
        <div
          style={{
            background: "var(--surface2)",
            border: "1px solid var(--border)",
            borderRadius: 10,
            padding: "18px 20px",
          }}
        >
          <div
            style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}
          >
            <div>
              <div
                style={{
                  color: "var(--text)",
                  fontSize: 16,
                  fontWeight: 600,
                  fontFamily: "var(--mono)",
                }}
              >
                {m.name}
              </div>
              <div style={{ color: "var(--text3)", fontSize: 12, marginTop: 3 }}>{m.ts}</div>
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button
                style={{
                  background: "var(--accent-dim)",
                  color: "var(--accent)",
                  border: "none",
                  borderRadius: 6,
                  padding: "6px 14px",
                  fontSize: 12,
                  cursor: "pointer",
                }}
              >
                Export
              </button>
              <button
                style={{
                  background: "rgba(244,63,94,0.1)",
                  color: "var(--red)",
                  border: "none",
                  borderRadius: 6,
                  padding: "6px 14px",
                  fontSize: 12,
                  cursor: "pointer",
                }}
              >
                Delete
              </button>
            </div>
          </div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(4,1fr)",
              gap: 10,
              marginTop: 16,
            }}
          >
            {[
              ["Architecture", m.arch, "var(--text)"],
              ["Parameters", "25.6M", "var(--accent)"],
              ["Val Loss", m.loss, "var(--accent)"],
              ["Val Acc", m.acc, "var(--accent2)"],
            ].map(([k, v, c]) => (
              <div
                key={k}
                style={{
                  background: "var(--surface3)",
                  borderRadius: 7,
                  padding: "10px 12px",
                }}
              >
                <div style={{ color: "var(--text3)", fontSize: 11 }}>{k}</div>
                <div
                  style={{
                    color: c,
                    fontFamily: "var(--mono)",
                    fontSize: 15,
                    fontWeight: 600,
                    marginTop: 4,
                  }}
                >
                  {v}
                </div>
              </div>
            ))}
          </div>
        </div>
        <div
          style={{
            background: "var(--surface2)",
            border: "1px solid var(--border)",
            borderRadius: 8,
            padding: "14px 18px",
            flex: 1,
          }}
        >
          <span
            style={{
              color: "var(--text3)",
              fontSize: 11,
              textTransform: "uppercase",
              letterSpacing: "0.07em",
              display: "block",
              marginBottom: 12,
            }}
          >
            Training Curve
          </span>
          <svg width="100%" height="160" viewBox="0 0 600 160" preserveAspectRatio="none">
            <defs>
              <linearGradient id="loss-grad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.3" />
                <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0" />
              </linearGradient>
            </defs>
            <polyline
              points={Array.from(
                { length: 13 },
                (_, i) => `${i * 50},${10 + (130 * (2.31 * Math.exp(-i * 0.22))) / 2.31}`,
              ).join(" ")}
              fill="none"
              stroke="#8b5cf6"
              strokeWidth="2"
              strokeLinejoin="round"
            />
            <polyline
              points={Array.from(
                { length: 13 },
                (_, i) =>
                  `${i * 50},${10 + (130 * (2.45 * Math.exp(-i * 0.19) + 0.05)) / 2.45}`,
              ).join(" ")}
              fill="none"
              stroke="#a78bfa"
              strokeWidth="1.5"
              strokeLinejoin="round"
              strokeDasharray="4,3"
            />
            <line
              x1={m.epoch * 50}
              y1="0"
              x2={m.epoch * 50}
              y2="160"
              stroke="var(--accent2)"
              strokeWidth="1"
              strokeDasharray="3,3"
            />
          </svg>
          <div style={{ display: "flex", gap: 16, marginTop: 4 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <div style={{ width: 16, height: 2, background: "#8b5cf6" }} />
              <span style={{ color: "var(--text3)", fontSize: 11 }}>train loss</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <div
                style={{
                  width: 16,
                  height: 2,
                  background: "#a78bfa",
                  borderTop: "2px dashed #a78bfa",
                }}
              />
              <span style={{ color: "var(--text3)", fontSize: 11 }}>val loss</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
