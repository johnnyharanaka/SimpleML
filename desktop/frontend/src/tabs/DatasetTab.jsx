import React, { useState } from "react";

const DATASET_FILES = [
  { name: "imagenet_subset/", type: "folder", size: "12,800 imgs" },
  { name: "train/", type: "folder", size: "10,240 imgs" },
  { name: "val/", type: "folder", size: "1,280 imgs" },
  { name: "test/", type: "folder", size: "1,280 imgs" },
  { name: "labels.json", type: "file", size: "48 KB" },
  { name: "classes.txt", type: "file", size: "14 KB" },
  { name: "meta.yaml", type: "file", size: "1.2 KB" },
];

const CLASSES_PREVIEW = [
  "golden_retriever",
  "Labrador_retriever",
  "tabby_cat",
  "macaw",
  "sports_car",
  "mountain_bike",
  "espresso",
  "pizza",
  "lighthouse",
  "space_shuttle",
];

export default function DatasetTab() {
  const [selected, setSelected] = useState(null);
  return (
    <div style={{ display: "flex", height: "100%", gap: 14, padding: 16 }}>
      <div style={{ flex: "0 0 320px", display: "flex", flexDirection: "column", gap: 10 }}>
        <div
          style={{
            background: "var(--surface2)",
            border: "1px solid var(--border)",
            borderRadius: 8,
            padding: "10px 14px",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: 10,
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
              Dataset Files
            </span>
            <button
              style={{
                background: "var(--accent-dim)",
                color: "var(--accent)",
                border: "none",
                borderRadius: 5,
                padding: "4px 10px",
                fontSize: 11,
                cursor: "pointer",
              }}
            >
              + Load
            </button>
          </div>
          {DATASET_FILES.map((f, i) => (
            <div
              key={i}
              onClick={() => setSelected(f)}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 10,
                padding: "8px 10px",
                borderRadius: 6,
                cursor: "pointer",
                marginBottom: 2,
                background: selected?.name === f.name ? "var(--accent-dim)" : "transparent",
                border: `1px solid ${selected?.name === f.name ? "var(--accent)" : "transparent"}`,
              }}
            >
              <svg
                width="15"
                height="15"
                viewBox="0 0 24 24"
                fill="none"
                stroke={f.type === "folder" ? "#eab308" : "var(--text3)"}
                strokeWidth="1.5"
              >
                {f.type === "folder" ? (
                  <path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z" />
                ) : (
                  <>
                    <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
                    <polyline points="14 2 14 8 20 8" />
                  </>
                )}
              </svg>
              <span
                style={{
                  color: "var(--text)",
                  flex: 1,
                  fontFamily: "var(--mono)",
                  fontSize: 12,
                }}
              >
                {f.name}
              </span>
              <span style={{ color: "var(--text3)", fontSize: 11 }}>{f.size}</span>
            </div>
          ))}
        </div>
        <div
          style={{
            background: "var(--surface2)",
            border: "1px solid var(--border)",
            borderRadius: 8,
            padding: "12px 14px",
            display: "flex",
            flexDirection: "column",
            gap: 8,
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
            Split Summary
          </span>
          {[
            ["Train", "10,240", "80%", "#8b5cf6"],
            ["Val", "1,280", "10%", "#06b6d4"],
            ["Test", "1,280", "10%", "#22c55e"],
          ].map(([s, c, p, col]) => (
            <div key={s} style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span
                style={{
                  color: col,
                  fontSize: 11,
                  fontFamily: "var(--mono)",
                  width: 40,
                }}
              >
                {s}
              </span>
              <div
                style={{
                  flex: 1,
                  height: 4,
                  background: "var(--surface3)",
                  borderRadius: 2,
                  overflow: "hidden",
                }}
              >
                <div
                  style={{ width: p, height: "100%", background: col, borderRadius: 2 }}
                />
              </div>
              <span
                style={{
                  color: "var(--text3)",
                  fontSize: 11,
                  width: 60,
                  textAlign: "right",
                }}
              >
                {c} imgs
              </span>
            </div>
          ))}
        </div>
      </div>
      <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 10, overflowY: "auto" }}>
        <div
          style={{
            background: "var(--surface2)",
            border: "1px solid var(--border)",
            borderRadius: 8,
            padding: "12px 14px",
          }}
        >
          <span
            style={{
              color: "var(--text3)",
              fontSize: 11,
              textTransform: "uppercase",
              letterSpacing: "0.07em",
              display: "block",
              marginBottom: 10,
            }}
          >
            Class Labels (1,000 total)
          </span>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
            {CLASSES_PREVIEW.map((c, i) => (
              <div
                key={i}
                style={{
                  background: "var(--surface3)",
                  border: "1px solid var(--border)",
                  borderRadius: 5,
                  padding: "3px 10px",
                  fontFamily: "var(--mono)",
                  fontSize: 11,
                  color: "var(--text2)",
                }}
              >
                {c}
              </div>
            ))}
            <div
              style={{
                background: "var(--surface3)",
                border: "1px solid var(--border)",
                borderRadius: 5,
                padding: "3px 10px",
                fontSize: 11,
                color: "var(--text3)",
              }}
            >
              +990 more...
            </div>
          </div>
        </div>
        <div
          style={{
            background: "var(--surface2)",
            border: "1px solid var(--border)",
            borderRadius: 8,
            padding: "12px 14px",
          }}
        >
          <span
            style={{
              color: "var(--text3)",
              fontSize: 11,
              textTransform: "uppercase",
              letterSpacing: "0.07em",
              display: "block",
              marginBottom: 10,
            }}
          >
            Image Grid Preview
          </span>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(8, 1fr)", gap: 4 }}>
            {Array.from({ length: 32 }).map((_, i) => (
              <div
                key={i}
                style={{
                  aspectRatio: "1",
                  background: `repeating-linear-gradient(${i * 30}deg,var(--surface3) 0,var(--surface3) 2px,var(--surface2) 2px,var(--surface2) 8px)`,
                  borderRadius: 4,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <span style={{ fontSize: 7, color: "var(--text3)" }}>img</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
