import React, { useState } from "react";
import { Icon, Icons } from "./components.jsx";
import TrainTab from "./tabs/TrainTab.jsx";
import InferenceTab from "./tabs/InferenceTab.jsx";
import DatasetTab from "./tabs/DatasetTab.jsx";
import ModelsTab from "./tabs/ModelsTab.jsx";
import SettingsTab from "./tabs/SettingsTab.jsx";

const NAV = [
  { id: "train", label: "Train", icon: Icons.train },
  { id: "infer", label: "Infer", icon: Icons.infer },
  { id: "dataset", label: "Dataset", icon: Icons.dataset },
  { id: "models", label: "Models", icon: Icons.models },
  { id: "settings", label: "Settings", icon: Icons.settings },
];

export default function App() {
  const [tab, setTab] = useState(() => localStorage.getItem("ml-tab") || "train");
  const selectTab = (t) => {
    setTab(t);
    localStorage.setItem("ml-tab", t);
  };

  return (
    <div
      style={{
        display: "flex",
        height: "100vh",
        background: "var(--bg)",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          width: 62,
          background: "var(--surface)",
          borderRight: "1px solid var(--border)",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          paddingTop: 12,
          gap: 4,
          flexShrink: 0,
        }}
      >
        <div
          style={{
            width: 36,
            height: 36,
            borderRadius: 9,
            background: "linear-gradient(135deg,#7c3aed,#06b6d4)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            marginBottom: 12,
            boxShadow: "0 0 18px rgba(139,92,246,0.4)",
          }}
        >
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="white"
            strokeWidth="2.2"
          >
            <path d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        </div>
        {NAV.map((n) => {
          const active = tab === n.id;
          return (
            <div
              key={n.id}
              onClick={() => selectTab(n.id)}
              title={n.label}
              style={{
                width: 44,
                height: 44,
                borderRadius: 10,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                cursor: "pointer",
                gap: 2,
                transition: "all 0.15s",
                background: active ? "var(--accent-dim)" : "transparent",
                border: `1px solid ${active ? "var(--accent)" : "transparent"}`,
                color: active ? "var(--accent)" : "var(--text3)",
              }}
            >
              <Icon d={n.icon} size={17} sw={active ? 2 : 1.5} />
              <span
                style={{
                  fontSize: 8,
                  fontFamily: "var(--mono)",
                  letterSpacing: "0.03em",
                }}
              >
                {n.label}
              </span>
            </div>
          );
        })}
      </div>

      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
        <div
          style={{
            height: 38,
            background: "var(--surface)",
            borderBottom: "1px solid var(--border)",
            display: "flex",
            alignItems: "center",
            padding: "0 16px",
            gap: 10,
            flexShrink: 0,
          }}
        >
          <span style={{ color: "var(--text2)", fontSize: 12, fontFamily: "var(--mono)" }}>
            simpleml
          </span>
          <span style={{ color: "var(--border2)" }}>›</span>
          <span style={{ color: "var(--accent)", fontSize: 12, fontFamily: "var(--mono)" }}>
            {tab}
          </span>
          <div style={{ flex: 1 }} />
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <div
              style={{
                width: 6,
                height: 6,
                borderRadius: "50%",
                background: "var(--green)",
              }}
            />
            <span
              style={{ color: "var(--text3)", fontSize: 11, fontFamily: "var(--mono)" }}
            >
              RTX 3090 · CUDA 11.8
            </span>
          </div>
        </div>
        <div style={{ flex: 1, overflow: "hidden" }}>
          {tab === "train" && <TrainTab />}
          {tab === "infer" && <InferenceTab />}
          {tab === "dataset" && <DatasetTab />}
          {tab === "models" && <ModelsTab />}
          {tab === "settings" && <SettingsTab />}
        </div>
      </div>
    </div>
  );
}
