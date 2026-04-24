import React, { useState } from "react";

const Toggle = ({ val, set }) => (
  <div
    onClick={() => set(!val)}
    style={{
      width: 36,
      height: 20,
      borderRadius: 10,
      background: val ? "var(--accent)" : "var(--surface3)",
      border: `1px solid ${val ? "var(--accent)" : "var(--border)"}`,
      cursor: "pointer",
      position: "relative",
      transition: "background 0.2s",
      flexShrink: 0,
    }}
  >
    <div
      style={{
        position: "absolute",
        top: 2,
        left: val ? 17 : 2,
        width: 14,
        height: 14,
        borderRadius: "50%",
        background: "white",
        transition: "left 0.2s",
      }}
    />
  </div>
);

const Row = ({ label, desc, children }) => (
  <div
    style={{
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      gap: 16,
      padding: "12px 0",
      borderBottom: "1px solid var(--border)",
    }}
  >
    <div>
      <div style={{ color: "var(--text)", fontSize: 13 }}>{label}</div>
      {desc && <div style={{ color: "var(--text3)", fontSize: 11, marginTop: 2 }}>{desc}</div>}
    </div>
    {children}
  </div>
);

const TextInput = ({ val, set }) => (
  <input
    value={val}
    onChange={(e) => set(e.target.value)}
    style={{
      background: "var(--surface3)",
      border: "1px solid var(--border)",
      borderRadius: 5,
      padding: "5px 10px",
      color: "var(--text)",
      fontFamily: "var(--mono)",
      fontSize: 12,
      outline: "none",
      width: 160,
    }}
  />
);

export default function SettingsTab() {
  const [ckptInterval, setCkptInterval] = useState("5");
  const [outputDir, setOutputDir] = useState("./runs/exp1");
  const [mixed, setMixed] = useState(true);
  const [workers, setWorkers] = useState("4");
  const [earlyStop, setEarlyStop] = useState(true);
  const [patience, setPatience] = useState("5");

  return (
    <div style={{ padding: 16, overflowY: "auto", height: "100%" }}>
      <div style={{ maxWidth: 600, display: "flex", flexDirection: "column", gap: 0 }}>
        <div
          style={{
            color: "var(--text2)",
            fontSize: 12,
            textTransform: "uppercase",
            letterSpacing: "0.07em",
            marginBottom: 12,
          }}
        >
          Training
        </div>
        <div
          style={{
            background: "var(--surface2)",
            border: "1px solid var(--border)",
            borderRadius: 8,
            padding: "0 16px",
          }}
        >
          <Row label="Output Directory" desc="Where runs, logs and checkpoints are saved">
            <TextInput val={outputDir} set={setOutputDir} />
          </Row>
          <Row label="Checkpoint Interval" desc="Save checkpoint every N epochs">
            <TextInput val={ckptInterval} set={setCkptInterval} />
          </Row>
          <Row
            label="Mixed Precision (FP16)"
            desc="Use AMP for faster training on compatible GPUs"
          >
            <Toggle val={mixed} set={setMixed} />
          </Row>
          <Row label="DataLoader Workers" desc="Number of parallel data loading processes">
            <TextInput val={workers} set={setWorkers} />
          </Row>
          <Row label="Early Stopping" desc="Stop training if validation loss doesn't improve">
            <Toggle val={earlyStop} set={setEarlyStop} />
          </Row>
          <Row label="Patience" desc="Epochs to wait before early stop triggers">
            <TextInput val={patience} set={setPatience} />
          </Row>
        </div>

        <div
          style={{
            color: "var(--text2)",
            fontSize: 12,
            textTransform: "uppercase",
            letterSpacing: "0.07em",
            marginTop: 20,
            marginBottom: 12,
          }}
        >
          Interface
        </div>
        <div
          style={{
            background: "var(--surface2)",
            border: "1px solid var(--border)",
            borderRadius: 8,
            padding: "0 16px",
          }}
        >
          <Row label="Log Refresh Rate" desc="Terminal update interval (ms)">
            <select
              style={{
                background: "var(--surface3)",
                border: "1px solid var(--border)",
                borderRadius: 5,
                padding: "5px 10px",
                color: "var(--text)",
                fontFamily: "var(--mono)",
                fontSize: 12,
                outline: "none",
              }}
            >
              <option>100ms</option>
              <option>250ms</option>
              <option>500ms</option>
            </select>
          </Row>
          <Row label="Chart History" desc="How many epochs to display in charts">
            <select
              style={{
                background: "var(--surface3)",
                border: "1px solid var(--border)",
                borderRadius: 5,
                padding: "5px 10px",
                color: "var(--text)",
                fontFamily: "var(--mono)",
                fontSize: 12,
                outline: "none",
              }}
            >
              <option>All</option>
              <option>Last 20</option>
              <option>Last 50</option>
            </select>
          </Row>
        </div>

        <button
          style={{
            marginTop: 16,
            padding: "9px 0",
            borderRadius: 7,
            border: "none",
            cursor: "pointer",
            background: "var(--accent-dim)",
            color: "var(--accent)",
            fontFamily: "var(--font)",
            fontSize: 13,
            fontWeight: 600,
          }}
        >
          Save Settings
        </button>
      </div>
    </div>
  );
}
