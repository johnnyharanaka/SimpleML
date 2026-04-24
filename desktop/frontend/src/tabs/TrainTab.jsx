import React, { useCallback, useEffect, useRef, useState } from "react";
import { Gauge, LineChart, MetricCard, Terminal } from "../components.jsx";

const LOG_INIT = [
  { time: "00:00:01", type: "hi", msg: "Initializing training session..." },
  { time: "00:00:02", type: "", msg: "Loading dataset: imagenet_subset (12,800 samples)" },
  { time: "00:00:03", type: "", msg: "Model: ResNet-50  |  Parameters: 25.6M" },
  { time: "00:00:04", type: "ok", msg: "CUDA device found: NVIDIA RTX 3090 (24GB)" },
  { time: "00:00:05", type: "", msg: "Optimizer: AdamW  lr=1e-3  wd=1e-4" },
];

const API_URL = "http://127.0.0.1:8765";

const COMMON_PARAMS = [
  { key: "lr", label: "Learning Rate", default: "1e-3" },
  { key: "bs", label: "Batch Size", default: "32" },
  { key: "epochs", label: "Epochs", default: "20" },
  { key: "weight_decay", label: "Weight Decay", default: "1e-4" },
];

const REGISTRY_SELECTS = [
  { key: "models", label: "Architecture", stateKey: "arch" },
  { key: "optimizers", label: "Optimizer", stateKey: "optimizer" },
  { key: "losses", label: "Loss", stateKey: "loss" },
  { key: "schedulers", label: "Scheduler", stateKey: "scheduler" },
];

function DynamicConfig({ arch, setArch, lr, setLr, bs, setBs, epMax, setEpMax, running }) {
  const [registries, setRegistries] = useState({});
  const [loadError, setLoadError] = useState(null);
  const [optimizer, setOptimizer] = useState("");
  const [loss, setLoss] = useState("");
  const [scheduler, setScheduler] = useState("");
  const [extras, setExtras] = useState({});

  const setters = { arch: setArch, optimizer: setOptimizer, loss: setLoss, scheduler: setScheduler };
  const values = { arch, optimizer, loss, scheduler };

  useEffect(() => {
    fetch(`${API_URL}/registries`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data) => {
        setRegistries(data);
        for (const { key, stateKey } of REGISTRY_SELECTS) {
          const items = data[key] || [];
          if (items.length > 0 && !items.includes(values[stateKey])) {
            setters[stateKey](items[0]);
          }
        }
      })
      .catch((e) => setLoadError(e.message));
  }, []);

  const handleChange = (key, val) => {
    if (key === "lr") setLr(val);
    else if (key === "bs") setBs(val);
    else if (key === "epochs") setEpMax(val);
    else setExtras((e) => ({ ...e, [key]: val }));
  };
  const getValue = (key) => {
    if (key === "lr") return lr;
    if (key === "bs") return bs;
    if (key === "epochs") return epMax;
    return extras[key] ?? COMMON_PARAMS.find((p) => p.key === key)?.default ?? "";
  };
  const taskColor = "var(--accent)";
  const totalRegistered = Object.values(registries).reduce((n, xs) => n + xs.length, 0);

  return (
    <div
      style={{
        background: "var(--surface2)",
        border: "1px solid var(--border)",
        borderRadius: 8,
        padding: "12px 14px",
        display: "flex",
        flexDirection: "column",
        gap: 10,
        flexShrink: 0,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <span
          style={{
            color: "var(--text3)",
            fontSize: 11,
            textTransform: "uppercase",
            letterSpacing: "0.07em",
          }}
        >
          Config
        </span>
        <span
          style={{
            background: taskColor + "22",
            color: taskColor,
            fontSize: 10,
            padding: "2px 8px",
            borderRadius: 10,
          }}
        >
          {totalRegistered} registered
        </span>
      </div>
      {REGISTRY_SELECTS.map(({ key, label, stateKey }) => {
        const items = registries[key] || [];
        return (
          <div key={key} style={{ display: "flex", flexDirection: "column", gap: 3 }}>
            <span style={{ color: "var(--text3)", fontSize: 10 }}>{label}</span>
            <select
              value={values[stateKey]}
              onChange={(e) => setters[stateKey](e.target.value)}
              disabled={running}
              style={{
                background: "var(--surface3)",
                border: "1px solid var(--border)",
                borderRadius: 4,
                padding: "5px 8px",
                color: "var(--text)",
                fontFamily: "var(--mono)",
                fontSize: 12,
                outline: "none",
                opacity: running ? 0.5 : 1,
              }}
            >
              {items.length === 0 && (
                <option value="">
                  {loadError ? `error: ${loadError}` : "loading..."}
                </option>
              )}
              {items.map((a) => (
                <option key={a}>{a}</option>
              ))}
            </select>
          </div>
        );
      })}
      {COMMON_PARAMS.map((p) => (
        <div key={p.key} style={{ display: "flex", flexDirection: "column", gap: 3 }}>
          <span style={{ color: "var(--text3)", fontSize: 10 }}>{p.label}</span>
          <input
            value={getValue(p.key)}
            onChange={(e) => handleChange(p.key, e.target.value)}
            disabled={running}
            style={{
              background: "var(--surface3)",
              border: "1px solid var(--border)",
              borderRadius: 4,
              padding: "4px 8px",
              color: "var(--text)",
              fontFamily: "var(--mono)",
              fontSize: 12,
              outline: "none",
              opacity: running ? 0.5 : 1,
            }}
          />
        </div>
      ))}
    </div>
  );
}

export default function TrainTab() {
  const [running, setRunning] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [step, setStep] = useState(0);
  const totalEpochs = 20;
  const stepsPerEpoch = 100;
  const [lossHistory, setLossHistory] = useState([2.31]);
  const [accHistory, setAccHistory] = useState([0.12]);
  const [valLossHistory, setValLossHistory] = useState([2.45]);
  const [logs, setLogs] = useState(LOG_INIT);
  const [gpu, setGpu] = useState(34);
  const [ram, setRam] = useState(42);
  const [vram, setVram] = useState(28);
  const [lr, setLr] = useState("1e-3");
  const [bs, setBs] = useState("32");
  const [epMax, setEpMax] = useState("20");
  const [arch, setArch] = useState("");
  const [checkpoints, setCheckpoints] = useState([
    { name: "checkpoint_e01.pt", epoch: 1, loss: "2.10", acc: "0.31", ts: "14:22" },
    { name: "checkpoint_e05.pt", epoch: 5, loss: "1.45", acc: "0.58", ts: "14:38" },
  ]);

  const timerRef = useRef();
  const now = () => {
    const d = new Date();
    return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}:${String(d.getSeconds()).padStart(2, "0")}`;
  };

  const addLog = useCallback((msg, type = "") => {
    setLogs((l) => [...l.slice(-200), { time: now(), type, msg }]);
  }, []);

  useEffect(() => {
    if (!running) {
      clearInterval(timerRef.current);
      return;
    }
    timerRef.current = setInterval(() => {
      setStep((s) => {
        const ns = s + 1;
        if (ns > stepsPerEpoch) {
          setEpoch((e) => {
            const ne = e + 1;
            const loss = Math.max(
              0.05,
              2.31 * Math.exp(-ne * 0.18) + (Math.random() - 0.5) * 0.05,
            );
            const acc = Math.min(
              0.99,
              0.12 + ne * 0.043 + (Math.random() - 0.5) * 0.02,
            );
            const vloss = loss + 0.1 + (Math.random() - 0.5) * 0.08;
            setLossHistory((h) => [...h, +loss.toFixed(4)]);
            setAccHistory((h) => [...h, +acc.toFixed(4)]);
            setValLossHistory((h) => [...h, +vloss.toFixed(4)]);
            addLog(
              `Epoch ${ne}/${totalEpochs} — loss: ${loss.toFixed(4)}  acc: ${(acc * 100).toFixed(1)}%  val_loss: ${vloss.toFixed(4)}`,
              ne % 5 === 0 ? "ok" : "",
            );
            if (ne % 5 === 0) {
              const ckpt = `checkpoint_e${String(ne).padStart(2, "0")}.pt`;
              setCheckpoints((c) => [
                ...c,
                { name: ckpt, epoch: ne, loss: loss.toFixed(2), acc: acc.toFixed(2), ts: now() },
              ]);
              addLog(`  ↳ Saved ${ckpt}`, "hi");
            }
            if (ne >= totalEpochs) {
              setRunning(false);
              addLog("Training complete!", "ok");
            }
            return ne;
          });
          return 0;
        }
        if (ns % 20 === 0) {
          setEpoch((e) => {
            addLog(
              `  step ${ns}/${stepsPerEpoch}  batch_loss: ${(1.8 - e * 0.08 + Math.random() * 0.2).toFixed(4)}`,
            );
            return e;
          });
        }
        setGpu(Math.round(65 + Math.random() * 20));
        setVram(Math.round(55 + Math.random() * 15));
        setRam(Math.round(40 + Math.random() * 10));
        return ns;
      });
    }, 120);
    return () => clearInterval(timerRef.current);
  }, [running, addLog]);

  const curLoss = lossHistory[lossHistory.length - 1];
  const curAcc = accHistory[accHistory.length - 1];
  const epochPct = epoch > 0 ? (epoch / totalEpochs) * 100 : 0;
  const stepPct = (step / stepsPerEpoch) * 100;

  const cancelTraining = () => {
    setRunning(false);
    setEpoch(0);
    setStep(0);
    setLossHistory([2.31]);
    setAccHistory([0.12]);
    setValLossHistory([2.45]);
    setCheckpoints([
      { name: "checkpoint_e01.pt", epoch: 1, loss: "2.10", acc: "0.31", ts: "14:22" },
      { name: "checkpoint_e05.pt", epoch: 5, loss: "1.45", acc: "0.58", ts: "14:38" },
    ]);
    addLog("Training cancelled. State reset.", "err");
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        gap: 12,
        padding: 16,
        overflow: "hidden",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          background: "var(--surface2)",
          border: "1px solid var(--border)",
          borderRadius: 8,
          padding: "10px 14px",
          flexShrink: 0,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8, flex: 1 }}>
          <div
            style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              background: running ? "var(--green)" : epoch > 0 ? "var(--yellow)" : "var(--text3)",
              boxShadow: running ? "0 0 6px var(--green)" : "",
            }}
          />
          <span style={{ color: "var(--text2)", fontSize: 12, fontFamily: "var(--mono)" }}>
            {running
              ? `Training · epoch ${epoch}/${totalEpochs} · step ${step}/${stepsPerEpoch}`
              : epoch === 0
              ? "Ready to train"
              : `Paused · epoch ${epoch}/${totalEpochs}`}
          </span>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {epoch > 0 && !running && (
            <button
              onClick={cancelTraining}
              style={{
                padding: "7px 18px",
                borderRadius: 6,
                border: "1px solid rgba(244,63,94,0.3)",
                cursor: "pointer",
                fontFamily: "var(--font)",
                fontSize: 12,
                fontWeight: 600,
                background: "rgba(244,63,94,0.1)",
                color: "var(--red)",
                transition: "all 0.2s",
              }}
            >
              ✕ Cancel
            </button>
          )}
          <button
            onClick={() => {
              if (!running) {
                setRunning(true);
                addLog(epoch === 0 ? "Starting training..." : "Resuming training...", "hi");
              } else {
                setRunning(false);
                addLog("Training paused.", "");
              }
            }}
            style={{
              padding: "7px 24px",
              borderRadius: 6,
              border: "none",
              cursor: "pointer",
              fontFamily: "var(--font)",
              fontSize: 12,
              fontWeight: 600,
              transition: "all 0.2s",
              background: running
                ? "rgba(244,63,94,0.15)"
                : "linear-gradient(135deg,#7c3aed,#06b6d4)",
              color: running ? "var(--red)" : "white",
              boxShadow: running ? "" : "0 0 16px rgba(139,92,246,0.35)",
            }}
          >
            {running ? "⏸ Pause" : epoch === 0 ? "▶ Start Training" : "▶ Resume"}
          </button>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 10 }}>
        <MetricCard
          title="Train Loss"
          value={curLoss?.toFixed(4)}
          color="var(--accent)"
          chart={<LineChart data={lossHistory} color="#8b5cf6" label="loss" w={90} h={36} />}
        />
        <MetricCard
          title="Accuracy"
          value={`${(curAcc * 100).toFixed(1)}%`}
          color="var(--accent2)"
          chart={<LineChart data={accHistory} color="#06b6d4" label="acc" w={90} h={36} />}
        />
        <MetricCard
          title="Val Loss"
          value={valLossHistory[valLossHistory.length - 1]?.toFixed(4)}
          color="#a78bfa"
          chart={<LineChart data={valLossHistory} color="#a78bfa" label="vloss" w={90} h={36} />}
        />
        <MetricCard
          title="Epoch"
          value={`${epoch}/${totalEpochs}`}
          color="var(--text)"
          sub={`step ${step}`}
        />
      </div>

      <div
        style={{
          background: "var(--surface2)",
          border: "1px solid var(--border)",
          borderRadius: 8,
          padding: "10px 14px",
          display: "flex",
          flexDirection: "column",
          gap: 8,
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <span style={{ color: "var(--text2)", fontSize: 11, fontFamily: "var(--mono)" }}>
            EPOCH PROGRESS
          </span>
          <span style={{ color: "var(--accent)", fontSize: 11, fontFamily: "var(--mono)" }}>
            {epochPct.toFixed(0)}%
          </span>
        </div>
        <div
          style={{
            height: 6,
            background: "var(--surface3)",
            borderRadius: 3,
            overflow: "hidden",
          }}
        >
          <div
            style={{
              height: "100%",
              width: `${epochPct}%`,
              background: "linear-gradient(90deg,#7c3aed,#8b5cf6)",
              borderRadius: 3,
              transition: "width 0.5s ease",
              boxShadow: "0 0 8px #8b5cf688",
            }}
          />
        </div>
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <span style={{ color: "var(--text3)", fontSize: 11, fontFamily: "var(--mono)" }}>
            STEP PROGRESS
          </span>
          <span style={{ color: "var(--accent2)", fontSize: 11, fontFamily: "var(--mono)" }}>
            {stepPct.toFixed(0)}%
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
              width: `${stepPct}%`,
              background: "linear-gradient(90deg,#0891b2,#06b6d4)",
              borderRadius: 2,
              transition: "width 0.1s linear",
            }}
          />
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 270px",
          gap: 10,
          minHeight: 0,
          flex: 1,
        }}
      >
        <div
          style={{
            background: "var(--surface)",
            border: "1px solid var(--border)",
            borderRadius: 8,
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
          }}
        >
          <div
            style={{
              padding: "8px 14px",
              borderBottom: "1px solid var(--border)",
              display: "flex",
              alignItems: "center",
              gap: 8,
            }}
          >
            <div
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: running ? "var(--green)" : "var(--text3)",
              }}
            />
            <span
              style={{
                color: "var(--text3)",
                fontSize: 11,
                fontFamily: "var(--mono)",
                letterSpacing: "0.05em",
              }}
            >
              TRAINING LOG
            </span>
            <div style={{ flex: 1 }} />
            <span style={{ color: "var(--text3)", fontSize: 10 }}>{logs.length} lines</span>
          </div>
          <Terminal logs={logs} />
        </div>

        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 10,
            overflowY: "auto",
            paddingRight: 2,
          }}
        >
          <div
            style={{
              background: "var(--surface2)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              padding: "12px 14px",
              display: "flex",
              flexDirection: "column",
              gap: 10,
              flexShrink: 0,
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
              System Resources
            </span>
            <Gauge value={gpu} label="GPU" color="#8b5cf6" />
            <Gauge value={vram} label="VRAM" color="#a78bfa" />
            <Gauge value={ram} label="RAM" color="#06b6d4" />
          </div>

          <DynamicConfig
            arch={arch}
            setArch={setArch}
            lr={lr}
            setLr={setLr}
            bs={bs}
            setBs={setBs}
            epMax={epMax}
            setEpMax={setEpMax}
            running={running}
          />

          <div
            style={{
              background: "var(--surface2)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              padding: "12px 14px",
              flexShrink: 0,
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
              <span
                style={{
                  color: "var(--text3)",
                  fontSize: 11,
                  textTransform: "uppercase",
                  letterSpacing: "0.07em",
                }}
              >
                Checkpoints
              </span>
              <span
                style={{
                  background: "var(--accent-dim)",
                  color: "var(--accent)",
                  fontSize: 10,
                  padding: "1px 6px",
                  borderRadius: 10,
                  fontFamily: "var(--mono)",
                }}
              >
                {checkpoints.length}
              </span>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {checkpoints
                .slice()
                .reverse()
                .slice(0, 5)
                .map((c, i) => (
                  <div
                    key={i}
                    style={{
                      background: "var(--surface3)",
                      border: "1px solid var(--border)",
                      borderRadius: 6,
                      padding: "7px 10px",
                      fontSize: 11,
                      display: "flex",
                      flexDirection: "column",
                      gap: 2,
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                      }}
                    >
                      <span
                        style={{
                          color: "var(--accent2)",
                          fontFamily: "var(--mono)",
                          fontSize: 11,
                        }}
                      >
                        {c.name}
                      </span>
                      <span style={{ color: "var(--text3)", fontSize: 10 }}>{c.ts}</span>
                    </div>
                    <span style={{ color: "var(--text3)", fontSize: 10 }}>
                      ep {c.epoch} · loss {c.loss} · acc {c.acc}
                    </span>
                  </div>
                ))}
              {checkpoints.length === 0 && (
                <span style={{ color: "var(--text3)", fontSize: 11 }}>No checkpoints yet</span>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
