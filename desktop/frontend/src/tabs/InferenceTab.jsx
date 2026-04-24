import React, { useState } from "react";
import { DetectionCanvas } from "../components.jsx";

const CV_TASKS = [
  { id: "classify", label: "Classification" },
  { id: "detect", label: "Object Detection" },
  { id: "segment", label: "Segmentation" },
  { id: "pose", label: "Pose Estimation" },
];

const CLS_RESULTS = [
  { label: "golden retriever", conf: 0.923 },
  { label: "Labrador retriever", conf: 0.041 },
  { label: "cocker spaniel", conf: 0.018 },
  { label: "kuvasz", conf: 0.009 },
  { label: "clumber spaniel", conf: 0.004 },
];

const DET_RESULTS = [
  { label: "person", conf: 0.97, x: 0.08, y: 0.1, w: 0.22, h: 0.75, color: "#8b5cf6" },
  { label: "car", conf: 0.94, x: 0.38, y: 0.42, w: 0.35, h: 0.38, color: "#06b6d4" },
  { label: "bicycle", conf: 0.88, x: 0.68, y: 0.35, w: 0.22, h: 0.48, color: "#22c55e" },
  { label: "person", conf: 0.76, x: 0.55, y: 0.12, w: 0.14, h: 0.55, color: "#8b5cf6" },
  { label: "dog", conf: 0.61, x: 0.82, y: 0.52, w: 0.14, h: 0.3, color: "#eab308" },
];

const SEG_RESULTS = [
  { label: "person", area: "18.4%", color: "#8b5cf6" },
  { label: "road", area: "34.1%", color: "#06b6d4" },
  { label: "vegetation", area: "22.7%", color: "#22c55e" },
  { label: "vehicle", area: "11.2%", color: "#f4743b" },
  { label: "sky", area: "13.6%", color: "#eab308" },
];

const POSE_KEYPOINTS = [
  { x: 50, y: 12, r: 4, label: "nose" },
  { x: 50, y: 20, r: 3, label: "neck" },
  { x: 38, y: 22, r: 3, label: "l_shoulder" },
  { x: 62, y: 22, r: 3, label: "r_shoulder" },
  { x: 30, y: 36, r: 3, label: "l_elbow" },
  { x: 70, y: 36, r: 3, label: "r_elbow" },
  { x: 26, y: 50, r: 3, label: "l_wrist" },
  { x: 74, y: 50, r: 3, label: "r_wrist" },
  { x: 42, y: 46, r: 3, label: "l_hip" },
  { x: 58, y: 46, r: 3, label: "r_hip" },
  { x: 38, y: 64, r: 3, label: "l_knee" },
  { x: 62, y: 64, r: 3, label: "r_knee" },
  { x: 36, y: 80, r: 3, label: "l_ankle" },
  { x: 64, y: 80, r: 3, label: "r_ankle" },
];
const POSE_BONES = [
  [0, 1],
  [1, 2],
  [1, 3],
  [2, 4],
  [4, 6],
  [3, 5],
  [5, 7],
  [2, 8],
  [3, 9],
  [8, 9],
  [8, 10],
  [10, 12],
  [9, 11],
  [11, 13],
];

export default function InferenceTab() {
  const [task, setTask] = useState("classify");
  const [dragging, setDragging] = useState(false);
  const [imageUrl, setImageUrl] = useState(null);
  const [running, setRunning] = useState(false);
  const [isDone, setIsDone] = useState(false);
  const [topK, setTopK] = useState(5);
  const [latency, setLatency] = useState(null);
  const [model, setModel] = useState("best_model.pt");
  const [confThresh, setConfThresh] = useState(0.5);
  const [iouThresh, setIouThresh] = useState(0.45);

  const handleFile = (file) => {
    if (!file || !file.type.startsWith("image/")) return;
    setImageUrl(URL.createObjectURL(file));
    setIsDone(false);
  };

  const runInference = () => {
    setRunning(true);
    setIsDone(false);
    const t0 = performance.now();
    setTimeout(() => {
      setLatency((performance.now() - t0).toFixed(0));
      setRunning(false);
      setIsDone(true);
    }, 700 + Math.random() * 500);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  };
  const detFiltered = DET_RESULTS.filter((r) => r.conf >= confThresh);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      <div
        style={{
          display: "flex",
          gap: 6,
          padding: "10px 16px 0",
          flexShrink: 0,
          alignItems: "center",
        }}
      >
        {CV_TASKS.map((t) => (
          <button
            key={t.id}
            onClick={() => {
              setTask(t.id);
              setIsDone(false);
            }}
            style={{
              padding: "6px 16px",
              borderRadius: 6,
              border: "1px solid " + (task === t.id ? "var(--accent)" : "var(--border)"),
              background: task === t.id ? "var(--accent-dim)" : "transparent",
              color: task === t.id ? "var(--accent)" : "var(--text3)",
              cursor: "pointer",
              fontSize: 12,
              fontWeight: task === t.id ? 600 : 400,
              transition: "all 0.15s",
            }}
          >
            {t.label}
          </button>
        ))}
        <div style={{ flex: 1 }} />
        <select
          value={model}
          onChange={(e) => setModel(e.target.value)}
          style={{
            background: "var(--surface3)",
            border: "1px solid var(--border)",
            borderRadius: 6,
            padding: "5px 10px",
            color: "var(--text2)",
            fontFamily: "var(--mono)",
            fontSize: 11,
            outline: "none",
          }}
        >
          {["checkpoint_e01.pt", "checkpoint_e05.pt", "checkpoint_e10.pt", "best_model.pt"].map(
            (m) => (
              <option key={m}>{m}</option>
            ),
          )}
        </select>
      </div>

      <div
        style={{
          display: "flex",
          flex: 1,
          gap: 14,
          padding: 12,
          overflow: "hidden",
          minHeight: 0,
        }}
      >
        <div
          style={{
            flexShrink: 0,
            width: 240,
            display: "flex",
            flexDirection: "column",
            gap: 10,
            overflowY: "auto",
          }}
        >
          <div
            onDrop={onDrop}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onClick={() => document.getElementById("infer-file").click()}
            style={{
              border: "2px dashed " + (dragging ? "var(--accent)" : "var(--border2)"),
              borderRadius: 10,
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              gap: 8,
              background: dragging ? "var(--accent-dim)" : "var(--surface2)",
              cursor: "pointer",
              transition: "all 0.2s",
              height: 150,
              overflow: "hidden",
              position: "relative",
              flexShrink: 0,
            }}
          >
            {imageUrl ? (
              <img
                src={imageUrl}
                alt=""
                style={{ width: "100%", height: "100%", objectFit: "cover", borderRadius: 8 }}
              />
            ) : (
              <>
                <svg
                  width="28"
                  height="28"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="var(--text3)"
                  strokeWidth="1.5"
                >
                  <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12" />
                </svg>
                <span style={{ color: "var(--text2)", fontSize: 12 }}>Drop image here</span>
                <span style={{ color: "var(--text3)", fontSize: 10 }}>PNG · JPG · WEBP</span>
              </>
            )}
            <input
              id="infer-file"
              type="file"
              accept="image/*"
              style={{ display: "none" }}
              onChange={(e) => handleFile(e.target.files[0])}
            />
          </div>

          <div
            style={{
              background: "var(--surface2)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              padding: "12px 14px",
              display: "flex",
              flexDirection: "column",
              gap: 10,
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
              Parameters
            </span>
            {task === "classify" && (
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                <span style={{ color: "var(--text3)", fontSize: 11 }}>Top-K</span>
                <div style={{ display: "flex", gap: 5 }}>
                  {[1, 3, 5, 10].map((k) => (
                    <button
                      key={k}
                      onClick={() => setTopK(k)}
                      style={{
                        flex: 1,
                        padding: "4px 0",
                        borderRadius: 5,
                        border: "1px solid " + (topK === k ? "var(--accent)" : "var(--border)"),
                        cursor: "pointer",
                        fontSize: 11,
                        fontFamily: "var(--mono)",
                        background: topK === k ? "var(--accent-dim)" : "transparent",
                        color: topK === k ? "var(--accent)" : "var(--text2)",
                      }}
                    >
                      {k}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {(task === "detect" || task === "segment") && (
              <>
                <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ color: "var(--text3)", fontSize: 11 }}>Conf Threshold</span>
                    <span
                      style={{
                        color: "var(--accent2)",
                        fontSize: 11,
                        fontFamily: "var(--mono)",
                      }}
                    >
                      {confThresh.toFixed(2)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="0.99"
                    step="0.01"
                    value={confThresh}
                    onChange={(e) => setConfThresh(+e.target.value)}
                    style={{ width: "100%", accentColor: "var(--accent)" }}
                  />
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ color: "var(--text3)", fontSize: 11 }}>IoU (NMS)</span>
                    <span
                      style={{
                        color: "var(--accent2)",
                        fontSize: 11,
                        fontFamily: "var(--mono)",
                      }}
                    >
                      {iouThresh.toFixed(2)}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="0.99"
                    step="0.01"
                    value={iouThresh}
                    onChange={(e) => setIouThresh(+e.target.value)}
                    style={{ width: "100%", accentColor: "var(--accent)" }}
                  />
                </div>
              </>
            )}
            {task === "pose" && (
              <div style={{ color: "var(--text3)", fontSize: 11, lineHeight: 1.7 }}>
                14 keypoints · COCO skeleton
                <br />
                <span style={{ color: "var(--accent2)" }}>PCKh@0.5</span> metric
              </div>
            )}
          </div>

          <button
            onClick={runInference}
            disabled={!imageUrl || running}
            style={{
              padding: "9px 0",
              borderRadius: 7,
              border: "none",
              cursor: imageUrl ? "pointer" : "not-allowed",
              fontFamily: "var(--font)",
              fontSize: 12,
              fontWeight: 600,
              background: imageUrl
                ? "linear-gradient(135deg,#7c3aed,#06b6d4)"
                : "var(--surface3)",
              color: imageUrl ? "white" : "var(--text3)",
              opacity: running ? 0.7 : 1,
              transition: "all 0.2s",
              boxShadow: imageUrl ? "0 0 20px rgba(139,92,246,0.3)" : "",
            }}
          >
            {running ? "⚙ Running..." : "▶ Run Inference"}
          </button>

          {isDone && (
            <div
              style={{
                background: "var(--surface2)",
                border: "1px solid var(--border)",
                borderRadius: 8,
                padding: "10px 14px",
                display: "flex",
                flexDirection: "column",
                gap: 4,
              }}
            >
              {[
                ["Task", CV_TASKS.find((t2) => t2.id === task)?.label],
                ["Model", model],
                ["Device", "CUDA:0"],
                ["Latency", latency + "ms"],
                ["Input", "640×640"],
              ].map(([k, v]) => (
                <div key={k} style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--text3)", fontSize: 11 }}>{k}</span>
                  <span
                    style={{
                      color: "var(--accent2)",
                      fontFamily: "var(--mono)",
                      fontSize: 11,
                    }}
                  >
                    {v}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 8, minWidth: 0 }}>
          <div
            style={{
              flex: 1,
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: 10,
              overflow: "hidden",
              position: "relative",
              minHeight: 0,
            }}
          >
            {running && (
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  background: "rgba(12,12,18,0.75)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  zIndex: 10,
                  flexDirection: "column",
                  gap: 12,
                }}
              >
                <div
                  style={{
                    width: 32,
                    height: 32,
                    border: "3px solid var(--accent)",
                    borderTopColor: "transparent",
                    borderRadius: "50%",
                    animation: "spin 0.8s linear infinite",
                  }}
                />
                <span
                  style={{ color: "var(--text2)", fontSize: 12, fontFamily: "var(--mono)" }}
                >
                  Running {CV_TASKS.find((t2) => t2.id === task)?.label}...
                </span>
              </div>
            )}
            <DetectionCanvas
              imageUrl={imageUrl}
              results={detFiltered}
              task={isDone ? task : "none"}
              segResults={SEG_RESULTS}
              poseKpts={POSE_KEYPOINTS}
              poseBones={POSE_BONES}
            />
          </div>
          {!imageUrl && (
            <div style={{ textAlign: "center", color: "var(--text3)", fontSize: 11 }}>
              Upload an image · {CV_TASKS.find((t2) => t2.id === task)?.label} mode active
            </div>
          )}
        </div>

        <div
          style={{
            flexShrink: 0,
            width: 230,
            display: "flex",
            flexDirection: "column",
            gap: 10,
            overflowY: "auto",
          }}
        >
          {!isDone && (
            <div
              style={{
                background: "var(--surface2)",
                border: "1px solid var(--border)",
                borderRadius: 8,
                padding: "14px",
                color: "var(--text3)",
                fontSize: 12,
                textAlign: "center",
                marginTop: 8,
                lineHeight: 1.6,
              }}
            >
              Results appear here
              <br />
              after inference
            </div>
          )}

          {isDone && task === "classify" && (
            <>
              <div
                style={{
                  background: "var(--surface2)",
                  border: "1px solid var(--accent)",
                  borderRadius: 10,
                  padding: "14px",
                }}
              >
                <div
                  style={{
                    color: "var(--text3)",
                    fontSize: 10,
                    textTransform: "uppercase",
                    letterSpacing: "0.07em",
                    marginBottom: 6,
                  }}
                >
                  Top Prediction
                </div>
                <div style={{ fontSize: 15, fontWeight: 700, color: "var(--text)" }}>
                  {CLS_RESULTS[0].label}
                </div>
                <div
                  style={{
                    color: "var(--accent2)",
                    fontFamily: "var(--mono)",
                    fontSize: 13,
                    marginTop: 3,
                  }}
                >
                  {(CLS_RESULTS[0].conf * 100).toFixed(1)}%
                </div>
              </div>
              <div
                style={{
                  background: "var(--surface2)",
                  border: "1px solid var(--border)",
                  borderRadius: 8,
                  padding: "12px 14px",
                  display: "flex",
                  flexDirection: "column",
                  gap: 9,
                }}
              >
                <span
                  style={{
                    color: "var(--text3)",
                    fontSize: 10,
                    textTransform: "uppercase",
                    letterSpacing: "0.07em",
                  }}
                >
                  Top-{topK} Classes
                </span>
                {CLS_RESULTS.slice(0, topK).map((c, i) => (
                  <div key={i} style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                    <div style={{ display: "flex", justifyContent: "space-between" }}>
                      <span
                        style={{
                          color: i === 0 ? "var(--text)" : "var(--text2)",
                          fontSize: 11,
                          fontWeight: i === 0 ? 600 : 400,
                        }}
                      >
                        {c.label}
                      </span>
                      <span
                        style={{
                          fontFamily: "var(--mono)",
                          fontSize: 10,
                          color: i === 0 ? "var(--accent2)" : "var(--text3)",
                        }}
                      >
                        {(c.conf * 100).toFixed(1)}%
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
                          width: c.conf * 100 + "%",
                          background:
                            i === 0
                              ? "linear-gradient(90deg,#7c3aed,#06b6d4)"
                              : "var(--border2)",
                          borderRadius: 2,
                          transition: "width 0.8s ease",
                          transitionDelay: i * 0.08 + "s",
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}

          {isDone && task === "detect" && (
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
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}
              >
                <span
                  style={{
                    color: "var(--text3)",
                    fontSize: 10,
                    textTransform: "uppercase",
                    letterSpacing: "0.07em",
                  }}
                >
                  Detections
                </span>
                <span
                  style={{
                    background: "var(--accent-dim)",
                    color: "var(--accent)",
                    fontSize: 10,
                    padding: "1px 7px",
                    borderRadius: 10,
                    fontFamily: "var(--mono)",
                  }}
                >
                  {detFiltered.length}
                </span>
              </div>
              {detFiltered.map((r, i) => (
                <div
                  key={i}
                  style={{
                    padding: "7px 10px",
                    background: "var(--surface3)",
                    borderRadius: 6,
                    borderLeft: "3px solid " + r.color,
                  }}
                >
                  <div style={{ color: "var(--text)", fontSize: 12, fontWeight: 500 }}>
                    {r.label}
                  </div>
                  <div
                    style={{
                      color: "var(--text3)",
                      fontSize: 10,
                      fontFamily: "var(--mono)",
                      marginTop: 2,
                    }}
                  >
                    conf {(r.conf * 100).toFixed(0)}%
                  </div>
                  <div
                    style={{
                      color: "var(--text3)",
                      fontSize: 10,
                      fontFamily: "var(--mono)",
                    }}
                  >
                    [{r.x.toFixed(2)},{r.y.toFixed(2)},{(r.x + r.w).toFixed(2)},
                    {(r.y + r.h).toFixed(2)}]
                  </div>
                </div>
              ))}
              {detFiltered.length === 0 && (
                <span style={{ color: "var(--text3)", fontSize: 12 }}>
                  No detections above threshold
                </span>
              )}
            </div>
          )}

          {isDone && task === "segment" && (
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
                  fontSize: 10,
                  textTransform: "uppercase",
                  letterSpacing: "0.07em",
                }}
              >
                Segments
              </span>
              {SEG_RESULTS.map((s, i) => (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <div
                    style={{
                      width: 10,
                      height: 10,
                      borderRadius: 2,
                      background: s.color,
                      flexShrink: 0,
                    }}
                  />
                  <span style={{ color: "var(--text)", fontSize: 12, flex: 1 }}>{s.label}</span>
                  <span
                    style={{
                      color: "var(--accent2)",
                      fontFamily: "var(--mono)",
                      fontSize: 11,
                    }}
                  >
                    {s.area}
                  </span>
                </div>
              ))}
            </div>
          )}

          {isDone && task === "pose" && (
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
                  fontSize: 10,
                  textTransform: "uppercase",
                  letterSpacing: "0.07em",
                }}
              >
                Keypoints (14)
              </span>
              {POSE_KEYPOINTS.slice(0, 7).map((k, i) => (
                <div
                  key={i}
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    gap: 6,
                  }}
                >
                  <span
                    style={{
                      color: "var(--text2)",
                      fontSize: 11,
                      fontFamily: "var(--mono)",
                      flex: 1,
                    }}
                  >
                    {k.label}
                  </span>
                  <span
                    style={{
                      color: "var(--accent)",
                      fontSize: 10,
                      fontFamily: "var(--mono)",
                    }}
                  >
                    {k.x},{k.y}
                  </span>
                  <div
                    style={{
                      width: 24,
                      height: 3,
                      background: "var(--surface3)",
                      borderRadius: 2,
                      overflow: "hidden",
                      flexShrink: 0,
                    }}
                  >
                    <div style={{ width: "85%", height: "100%", background: "#8b5cf6" }} />
                  </div>
                </div>
              ))}
              <span style={{ color: "var(--text3)", fontSize: 10 }}>+7 more keypoints</span>
              <div style={{ padding: "7px 10px", background: "var(--surface3)", borderRadius: 6 }}>
                <span style={{ color: "var(--text3)", fontSize: 11 }}>PCKh@0.5: </span>
                <span
                  style={{
                    color: "var(--green)",
                    fontFamily: "var(--mono)",
                    fontSize: 12,
                    fontWeight: 600,
                  }}
                >
                  87.3%
                </span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
