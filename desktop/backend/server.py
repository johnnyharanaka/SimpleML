"""FastAPI backend for the SimpleML desktop app.

Spawned by the Tauri shell as a local-only process. Exposes the SimpleML
registries and training entrypoints to the webview frontend.
"""

from __future__ import annotations

import argparse
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from simpleml import API
from simpleml.registries import (
    DATASETS,
    LOSSES,
    METRICS,
    MODELS,
    OPTIMIZERS,
    SCHEDULERS,
)

REGISTRIES = {
    "models": MODELS,
    "losses": LOSSES,
    "metrics": METRICS,
    "datasets": DATASETS,
    "optimizers": OPTIMIZERS,
    "schedulers": SCHEDULERS,
}

app = FastAPI(title="SimpleML Desktop Backend")

# The Tauri webview loads from tauri://localhost in prod and http://localhost:1420
# (Vite default) in dev. Wide-open CORS is safe here because the server only
# binds to loopback.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainRequest(BaseModel):
    config: dict[str, Any]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/registries")
def list_registries() -> dict[str, list[str]]:
    return {name: reg.list() for name, reg in REGISTRIES.items()}


@app.get("/registries/{category}")
def list_registry(category: str) -> dict[str, list[str]]:
    if category not in REGISTRIES:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown registry {category!r}. Available: {list(REGISTRIES)}",
        )
    return {"items": REGISTRIES[category].list()}


@app.post("/train")
def train(request: TrainRequest) -> dict[str, Any]:
    api = API()
    api._config = request.config
    return api.fit()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
