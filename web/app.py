from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
STATIC_DIR = BASE_DIR / "static"
RUNTIME_DIR = BASE_DIR / ".runtime"
UPLOAD_DIR = RUNTIME_DIR / "uploads"

for directory in (RUNTIME_DIR, UPLOAD_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def sanitize_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._")
    return cleaned or "program.ptx"


def resolve_runner() -> Path:
    override = os.environ.get("PTX_WEB_RUNNER")
    candidates = []
    if override:
        candidates.append(Path(override))

    candidates.extend(
        [
            REPO_ROOT / "build" / "ptx_web_runner",
            REPO_ROOT / "build" / "web" / "ptx_web_runner",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise RuntimeError(
        "Could not find ptx_web_runner. Build the project first, for example: "
        "cmake -S . -B build && cmake --build build"
    )


class PointerRequest(BaseModel):
    buffer_type: str
    element_count: int = Field(gt=0)
    values: str = ""


class RunRequest(BaseModel):
    kernel: str
    grid: List[int] = Field(default_factory=lambda: [1, 1, 1], min_length=3, max_length=3)
    block: List[int] = Field(default_factory=lambda: [32, 1, 1], min_length=3, max_length=3)
    scalars: Dict[str, str] = Field(default_factory=dict)
    pointers: Dict[str, PointerRequest] = Field(default_factory=dict)


app = FastAPI(title="PTX VM Web")
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

PROGRAMS: Dict[str, Path] = {}


def run_runner(arguments: List[str]) -> dict:
    runner = resolve_runner()
    process = subprocess.run(
        [str(runner), *arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    stdout = process.stdout.strip()
    if not stdout:
        detail = process.stderr.strip() or "Runner returned no output"
        raise HTTPException(status_code=500, detail=detail)

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Runner returned invalid JSON",
                "stdout": stdout,
                "stderr": process.stderr.strip(),
                "error": str(error),
            },
        ) from error

    if process.returncode != 0 and payload.get("ok", True):
        payload["ok"] = False
        payload["error"] = payload.get("error") or process.stderr.strip() or "PTX runner failed"

    return payload


@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/programs/inspect")
async def inspect_program(file: UploadFile = File(...)) -> dict:
    filename = sanitize_filename(file.filename or "program.ptx")
    if not filename.lower().endswith(".ptx"):
        raise HTTPException(status_code=400, detail="Please upload a .ptx file")

    program_id = uuid.uuid4().hex
    destination = UPLOAD_DIR / f"{program_id}_{filename}"
    contents = await file.read()
    destination.write_bytes(contents)
    PROGRAMS[program_id] = destination

    payload = run_runner(["inspect", str(destination)])
    if not payload.get("ok"):
        raise HTTPException(status_code=400, detail=payload)

    payload["program_id"] = program_id
    return payload


@app.post("/api/programs/{program_id}/run")
def run_program(program_id: str, request: RunRequest) -> dict:
    program_path = PROGRAMS.get(program_id)
    if program_path is None or not program_path.exists():
        raise HTTPException(status_code=404, detail="Uploaded PTX file was not found on the server")

    command = [
        "run",
        str(program_path),
        "--kernel",
        request.kernel,
        "--grid",
        ",".join(str(value) for value in request.grid),
        "--block",
        ",".join(str(value) for value in request.block),
    ]

    for name, value in request.scalars.items():
        command.extend(["--scalar", f"{name}={value}"])

    for name, pointer in request.pointers.items():
        command.extend(
            [
                "--pointer",
                f"{name}={pointer.buffer_type}@{pointer.element_count}:{pointer.values.strip()}",
            ]
        )

    payload = run_runner(command)
    if not payload.get("ok"):
        raise HTTPException(status_code=400, detail=payload)
    return payload


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
