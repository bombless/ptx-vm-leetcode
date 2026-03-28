from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
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
SOURCE_DIR = RUNTIME_DIR / "solutions"

for directory in (RUNTIME_DIR, UPLOAD_DIR, SOURCE_DIR):
    directory.mkdir(parents=True, exist_ok=True)


PROBLEM_ID = "gpu-vector-add-f32"

STARTER_CODE = """// PTX VM Challenge 001
// Keep the solve kernel signature unchanged.
// Write the final result into vector C.

.version 7.0
.target sm_50
.address_size 64

.entry solve(
    .param .u64 A,
    .param .u64 B,
    .param .u64 C,
    .param .u32 N
)
{
    .reg .pred %p<2>;
    .reg .s32 %r<10>;
    .reg .u64 %rd<10>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [A];
    ld.param.u64 %rd2, [B];
    ld.param.u64 %rd3, [C];
    ld.param.u32 %r1, [N];

    mov.s32 %r2, 0;

LOOP:
    // TODO: if i >= N, branch to DONE

    // TODO: compute byte offset = i * 4
    // TODO: load A[i] and B[i]
    // TODO: add them as float32 values
    // TODO: store the answer into C[i]

    // TODO: i++
    bra LOOP;

DONE:
    exit;
}
"""

REFERENCE_SOLUTION = """// Reference solution used by the local judge.
.version 7.0
.target sm_50
.address_size 64

.entry solve(
    .param .u64 A,
    .param .u64 B,
    .param .u64 C,
    .param .u32 N
)
{
    .reg .pred %p<3>;
    .reg .s32 %r<10>;
    .reg .u64 %rd<10>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [A];
    ld.param.u64 %rd2, [B];
    ld.param.u64 %rd3, [C];
    ld.param.u32 %r1, [N];

    mov.s32 %r2, 0;

LOOP:
    setp.ge.s32 %p1, %r2, %r1;
    @%p1 bra DONE;

    mul.lo.s32 %r3, %r2, 4;
    cvt.u64.s32 %rd4, %r3;
    add.u64 %rd5, %rd1, %rd4;
    add.u64 %rd6, %rd2, %rd4;
    add.u64 %rd7, %rd3, %rd4;

    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    add.f32 %f3, %f1, %f2;
    st.global.f32 [%rd7], %f3;

    add.s32 %r2, %r2, 1;
    bra LOOP;

DONE:
    exit;
}
"""

PROBLEM = {
    "id": PROBLEM_ID,
    "slug": PROBLEM_ID,
    "number": 1,
    "title": "Vector Add on PTX VM",
    "difficulty": "Medium",
    "category": "GPU / PTX",
    "signature": ".entry solve(.param .u64 A, .param .u64 B, .param .u64 C, .param .u32 N)",
    "statement": (
        "Write a GPU program that performs element-wise addition of two vectors containing "
        "32-bit floating point numbers. The program should take two input vectors of equal "
        "length and produce a single output vector containing their sum."
    ),
    "implementation_requirements": [
        "External libraries are not permitted.",
        "The solve function signature must remain unchanged.",
        "The final result must be stored in vector C.",
    ],
    "examples": [
        {
            "title": "Example 1",
            "input": [
                "A = [1.0, 2.0, 3.0, 4.0]",
                "B = [5.0, 6.0, 7.0, 8.0]",
            ],
            "output": "C = [6.0, 8.0, 10.0, 12.0]",
        },
        {
            "title": "Example 2",
            "input": [
                "A = [1.5, 1.5, 1.5]",
                "B = [2.3, 2.3, 2.3]",
            ],
            "output": "C = [3.8, 3.8, 3.8]",
        },
    ],
    "constraints": [
        "Input vectors A and B have identical lengths.",
        "1 <= N <= 100,000,000",
        "Performance is measured with N = 25,000,000",
    ],
    "notes": [
        "This judge runs on the PTX VM in this repository.",
        "The evaluation harness launches a single logical thread, so iterate over N inside solve instead of relying on %tid.x.",
        "Parameter names and types are checked exactly against the required solve signature.",
    ],
    "starter_code": STARTER_CODE,
}

SAMPLE_CASES = [
    {
        "name": "Example 1",
        "hidden": False,
        "A": [1.0, 2.0, 3.0, 4.0],
        "B": [5.0, 6.0, 7.0, 8.0],
    },
    {
        "name": "Example 2",
        "hidden": False,
        "A": [1.5, 1.5, 1.5],
        "B": [2.3, 2.3, 2.3],
    },
]

HIDDEN_CASES = [
    {
        "name": "Hidden 1",
        "hidden": True,
        "A": [0.0, -3.5, 8.25, 1.125],
        "B": [4.0, 3.5, -2.25, 9.875],
    },
    {
        "name": "Hidden 2",
        "hidden": True,
        "A": [float(index) * 0.5 for index in range(16)],
        "B": [1.0 - float(index) * 0.25 for index in range(16)],
    },
    {
        "name": "Hidden 3",
        "hidden": True,
        "A": [7.0, 0.25, -1.25, 2.0, 5.5, -9.0, 3.75],
        "B": [-2.0, 4.75, 8.0, -3.0, 1.5, 9.0, 0.25],
    },
]


def sanitize_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._")
    return cleaned or "program.ptx"


def serialize_values(values: List[float]) -> str:
    return ",".join(format(value, ".9g") for value in values)


def resolve_runner() -> Path:
    override = os.environ.get("PTX_WEB_RUNNER")
    candidates: List[Path] = []
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


def compact_logs(logs: str, max_chars: int = 6000) -> str:
    if len(logs) <= max_chars:
        return logs
    head = logs[: max_chars // 2]
    tail = logs[-max_chars // 2 :]
    return f"{head}\n\n... log truncated ...\n\n{tail}"


def build_expected(values_a: List[float], values_b: List[float]) -> List[float]:
    return [left + right for left, right in zip(values_a, values_b)]


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


class SolutionRequest(BaseModel):
    source: str = Field(min_length=1, max_length=200000)


app = FastAPI(title="PTX Challenge Arena")
app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

PROGRAMS: Dict[str, Path] = {}


def invoke_runner(arguments: List[str]) -> dict:
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


def write_solution_source(source: str) -> Path:
    file_path = SOURCE_DIR / f"{uuid.uuid4().hex}.ptx"
    file_path.write_text(source, encoding="utf-8")
    return file_path


def validate_signature(inspect_payload: dict) -> str | None:
    required_parameters = [
        {"name": "A", "type": ".u64", "is_pointer": True},
        {"name": "B", "type": ".u64", "is_pointer": True},
        {"name": "C", "type": ".u64", "is_pointer": True},
        {"name": "N", "type": ".u32", "is_pointer": False},
    ]

    solve_kernel = None
    for kernel in inspect_payload.get("kernels", []):
        if kernel.get("name") == "solve":
            solve_kernel = kernel
            break

    if solve_kernel is None:
        return "Expected an entry kernel named solve."

    parameters = solve_kernel.get("parameters", [])
    if len(parameters) != len(required_parameters):
        return (
            "solve must declare exactly four parameters: "
            "(.u64 A, .u64 B, .u64 C, .u32 N)."
        )

    for actual, expected in zip(parameters, required_parameters):
        if (
            actual.get("name") != expected["name"]
            or actual.get("type") != expected["type"]
            or actual.get("is_pointer") != expected["is_pointer"]
        ):
            return (
                "solve must keep the exact signature "
                "(.u64 A, .u64 B, .u64 C, .u32 N)."
            )

    return None


def extract_output_vector(run_payload: dict) -> List[float]:
    for buffer in run_payload.get("pointer_buffers", []):
        if buffer.get("name") == "C":
            return [float(value) for value in buffer.get("after", [])]
    raise RuntimeError("The judge could not locate vector C in the runner output.")


def compare_vectors(actual: List[float], expected: List[float], tolerance: float = 1e-5) -> str | None:
    if len(actual) != len(expected):
        return f"Expected {len(expected)} values in C, but the program produced {len(actual)}."

    for index, (got, want) in enumerate(zip(actual, expected)):
        if not math.isclose(got, want, rel_tol=tolerance, abs_tol=tolerance):
            return f"Mismatch at index {index}: expected {want}, got {got}."

    return None


def build_case_result(case: dict, passed: bool, expected: List[float], actual: List[float] | None, message: str) -> dict:
    payload = {
        "name": case["name"],
        "hidden": case["hidden"],
        "passed": passed,
        "n": len(case["A"]),
        "message": message,
    }

    if not case["hidden"]:
        payload["input"] = {"A": case["A"], "B": case["B"]}
        payload["expected"] = expected
        payload["actual"] = actual

    return payload


def run_case(source_path: Path, case: dict) -> tuple[dict | None, str | None]:
    command = [
        "run",
        str(source_path),
        "--kernel",
        "solve",
        "--grid",
        "1,1,1",
        "--block",
        "1,1,1",
        "--pointer",
        f"A=float32@{len(case['A'])}:{serialize_values(case['A'])}",
        "--pointer",
        f"B=float32@{len(case['B'])}:{serialize_values(case['B'])}",
        "--pointer",
        f"C=float32@{len(case['A'])}:{serialize_values([0.0] * len(case['A']))}",
        "--scalar",
        f"N={len(case['A'])}",
    ]

    payload = invoke_runner(command)
    if not payload.get("ok"):
        return None, f"{payload.get('error', 'Runtime error')}\n\n{compact_logs(payload.get('logs', ''))}"
    return payload, None


def judge_solution(source: str, cases: List[dict], mode: str) -> dict:
    started_at = time.perf_counter()
    source_path = write_solution_source(source)

    inspect_payload = invoke_runner(["inspect", str(source_path)])
    if not inspect_payload.get("ok"):
        return {
            "ok": False,
            "mode": mode,
            "status": "compile_error",
            "summary": inspect_payload.get("error", "Failed to parse PTX source."),
            "passed": 0,
            "total": len(cases),
            "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "cases": [],
            "logs": compact_logs(inspect_payload.get("logs", "")),
        }

    signature_error = validate_signature(inspect_payload)
    if signature_error is not None:
        return {
            "ok": False,
            "mode": mode,
            "status": "signature_error",
            "summary": signature_error,
            "passed": 0,
            "total": len(cases),
            "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "cases": [],
            "logs": "",
        }

    passed = 0
    case_results = []
    combined_logs: List[str] = []

    for case in cases:
        expected = build_expected(case["A"], case["B"])
        run_payload, runtime_error = run_case(source_path, case)

        if runtime_error is not None:
            case_results.append(build_case_result(case, False, expected, None, "Runtime error during execution."))
            combined_logs.append(runtime_error)
            return {
                "ok": False,
                "mode": mode,
                "status": "runtime_error",
                "summary": f"Execution failed on {case['name']}.",
                "passed": passed,
                "total": len(cases),
                "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
                "cases": case_results,
                "logs": compact_logs("\n\n".join(combined_logs)),
            }

        actual = extract_output_vector(run_payload)
        mismatch = compare_vectors(actual, expected)
        if mismatch is None:
            passed += 1
            case_results.append(
                build_case_result(case, True, expected, actual if not case["hidden"] else None, "Output matches expected vector.")
            )
        else:
            case_results.append(
                build_case_result(case, False, expected, actual if not case["hidden"] else None, mismatch)
            )
            combined_logs.append(run_payload.get("logs", ""))
            return {
                "ok": False,
                "mode": mode,
                "status": "wrong_answer",
                "summary": f"Wrong answer on {case['name']}.",
                "passed": passed,
                "total": len(cases),
                "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
                "cases": case_results,
                "logs": compact_logs("\n\n".join(combined_logs)),
            }

        combined_logs.append(run_payload.get("logs", ""))

    label = "sample" if mode == "sample" else "submission"
    return {
        "ok": True,
        "mode": mode,
        "status": "accepted",
        "summary": f"Accepted. Passed {passed} / {len(cases)} {label} tests.",
        "passed": passed,
        "total": len(cases),
        "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
        "cases": case_results,
        "logs": compact_logs("\n\n".join(combined_logs)),
    }


@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/problems")
def list_problems() -> dict:
    return {
        "ok": True,
        "problems": [
            {
                "id": PROBLEM["id"],
                "number": PROBLEM["number"],
                "title": PROBLEM["title"],
                "difficulty": PROBLEM["difficulty"],
                "category": PROBLEM["category"],
            }
        ],
    }


@app.get(f"/api/problems/{PROBLEM_ID}")
def get_problem() -> dict:
    return {"ok": True, "problem": PROBLEM}


@app.post(f"/api/problems/{PROBLEM_ID}/run-samples")
def run_samples(request: SolutionRequest) -> dict:
    return judge_solution(request.source, SAMPLE_CASES, mode="sample")


@app.post(f"/api/problems/{PROBLEM_ID}/submit")
def submit_solution(request: SolutionRequest) -> dict:
    return judge_solution(request.source, SAMPLE_CASES + HIDDEN_CASES, mode="submit")


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

    payload = invoke_runner(["inspect", str(destination)])
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

    payload = invoke_runner(command)
    if not payload.get("ok"):
        raise HTTPException(status_code=400, detail=payload)
    return payload


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
