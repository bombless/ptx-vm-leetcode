from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
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
JUDGE_JOB_DIR = RUNTIME_DIR / "judge_jobs"

for directory in (RUNTIME_DIR, UPLOAD_DIR, SOURCE_DIR, JUDGE_JOB_DIR):
    directory.mkdir(parents=True, exist_ok=True)


PROBLEM_ID = "gpu-vector-add-f32"
NVCC_TIMEOUT_SECONDS = 90
RUN_TIMEOUT_SECONDS = 20

STARTER_CODE = """#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
"""

REFERENCE_SOLUTION = """#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        C[index] = A[index] + B[index];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
"""

JUDGE_DRIVER_SOURCE = """#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

extern "C" void solve(const float* A, const float* B, float* C, int N);

static std::string escape_json(const std::string& text) {
    std::string escaped;
    escaped.reserve(text.size());
    for (char ch : text) {
        switch (ch) {
            case '\\\\':
                escaped += "\\\\\\\\";
                break;
            case '\"':
                escaped += "\\\\\\\"";
                break;
            case '\\n':
                escaped += "\\\\n";
                break;
            case '\\r':
                escaped += "\\\\r";
                break;
            case '\\t':
                escaped += "\\\\t";
                break;
            default:
                escaped += ch;
                break;
        }
    }
    return escaped;
}

static void print_error(const std::string& message) {
    std::cout << "{\\\"ok\\\":false,\\\"error\\\":\\\"" << escape_json(message) << "\\\"}";
}

static std::vector<float> parse_csv(const std::string& text) {
    std::vector<float> values;
    if (text.empty()) {
        return values;
    }

    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (token.empty()) {
            continue;
        }
        values.push_back(std::stof(token));
    }
    return values;
}

static bool check_cuda(cudaError_t status, const std::string& step) {
    if (status == cudaSuccess) {
        return true;
    }

    print_error(step + ": " + cudaGetErrorString(status));
    return false;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        print_error("Expected two CSV arguments: A and B.");
        return 1;
    }

    std::vector<float> host_a = parse_csv(argv[1]);
    std::vector<float> host_b = parse_csv(argv[2]);
    if (host_a.size() != host_b.size()) {
        print_error("Input vectors must have identical lengths.");
        return 1;
    }

    const int count = static_cast<int>(host_a.size());
    std::vector<float> host_c(static_cast<size_t>(count), 0.0f);
    const size_t bytes = static_cast<size_t>(count) * sizeof(float);

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;

    if (!check_cuda(cudaMalloc(&device_a, bytes), "cudaMalloc(A)")) {
        return 1;
    }
    if (!check_cuda(cudaMalloc(&device_b, bytes), "cudaMalloc(B)")) {
        cudaFree(device_a);
        return 1;
    }
    if (!check_cuda(cudaMalloc(&device_c, bytes), "cudaMalloc(C)")) {
        cudaFree(device_a);
        cudaFree(device_b);
        return 1;
    }
    if (!check_cuda(cudaMemcpy(device_a, host_a.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy(A)")) {
        cudaFree(device_a);
        cudaFree(device_b);
        cudaFree(device_c);
        return 1;
    }
    if (!check_cuda(cudaMemcpy(device_b, host_b.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy(B)")) {
        cudaFree(device_a);
        cudaFree(device_b);
        cudaFree(device_c);
        return 1;
    }

    solve(device_a, device_b, device_c, count);

    if (!check_cuda(cudaGetLastError(), "Kernel launch")) {
        cudaFree(device_a);
        cudaFree(device_b);
        cudaFree(device_c);
        return 1;
    }
    if (!check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
        cudaFree(device_a);
        cudaFree(device_b);
        cudaFree(device_c);
        return 1;
    }
    if (!check_cuda(cudaMemcpy(host_c.data(), device_c, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(C)")) {
        cudaFree(device_a);
        cudaFree(device_b);
        cudaFree(device_c);
        return 1;
    }

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    std::cout << "{\\\"ok\\\":true,\\\"output\\\":[";
    for (size_t index = 0; index < host_c.size(); ++index) {
        if (index != 0) {
            std::cout << ',';
        }
        std::cout << std::setprecision(9) << host_c[index];
    }
    std::cout << "]}";
    return 0;
}
"""

SOLUTION_SIGNATURE_PATTERN = re.compile(
    r'extern\s+"C"\s+void\s+solve\s*\(\s*'
    r'const\s+float\s*\*\s*A\s*,\s*'
    r'const\s+float\s*\*\s*B\s*,\s*'
    r'float\s*\*\s*C\s*,\s*'
    r'int\s+N\s*'
    r'\)',
    re.MULTILINE,
)

PROBLEM = {
    "id": PROBLEM_ID,
    "slug": PROBLEM_ID,
    "number": 1,
    "title": "Vector Add in CUDA",
    "difficulty": "Medium",
    "category": "GPU / CUDA",
    "signature": (
        'solution.cu\n'
        '__global__ void vector_add(const float* A, const float* B, float* C, int N);\n'
        'extern "C" void solve(const float* A, const float* B, float* C, int N);'
    ),
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
        "Submit CUDA source as solution.cu, or paste it directly into the editor.",
        "A, B, and C are device pointers that already point to GPU memory.",
        "The provided solve wrapper already launches the kernel, so most solutions only need to fill vector_add.",
        "The local judge compiles your CUDA code with nvcc and executes it on this machine's NVIDIA GPU.",
    ],
    "starter_code": STARTER_CODE,
    "starter_filename": "solution.cu",
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


def resolve_nvcc() -> Path:
    override = os.environ.get("PTX_WEB_NVCC")
    candidates: List[Path] = []
    if override:
        candidates.append(Path(override))

    which_nvcc = shutil.which("nvcc")
    if which_nvcc:
        candidates.append(Path(which_nvcc))

    candidates.append(Path("/opt/cuda/bin/nvcc"))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise RuntimeError(
        "Could not find nvcc. Install the CUDA toolkit or set PTX_WEB_NVCC=/path/to/nvcc."
    )


def resolve_cuda_arch() -> str:
    override = os.environ.get("PTX_WEB_CUDA_ARCH")
    if override:
        return override

    try:
        process = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "native"

    if process.returncode != 0:
        return "native"

    first_line = process.stdout.strip().splitlines()
    if not first_line:
        return "native"

    digits = first_line[0].strip().replace(".", "")
    if digits.isdigit():
        return f"sm_{digits}"

    return "native"


def compact_logs(logs: str, max_chars: int = 6000) -> str:
    if len(logs) <= max_chars:
        return logs
    head = logs[: max_chars // 2]
    tail = logs[-max_chars // 2 :]
    return f"{head}\n\n... log truncated ...\n\n{tail}"


def build_expected(values_a: List[float], values_b: List[float]) -> List[float]:
    return [left + right for left, right in zip(values_a, values_b)]


def format_process_logs(command: List[str], stdout: str, stderr: str) -> str:
    parts = [f"$ {shlex.join(command)}"]
    if stdout.strip():
        parts.append(f"stdout:\n{stdout.strip()}")
    if stderr.strip():
        parts.append(f"stderr:\n{stderr.strip()}")
    return "\n\n".join(parts)


def validate_cuda_source_signature(source: str) -> str | None:
    if SOLUTION_SIGNATURE_PATTERN.search(source):
        return None

    return (
        'solve must keep the exact signature '
        'extern "C" void solve(const float* A, const float* B, float* C, int N).'
    )


def create_job_dir() -> Path:
    job_dir = JUDGE_JOB_DIR / uuid.uuid4().hex
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def write_cuda_solution(source: str, job_dir: Path) -> Path:
    solution_path = job_dir / "solution.cu"
    solution_path.write_text(source, encoding="utf-8")
    return solution_path


def write_judge_driver(job_dir: Path) -> Path:
    driver_path = job_dir / "judge_driver.cpp"
    driver_path.write_text(JUDGE_DRIVER_SOURCE, encoding="utf-8")
    return driver_path


def compile_cuda_solution(source: str) -> tuple[Path | None, str]:
    job_dir = create_job_dir()
    solution_path = write_cuda_solution(source, job_dir)
    driver_path = write_judge_driver(job_dir)
    executable_path = job_dir / "judge_runner"

    nvcc = resolve_nvcc()
    arch = resolve_cuda_arch()
    command = [
        str(nvcc),
        "-std=c++17",
        "-O2",
        f"-arch={arch}",
        "-o",
        str(executable_path),
        str(solution_path),
        str(driver_path),
    ]

    try:
        process = subprocess.run(
            command,
            cwd=job_dir,
            capture_output=True,
            text=True,
            check=False,
            timeout=NVCC_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return None, format_process_logs(command, "", "Compilation timed out.")

    logs = format_process_logs(command, process.stdout, process.stderr)
    if process.returncode != 0 or not executable_path.exists():
        return None, logs

    return executable_path, logs


def extract_cuda_output(run_payload: dict) -> List[float]:
    return [float(value) for value in run_payload.get("output", [])]


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


def run_case(executable_path: Path, case: dict) -> tuple[dict | None, str, str | None]:
    command = [
        str(executable_path),
        serialize_values(case["A"]),
        serialize_values(case["B"]),
    ]

    try:
        process = subprocess.run(
            command,
            cwd=executable_path.parent,
            capture_output=True,
            text=True,
            check=False,
            timeout=RUN_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        logs = format_process_logs(command, "", "Execution timed out.")
        return None, logs, "Execution timed out."

    logs = format_process_logs(command, process.stdout, process.stderr)
    stdout = process.stdout.strip()
    if not stdout:
        error_message = process.stderr.strip() or "Judge runner returned no output."
        return None, logs, error_message

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as error:
        return None, logs, f"Judge runner returned invalid JSON: {error}"

    if process.returncode != 0 or not payload.get("ok"):
        return None, logs, payload.get("error", "Runtime error during CUDA execution.")

    return payload, logs, None


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


app = FastAPI(title="CUDA Challenge Arena")
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


def judge_solution(source: str, cases: List[dict], mode: str) -> dict:
    started_at = time.perf_counter()

    signature_error = validate_cuda_source_signature(source)
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

    executable_path, compile_logs = compile_cuda_solution(source)
    if executable_path is None:
        return {
            "ok": False,
            "mode": mode,
            "status": "compile_error",
            "summary": "nvcc failed to compile solution.cu.",
            "passed": 0,
            "total": len(cases),
            "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
            "cases": [],
            "logs": compact_logs(compile_logs),
        }

    passed = 0
    case_results = []
    combined_logs: List[str] = [compile_logs]

    for case in cases:
        expected = build_expected(case["A"], case["B"])
        run_payload, case_logs, runtime_error = run_case(executable_path, case)
        combined_logs.append(case_logs)

        if runtime_error is not None:
            case_results.append(
                build_case_result(case, False, expected, None, f"Runtime error: {runtime_error}")
            )
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

        actual = extract_cuda_output(run_payload)
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
