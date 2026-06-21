"""
Private web upload app for the Douyin/Xiaohongshu editor.

Drag in a raw clip, pick options, a background thread runs the shared pipeline
(`src.pipeline.run_job`), poll for progress, download the result. Single-user
HTTP Basic auth (APP_PASSWORD). AI restore is offloaded to a GPU backend when
RESTORE_BACKEND=modal (see webapp/modal_restore.py); otherwise it runs locally.
"""
from __future__ import annotations
import os
import secrets
import shutil
import sys
import threading
import uuid
from pathlib import Path

from fastapi import (FastAPI, UploadFile, File, Form, Depends, HTTPException)
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.pipeline import run_job, JobOptions  # noqa: E402
from src.config import Settings  # noqa: E402

JOBS_DIR = ROOT / "webapp" / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
PASSWORD = os.environ.get("APP_PASSWORD", "changeme")
INDEX = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")

app = FastAPI(title="Douyin Editor")
security = HTTPBasic()
JOBS: dict[str, dict] = {}


def auth(creds: HTTPBasicCredentials = Depends(security)) -> bool:
    if not secrets.compare_digest(creds.password, PASSWORD):
        raise HTTPException(status_code=401, detail="unauthorized",
                            headers={"WWW-Authenticate": "Basic"})
    return True


def _restore_backend():
    if os.environ.get("RESTORE_BACKEND") == "modal":
        from webapp.modal_restore import restore_via_modal
        return restore_via_modal
    return None  # local GFPGAN


def _run(job_id: str, src: str, opts: JobOptions):
    job = JOBS[job_id]
    try:
        jdir = JOBS_DIR / job_id
        out = str(jdir / "output.mp4")
        run_job(src, out, opts, tmpdir=str(jdir / "tmp"),
                progress=lambda m: job.update(message=m),
                restore_backend=_restore_backend())
        job.update(status="done", message="Done", output=out)
    except Exception as e:  # surface failures to the UI
        job.update(status="error", message=f"Error: {e}", error=str(e))


@app.get("/", response_class=HTMLResponse)
def index(_: bool = Depends(auth)):
    return INDEX


@app.post("/jobs")
async def create_job(_: bool = Depends(auth), file: UploadFile = File(...),
                     trim: str = Form("0"), restore: str = Form("0"),
                     restore_intensity: float = Form(0.35), beauty: str = Form("0"),
                     subtitle: str = Form("1"), hdr: str = Form("0"),
                     model: str = Form("small")):
    yes = lambda v: str(v).lower() in ("1", "true", "on", "yes")
    job_id = uuid.uuid4().hex[:12]
    jdir = JOBS_DIR / job_id
    (jdir / "tmp").mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename or "input.mp4").suffix or ".mp4"
    src = str(jdir / f"input{ext}")
    with open(src, "wb") as f:
        shutil.copyfileobj(file.file, f)

    opts = JobOptions(trim=yes(trim), restore=yes(restore),
                      restore_intensity=float(restore_intensity), beauty=yes(beauty),
                      subtitle=yes(subtitle), hdr=yes(hdr), model=model)
    JOBS[job_id] = {"status": "running", "message": "Queued…",
                    "name": Path(file.filename or "clip").stem}
    threading.Thread(target=_run, args=(job_id, src, opts), daemon=True).start()
    return {"job_id": job_id}


@app.get("/jobs/{job_id}")
def job_status(job_id: str, _: bool = Depends(auth)):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    return JSONResponse({k: v for k, v in job.items() if k != "output"})


@app.get("/jobs/{job_id}/download")
def job_download(job_id: str, _: bool = Depends(auth)):
    job = JOBS.get(job_id)
    if not job or job.get("status") != "done":
        raise HTTPException(404, "not ready")
    return FileResponse(job["output"], media_type="video/mp4",
                        filename=f"{job.get('name','clip')}_edited.mp4")


@app.get("/healthz")
def healthz():
    return {"ok": True}
