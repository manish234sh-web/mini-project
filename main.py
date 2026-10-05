import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
try:
    from supabase import Client, create_client  # type: ignore[reportMissingImports]
except ImportError as exc:
    print("[FATAL] The 'supabase' package is not installed for this Python.")
    print(f"[FATAL] Run:  {sys.executable} -m pip install supabase python-dotenv")
    raise SystemExit(1) from exc
import cv2
import numpy as np
import time
import math
import threading
import base64
import traceback
from io import BytesIO
from PIL import Image
from collections import deque
from datetime import datetime
from ultralytics import YOLO
from typing import Optional, Literal

app = FastAPI(title="Jan Suraksha Core API")

# Comma-separated list of allowed frontend origins, e.g.
# ALLOWED_ORIGINS=https://mini-project-eight-sepia.vercel.app
# Defaults to "*" (any origin) if unset. The frontend sends no cookies, so credentials stay off.
_origins_env = os.environ.get("ALLOWED_ORIGINS", "*").strip()
ALLOWED_ORIGINS = ["*"] if _origins_env in ("", "*") else [o.strip().rstrip("/") for o in _origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- DATABASE SETUP (SUPABASE) -----------------
import hashlib
import secrets
import sys

def hash_auth_key(raw_key: str, salt: Optional[str] = None) -> str:
    """PBKDF2-HMAC-SHA256 with a per-credential salt, stdlib-only (no new
    dependency needed). Storing/comparing plaintext operator passwords for a
    system that dispatches police/EMS is a real risk, not a style nitpick -
    this is the minimal fix that doesn't require adding bcrypt/passlib."""
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", raw_key.encode("utf-8"), bytes.fromhex(salt), 200_000)
    return f"{salt}${digest.hex()}"

def verify_auth_key(raw_key: str, stored: str) -> bool:
    if not stored or "$" not in stored:
        return False
    salt, _ = stored.split("$", 1)
    return secrets.compare_digest(hash_auth_key(raw_key, salt), stored)

# Load simple KEY=VALUE entries from .env without requiring python-dotenv.
env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.isfile(env_file):
    with open(env_file, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if value[:1] == value[-1:] and value[:1] in {"'", '"'}:
                value = value[1:-1]
            os.environ.setdefault(key, value)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
# Service-role key: backend only. It bypasses Row Level Security, so it must
# never appear in any .html file or be committed to git.
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()

if not SUPABASE_URL or not SUPABASE_KEY:
    print("[FATAL] Set SUPABASE_URL and SUPABASE_SERVICE_KEY (env vars or a .env file).")
    sys.exit(1)

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    supabase.table("admins").select("id").limit(1).execute()
except Exception as e:
    print(f"[FATAL] Could not connect to Supabase - {e}")
    print("[FATAL] Check SUPABASE_URL / SUPABASE_SERVICE_KEY and that the tables exist.")
    sys.exit(1)

def haversine_m(lat1, lng1, lat2, lng2):
    """Great-circle distance in metres (replaces Mongo's $nearSphere)."""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))

def _count(table, statuses, exclude=False):
    q = supabase.table(table).select("id", count="exact", head=True)
    q = q.not_.in_("status", statuses) if exclude else q.in_("status", statuses)
    return q.execute().count or 0

# Public URL of this backend, used to build evidence links. Render sets RENDER_EXTERNAL_URL automatically.
PUBLIC_BASE_URL = (os.environ.get("PUBLIC_BASE_URL") or "https://browse-header-fountain.ngrok-free.dev").rstrip("/")

EVIDENCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evidence")
os.makedirs(EVIDENCE_DIR, exist_ok=True)
app.mount("/evidence", StaticFiles(directory=EVIDENCE_DIR), name="evidence")

def generate_id():
    return int(time.time() * 1000)

def init_db():
    try:
        if _table_empty("admins"):
            supabase.table("admins").insert({
                "id": generate_id(),
                "operator_id": "JS-OP-9014",
                "auth_key": hash_auth_key("admin123")
            }).execute()
            print("[SYSTEM] Default admin created (JS-OP-9014 / admin123) - change this credential.")
        if _table_empty("responders"):
            base = generate_id()
            supabase.table("responders").insert([
                {"id": base,     "operator_id": "JS-POL-001", "auth_key": hash_auth_key("police123"), "department": "POLICE"},
                {"id": base + 1, "operator_id": "JS-EMS-002", "auth_key": hash_auth_key("ems123"), "department": "EMS"},
                {"id": base + 2, "operator_id": "JS-CIV-003", "auth_key": hash_auth_key("civic123"), "department": "CIVIC"}
            ]).execute()
            print("[SYSTEM] Default responders created - change these credentials.")
        print("[SYSTEM] Supabase connected and initialized.")
    except Exception as e:
        print(f"[FATAL] Database initialization failed: {e}")
        sys.exit(1)

def _table_empty(table):
    return (supabase.table(table).select("id", count="exact", head=True).execute().count or 0) == 0

init_db()

def format_doc(doc):
    # Supabase rows already carry "id"; kept so callers don't change.
    return doc

# ----------------- HARDWARE TELEMETRY CACHE -----------------
hardware_state = {
    "water_raw": 0,
    "rain_raw": 0,
    "overall_state": "NORMAL",
    "last_update": 0
    
}
last_hardware_incident_time = 0.0
HARDWARE_INCIDENT_COOLDOWN_SEC = 60.0

CCTV_NODES = [
    {"id": "CAM-SMART-ROAD-01", "lat": 26.4499, "lng": 80.3319, "name": "Model Underpass & Footpath (Sector 3)"},
    {"id": "CAM-HQ-00", "lat": 26.4505, "lng": 80.3300, "name": "Central Police Headquarters"}
]

# ----------------- AI SUPPRESSION AUDIT COUNTERS -----------------
# Every time the "two people sitting/settling down together" veto (see
# _co_settled_recently below) stops a fall or altercation alert from firing,
# it's counted here instead of just silently vanishing. A false-positive fix
# you can't verify is happening is not a fix you can trust - this gives the
# admin dashboard something concrete to check ("suppressions are climbing on
# CAM-X, the veto is doing real work") rather than just hoping the tuning is
# right.
ai_suppression_stats = {"fall_co_settle": 0, "altercation_co_settle": 0}

@app.get("/api/ai/suppressed_stats")
def get_suppressed_stats():
    return ai_suppression_stats

# ----------------- DATA MODELS -----------------
class ESP32Telemetry(BaseModel):
    # Data continuously sent by the ESP32 environmental module.
    # Ultrasonic distance is intentionally local-only and is not received here.
    water_raw: int
    rain_raw: int
    overall_state: Literal["NORMAL", "WARNING", "CRITICAL"]

class StatusUpdate(BaseModel):
    status: str
    resolution_media: Optional[str] = None

class DispatchUpdate(BaseModel):
    department: str
    collection: str

class CitizenReport(BaseModel):
    concern_type: str
    severity: str
    landmark: str
    details: str
    lat: Optional[float] = None
    lng: Optional[float] = None
    media_url: Optional[str] = None

class OfficialLogin(BaseModel):
    operator_id: str
    auth_key: str

# ----------------- IMAGE VERIFICATION -----------------
def decode_base64_image(base64_str: str) -> Optional[np.ndarray]:
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        img_bytes = base64.b64decode(base64_str)
        pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        return None

def check_image_quality(frame: np.ndarray) -> tuple[bool, str]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 20.0: return False, "Image too blurry."
    mean_brightness = np.mean(gray)
    if mean_brightness < 15.0: return False, "Photo is pitch dark."
    if mean_brightness > 245.0: return False, "Photo is completely overexposed."
    return True, "Quality OK"

def verify_semantic_relevance(frame: np.ndarray, category: str) -> tuple[bool, str, float]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]
    cat_lower = category.lower()

    if "water" in cat_lower or "flood" in cat_lower:
        water_mask = cv2.inRange(hsv, np.array([85, 30, 30]), np.array([135, 255, 255]))
        mud_mask = cv2.inRange(hsv, np.array([10, 40, 40]), np.array([30, 200, 200]))
        ratio = (cv2.countNonZero(water_mask) + cv2.countNonZero(mud_mask)) / (h * w)
        if ratio < 0.04: return False, "No standing water detected.", 15.0
        return True, "Water accumulation verified.", min(99.0, 70.0 + (ratio * 100))

    elif "fall" in cat_lower or "medical" in cat_lower:
        res = ai_pose_model(frame, imgsz=320, device='cpu', verbose=False)
        if not any(r.boxes is not None and len(r.boxes) > 0 for r in res):
            return False, "No human detected in submitted proof.", 10.0
        return True, "Human presence confirmed.", 92.0

    return True, "Visual features conform to incident report.", 85.0

def find_nearby_duplicate(lat: float, lng: float, concern_type: str, max_meters: float = 50.0):
    if not lat or not lng: return None
    rows = (supabase.table("reports")
            .select("id,lat,lng,status,citizen_upvotes,supplementary_evidence")
            .eq("concern_type", concern_type)
            .not_.is_("lat", "null").not_.is_("lng", "null")
            .order("id", desc=True).limit(500).execute().data) or []
    best, best_d = None, max_meters
    for r in rows:
        st = r.get("status") or ""
        if not (st in ("Pending", "Pending Review") or "DISPATCHED" in st):
            continue
        d = haversine_m(lat, lng, r["lat"], r["lng"])
        if d <= best_d:
            best, best_d = r, d
    return best

def dispatch_timeout_monitor():
    while True:
        try:
            now = time.time()
            stale_incidents = (supabase.table("incidents").select("id,status")
                               .like("status", "%DISPATCHED%")
                               .lt("dispatch_time", now - 45).execute().data) or []
            for incident in stale_incidents:
                depts = []
                if "POLICE" in incident["status"]: depts.append("POLICE BACKUP")
                if "EMS" in incident["status"]: depts.append("EMS BACKUP")
                if "CIVIC" in incident["status"]: depts.append("CIVIC BACKUP")
                dept = " + ".join(depts) if depts else "CENTRAL COMMAND"
                supabase.table("incidents").update(
                    {"status": f"🚨 RE-ROUTED TO {dept}", "dispatch_time": now}
                ).eq("id", incident["id"]).execute()
        except Exception as e:
            # Network hiccups are now possible (remote DB) - never let the monitor thread die.
            print(f"[MONITOR] Dispatch timeout check failed: {e}")
        time.sleep(10)

threading.Thread(target=dispatch_timeout_monitor, daemon=True).start()

# ----------------- ENDPOINTS -----------------
def _store_esp32_state(data: ESP32Telemetry):
    """Store the latest ESP32 environmental telemetry in the in-memory cache and trigger camera."""
    global hardware_state, last_hardware_incident_time
    hardware_state.update({
        "water_raw": data.water_raw,
        "rain_raw": data.rain_raw,
        "overall_state": data.overall_state,
        "last_update": time.time()
    })

    # --- NEW: Hardware-Triggered Camera Snapshot ---
    if data.overall_state in ["WARNING", "CRITICAL"]:
        current_time = time.time()
        
        # Only trigger a photo if 60 seconds have passed since the last hardware alert
        if current_time - last_hardware_incident_time > HARDWARE_INCIDENT_COOLDOWN_SEC:
            last_hardware_incident_time = current_time

            snapshot_b64 = None
            # 1. Grab the latest frame directly from the live camera memory
            with camera_state_lock:
                if latest_encoded_frame is not None:
                    import base64
                    b64_str = base64.b64encode(latest_encoded_frame).decode('utf-8')
                    snapshot_b64 = f"data:image/jpeg;base64,{b64_str}"

            # 2. If we got a photo, commit it to the database as a new incident
            if snapshot_b64:
                event_label = "Sensor Hazard: CRITICAL" if data.overall_state == "CRITICAL" else "Sensor Hazard: WARNING"
                conf_score = "99%" if data.overall_state == "CRITICAL" else "85%"
                
                # Run the commit process in the background so the ESP32 doesn't time out
                threading.Thread(
                    target=commit_incident_in_memory, 
                    args=(event_label, conf_score, [], [snapshot_b64]), 
                    daemon=True
                ).start()
                
                print(f"[HARDWARE TRIGGER] {data.overall_state} detected! Snapshot captured and incident published.")


@app.post("/api/hardware/telemetry")
def receive_hardware_telemetry(data: ESP32Telemetry):
    _store_esp32_state(data)
    return {"success": True}


@app.post("/api/update-state")
def update_state(data: ESP32Telemetry):
    """ESP32-compatible telemetry endpoint used by the current firmware."""
    _store_esp32_state(data)
    return {"success": True, "message": "ESP32 state received"}


def _hardware_snapshot():
    """Telemetry plus a server-computed age, so the dashboard never depends on
    the viewer's clock matching the server's clock."""
    snap = dict(hardware_state)
    last = snap.get("last_update") or 0
    snap["age_seconds"] = (time.time() - last) if last else None
    return snap

@app.get("/api/hardware/status")
def get_hardware_status():
    return _hardware_snapshot()

@app.get("/api/cameras")
def get_cameras():
    return CCTV_NODES

@app.get("/api/camera/diagnostics")
def camera_diagnostics():
    """Probes camera indices 0-4 with both DSHOW and default backends and reports
    which ones actually open AND return a real frame. Use this to find the
    right index before relying on the live /video_feed stream."""
    results = []
    candidates = [
        (0, cv2.CAP_DSHOW, "index 0 (DSHOW)"),
        (1, cv2.CAP_DSHOW, "index 1 (DSHOW)"),
        (0, cv2.CAP_ANY,   "index 0 (default backend)"),
        (1, cv2.CAP_ANY,   "index 1 (default backend)"),
        (2, cv2.CAP_ANY,   "index 2 (default backend)"),
        (3, cv2.CAP_ANY,   "index 3 (default backend)"),
        (4, cv2.CAP_ANY,   "index 4 (default backend)"),
    ]
    for idx, backend, label in candidates:
        entry = {"label": label, "index": idx, "opened": False, "frame_read": False, "resolution": None, "error": None}
        try:
            cam = cv2.VideoCapture(idx, backend)
            entry["opened"] = cam.isOpened()
            if entry["opened"]:
                ok, frame = cam.read()
                entry["frame_read"] = bool(ok and frame is not None)
                if entry["frame_read"]:
                    entry["resolution"] = f"{frame.shape[1]}x{frame.shape[0]}"
            cam.release()
        except Exception as e:
            entry["error"] = str(e)
        results.append(entry)
    working = [r for r in results if r["frame_read"]]
    return {"working_cameras": working, "all_attempts": results}

def _find_operator(table: str, clean_id: str):
    pattern = clean_id.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    rows = supabase.table(table).select("*").ilike("operator_id", pattern).limit(5).execute().data or []
    for r in rows:
        if str(r.get("operator_id", "")).lower() == clean_id.lower():
            return r
    return None

@app.post("/api/auth/login")
def unified_login(login_data: OfficialLogin):
    clean_id, clean_key = login_data.operator_id.strip(), login_data.auth_key.strip()
    admin = _find_operator("admins", clean_id)
    if admin and verify_auth_key(clean_key, admin.get("auth_key", "")):
        return {"success": True, "role": "admin"}
    responder = _find_operator("responders", clean_id)
    if responder and verify_auth_key(clean_key, responder.get("auth_key", "")):
        return {"success": True, "role": "responder", "department": (responder.get("department") or "POLICE").upper()}
    raise HTTPException(status_code=401, detail="INVALID CREDENTIALS")

@app.get("/api/incidents")
async  def get_incidents(limit: int = 200):
    # Unbounded queries here were the main cause of the admin dashboard
    # getting slower over time: every incident ever recorded was being sent
    # and re-sorted on every 4-second poll. The dashboard only ever shows the
    # most recent ones anyway, so cap it server-side.
    return supabase.table("incidents").select("*").order("timestamp", desc=True).limit(max(1, min(limit, 500))).execute().data or []

@app.get("/api/reports")
async def get_reports(limit: int = 200):
    """Fetches the latest citizen reports for the admin dashboard."""
    return supabase.table("reports").select("*").order("timestamp", desc=True).limit(max(1, min(limit, 500))).execute().data or []

@app.patch("/api/incidents/{incident_id}/status")
def update_incident_status(incident_id: int, update_data: StatusUpdate):
    update_doc = {"status": update_data.status}
    if update_data.resolution_media: update_doc["resolution_media"] = update_data.resolution_media
    supabase.table("incidents").update(update_doc).eq("id", incident_id).execute()
    return {"message": "Updated"}

@app.post("/api/reports")
def submit_report(report: CitizenReport):
    # NOTE: plain `def` (not `async def`) on purpose. This handler does blocking work
    # (OpenCV, YOLO, Supabase calls). Inside `async def` it froze the whole event loop,
    # so every other request (stats, video, ESP32) stalled while a photo was processed.
    try:
        return _submit_report_impl(report)
    except HTTPException:
        raise
    except Exception as e:
        # Return a proper JSON error: an unhandled 500 carries no CORS headers, which the
        # browser reports as a vague network failure instead of the real reason.
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server could not save the report: {e}")

def _submit_report_impl(report: CitizenReport):
    if report.lat is not None and not (-90.0 <= report.lat <= 90.0):
        raise HTTPException(status_code=422, detail="Invalid latitude.")
    if report.lng is not None and not (-180.0 <= report.lng <= 180.0):
        raise HTTPException(status_code=422, detail="Invalid longitude.")

    new_id = generate_id()
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if report.lat and report.lng:
        dup = find_nearby_duplicate(report.lat, report.lng, report.concern_type, max_meters=50.0)
        if dup:
            patch = {"citizen_upvotes": (dup.get("citizen_upvotes") or 1) + 1}
            
            supabase.table("reports").update(patch).eq("id", dup["id"]).execute()
            return {"success": True, "ref": f"JS-{str(dup['id'])[-4:]}", "merged": True, "message": "Incident merged."}

    is_valid, decline_reason, ai_conf = True, None, 85.0
    final_media_url = ""

    if report.media_url and report.media_url.startswith("data:image"):
        frame = decode_base64_image(report.media_url)
        if frame is not None:
            q_ok, q_msg = check_image_quality(frame)
            if not q_ok: is_valid, decline_reason = False, f"Image Discarded: {q_msg}"
            else:
                s_ok, s_msg, s_conf = verify_semantic_relevance(frame, report.concern_type)
                ai_conf = s_conf
                if not s_ok: is_valid, decline_reason = False, f"Relevance Mismatch: {s_msg}"
                else:
                    # PROPER STORAGE ROUTING: Save to Supabase Storage bucket instead of DB row
                    filename = f"report_{new_id}.jpg"
                    local_path = os.path.join(EVIDENCE_DIR, filename)
                    cv2.imwrite(local_path, frame)
                    cloud_url = upload_evidence_to_supabase(local_path, filename)
                    
                    if cloud_url:
                        final_media_url = cloud_url
                        try:
                            os.remove(local_path)
                        except OSError:
                            pass
                    else:
                        final_media_url = f"{PUBLIC_BASE_URL}/evidence/{filename}"
        else:
            is_valid, decline_reason = False, "Corrupted image payload."

    initial_status = "Pending Review" if is_valid else f"Auto-Declined: {decline_reason}"

    supabase.table("reports").insert({
       "id": new_id, "concern_type": report.concern_type, "severity": report.severity,
        "landmark": report.landmark, "details": report.details, "lat": report.lat, "lng": report.lng,
        "media_url": final_media_url, # Now saves a short, clean URL instead of a 14MB string
        "timestamp": timestamp_str, "status": initial_status,
        "ai_verified": is_valid, "ai_confidence": f"{int(ai_conf)}%", "rejection_reason": decline_reason if not is_valid else None,
        "citizen_upvotes": 1,
    }).execute() 
    
    return {"success": True, "ref": f"JS-{str(new_id)[-4:]}", "verified": is_valid, "status": initial_status}

@app.get("/api/reports/track/{ref_id}")
def track_report(ref_id: str):
    clean_id = ref_id.replace("JS-", "").strip()
    # Supabase caps a response at 1000 rows by default, so scan the newest 1000.
    for r in (supabase.table("reports").select("*").order("id", desc=True).limit(1000).execute().data or []):
        if str(r["id"]).endswith(clean_id):
            return {
                "success": True, "status": r.get("status", "Unknown"), "concern": r.get("concern_type", "Unknown"), 
                "date": r.get("timestamp", "Unknown"), "rejection_reason": r.get("rejection_reason"),
                "citizen_upvotes": r.get("citizen_upvotes", 1), "ai_confidence": r.get("ai_confidence", "N/A")
            }
    raise HTTPException(status_code=404, detail="Report not found")

@app.patch("/api/reports/{report_id}/status")
def update_report_status(report_id: int, update_data: StatusUpdate):
    update_doc = {"status": update_data.status}
    if update_data.resolution_media: update_doc["resolution_media"] = update_data.resolution_media
    supabase.table("reports").update(update_doc).eq("id", report_id).execute()
    return {"message": "Updated"}

@app.patch("/api/dispatch/{doc_id}")
def dispatch_official(doc_id: int, data: DispatchUpdate):
    table = "incidents" if data.collection == "incidents" else "reports"
    found = supabase.table(table).select("id").eq("id", doc_id).limit(1).execute().data
    if not found: raise HTTPException(status_code=404, detail="Record not found")
        
    dept_labels = {"police": "🚓 DISPATCHED (POLICE)", "ems": "🚑 DISPATCHED (EMS)", "civic": "🚜 DISPATCHED (CIVIC)"}
    status_txt = dept_labels.get(data.department.lower(), "🚨 DISPATCHED")
    
    supabase.table(table).update({"status": status_txt, "dispatch_time": time.time()}).eq("id", doc_id).execute()
    return {"message": "Dispatched", "status": status_txt}

@app.get("/api/stats")
async def get_stats():
    open_states = ["Pending", "Pending Review"]
    pending = _count("incidents", open_states) + _count("reports", open_states)
    solved = _count("incidents", open_states, exclude=True) + _count("reports", open_states, exclude=True)
    return {"pending_reviews": pending, "cases_solved": solved, "active_cctv_nodes": len(CCTV_NODES), "hardware_telemetry": _hardware_snapshot()}

@app.get("/")
def root():
    return {"status": "ok", "service": "Jan Suraksha Core API", "docs": "/docs", "health": "/api/health"}

@app.get("/api/health")
def health_check():
    """Quick liveness/readiness probe - useful for deployment/monitoring so a
    failed Supabase connection or a missing model file surfaces immediately
    instead of only showing up as a mysterious 500 later."""
    db_ok = True
    try:
        supabase.table("admins").select("id").limit(1).execute()
    except Exception:
        db_ok = False
    return {
        "status": "ok" if db_ok else "degraded",
        "database": "connected" if db_ok else "unreachable",
        "pose_model_loaded": ai_pose_model is not None,
    }

# ----------------- EVIDENCE STORAGE (Supabase Storage) -----------------
# Render's disk is wiped on every deploy/restart, so finished clips are uploaded to a
# PUBLIC Supabase Storage bucket and the permanent public URL is saved in the incident row.
# If the upload fails, the incident falls back to the (temporary) local /evidence link.
EVIDENCE_BUCKET = os.environ.get("EVIDENCE_BUCKET", "evidence")
EVIDENCE_CONTENT_TYPES = {"webm": "video/webm", "mp4": "video/mp4", "avi": "video/x-msvideo"}

def upload_evidence_to_supabase(local_path, filename):
    """Upload a clip to Supabase Storage. Returns its public URL, or None on failure."""
    try:
        ext = filename.rsplit(".", 1)[-1].lower()
        with open(local_path, "rb") as fh:
            data = fh.read()
        bucket = supabase.storage.from_(EVIDENCE_BUCKET)
        bucket.upload(
            path=filename,
            file=data,
            file_options={"content-type": EVIDENCE_CONTENT_TYPES.get(ext, "application/octet-stream"), "upsert": "true"},
        )
        return str(bucket.get_public_url(filename)).rstrip("?")
    except Exception as e:
        print(f"[EVIDENCE] Supabase Storage upload failed for {filename}: {e}")
        return None

# ----------------- BACKGROUND WEBM VIDEO COMMIT ENGINE -----------------
def commit_incident_in_memory(event_type, conf_score, captured_frames, snapshots):
    try:
        incident_id = generate_id()
        raw_num = "".join(filter(str.isdigit, str(conf_score)))
        conf_int = int(raw_num) if raw_num else 88

        assigned_status = "Pending"
        is_auto = False
        dispatch_timestamp = None
        name_lower = event_type.lower()

        # Harassment/possible-molestation labels NEVER auto-dispatch, no matter
        # how high the confidence turns out to be after future tuning - a human
        # must confirm before anything routes to responders. This is a hard
        # rule, not a threshold, precisely because a wrong auto-dispatch here
        # (false accusation) or a wrong auto-dismissal (missed real harassment)
        # both carry serious consequences.
        never_auto = ("harassment" in name_lower) or ("molestation" in name_lower)

        if conf_int >= 85 and not never_auto:
            is_auto = True
            dispatch_timestamp = time.time()
            is_medical = "medical" in name_lower or "fall" in name_lower or "unconscious" in name_lower
            is_police = "fight" in name_lower or "assault" in name_lower or "altercation" in name_lower
            # A person knocked down mid-fight (event_type "Assault - Person Down /
            # Medical Emergency") matches both keyword groups on purpose - it needs
            # both departments, not whichever the if/elif happened to check first.
            if is_medical and is_police: assigned_status = "🚓🚑 DISPATCHED (POLICE + EMS)"
            elif is_medical: assigned_status = "🚑 DISPATCHED (EMS)"
            elif "water" in name_lower or "tree" in name_lower: assigned_status = "🚜 DISPATCHED (CIVIC)"
            elif is_police: assigned_status = "🚓 DISPATCHED (POLICE)"

        video_url = ""
        if captured_frames:
            height, width, _ = captured_frames[0].shape
            # vp80/webm isn't guaranteed to be available on every OpenCV build -
            # if it silently fails to open, you get a 0-byte "proof" video for a
            # real incident and never know. Try a couple of codecs and actually
            # check isOpened() before trusting it.
            codec_candidates = [("vp80", "webm"), ("mp4v", "mp4"), ("MJPG", "avi")]
            out, video_filename = None, None
            for fourcc_name, ext in codec_candidates:
                candidate_filename = f"incident_{incident_id}.{ext}"
                candidate_path = os.path.join(EVIDENCE_DIR, candidate_filename)
                candidate_out = cv2.VideoWriter(candidate_path, cv2.VideoWriter_fourcc(*fourcc_name), 10.0, (width, height))
                if candidate_out.isOpened():
                    out, video_filename = candidate_out, candidate_filename
                    break
                candidate_out.release()

            if out is not None:
                for frame in captured_frames:
                    out.write(frame)
                out.release()
                local_path = os.path.join(EVIDENCE_DIR, video_filename)
                cloud_url = upload_evidence_to_supabase(local_path, video_filename)
                if cloud_url:
                    video_url = cloud_url
                    try:
                        os.remove(local_path)  # no need to keep a copy on the server
                    except OSError:
                        pass
                else:
                    video_url = f"{PUBLIC_BASE_URL}/evidence/{video_filename}"
            else:
                print(f"[EVIDENCE] No working video codec found - incident {incident_id} has snapshots only, no video.")

        supabase.table("incidents").insert({
            "id": incident_id, "event_type": event_type, "risk_score": f"{conf_int}%",
            "location": "CAM-SMART-ROAD-01 (Sector 3 Model, 208016)", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": assigned_status, "dispatch_time": dispatch_timestamp, "media_url": video_url,
            "snapshots": snapshots, "auto_routed": is_auto
        }).execute()
        print(f"✅ Successfully committed incident {incident_id} [{event_type}] to Supabase!")
        return {"id": incident_id, "media_url": video_url}
    except Exception as e:
        print(f"❌ Error committing incident to Supabase: {e}")
        return None

# ----------------- REAL-TIME AI & TRACKING -----------------
_pose_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov8n-pose.pt')
ai_pose_model = YOLO(_pose_path if os.path.isfile(_pose_path) else 'yolov8n-pose.pt')
print("[SYSTEM] Running on CPU Mode.")

# --- MULTI-OBJECT TRACKER (fall + altercation detection ONLY) ---
# TRACKER_TYPE=botsort (default) or bytetrack. Both ship with Ultralytics.
# A SEPARATE model instance is used for tracking on purpose: .track() keeps tracker
# state inside the model, so sharing ai_pose_model would let report-verification and
# ESP32 jobs (which call ai_pose_model on one-off images) corrupt the live track IDs.
# ai_pose_model above is left exactly as it was for those callers.
TRACKER_TYPE = os.environ.get("TRACKER_TYPE", "botsort").strip().lower()
if TRACKER_TYPE not in ("botsort", "bytetrack"):
    print(f"[TRACKER] Unknown TRACKER_TYPE '{TRACKER_TYPE}', using botsort.")
    TRACKER_TYPE = "botsort"
TRACKER_CONFIG = f"{TRACKER_TYPE}.yaml"
# Low detection floor on purpose: the tracker splits detections into high (>=0.25) and low
# (0.10-0.25) confidence. Low ones can only EXTEND an existing track (never start a new one),
# which keeps IDs alive through blur / occlusion / a person going horizontal mid-fall.
TRACK_DET_CONF = 0.10
ai_track_model = YOLO(_pose_path if os.path.isfile(_pose_path) else 'yolov8n-pose.pt')
print(f"[SYSTEM] Fall/altercation tracker: {TRACKER_TYPE}")

def reset_tracker_state():
    """Drop all tracker memory (called after the scene has been idle, so a stale lost
    track can never be re-matched to someone who walks in much later)."""
    try:
        predictor = getattr(ai_track_model, "predictor", None)
        for t in (getattr(predictor, "trackers", None) or []):
            t.reset()
    except Exception as e:
        print(f"[TRACKER] reset skipped: {e}")

person_kinematic_cache = {}
PRE_ROLL_SECONDS = 5
FRAME_RATE = 10.0
frame_buffer = deque(maxlen=int(PRE_ROLL_SECONDS * FRAME_RATE))

# --- SHARED CAMERA STATE (single producer, many consumers) ---
# The camera is read and processed by exactly ONE background thread for the
# whole life of the server. Every browser tab / page refresh just reads the
# latest already-processed frame from here - it never touches the camera
# device itself. This is what fixes the "keeps loading in a loop on
# refresh" bug: the old code opened a brand-new cv2.VideoCapture on every
# single connection to /video_feed, so refreshing the page raced the still-
# open handle from the previous load for exclusive access to the same
# physical camera, usually lost, and the frontend's own retry logic then
# hammered that same losing race every 3 seconds forever. It also means
# detection/recording now keeps running even when nobody has the dashboard
# open, instead of only while someone was actively watching the stream.
camera_state_lock = threading.Lock()
latest_encoded_frame = None
camera_worker_lock = threading.Lock()
camera_worker_started = False
last_live_frame_time = 0.0   # updated by the camera loop on every real frame

# --- ALTERCATION / HARASSMENT DETECTION STATE ---
# Keyed by a stable pair id (frozenset-like tuple of the two track ids), not by
# frame. This is what prevents the same ongoing fight from being reported
# over and over: an incident is only allowed to fire once per CONFIRMED
# episode, and the pair can't start a new one until they've actually
# separated for ALTERCATION_SEPARATION_SECONDS.
altercation_cache = {}
ALTERCATION_CONFIRM_CYCLES = 8           # ~2.7s of sustained signal (at ~3Hz analysis) before confirming - a bit longer than before, on purpose, to ride out brief spikes
ALTERCATION_SCORE_THRESHOLD = 0.65       # smoothed score needed to confirm an incident (raised from 0.55 - real streets have a lot of borderline motion; this trades a little detection speed for far fewer false alarms)
ALTERCATION_SEPARATION_SECONDS = 4.0    # how long the pair must stay apart before a NEW incident can start
ALTERCATION_STALE_SECONDS = 20.0        # purge pair state if not seen this long

# --- FALL CONFIRMATION WINDOW ---
# How long we'll wait, after a sudden-drop signature fires, for the person to
# actually settle and stay down before confirming. See the PENDING state
# machine below for why this has to be decoupled from the drop signature
# itself.
FALL_CONFIRM_WINDOW_SECONDS = 3.0

# How long a person who was just part of a confirmed altercation is still
# treated as "possibly knocked down mid-fight" for fall-detection purposes
# (see in_altercation handling in evaluate_fall_state_angle_proof) - a real
# fight-induced fall is often a shove/stumble, not a clean vertical drop, so
# it needs a slightly different (more forgiving) fall signature than a solo
# medical collapse.
FIGHT_CONTEXT_GRACE_SECONDS = 6.0

def apply_night_vision(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def _torso_angle_from_vertical(kpts, kconf, min_conf=0.15):
    """Angle (deg) of the shoulder-midpoint -> hip-midpoint line from vertical.
    ~0 = standing upright, ~90 = lying flat. Returns None if too few confident
    joints to compute it (perfectly fine - caller treats that as 'no opinion',
    never as 'not fallen')."""
    ls, rs, lh, rh = kpts[5], kpts[6], kpts[11], kpts[12]
    cls, crs, clh, crh = kconf[5], kconf[6], kconf[11], kconf[12]
    def ok(c): return c == -1 or c >= min_conf
    valid_sh = [p for p, c in [(ls, cls), (rs, crs)] if p[0] > 0 and ok(c)]
    valid_hip = [p for p, c in [(lh, clh), (rh, crh)] if p[0] > 0 and ok(c)]
    if not valid_sh or not valid_hip:
        return None
    sx = sum(p[0] for p in valid_sh) / len(valid_sh)
    sy = sum(p[1] for p in valid_sh) / len(valid_sh)
    hx = sum(p[0] for p in valid_hip) / len(valid_hip)
    hy = sum(p[1] for p in valid_hip) / len(valid_hip)
    dx, dy = hx - sx, hy - sy
    if dx == 0 and dy == 0:
        return None
    return math.degrees(math.atan2(abs(dx), abs(dy)))

# --- STRICT PHYSICS ENGINE (Skeleton-Verified) ---
def evaluate_fall_state_angle_proof(history, frame_height, current_w, current_h, torso_angle=None, in_altercation=False):
    if len(history) < 4: 
        return False, 0, "Normal", 0.0
    
    old_time, old_cx, old_cy, old_w, old_h = history[-4]
    curr_time, curr_cx, curr_cy, curr_w, curr_h = history[-1]
    
    dt = curr_time - old_time
    if dt <= 0: return False, 0, "Normal", 0.0

    drop_distance = curr_cy - old_cy
    
    old_area = old_w * old_h
    curr_area = curr_w * curr_h
    area_change_ratio = abs(curr_area - old_area) / float(max(1, old_area))
    aspect_ratio = current_w / float(max(1, current_h))

    is_fall = False
    conf = 0
    drop_ratio = max(0.0, drop_distance) / float(max(1, old_h))

    # A solo collapse is (almost) always a clean vertical drop - a person's
    # centre of mass simply falls straight down. A fight-induced fall is
    # usually diagonal: shoved sideways, stumbling, going down at an angle
    # rather than straight down. Requiring a purely-vertical drop_distance
    # (the check below) would miss most of those, so during/just after a
    # confirmed altercation we also compute a 2D (diagonal) displacement and
    # accept that as an alternative signature.
    if drop_distance < (frame_height * 0.04) and not in_altercation:
        return False, 0, "Normal", 0.0

    if drop_distance >= (frame_height * 0.04):
        vertical_velocity = drop_distance / dt

        if drop_ratio > 0.35 and vertical_velocity > (frame_height * 0.10):
            is_fall = True
            conf = min(99, int(75 + (drop_ratio * 50)))

        elif aspect_ratio > 1.5 and curr_cy > (frame_height * 0.45) and drop_ratio > 0.15:
            is_fall = True
            conf = min(99, int(80 + (aspect_ratio * 10)))

        elif area_change_ratio > 0.45 and drop_distance > (frame_height * 0.08) and aspect_ratio > 1.0:
            is_fall = True
            conf = min(99, int(70 + (area_change_ratio * 20)))

    if not is_fall and in_altercation:
        total_disp = math.hypot(curr_cx - old_cx, curr_cy - old_cy)
        total_disp_ratio = total_disp / float(max(1, old_h))
        total_velocity = total_disp / dt
        # Deliberately a bit more forgiving than the vertical-only path above
        # (fight footage is noisier and the drop is rarely clean), but capped
        # at a lower starting confidence for the same reason - this needs a
        # human glance either way.
        if total_disp_ratio > 0.40 and total_velocity > (frame_height * 0.12) and aspect_ratio > 0.9:
            is_fall = True
            drop_ratio = total_disp_ratio
            conf = min(90, int(55 + (total_disp_ratio * 40)))

    if is_fall:
        # Cross-check against actual body orientation when we have confident
        # keypoints for it - this is the real "skeleton-verified" step that was
        # missing before (the function only ever looked at the bounding box).
        # It boosts confidence when the torso is confirmed horizontal, and
        # pulls confidence down (never silently discards) when keypoints
        # confidently say the person is still upright - catches things like a
        # fast sit-down/crouch that fools the box math alone. Mid-fight,
        # keypoints are frequently occluded/noisy from grappling, so the
        # penalty for an "upright" reading is softer and the rejection floor
        # lower - we don't want a single bad occluded frame to erase a real
        # fight-induced fall.
        if torso_angle is not None:
            if torso_angle > 55:
                conf = min(99, conf + 10)
            elif torso_angle > 35:
                # Ambiguous band. The old code left 20-55 completely untouched,
                # which is exactly the gap that let a quick sit-down-and-slouch
                # (very common - curbs, steps, stools) sail through with
                # whatever confidence the box math alone produced. A modest
                # haircut reflects the genuine uncertainty here without
                # outright rejecting a real partial/side fall.
                conf = max(0, conf - 8)
            else:
                penalty = 10 if in_altercation else 25
                reject_floor = 35 if in_altercation else 50
                conf = max(0, conf - penalty)
                if conf < reject_floor:
                    return False, 0, "Normal", drop_ratio
        label = "Assault - Person Down / Medical Emergency" if in_altercation else "Sudden Fall / Medical Emergency"
        return True, conf, label, drop_ratio
        
    return False, 0, "Normal", drop_ratio

def frame_to_base64_jpeg(img):
    try:
        ret, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 85]) 
        return f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}" if ret else ""
    except Exception: return ""

# --- ALTERCATION / FIGHT / HARASSMENT DETECTION (heuristic, pose-based) ---
# IMPORTANT: this is a triage screen, not a verdict. It tells you "these two
# people are physically interacting in a way worth a human looking at", and
# gives its best guess at *what kind* of interaction - it does not, and
# cannot from pose alone, confirm intent, consent, or wrongdoing.

def pair_key(pid_a, pid_b):
    return tuple(sorted((pid_a, pid_b)))

def _iou_from_center(cx1, cy1, w1, h1, cx2, cy2, w2, h2):
    """IoU of two boxes given as (center, width, height) - used for tracking
    association, which is far more stable than centroid distance once two
    people are standing close together (fights, crowds, harassment)."""
    ax1, ay1, ax2, ay2 = cx1 - w1/2, cy1 - h1/2, cx1 + w1/2, cy1 + h1/2
    bx1, by1, bx2, by2 = cx2 - w2/2, cy2 - h2/2, cx2 + w2/2, cy2 + h2/2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0

def _motion_energy(history):
    """Average speed, 'jerk' (peak-to-trough speed swing), and a 'steadiness'
    ratio over the last few tracked points, all normalized by the person's
    on-screen height. Steadiness = net displacement / total path length: near
    0 means the box center wobbled back and forth without the person actually
    travelling anywhere (classic signature of gesturing with hands while
    standing still - raising an arm grows the box and shifts its center even
    though the body hasn't moved); near 1 means real, sustained movement in
    one direction (stepping in for a hit, being shoved, falling)."""
    pts = list(history)[-5:]
    if len(pts) < 3:
        return 0.0, 0.0, 0.0
    speeds = []
    path_length = 0.0
    for k in range(1, len(pts)):
        t0, x0, y0, _, h0 = pts[k-1]
        t1, x1, y1, _, h1 = pts[k]
        dt = max(0.01, t1 - t0)
        scale = max(20.0, (h0 + h1) / 2.0)
        step_dist = math.hypot(x1 - x0, y1 - y0)
        path_length += (step_dist / scale) * 100.0
        speeds.append((step_dist / dt) / scale * 100.0)
    avg_speed = sum(speeds) / len(speeds)
    jerk = (max(speeds) - min(speeds)) if len(speeds) > 1 else 0.0

    first_x, first_y, first_h = pts[0][1], pts[0][2], pts[0][4]
    last_x, last_y, last_h = pts[-1][1], pts[-1][2], pts[-1][4]
    scale0 = max(20.0, (first_h + last_h) / 2.0)
    net_disp = (math.hypot(last_x - first_x, last_y - first_y) / scale0) * 100.0
    steadiness = net_disp / max(1.0, path_length)
    return avg_speed, jerk, steadiness

def _heading_unit(history):
    """Unit direction vector of this person's own net travel over the recent
    window, or None if they haven't actually gone anywhere (standing still /
    just gesturing - no reliable heading to speak of)."""
    pts = list(history)[-5:]
    if len(pts) < 3:
        return None
    x0, y0, h0 = pts[0][1], pts[0][2], pts[0][4]
    x1, y1, h1 = pts[-1][1], pts[-1][2], pts[-1][4]
    scale = max(20.0, (h0 + h1) / 2.0)
    dx, dy = x1 - x0, y1 - y0
    if (math.hypot(dx, dy) / scale) < 0.08:
        return None
    mag = math.hypot(dx, dy)
    return (dx / mag, dy / mag)

def _shares_travel_heading(hist_a, hist_b):
    """Two people simply walking somewhere near each other - crossing paths,
    or walking together/alongside each other (companions, family, a queue
    shuffling forward) - produce the exact same raw signature as the start of
    a scuffle for a moment: close together, real (non-jittery) combined
    motion. What actually tells them apart is heading. A crossing pair each
    keep travelling their own original, opposed, straight-line path; a pair
    walking together keep travelling the SAME straight-line path. Either way,
    neither is oriented toward and reacting to the other. A fight or
    harassment pair's motion stays centred ON the other person (closing in,
    circling, push-then-recoil, retreating and closing again) - not a stable
    heading correlation in either direction. This matters especially in
    Indian street/footpath/underpass footage, where routine crossings and
    people walking in groups happen well within arm's length of each other."""
    ha, hb = _heading_unit(hist_a), _heading_unit(hist_b)
    if ha is None or hb is None:
        return False
    dot = ha[0] * hb[0] + ha[1] * hb[1]
    return abs(dot) > 0.45

CO_SETTLE_WINDOW_SECONDS = 4.0  # how close together in time two people's "just went still" moments have to be to count as sitting down together, not one collapsing near an unrelated bystander

def _co_settled_recently(data_a, data_b, now, window=CO_SETTLE_WINDOW_SECONDS):
    """True if BOTH people independently transitioned from moving to still
    within the last `window` seconds of each other. This is the actual
    signature that separates "two people sitting/lying down together" from
    "one person collapsed near someone who happens to be standing/sitting
    nearby": a real bystander was ALREADY still before the other person
    arrived or fell (their own settle timestamp is old or absent), whereas
    two companions who walk up and sit down produce two fresh, closely-timed
    settle events. Doesn't fire for a bystander who's been standing there for
    the last two minutes, and doesn't fire for a fall next to someone who is
    currently moving/reacting (no recent settle timestamp for them either)."""
    ts_a = data_a.get("just_settled_at")
    ts_b = data_b.get("just_settled_at")
    if ts_a is None or ts_b is None:
        return False
    return (now - ts_a) <= window and (now - ts_b) <= window and abs(ts_a - ts_b) <= window

def _count_bystanders(active_people, pair_pids, mid_x, mid_y, radius):
    count = 0
    for pid, cx, cy, bw, bh in active_people:
        if pid in pair_pids: continue
        if math.hypot(cx - mid_x, cy - mid_y) <= radius:
            count += 1
    return count

def analyze_altercation_pairs(active_people, kinematic_cache, now):
    """
    Scores every pair of currently-tracked people close enough to be
    physically interacting. Returns [(pair_id, raw_score_0_1, signals), ...].
    """
    findings = []
    n = len(active_people)
    for i in range(n):
        pid_a, cx_a, cy_a, bw_a, bh_a = active_people[i]
        for j in range(i + 1, n):
            pid_b, cx_b, cy_b, bw_b, bh_b = active_people[j]

            dist = math.hypot(cx_a - cx_b, cy_a - cy_b)
            contact_radius = 0.75 * ((max(bw_a, bh_a) + max(bw_b, bh_b)) / 2.0)
            if dist > contact_radius:
                continue  # not close enough to be interacting

            data_a = kinematic_cache.get(pid_a, {})
            data_b = kinematic_cache.get(pid_b, {})
            hist_a = data_a.get("history")
            hist_b = data_b.get("history")
            if not hist_a or not hist_b or len(hist_a) < 3 or len(hist_b) < 3:
                continue

            speed_a, jerk_a, steady_a = _motion_energy(hist_a)
            speed_b, jerk_b, steady_b = _motion_energy(hist_b)
            total_energy = speed_a + speed_b
            peak_jerk = max(jerk_a, jerk_b)
            peak_steadiness = max(steady_a, steady_b)
            asymmetry = abs(speed_a - speed_b) / max(1.0, total_energy)

            raw_score = 0.0
            # Thresholds are in "% of body height moved per second" now (see
            # _motion_energy) so they hold up regardless of how close/far the
            # person is from the camera. The steadiness gate is what filters
            # out two people standing and talking (lots of jerk from hand
            # gestures shifting the box, but no real net movement) - without
            # it, an animated conversation reads exactly like a scuffle.
            if total_energy > 25 and peak_jerk > 35 and peak_steadiness > 0.35:
                raw_score = min(1.0, (total_energy / 220.0) + (peak_jerk / 180.0))

                if _shares_travel_heading(hist_a, hist_b):
                    # Both are travelling a stable heading (same direction, or
                    # squarely opposite) rather than reacting to one another -
                    # a crossing or a pair walking together, not contact. Zero
                    # it out rather than dampening it; this is a categorical
                    # difference, not a matter of degree.
                    raw_score = 0.0
                elif _co_settled_recently(data_a, data_b, now):
                    # Both people independently went still within the same short
                    # window (see _co_settled_recently) - the signature of sitting
                    # or lying down together, not one person reacting to/attacking
                    # the other. Zeroed for the same "categorical, not graded"
                    # reason as the shared-heading case above.
                    raw_score = 0.0
                    ai_suppression_stats["altercation_co_settle"] += 1
                else:
                    # Graded by how deep INTO contact_radius they actually
                    # are. A graze at the outer edge (typical of two people
                    # squeezing past each other in a narrow space) should
                    # count for a lot less than genuine close/overlapping
                    # contact - not zero (sustained edge-of-radius contact
                    # over many cycles can still be real), just discounted.
                    proximity_factor = max(0.0, 1.0 - (dist / max(1.0, contact_radius)))
                    raw_score *= (0.5 + 0.5 * proximity_factor)

            findings.append((
                pair_key(pid_a, pid_b), raw_score,
                {"dist": dist, "contact_radius": contact_radius,
                 "speed_a": speed_a, "speed_b": speed_b, "asymmetry": asymmetry,
                 "jerk": peak_jerk, "total_energy": total_energy, "steadiness": peak_steadiness,
                 "mid_x": (cx_a + cx_b) / 2.0, "mid_y": (cy_a + cy_b) / 2.0}
            ))
    return findings

def update_altercation_state(pair_id, raw_score, now):
    """
    Per-pair state machine: MONITORING -> CONFIRMED -> (back to MONITORING
    only once the pair has been apart/inactive for ALTERCATION_SEPARATION_SECONDS).
    Returns one of "MONITORING", "NEW_INCIDENT", "ONGOING", "ENDED".
    A caller only ever gets ONE "NEW_INCIDENT" per real episode, however long
    that episode lasts - that's the fix for repeat-recording.
    """
    entry = altercation_cache.get(pair_id)
    fresh_contact = entry is None or len(entry["score_history"]) == 0
    if entry is None:
        entry = {"score_history": deque(maxlen=ALTERCATION_CONFIRM_CYCLES),
                  "state": "MONITORING", "low_since": None, "contact_since": now}
        altercation_cache[pair_id] = entry
    if fresh_contact:
        entry["contact_since"] = now

    entry["last_seen"] = now
    entry["score_history"].append(raw_score)
    smoothed = sum(entry["score_history"]) / len(entry["score_history"])

    if entry["state"] == "MONITORING":
        full_window = len(entry["score_history"]) == entry["score_history"].maxlen
        if full_window and smoothed >= ALTERCATION_SCORE_THRESHOLD:
            entry["state"] = "CONFIRMED"
            entry["low_since"] = None
            return "NEW_INCIDENT", smoothed
        return "MONITORING", smoothed

    # state == CONFIRMED: never re-fire, just track whether it's ended
    if smoothed < ALTERCATION_SCORE_THRESHOLD * 0.4:
        entry["low_since"] = entry["low_since"] or now
        if now - entry["low_since"] > ALTERCATION_SEPARATION_SECONDS:
            entry["state"] = "MONITORING"
            entry["score_history"].clear()
            entry["low_since"] = None
            return "ENDED", smoothed
    else:
        entry["low_since"] = None
    return "ONGOING", smoothed

def classify_altercation(signals, bystanders, contact_seconds):
    """
    Mutually-exclusive branches -> exactly one label, one confidence band.
    This is the fix for 'confused naming' and 'wrong confidence': the label
    can't waffle between calls, and confidence reflects how much the
    evidence actually supports (harassment is deliberately capped low -
    see the commit-time guard that refuses to auto-dispatch it regardless).
    """
    asymmetry = signals["asymmetry"]
    total_energy = signals["total_energy"]

    if asymmetry > 0.55 and total_energy > 70 and signals["jerk"] > 50:
        # One person doing almost all the hitting/lunging, the other mostly
        # static or retreating, AND a genuine sharp jerk (not just a
        # consistently higher average) - a one-sided pattern.
        return "Serious Fight / Possible Assault", 78

    if total_energy < 25 and contact_seconds > 5.0:
        # Little exchanged motion but sustained close-range contact - more
        # consistent with grabbing/restraint than mutual fighting. The much
        # more common false trigger here - two people who simply sat/lay down
        # together and are now calmly still - is filtered upstream in
        # analyze_altercation_pairs (_co_settled_recently) before the pair
        # ever reaches CONFIRMED, so a pair only gets here after genuinely
        # elevated, sustained motion energy first pushed them into this
        # state. Confidence stays capped low on purpose either way: enough to
        # flag for a human, not to accuse.
        return "Sustained Close Contact - Possible Harassment (Needs Review)", 32

    if bystanders >= 2 and asymmetry < 0.3 and total_energy < 150:
        # Roughly equal energy on both sides, a small ring of onlookers, no
        # clear one-sided victim - most consistent with boys sparring/showing
        # off rather than a real attack. Still logged, just low priority.
        return "Playful Scuffle (Low Priority)", 40

    return "Physical Altercation - Needs Review", 55


def make_status_frame(text_lines):
    """Render a plain 640x480 frame with status text, so the browser always gets
    something to show instead of an instantly-dead stream."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    y = 200
    for line in text_lines:
        cv2.putText(img, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y += 30
    ret, buf = cv2.imencode('.jpg', img)
    return buf.tobytes() if ret else b""

# Optional: set VIDEO_SOURCE to a video file path or stream URL (RTSP/HTTP) to run detection without
# a physical camera - needed on Render, which has no camera. A local file loops forever.
VIDEO_SOURCE = os.environ.get("VIDEO_SOURCE", "").strip()

def open_camera():
    """Try a range of indices/backends and report exactly what was tried."""
    attempts = []
    if VIDEO_SOURCE:
        src = int(VIDEO_SOURCE) if VIDEO_SOURCE.isdigit() else VIDEO_SOURCE
        cam = cv2.VideoCapture(src)
        if cam.isOpened():
            ok, frame = cam.read()
            if ok and frame is not None:
                if os.path.isfile(VIDEO_SOURCE):
                    cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
                print(f"[CAMERA] Using VIDEO_SOURCE={VIDEO_SOURCE}")
                return cam
        cam.release()
        print(f"[CAMERA] VIDEO_SOURCE={VIDEO_SOURCE} could not be opened, falling back to local cameras.")
    # CAP_DSHOW only exists/works on Windows; on Linux/Mac these attempts fail fast and harmlessly.
    candidates = [
        (0, cv2.CAP_DSHOW, "index 0 (DSHOW)"),
        (1, cv2.CAP_DSHOW, "index 1 (DSHOW)"),
        (0, cv2.CAP_ANY,   "index 0 (default backend)"),
        (1, cv2.CAP_ANY,   "index 1 (default backend)"),
        (2, cv2.CAP_ANY,   "index 2 (default backend)"),
    ]
    for idx, backend, label in candidates:
        cam = cv2.VideoCapture(idx, backend)
        opened = cam.isOpened()
        attempts.append(f"{label}: {'OK' if opened else 'failed'}")
        if opened:
            # Confirm we can actually pull a frame, not just that the handle opened
            ok, frame = cam.read()
            if ok and frame is not None:
                print(f"[CAMERA] Connected via {label}. Frame shape: {frame.shape}")
                for a in attempts:
                    print(f"[CAMERA]   attempt -> {a}")
                return cam
            else:
                print(f"[CAMERA] {label} opened but returned no frame on read(). Releasing and trying next.")
                cam.release()
    print("[SYSTEM ERROR] OpenCV cannot open any video stream device. Attempts:")
    for a in attempts:
        print(f"[SYSTEM ERROR]   {a}")
    return None

def camera_processing_loop():
    """The single owner of the physical camera for the whole process
    lifetime. Runs forever in a background thread; never yields to an HTTP
    client directly (see mjpeg_stream_generator for that)."""
    global frame_buffer, latest_encoded_frame, last_live_frame_time

    camera = open_camera()
    while camera is None:
        with camera_state_lock:
            latest_encoded_frame = make_status_frame(["NO CAMERA DETECTED", "Check USB connection / index", "On a server: set VIDEO_SOURCE env var", "Retrying every 5s..."])
        time.sleep(5.0)
        camera = open_camera()

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    t_start = time.time()
    # NOTE: fall alerts now use per-person state (data["fall_state"] in
    # person_kinematic_cache) instead of a global cooldown - see the fall
    # trigger loop below.
    
    prev_water_ratio = 0.0
    is_recording, recording_start = False, 0
    recorded_frames, active_event_type, active_conf_str = [], "", ""
    snap_1, snap_2, snap_3 = None, None, None
    frame_count = 0
    
    cached_humans, cached_kpts, cached_kconfs = [], [], []
    cached_fall, cached_water = None, None
    cached_altercation = None
    prev_gray_for_motion = None
    MOTION_SKIP_THRESHOLD = 2.0   # mean pixel diff (0-255) below which we call the scene static

    is_alt_recording, alt_recording_start = False, 0
    alt_recorded_frames, active_alt_label, active_alt_conf_str = [], "", ""
    alt_snap_1 = None
    
    tracked_incident_id = None
    abort_msg_until = 0
    consecutive_read_failures = 0
    tracker_was_reset = False

    while True:
        success, raw_frame = camera.read()
        if not success or raw_frame is None:
            if VIDEO_SOURCE and os.path.isfile(VIDEO_SOURCE):
                camera.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop the demo video
                time.sleep(0.05)
                continue
            consecutive_read_failures += 1
            if consecutive_read_failures == 1 or consecutive_read_failures % 50 == 0:
                print(f"[CAMERA] read() failed ({consecutive_read_failures} consecutive failures) — is the USB camera still connected?")
            if consecutive_read_failures > 150:  # ~7-8s of continuous failure
                print("[CAMERA] Camera appears disconnected. Attempting to reconnect...")
                with camera_state_lock:
                    latest_encoded_frame = make_status_frame(["CAMERA DISCONNECTED", "Attempting to reconnect...", "Reconnect USB and re-sync"])
                camera.release()
                camera = open_camera()
                while camera is None:
                    time.sleep(5.0)
                    camera = open_camera()
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                consecutive_read_failures = 0
                print("[CAMERA] Reconnected successfully.")
                continue
            time.sleep(0.05)
            continue
        consecutive_read_failures = 0

        try:
            frame = cv2.resize(raw_frame, (640, 480))
        except Exception:
            print("[CAMERA] Failed to resize captured frame:")
            traceback.print_exc()
            continue
        
        if np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) < 30: 
            frame = apply_night_vision(frame)
            cv2.putText(frame, "NIGHT VISION ACTIVE", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        h, w = frame.shape[:2]
        now = time.time()
        frame_buffer.append(frame.copy())
        last_live_frame_time = now
        frame_count += 1

        if now < abort_msg_until:
            cv2.putText(frame, "SUBJECT RECOVERED - ALARM ABORTED", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Motion gate: only bother running the (expensive) pose model when
        # either something is already being tracked, or the frame actually
        # changed since last time. On a quiet street camera the scene is
        # static almost all the time, so this is the single biggest
        # efficiency win available without touching detection quality - it
        # never skips while anyone is actively tracked, moving or not.
        gray_now = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scene_is_idle = False
        if prev_gray_for_motion is not None and len(person_kinematic_cache) == 0 and not is_recording and not is_alt_recording:
            motion_score = float(np.mean(cv2.absdiff(gray_now, prev_gray_for_motion)))
            scene_is_idle = motion_score < MOTION_SKIP_THRESHOLD
        prev_gray_for_motion = gray_now

        if scene_is_idle:
            if not tracker_was_reset:
                reset_tracker_state()
                tracker_was_reset = True
        else:
            tracker_was_reset = False

        if frame_count % 3 == 0 and not scene_is_idle:
            try:
                cached_humans.clear(); cached_kpts.clear(); cached_kconfs.clear()
                cached_fall = None; cached_water = None; cached_altercation = None

                results = ai_track_model.track(frame, persist=True, tracker=TRACKER_CONFIG, conf=TRACK_DET_CONF, imgsz=320, device='cpu', verbose=False)
                detected_pids = set()
                active_people = []
                pid_to_pose = {}

                for r in results:
                    if r.boxes is None or r.boxes.id is None: continue   # no confirmed tracks this cycle
                    boxes = r.boxes.xyxy.cpu().numpy()
                    track_ids = r.boxes.id.int().cpu().tolist()
                    # Keypoints may be sparse/low-confidence exactly when someone has fallen
                    # (occluded limbs, horizontal body, odd viewing angle). Never let missing
                    # keypoints throw away the detection itself - the fall math below only
                    # needs the bounding box. Keypoints are used purely for the optional
                    # skeleton overlay, guarded by their own per-joint confidence.
                    has_kpts = r.keypoints is not None and r.keypoints.xy is not None
                    kpts_xy = r.keypoints.xy.cpu().numpy() if has_kpts else None
                    has_kconf = has_kpts and r.keypoints.conf is not None
                    kpts_conf = r.keypoints.conf.cpu().numpy() if has_kconf else None

                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box[:4])
                        box_width = max(1, x2 - x1)
                        box_height = max(1, y2 - y1)
                    
                        if box_width < 40 and box_height < 40: continue 

                        this_kpts = kpts_xy[i] if kpts_xy is not None else np.zeros((17, 2))
                        this_kconf = kpts_conf[i] if kpts_conf is not None else np.full(17, -1.0)

                        cached_humans.append((x1, y1, x2, y2))
                        cached_kpts.append(this_kpts)
                        cached_kconfs.append(this_kconf)
                    
                        cx = (x1 + x2) / 2.0
                        cy = (y1 + y2) / 2.0

                        # Identity now comes from BoT-SORT / ByteTrack (Kalman prediction +
                        # one-to-one Hungarian matching + low-confidence recovery) instead of
                        # the old greedy per-box IoU matcher.
                        track_id = f"t{track_ids[i]}"
                        if track_id not in person_kinematic_cache:
                            person_kinematic_cache[track_id] = {
                                "history": deque(maxlen=20),
                                "stationary_time": 0.0
                            }
                        detected_pids.add(track_id)
                        active_people.append((track_id, cx, cy, box_width, box_height))
                        pid_to_pose[track_id] = (this_kpts, this_kconf)
                        prev = person_kinematic_cache[track_id]

                        # Exponential moving average on position damps single-frame detection
                        # jitter (a few pixels of box noise) so it doesn't get mistaken for real
                        # velocity/jerk by the fall and altercation math further down. Still
                        # reacts fast enough (alpha=0.6) to catch a genuine sudden fall/strike.
                        EMA_ALPHA = 0.6
                        if "smoothed_cx" in prev:
                            sm_cx = EMA_ALPHA * cx + (1 - EMA_ALPHA) * prev["smoothed_cx"]
                            sm_cy = EMA_ALPHA * cy + (1 - EMA_ALPHA) * prev["smoothed_cy"]
                        else:
                            sm_cx, sm_cy = cx, cy
                        prev["smoothed_cx"], prev["smoothed_cy"] = sm_cx, sm_cy
                        prev["history"].append((now, sm_cx, sm_cy, box_width, box_height))

                        if len(prev["history"]) > 1:
                            movement = math.hypot(cx - prev["history"][-2][1], cy - prev["history"][-2][2])
                        else:
                            movement = 100.0

                        if movement < (h * 0.05):
                            # The instant stationary_time goes from 0 -> nonzero is the
                            # instant this specific person just stopped moving. Recording
                            # it (once, on the transition - not every still frame after)
                            # is what lets _co_settled_recently tell "two people who just
                            # sat down together" apart from "one person still from minutes
                            # ago". Re-arms itself: goes back to None the moment they move
                            # again, so a later, unrelated stillness doesn't stay "recent".
                            if prev["stationary_time"] == 0.0:
                                prev["just_settled_at"] = now
                            prev["stationary_time"] += max(0.01, now - prev["history"][-2][0])
                        else:
                            prev["stationary_time"] = 0.0
                            prev["just_settled_at"] = None

                for pid, data in person_kinematic_cache.items():
                    if pid not in detected_pids and len(data["history"]) > 0:
                        last_time, l_cx, l_cy, l_w, l_h = data["history"][-1]
                        data["history"].append((now, l_cx, l_cy, l_w, l_h))
                        data["stationary_time"] += max(0.01, now - last_time)

                    if len(data["history"]) > 5:
                        last_record = data["history"][-1]
                        torso_angle = None
                        pose_for_pid = pid_to_pose.get(pid)
                        if pose_for_pid is not None:
                            torso_angle = _torso_angle_from_vertical(pose_for_pid[0], pose_for_pid[1])
                        in_alt = data.get("in_altercation_until", 0) > now
                        is_fall, fall_prob, fall_lbl, drop_ratio = evaluate_fall_state_angle_proof(data["history"], h, last_record[3], last_record[4], torso_angle, in_alt)

                        # --- Fall state machine: MONITORING -> PENDING -> ALERTED ---
                        # PENDING is what fixes the missed-fall bug. is_fall (the sudden-drop
                        # signature) can only be True WHILE the body is still moving fast; the
                        # old code also demanded stationary_time > 0.8s on that exact same
                        # frame, which only becomes true AFTER the body has stopped moving -
                        # by which point the recent history shows no more drop and is_fall has
                        # already flipped back to False. The two conditions almost never held
                        # at once, so real falls were detected and un-detected a frame later.
                        # Splitting it in two removes that race: the drop signature only ARMS a
                        # short pending window, and confirmation is checked independently on
                        # every subsequent frame against fresh stillness data, not against the
                        # instant of impact.
                        fall_state = data.get("fall_state", "MONITORING")

                        if is_fall and fall_state == "MONITORING":
                            fall_state = data["fall_state"] = "PENDING"
                            data["fall_impact_time"] = now
                            data["fall_prob_at_impact"] = fall_prob
                            data["fall_lbl_at_impact"] = fall_lbl
                            data["fall_box_at_impact"] = (last_record[1], last_record[2], last_record[3], last_record[4])

                        if fall_state == "PENDING":
                            movement_now = 0.0
                            if len(data["history"]) > 1:
                                prev_pt = data["history"][-2]
                                movement_now = math.hypot(last_record[1] - prev_pt[1], last_record[2] - prev_pt[2])
                            person_is_still_now = movement_now < (h * 0.05)

                            if person_is_still_now and data["stationary_time"] > 0.8:
                                # Confirmed IMPACT + STILLNESS - but that alone is also exactly
                                # what "sat down on a step/curb/stool and stayed seated" looks
                                # like, which is extremely common street behaviour, not an
                                # emergency. Re-check the settled pose itself, right now, before
                                # actually confirming: a real fall/collapse leaves the person
                                # sprawled - either a wide, flattened silhouette or (when
                                # keypoints are confident) a near-horizontal torso. A normal seated
                                # or crouched posture is neither. This is the check that was
                                # missing before: the old code confirmed off the drop+stillness
                                # alone and never looked at what the final pose actually was.
                                settle_aspect_ratio = last_record[3] / float(max(1, last_record[4]))
                                looks_fallen_now = (torso_angle is not None and torso_angle > 45) or settle_aspect_ratio > 1.15

                                # Even a genuinely sprawled/horizontal settle can still be two
                                # people sitting or lying down together rather than a solo
                                # collapse - check for a companion who ALSO just, independently,
                                # stopped moving at basically the same moment nearby (see
                                # _co_settled_recently). A pre-existing bystander who was already
                                # standing/sitting there doesn't have a fresh "just settled"
                                # timestamp, so this only catches genuine synchronized
                                # co-settling, not every fall that happens to have someone near.
                                has_co_settling_companion = False
                                if looks_fallen_now:
                                    co_settle_radius = 1.5 * max(last_record[3], last_record[4])
                                    for other_pid, ocx, ocy, _obw, _obh in active_people:
                                        if other_pid == pid:
                                            continue
                                        if math.hypot(ocx - last_record[1], ocy - last_record[2]) > co_settle_radius:
                                            continue
                                        if _co_settled_recently(data, person_kinematic_cache.get(other_pid, {}), now):
                                            has_co_settling_companion = True
                                            break

                                if looks_fallen_now and has_co_settling_companion:
                                    print(f"[AI PROCESSING] Suppressed fall alert for {pid} - synchronized co-settle with a nearby person (likely sitting/lying down together, not a collapse).")
                                    ai_suppression_stats["fall_co_settle"] += 1
                                    data["fall_state"] = "MONITORING"
                                elif looks_fallen_now:
                                    cx, cy, bw, bh = data["fall_box_at_impact"]
                                    fx1, fy1 = int(cx - bw/2), int(cy - bh/2)
                                    fx2, fy2 = int(cx + bw/2), int(cy + bh/2)
                                    cached_fall = (fx1, fy1, fx2, fy2, data["fall_prob_at_impact"])
                                    data["fall_state"] = "ALERTED"
                                    data["fall_cleared_since"] = None
                                    if not is_recording:
                                        is_recording, active_event_type, active_conf_str, recording_start = True, data["fall_lbl_at_impact"], f"{data['fall_prob_at_impact']}%", now
                                        tracked_incident_id = pid
                                        recorded_frames = list(frame_buffer)
                                        snap_1, snap_2, snap_3 = frame_to_base64_jpeg(frame), None, None
                                else:
                                    # Dropped and went still, but the settled pose is upright/
                                    # narrow - almost certainly just sat or crouched down
                                    # normally. Not a fall; back to monitoring.
                                    data["fall_state"] = "MONITORING"
                            elif now - data.get("fall_impact_time", now) > FALL_CONFIRM_WINDOW_SECONDS:
                                # Never settled within the grace window - got up quickly, bent
                                # over, a brief stumble. False alarm; back to normal.
                                data["fall_state"] = "MONITORING"

                        elif fall_state == "ALERTED":
                            cx, cy, bw, bh = data["fall_box_at_impact"]
                            fx1, fy1 = int(cx - bw/2), int(cy - bh/2)
                            fx2, fy2 = int(cx + bw/2), int(cy + bh/2)
                            cached_fall = (fx1, fy1, fx2, fy2, data["fall_prob_at_impact"])
                            # Only clear once confirmed NOT fallen for a few seconds straight
                            # (got up / was helped up) - stops a single flickering frame from
                            # instantly re-arming.
                            if not is_fall:
                                data["fall_cleared_since"] = data.get("fall_cleared_since") or now
                                if now - data["fall_cleared_since"] > 5.0:
                                    data["fall_state"] = "MONITORING"
                                    data["fall_cleared_since"] = None
                            else:
                                data["fall_cleared_since"] = None

                stale_ids = [pid for pid, data in person_kinematic_cache.items() if len(data["history"]) > 0 and now - data["history"][-1][0] > 3.0]
                for pid in stale_ids: del person_kinematic_cache[pid]

                # --- Fight / harassment screening pass ---
                if len(active_people) >= 2:
                    for pair_id, raw_score, signals in analyze_altercation_pairs(active_people, person_kinematic_cache, now):
                        pid_a, pid_b = pair_id
                        # A sudden fall produces exactly the signature this screen looks for
                        # (one person moving fast and hard, the other static nearby) - so
                        # anyone the fall detector has already confirmed as fallen is excluded
                        # here entirely. Without this, a solo fall near a bystander (or two
                        # strangers just standing near each other when one collapses) reads as
                        # a one-sided assault.
                        if person_kinematic_cache.get(pid_a, {}).get("fall_state") == "ALERTED" or \
                           person_kinematic_cache.get(pid_b, {}).get("fall_state") == "ALERTED":
                            continue

                        event, smoothed = update_altercation_state(pair_id, raw_score, now)
                        if event in ("NEW_INCIDENT", "ONGOING"):
                            entry = altercation_cache[pair_id]
                            contact_seconds = now - entry["contact_since"]
                            bystanders = _count_bystanders(active_people, set(pair_id), signals["mid_x"], signals["mid_y"], signals["contact_radius"] * 3.0)
                            label, conf = classify_altercation(signals, bystanders, contact_seconds)
                            cached_altercation = (int(signals["mid_x"]), int(signals["mid_y"]), label, conf)

                            # Tag both participants as "possibly about to be knocked down mid-
                            # fight" for a short grace window - the fall detector reads this to
                            # use a more forgiving, non-purely-vertical fall signature (see
                            # evaluate_fall_state_angle_proof / in_altercation below), since a
                            # shove-and-fall during a fight rarely looks like a clean collapse.
                            fight_grace = now + FIGHT_CONTEXT_GRACE_SECONDS
                            if pid_a in person_kinematic_cache: person_kinematic_cache[pid_a]["in_altercation_until"] = fight_grace
                            if pid_b in person_kinematic_cache: person_kinematic_cache[pid_b]["in_altercation_until"] = fight_grace

                            if event == "NEW_INCIDENT" and not is_alt_recording:
                                is_alt_recording = True
                                alt_recording_start = now
                                active_alt_label, active_alt_conf_str = label, f"{conf}%"
                                alt_recorded_frames = list(frame_buffer)
                                alt_snap_1 = frame_to_base64_jpeg(frame)

                # A pair that has separated beyond contact_radius entirely stops appearing
                # in analyze_altercation_pairs, so update_altercation_state's own score-decay
                # path (which is what normally clears CONFIRMED -> MONITORING) never runs for
                # them again. Left alone, a CONFIRMED entry would then sit locked until the
                # full ALTERCATION_STALE_SECONDS (20s) purge below - which meant the same two
                # people re-engaging within 20s of clearly separating couldn't trigger a new
                # incident at all. Time this out on the much shorter separation window instead.
                for pid, entry in altercation_cache.items():
                    if entry.get("state") == "CONFIRMED" and now - entry.get("last_seen", 0) > ALTERCATION_SEPARATION_SECONDS:
                        entry["state"] = "MONITORING"
                        entry["score_history"].clear()
                        entry["low_since"] = None

                stale_pairs = [pid for pid, data in altercation_cache.items() if now - data.get("last_seen", 0) > ALTERCATION_STALE_SECONDS]
                for pid in stale_pairs: del altercation_cache[pid]
            except Exception:
                print("[AI PROCESSING] Exception during pose/fall analysis this cycle — skipping, stream continues:")
                traceback.print_exc()

        if cached_fall:
            fx1, fy1, fx2, fy2, fprob = cached_fall
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 0, 255), 3)
            cv2.putText(frame, f"MEDICAL EMERGENCY: {fprob}%", (fx1, max(20, fy1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            # Low keypoint confidence is expected/normal for a person on the ground -
            # use a lenient confidence floor (not an all-or-nothing joint count) so the
            # skeleton still draws whatever joints the model could find.
            MIN_KPT_CONF = 0.15
            for kpts, kconf in zip(cached_kpts, cached_kconfs):
                for joint in [(5,6), (5,7), (7,9), (6,8), (8,10), (5,11), (6,12), (11,12), (11,13), (13,15), (12,14), (14,16)]:
                    pt1, pt2 = kpts[joint[0]], kpts[joint[1]]
                    c1, c2 = kconf[joint[0]], kconf[joint[1]]
                    # c == -1 means no confidence array was available at all (older/odd
                    # model output) - fall back to the coordinate check in that case only.
                    conf_ok = (c1 == -1 or c1 >= MIN_KPT_CONF) and (c2 == -1 or c2 >= MIN_KPT_CONF)
                    if conf_ok and pt1[0] > 0 and pt2[0] > 0:
                        cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (255, 0, 255), 2)
                        cv2.circle(frame, (int(pt1[0]), int(pt1[1])), 3, (0, 255, 255), -1)

        if cached_altercation:
            ax, ay, alabel, aconf = cached_altercation
            color = (0, 140, 255) if "Harassment" in alabel else (0, 0, 255) if "Assault" in alabel else (0, 200, 200)
            cv2.circle(frame, (ax, ay), 8, color, -1)
            cv2.putText(frame, f"{alabel}: {aconf}%", (max(10, ax - 140), max(20, ay - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if is_recording:
            elapsed = now - recording_start
            if recording_start == 0:
                recording_start = now
                recorded_frames = list(frame_buffer)
            else:
                recorded_frames.append(frame.copy())
                if elapsed >= 5.0:
                    is_recording = False
                    recording_start = 0
                    all_snaps = [s for s in [snap_1, snap_2, snap_3] if s]
                    threading.Thread(target=commit_incident_in_memory, args=(active_event_type, active_conf_str, list(recorded_frames), all_snaps), daemon=True).start()

        if is_alt_recording:
            elapsed = now - alt_recording_start
            alt_recorded_frames.append(frame.copy())
            if elapsed >= 8.0:  # fights/altercations tend to run longer than a fall - give it more tail
                is_alt_recording = False
                alt_snaps = [s for s in [alt_snap_1] if s]
                threading.Thread(target=commit_incident_in_memory, args=(active_alt_label, active_alt_conf_str, list(alt_recorded_frames), alt_snaps), daemon=True).start()

        ret, encoded_img = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ret:
            with camera_state_lock:
                latest_encoded_frame = encoded_img.tobytes()

def camera_supervisor():
    """Keeps the camera thread alive. Previously any unexpected exception inside
    camera_processing_loop killed the thread silently while camera_worker_started stayed
    True, so the dashboard feed froze/offline until the whole server was restarted."""
    import gc
    while True:
        try:
            camera_processing_loop()
        except Exception:
            print("[CAMERA] Processing loop crashed - restarting in 2s:")
            traceback.print_exc()
            gc.collect()   # drops the dead loop's VideoCapture so the device is freed
            time.sleep(2.0)

def start_camera_worker_if_needed():
    global camera_worker_started
    with camera_worker_lock:
        if not camera_worker_started:
            camera_worker_started = True
            threading.Thread(target=camera_supervisor, daemon=True).start()

@app.on_event("startup")
def _start_camera_on_boot():
    # Start the camera as soon as the server boots. Before, it only started when someone
    # opened /video_feed, so ESP32 hardware alerts had no frame to snapshot until then.
    start_camera_worker_if_needed()

def mjpeg_stream_generator():
    """Thin per-client consumer - never touches the camera. Re-sends the latest frame at
    least once a second, so proxies like ngrok never see an idle connection."""
    last_sent, last_sent_at = None, 0.0
    while True:
        with camera_state_lock:
            frame_bytes = latest_encoded_frame
        now = time.time()
        if frame_bytes is not None and (frame_bytes is not last_sent or now - last_sent_at > 1.0):
            last_sent, last_sent_at = frame_bytes, now
            yield (b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.05)

@app.get("/api/camera/snapshot")
def camera_snapshot():
    """Single latest JPEG. The dashboard polls this with fetch() instead of using a long-lived
    MJPEG <img>: an <img> tag cannot send the ngrok-skip-browser-warning header, and ngrok's
    free tier handles endless multipart streams poorly. Plain requests are reliable."""
    start_camera_worker_if_needed()
    with camera_state_lock:
        frame_bytes = latest_encoded_frame
    if frame_bytes is None:
        raise HTTPException(status_code=503, detail="Camera warming up")
    return Response(content=frame_bytes, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store, no-cache, must-revalidate", "Access-Control-Allow-Origin": "*"})

@app.get("/video_feed")
def video_feed():
    start_camera_worker_if_needed()
    return StreamingResponse(
        mjpeg_stream_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
    )


# ----------------- ESP32 TRIGGER -> CAPTURE -> DETECT -> SIGNAL BACK -----------------
# Flow:
#   1. ESP32 POSTs /api/esp32/trigger  (returns immediately with a job_id - never make the
#      ESP32 wait for video processing, its HTTP client will time out).
#   2. A background thread records a clip (pre-roll + live), checks it for waterlogging,
#      uploads the clip and saves an incident if water is confirmed.
#   3. ESP32 polls GET /api/esp32/job/{job_id} until "done" is true. Polling is used because
#      the backend on Render cannot open a connection INTO an ESP32 sitting behind a router.
ESP32_API_KEY = os.environ.get("ESP32_API_KEY", "").strip()               # set this on Render!
WATERLOG_LEVEL_CM = float(os.environ.get("WATERLOG_LEVEL_CM", "5.0"))     # sensor level counted as waterlogging - tune to your sensor
CLIP_MIN_SECONDS, CLIP_MAX_SECONDS = 3, 20
ESP32_JOB_TTL_SECONDS = 3600

esp32_jobs = {}
esp32_jobs_lock = threading.Lock()

class ESP32Trigger(BaseModel):
    device_id: str = "esp32-01"
    event_type: str = "waterlogging"
    duration_seconds: int = 8
    water_level_cm: Optional[float] = None

def _check_esp32_key(provided: Optional[str]):
    if ESP32_API_KEY and provided != ESP32_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Device-Key")

def _job_update(job_id, **fields):
    with esp32_jobs_lock:
        if job_id in esp32_jobs:
            esp32_jobs[job_id].update(fields)

def _prune_old_jobs():
    cutoff = time.time() - ESP32_JOB_TTL_SECONDS
    with esp32_jobs_lock:
        for jid in [j for j, v in esp32_jobs.items() if v["created"] < cutoff]:
            del esp32_jobs[jid]

def _run_esp32_job(job_id, device_id, event_type, duration, sensor_level):
    try:
        start_camera_worker_if_needed()
        # Give a cold camera a few seconds to produce its first real frame.
        wait_until = time.time() + 8
        while time.time() - last_live_frame_time > 5 and time.time() < wait_until:
            time.sleep(0.25)
        if time.time() - last_live_frame_time > 5:
            _job_update(job_id, status="failed", done=True, success=False,
                        message="Camera offline - no live video source (set VIDEO_SOURCE or connect the camera).")
            return

        # ---- 1. capture: pre-roll from the rolling buffer + live frames at 10 fps ----
        _job_update(job_id, status="capturing")
        frames = list(frame_buffer)
        t_end = time.time() + duration
        last_bytes, last_frame = None, None
        while time.time() < t_end:
            with camera_state_lock:
                jpg = latest_encoded_frame
            if jpg is not None and jpg is not last_bytes:
                decoded = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if decoded is not None:
                    last_bytes, last_frame = jpg, decoded
            if last_frame is not None:
                frames.append(last_frame.copy())   # one entry per 0.1s keeps playback at real speed
            time.sleep(1.0 / FRAME_RATE)
        if not frames:
            _job_update(job_id, status="failed", done=True, success=False, message="No frames captured.")
            return

        # ---- 2. detect waterlogging (camera colour check + optional sensor level) ----
        _job_update(job_id, status="analyzing")
        step = max(1, len(frames) // 12)
        confs, hits = [], 0
        for f in frames[::step]:
            ok, _msg, conf = verify_semantic_relevance(f, "waterlogging")
            confs.append(conf)
            hits += 1 if ok else 0
        visual_ok = hits >= max(1, len(confs) // 2)
        visual_conf = float(np.median(confs)) if confs else 0.0

        if sensor_level is None and time.time() - hardware_state.get("last_update", 0) < 30:
            sensor_level = hardware_state.get("water_level_cm")     # fall back to latest telemetry
        sensor_ok = sensor_level is not None and sensor_level >= WATERLOG_LEVEL_CM

        detected = visual_ok or sensor_ok
        if not detected:
            _job_update(job_id, status="done", done=True, success=True, detected=False,
                        confidence=round(visual_conf, 1), water_level_cm=sensor_level,
                        message="No waterlogging confirmed.")
            return
        confidence = min(99, int(visual_conf + (10 if (visual_ok and sensor_ok) else 0))) if visual_ok else 60

        # ---- 3. save clip + incident ----
        _job_update(job_id, status="saving")
        snaps = [frame_to_base64_jpeg(frames[len(frames) // 2])]
        result = commit_incident_in_memory("Water Logging Detected", f"{confidence}%", frames, snaps)
        if not result:
            _job_update(job_id, status="failed", done=True, success=False,
                        message="Waterlogging detected but saving the incident failed - see server logs.")
            return
        _job_update(job_id, status="done", done=True, success=True, detected=True, confidence=confidence,
                    water_level_cm=sensor_level, incident_id=result["id"], media_url=result["media_url"],
                    message="Waterlogging confirmed. Incident recorded.")
    except Exception as e:
        traceback.print_exc()
        _job_update(job_id, status="failed", done=True, success=False, message=f"Internal error: {e}")

@app.post("/api/esp32/trigger", status_code=202)
def esp32_trigger(data: ESP32Trigger, x_device_key: Optional[str] = Header(default=None)):
    _check_esp32_key(x_device_key)
    _prune_old_jobs()
    if "water" not in data.event_type.lower() and "flood" not in data.event_type.lower():
        raise HTTPException(status_code=400, detail="Only waterlogging events are supported by this endpoint.")
    duration = max(CLIP_MIN_SECONDS, min(CLIP_MAX_SECONDS, data.duration_seconds))
    with esp32_jobs_lock:
        for jid, j in esp32_jobs.items():      # ignore sensor bounce: one active job per device
            if j["device_id"] == data.device_id and not j["done"]:
                return {"job_id": jid, "status": j["status"], "already_running": True}
        job_id = f"job_{generate_id()}"
        esp32_jobs[job_id] = {"job_id": job_id, "device_id": data.device_id, "created": time.time(),
                              "status": "accepted", "done": False, "success": None, "message": "Trigger received."}
    threading.Thread(target=_run_esp32_job, args=(job_id, data.device_id, data.event_type, duration, data.water_level_cm), daemon=True).start()
    return {"job_id": job_id, "status": "accepted", "already_running": False, "poll": f"/api/esp32/job/{job_id}"}

@app.get("/api/esp32/job/{job_id}")
def esp32_job_status(job_id: str, x_device_key: Optional[str] = Header(default=None)):
    _check_esp32_key(x_device_key)
    with esp32_jobs_lock:
        job = esp32_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Unknown job_id")
        return {k: v for k, v in job.items() if k != "created"}