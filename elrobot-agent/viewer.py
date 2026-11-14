"""Viewer: live camera cards, what Claude saw, and its last 10 tool calls.

    uv run python viewer.py      ->  http://127.0.0.1:5050

Live frames are read from the elrobot-ik API (GET /cameras/{id}/image), which
decodes them from the Station stream. Read-only: this page cannot move the arm.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from flask import Flask, Response, abort, jsonify, send_from_directory

HERE = Path(__file__).resolve().parent
PHOTOS = HERE / "photos"
LOGS = HERE / "logs"
HOST, PORT = "127.0.0.1", 5050   # ADJUST: keep 127.0.0.1, the hackathon WiFi is open
LIVE_INTERVAL_MS = 250           # ADJUST: live refresh per camera (250 ms = 4 fps)

load_dotenv(HERE / ".env")
IK_URL = os.environ.get("IK_URL", "http://127.0.0.1:8020").rstrip("/")
MAC_CAMERA_URL = os.environ.get("MAC_CAMERA_URL", "http://127.0.0.1:5060").strip().rstrip("/")

app = Flask(__name__)
ik = httpx.Client(base_url=IK_URL + "/api/v1", timeout=httpx.Timeout(4.0, connect=1.0))

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>ElRobot agent viewer</title>
<style>
 body{font-family:-apple-system,sans-serif;background:#111;color:#ddd;margin:16px}
 h1{font-size:18px;margin:0 0 4px} h2{font-size:14px;color:#9ab;margin:22px 0 8px;text-transform:uppercase;letter-spacing:.05em}
 #status{font-size:12px;color:#999;margin-bottom:4px}
 .ok{color:#6d6} .bad{color:#f77}
 .cards{display:flex;gap:12px;flex-wrap:wrap}
 .card{background:#1b1d21;border:1px solid #333;border-radius:8px;padding:10px;width:min(480px,100%)}
 .card header{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:8px;font-size:13px}
 .card .meta{color:#888;font-size:11px}
 .card img{width:100%;border-radius:4px;background:#000;display:block;min-height:120px;image-rendering:auto}
 .card.off img{display:none} .card .hint{display:none;color:#777;font-size:12px;padding:30px 0;text-align:center}
 .card.off .hint{display:block}
 .switch{position:relative;width:38px;height:22px;flex:none}
 .switch input{opacity:0;width:0;height:0}
 .switch span{position:absolute;inset:0;background:#444;border-radius:11px;cursor:pointer;transition:.15s}
 .switch span:before{content:"";position:absolute;width:16px;height:16px;left:3px;top:3px;background:#ddd;border-radius:50%;transition:.15s}
 .switch input:checked+span{background:#2a8f4a} .switch input:checked+span:before{transform:translateX(16px)}
 .empty{color:#888;font-size:13px;border:1px dashed #444;border-radius:8px;padding:14px;max-width:640px}
 figure{margin:0} figure img{max-width:480px;width:100%;border:1px solid #444;border-radius:6px}
 figcaption{font-size:12px;color:#999;margin-top:4px}
 table{border-collapse:collapse;width:100%;font-size:13px}
 td,th{border-bottom:1px solid #333;padding:4px 8px;text-align:left;vertical-align:top}
 td.r{font-family:ui-monospace,monospace;word-break:break-word} .err{color:#f77}
</style></head><body>
<h1>ElRobot agent viewer</h1>
<div id="status">connecting...</div>

<h2>Live cameras</h2>
<div class="cards" id="live"></div>
<div class="empty" id="nocams" style="display:none">
 The IK API lists no cameras. Station is not streaming any: pick the camera in the dropdown on the bus card at
 <a href="http://localhost:8889" style="color:#8bf">localhost:8889</a>, check the hub port button, and allow camera
 access for the app that started Station.
</div>

<h2>Last photos Claude took</h2>
<div class="cards" id="photos"></div>

<h2>Last 10 tool calls <small id="log" style="color:#888;text-transform:none"></small></h2>
<table><thead><tr><th>time</th><th>#</th><th>tool</th><th>input</th><th>result</th></tr></thead>
<tbody id="rows"></tbody></table>

<script>
const INTERVAL = __INTERVAL__;
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const live = {};   // camera id -> {on, busy, card}
const saved = k => { try { return localStorage.getItem(k); } catch(e) { return null; } };
const save = (k, v) => { try { localStorage.setItem(k, v); } catch(e) {} };

function addCamera(cam, index){
  const card = document.createElement('div');
  card.className = 'card';
  card.innerHTML = '<header><div><b>Camera ' + index + '</b> ' + esc(cam.name) +
    '<div class="meta">id ' + esc(cam.id) + ' · ' + cam.width + 'x' + cam.height + ' · <span class="fps">-</span></div></div>' +
    '<label class="switch" title="live on/off"><input type="checkbox"><span></span></label></header>' +
    '<img alt="live"><div class="hint">live view off</div>';
  document.getElementById('live').appendChild(card);
  const state = {on: saved('cam_' + cam.id) !== '0', busy: false, card, last: 0};
  live[cam.id] = state;
  const box = card.querySelector('input');
  box.checked = state.on;
  card.classList.toggle('off', !state.on);
  box.addEventListener('change', () => {
    state.on = box.checked;
    save('cam_' + cam.id, state.on ? '1' : '0');
    card.classList.toggle('off', !state.on);
  });
}

function pump(){
  for(const [id, s] of Object.entries(live)){
    if(!s.on || s.busy || document.hidden) continue;
    s.busy = true;
    const img = new Image();
    const t0 = performance.now();
    img.onload = () => {
      s.card.querySelector('img').src = img.src;
      const now = performance.now();
      s.card.querySelector('.fps').innerHTML = '<span class="ok">live</span> ' + Math.round(now - t0) + ' ms';
      s.busy = false;
    };
    img.onerror = () => {
      s.card.querySelector('.fps').innerHTML = '<span class="bad">no frame</span>';
      s.busy = false;
    };
    img.src = '/live/' + encodeURIComponent(id) + '.jpg?t=' + Date.now();
  }
}

async function tick(){
  try{
    const s = await (await fetch('/api/state')).json();
    const st = document.getElementById('status');
    st.innerHTML = 'IK API: ' + (s.ik_ok ? '<span class="ok">ok</span>' : '<span class="bad">not reachable</span>') +
      ' · Station: ' + (s.station_connected ? '<span class="ok">connected</span>' : '<span class="bad">disconnected</span>') +
      ' · motor state age: ' + (s.frame_age_ms == null ? '-' :
        (s.frame_age_ms > 500 ? '<span class="bad">' + Math.round(s.frame_age_ms) + ' ms (stale)</span>' : Math.round(s.frame_age_ms) + ' ms'));
    s.cameras.forEach((cam, i) => { if(!live[cam.id]) addCamera(cam, i); });
    for(const id of Object.keys(live)){
      if(!s.cameras.some(c => c.id === id)){ live[id].card.remove(); delete live[id]; }
    }
    document.getElementById('nocams').style.display = s.cameras.length ? 'none' : 'block';
    document.getElementById('log').textContent = s.log ? '(' + s.log + ')' : '(no log yet)';
    const photos = document.getElementById('photos');
    for(const name of s.photos){
      let fig = document.getElementById('f_' + name);
      if(!fig){
        fig = document.createElement('figure'); fig.id = 'f_' + name;
        fig.innerHTML = '<img><figcaption>' + esc(name) + '</figcaption>';
        photos.appendChild(fig);
      }
      fig.querySelector('img').src = '/photo/' + encodeURIComponent(name) + '?t=' + Date.now();
    }
    document.getElementById('rows').innerHTML = s.calls.map(c =>
      '<tr><td>' + esc((c.ts||'').slice(11,19)) + '</td><td>' + esc(c.n ?? '') + '</td><td>' + esc(c.tool) +
      '</td><td class="r">' + esc(JSON.stringify(c.input)) + '</td><td class="r' + (c.is_error ? ' err' : '') +
      '">' + esc(c.result) + '</td></tr>').join('');
  }catch(e){
    document.getElementById('status').innerHTML = '<span class="bad">viewer backend not reachable</span>';
  }
}
tick(); setInterval(tick, 1000); setInterval(pump, INTERVAL);
</script></body></html>"""


@app.get("/")
def index():
    return PAGE.replace("__INTERVAL__", str(LIVE_INTERVAL_MS))


@app.get("/api/state")
def state():
    photos = sorted(p.name for p in PHOTOS.glob("latest_*.jpg")) if PHOTOS.exists() else []
    logs = sorted(LOGS.glob("session_*.jsonl")) if LOGS.exists() else []
    calls: list[dict] = []
    if logs:
        for line in logs[-1].read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if entry.get("kind") == "tool":
                calls.append(entry)
    ik_ok, station_connected, frame_age, cameras = False, False, None, []
    try:
        health = ik.get("/health").json()
        ik_ok = True
        station_connected = bool(health.get("station_connected"))
        frame_age = health.get("last_frame_age_ms")
        r = ik.get("/cameras")
        if r.status_code == 200:
            cameras = r.json()
    except (httpx.HTTPError, ValueError):
        pass
    try:                                    # optional MacBook side camera (mac_camera.py)
        info = httpx.get(MAC_CAMERA_URL + "/info", timeout=0.5).json()
        if info.get("width"):
            cameras = cameras + [{"id": "macbook", "name": "MacBook side camera", "unique_id": "macbook",
                                  "width": info["width"], "height": info["height"]}]
    except (httpx.HTTPError, ValueError):
        pass
    return jsonify(photos=photos, log=logs[-1].name if logs else None, calls=calls[-10:][::-1],
                   ik_ok=ik_ok, station_connected=station_connected, frame_age_ms=frame_age, cameras=cameras)


@app.get("/live/<camera_id>.jpg")
def live_frame(camera_id: str):
    """Newest frame of one camera, proxied from the IK API."""
    if not camera_id.replace("-", "").replace("_", "").isalnum():
        abort(404)
    try:
        if camera_id == "macbook":
            r = httpx.get(MAC_CAMERA_URL + "/frame.jpg", timeout=3.0)
        else:
            r = ik.get(f"/cameras/{camera_id}/image")
    except httpx.HTTPError:
        abort(503)
    if r.status_code != 200:
        abort(r.status_code if r.status_code in (404, 415, 503) else 502)
    return Response(r.content, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/photo/<name>")
def photo(name: str):
    if not (name.startswith("latest_") and name.endswith(".jpg")):
        abort(404)
    response = send_from_directory(PHOTOS, name)
    response.headers["Cache-Control"] = "no-store"
    return response


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
