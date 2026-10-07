"""
camera_presets.py
Catalog of stream URL patterns for common CCTV brands and phone-camera apps,
plus a builder and an auto-discovery routine.

NOTE: paths vary by model/firmware. If a preset doesn't connect, use
discover() (tries every pattern) or paste a full URL manually.
"""
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

from camera_source import probe, mask_source

# paths: "main" = full-quality stream, "sub" = lighter stream (faster AI).
# {ch} = channel number (1 for a single camera; NVR cameras are 1..N).
PRESETS = {
    # ── CCTV / IP cameras (RTSP) ─────────────────────────────────────
    "hikvision": {
        "label": "Hikvision / HiWatch", "scheme": "rtsp", "port": 554, "auth": True,
        "paths": {"main": "/Streaming/Channels/{ch}01", "sub": "/Streaming/Channels/{ch}02"},
        "note": "Also used by many Hikvision-based NVRs. Channel 1 main = 101, sub = 102.",
    },
    "dahua": {
        "label": "Dahua / CP Plus / Amcrest / Lorex", "scheme": "rtsp", "port": 554, "auth": True,
        "paths": {"main": "/cam/realmonitor?channel={ch}&subtype=0",
                  "sub": "/cam/realmonitor?channel={ch}&subtype=1"},
        "note": "CP Plus and Amcrest cameras are Dahua-based and use this format.",
    },
    "axis": {
        "label": "Axis", "scheme": "rtsp", "port": 554, "auth": True,
        "paths": {"main": "/axis-media/media.amp?camera={ch}",
                  "sub": "/axis-media/media.amp?resolution=640x480&camera={ch}"},
        "note": "",
    },
    "uniview": {
        "label": "Uniview (UNV)", "scheme": "rtsp", "port": 554, "auth": True,
        "paths": {"main": "/unicast/c{ch}/s0/live", "sub": "/unicast/c{ch}/s1/live"},
        "note": "",
    },
    "reolink": {
        "label": "Reolink", "scheme": "rtsp", "port": 554, "auth": True,
        "paths": {"main": "/h264Preview_{ch:02d}_main", "sub": "/h264Preview_{ch:02d}_sub"},
        "note": "Enable RTSP in the camera's network settings first.",
    },
    "tapo": {
        "label": "TP-Link Tapo", "scheme": "rtsp", "port": 554, "auth": True,
        "paths": {"main": "/stream1", "sub": "/stream2"},
        "note": "Create a 'Camera Account' in the Tapo app (Advanced Settings) and use those credentials.",
    },
    "foscam": {
        "label": "Foscam", "scheme": "rtsp", "port": 554, "auth": True,
        "paths": {"main": "/videoMain", "sub": "/videoSub"},
        "note": "Older models may use a different path; try discovery.",
    },
    "hanwha": {
        "label": "Hanwha Wisenet / Samsung", "scheme": "rtsp", "port": 554, "auth": True,
        "paths": {"main": "/profile1/media.smp", "sub": "/profile2/media.smp"},
        "note": "Profile numbers depend on how the camera is configured.",
    },
    "vivotek": {
        "label": "Vivotek", "scheme": "rtsp", "port": 554, "auth": True,
        "paths": {"main": "/live.sdp", "sub": "/live2.sdp"},
        "note": "",
    },
    "ezviz": {
        "label": "EZVIZ", "scheme": "rtsp", "port": 554, "auth": True,
        "paths": {"main": "/H.264", "sub": "/H.264"},
        "note": "Username is 'admin'; the password is the device verification code (on the label).",
    },
    "generic_rtsp": {
        "label": "Other / generic RTSP camera", "scheme": "rtsp", "port": 554, "auth": True,
        "paths": {"main": "/stream1", "sub": "/stream2"},
        "extra_paths": ["/live", "/h264", "/1", "/11", "/ch0", "/live/ch0",
                        "/onvif1", "/cam1/h264", "/media/video1"],
        "note": "Common paths used by low-cost cameras. Discovery tries all of them.",
    },
    # ── Phone camera apps (no CCTV hardware needed) ──────────────────
    "android_ipwebcam": {
        "label": "Android phone: 'IP Webcam' app", "scheme": "http", "port": 8080, "auth": False,
        "paths": {"main": "/video", "sub": "/video"},
        "note": "Open the app, tap 'Start server', and use the IP shown on screen. "
                "Phone and laptop must be on the same Wi-Fi (or laptop on the phone's hotspot).",
    },
    "droidcam": {
        "label": "Android/iPhone: DroidCam", "scheme": "http", "port": 4747, "auth": False,
        "paths": {"main": "/video", "sub": "/video"},
        "note": "Use the Wi-Fi IP shown in the DroidCam app.",
    },
}

_HOST_RE = re.compile(r"^[A-Za-z0-9.\-]{1,253}$")


def list_presets():
    return [
        {"id": k, "label": v["label"], "default_port": v["port"],
         "needs_login": v["auth"], "note": v["note"], "type": v["scheme"]}
        for k, v in PRESETS.items()
    ]


def _split_host(ip, port):
    ip = (ip or "").strip()
    if ":" in ip:                      # allow "192.168.1.64:554"
        ip, _, p = ip.rpartition(":")
        if p.isdigit():
            port = port or int(p)
    if not _HOST_RE.match(ip):
        raise ValueError("Invalid IP address or hostname")
    return ip, port


def build_url(brand, ip, user="", password="", port=None, channel=1,
              stream="main", path=None):
    if brand not in PRESETS:
        raise ValueError(f"Unknown camera brand '{brand}'")
    p = PRESETS[brand]
    ip, port = _split_host(ip, port)
    port = int(port or p["port"])
    channel = int(channel or 1)
    if not (1 <= port <= 65535):
        raise ValueError("Invalid port")

    path = path or p["paths"].get(stream, p["paths"]["main"])
    path = path.format(ch=channel)
    if not path.startswith("/"):
        path = "/" + path

    creds = ""
    if user:
        # URL-encode so passwords containing @ : / # ? etc. don't break the URL
        creds = f"{quote(user, safe='')}:{quote(password or '', safe='')}@"
    return f"{p['scheme']}://{creds}{ip}:{port}{path}"


def _candidates(ip, user, password, port, channel, brand=None):
    out = []
    brands = [brand] if brand else [k for k, v in PRESETS.items() if v["scheme"] == "rtsp"]
    for b in brands:
        p = PRESETS[b]
        paths = [("main", p["paths"]["main"])]
        if p["paths"]["sub"] != p["paths"]["main"]:
            paths.append(("sub", p["paths"]["sub"]))
        paths += [("custom", x) for x in p.get("extra_paths", [])]
        for stream, path in paths:
            out.append({
                "brand": b, "label": p["label"], "stream": stream,
                "path": path.format(ch=int(channel or 1)),
                "url": build_url(b, ip, user, password, port, channel, stream,
                                 path if stream == "custom" else None),
            })
    return out


def discover(ip, user="", password="", port=None, channel=1, brand=None,
             timeout=8, workers=4):
    """
    Try every known URL pattern against one camera and return those that work.
    Tests in small parallel batches and stops after the first batch with a hit.

    WARNING: if the username/password is wrong, every attempt fails the login,
    and some cameras (e.g. Hikvision) temporarily lock the account after a few
    failures. Verify the credentials first (e.g. in the camera's web page).
    """
    cands = _candidates(ip, user, password, port, channel, brand)
    working, tried = [], 0
    for i in range(0, len(cands), workers):
        batch = cands[i:i + workers]
        with ThreadPoolExecutor(workers) as ex:
            results = list(ex.map(lambda c: probe(c["url"], timeout), batch))
        for c, r in zip(batch, results):
            tried += 1
            if r["ok"]:
                working.append({
                    "brand": c["brand"], "label": c["label"], "stream": c["stream"],
                    "width": r.get("width"), "height": r.get("height"),
                    "url_preview": mask_source(c["url"]),
                    # send this back to POST /camera/source to select it
                    "select": {"brand": c["brand"], "stream": c["stream"],
                               "path": c["path"] if c["stream"] == "custom" else None},
                })
        if working:
            break
    # prefer the main stream (better for people-counting accuracy)
    working.sort(key=lambda w: 0 if w["stream"] == "main" else 1)
    return {"tried": tried, "total_patterns": len(cands), "working": working}
