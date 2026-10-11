"""A tiny fake eSCL (AirScan) scanner for tests."""

from __future__ import annotations

import threading
import uuid
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

CAPS = b"""<?xml version="1.0" encoding="UTF-8"?>
<scan:ScannerCapabilities xmlns:scan="http://schemas.hp.com/imaging/escl/2011/05/03"
    xmlns:pwg="http://www.pwg.org/schemas/2010/12/sm">
  <pwg:Version>2.63</pwg:Version>
  <pwg:MakeAndModel>Mock AirScan 3000</pwg:MakeAndModel>
  <scan:Platen>
    <scan:PlatenInputCaps>
      <scan:MinWidth>16</scan:MinWidth><scan:MaxWidth>2550</scan:MaxWidth>
      <scan:MinHeight>16</scan:MinHeight><scan:MaxHeight>3508</scan:MaxHeight>
      <scan:SettingProfiles><scan:SettingProfile>
        <scan:ColorModes>
          <scan:ColorMode>BlackAndWhite1</scan:ColorMode>
          <scan:ColorMode>Grayscale8</scan:ColorMode>
          <scan:ColorMode>RGB24</scan:ColorMode>
        </scan:ColorModes>
        <scan:DocumentFormats>
          <pwg:DocumentFormat>image/jpeg</pwg:DocumentFormat>
          <pwg:DocumentFormat>image/png</pwg:DocumentFormat>
          <pwg:DocumentFormat>application/pdf</pwg:DocumentFormat>
        </scan:DocumentFormats>
        <scan:SupportedResolutions><scan:DiscreteResolutions>
          <scan:DiscreteResolution><scan:XResolution>75</scan:XResolution><scan:YResolution>75</scan:YResolution></scan:DiscreteResolution>
          <scan:DiscreteResolution><scan:XResolution>150</scan:XResolution><scan:YResolution>150</scan:YResolution></scan:DiscreteResolution>
          <scan:DiscreteResolution><scan:XResolution>300</scan:XResolution><scan:YResolution>300</scan:YResolution></scan:DiscreteResolution>
          <scan:DiscreteResolution><scan:XResolution>600</scan:XResolution><scan:YResolution>600</scan:YResolution></scan:DiscreteResolution>
        </scan:DiscreteResolutions></scan:SupportedResolutions>
      </scan:SettingProfile></scan:SettingProfiles>
    </scan:PlatenInputCaps>
  </scan:Platen>
  <scan:Adf><scan:AdfSimplexInputCaps>
      <scan:MaxWidth>2550</scan:MaxWidth><scan:MaxHeight>4200</scan:MaxHeight>
      <scan:SettingProfiles><scan:SettingProfile>
        <scan:ColorModes><scan:ColorMode>RGB24</scan:ColorMode></scan:ColorModes>
        <scan:DocumentFormats><pwg:DocumentFormat>image/jpeg</pwg:DocumentFormat></scan:DocumentFormats>
      </scan:SettingProfile></scan:SettingProfiles>
  </scan:AdfSimplexInputCaps></scan:Adf>
</scan:ScannerCapabilities>"""

NS = {"scan": "http://schemas.hp.com/imaging/escl/2011/05/03", "pwg": "http://www.pwg.org/schemas/2010/12/sm"}


class MockEscl:
    """Runs a fake scanner on localhost; ``url`` is its eSCL base URL."""

    def __init__(self, busy_first: int = 1, adf_pages: int = 3):
        self.jobs: dict[str, dict] = {}
        self.settings: list[ET.Element] = []
        self.deleted: list[str] = []
        self.busy_first = busy_first
        self.adf_pages = adf_pages
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body=b"", ctype="text/xml", headers=None):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/eSCL/ScannerCapabilities":
                    return self._send(200, CAPS)
                if self.path.startswith("/eSCL/ScanJobs/") and self.path.endswith("/NextDocument"):
                    jid = self.path.split("/")[3]
                    job = mock.jobs.get(jid)
                    if job is None:
                        return self._send(404)
                    if job["busy"] > 0:
                        job["busy"] -= 1
                        return self._send(503)
                    if job["remaining"] <= 0:
                        return self._send(404)
                    job["remaining"] -= 1
                    return self._send(200, job["image"], job["ctype"])
                return self._send(404)

            def do_POST(self):
                if self.path != "/eSCL/ScanJobs":
                    return self._send(404)
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                root = ET.fromstring(body)
                mock.settings.append(root)
                dpi = int(root.find("scan:XResolution", NS).text)
                region = root.find("pwg:ScanRegions/pwg:ScanRegion", NS)
                w = int(region.find("pwg:Width", NS).text) * dpi // 300
                h = int(region.find("pwg:Height", NS).text) * dpi // 300
                fmt = root.find("pwg:DocumentFormat", NS).text
                src = root.find("pwg:InputSource", NS).text
                img = np.full((h, w, 3), (200, 120, 40), np.uint8)
                ext = ".png" if fmt == "image/png" else ".jpg"
                ok, enc = cv2.imencode(ext, img)
                jid = uuid.uuid4().hex
                mock.jobs[jid] = {
                    "image": enc.tobytes(),
                    "ctype": fmt,
                    "busy": mock.busy_first,
                    "remaining": mock.adf_pages if src == "Feeder" else 1,
                }
                return self._send(201, b"", headers={"Location": f"/eSCL/ScanJobs/{jid}"})

            def do_DELETE(self):
                mock.deleted.append(self.path)
                return self._send(200)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/eSCL"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
