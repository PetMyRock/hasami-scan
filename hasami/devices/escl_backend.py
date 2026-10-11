"""eSCL / AirScan backend: driverless scanning over the network.

Most network printers and scanners made in the last decade (anything that
works with Apple AirPrint scanning or Mopria) speak eSCL: plain HTTP with XML
job tickets.  No drivers are needed, so this backend works the same on
Windows, macOS and Linux.

Devices are found with mDNS (``_uscan._tcp`` / ``_uscans._tcp``) when the
optional ``zeroconf`` package is installed.  Any device can also be used by
address: ``escl:http://192.168.1.20/eSCL`` (``SCANNER_ESCL`` environment
variable: comma-separated URLs that are always listed).
"""

from __future__ import annotations

import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Iterator

from .base import (
    MM_PER_INCH,
    Backend,
    Capabilities,
    DeviceInfo,
    ProgressFn,
    ScanCancelled,
    Scanner,
    ScannerError,
    ScanRequest,
    ScanResult,
    decode_image,
)

NS = {
    "scan": "http://schemas.hp.com/imaging/escl/2011/05/03",
    "pwg": "http://www.pwg.org/schemas/2010/12/sm",
}
UNITS_PER_INCH = 300.0  # eSCL regions are in 1/300 inch
SOURCES = {"Flatbed": "Platen", "ADF": "Feeder", "ADF Duplex": "Feeder"}
CAPS_PATH = {"Platen": "scan:Platen/scan:PlatenInputCaps",
             "Feeder": "scan:Adf/scan:AdfSimplexInputCaps"}


def _text(el, path: str, default: str = "") -> str:
    found = el.find(path, NS) if el is not None else None
    return found.text.strip() if found is not None and found.text else default


class EsclScanner(Scanner):
    def __init__(self, base_url: str, name: str = "", timeout: float = 30.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self._caps_xml: ET.Element | None = None
        self.info = DeviceInfo(f"escl:{self.base}", name or self.base, "", "", "escl")
        caps = self._caps()
        model = _text(caps, "pwg:MakeAndModel")
        if model:
            self.info.model = model
            self.info.name = name or model

    # ------------------------------------------------------------------ http
    def _request(self, method: str, url: str, data: bytes | None = None,
                 headers: dict | None = None):
        req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
        return urllib.request.urlopen(req, timeout=self.timeout)

    def _caps(self) -> ET.Element:
        if self._caps_xml is None:
            try:
                with self._request("GET", f"{self.base}/ScannerCapabilities") as r:
                    self._caps_xml = ET.fromstring(r.read())
            except (urllib.error.URLError, OSError, ET.ParseError) as e:
                raise ScannerError(f"cannot reach eSCL scanner at {self.base}: {e}") from e
        return self._caps_xml

    # ---------------------------------------------------------- capabilities
    def _input_caps(self, source: str):
        return self._caps().find(CAPS_PATH.get(source, CAPS_PATH["Platen"]), NS)

    def _color_modes(self, source: str) -> list[str]:
        ic = self._input_caps(source)
        if ic is None:
            return []
        return [m.text.strip() for m in ic.findall(".//scan:ColorModes/scan:ColorMode", NS) if m.text]

    def _formats(self, source: str) -> list[str]:
        ic = self._input_caps(source)
        if ic is None:
            return []
        out = [f.text.strip() for f in ic.findall(".//pwg:DocumentFormat", NS) if f.text]
        out += [f.text.strip() for f in ic.findall(".//scan:DocumentFormatExt", NS) if f.text]
        return out

    def capabilities(self) -> Capabilities:
        caps = self._caps()
        sources = []
        if caps.find("scan:Platen", NS) is not None:
            sources.append("Flatbed")
        if caps.find("scan:Adf", NS) is not None:
            sources.append("ADF")
            if caps.find("scan:Adf/scan:AdfDuplexInputCaps", NS) is not None:
                sources.append("ADF Duplex")
        src = "Platen" if "Flatbed" in sources or not sources else "Feeder"
        ic = self._input_caps(src)
        modes = self._color_modes(src)
        res = sorted({int(x.text) for x in ic.findall(".//scan:DiscreteResolution/scan:XResolution", NS)
                      if x.text}) if ic is not None else []
        if not res and ic is not None:  # range form
            lo = _text(ic, ".//scan:XResolutionRange/scan:Min")
            hi = _text(ic, ".//scan:XResolutionRange/scan:Max")
            if lo and hi:
                res = [r for r in (75, 100, 150, 200, 300, 600, 1200, 2400) if int(lo) <= r <= int(hi)]
        w = float(_text(ic, "scan:MaxWidth", "2550")) / UNITS_PER_INCH * MM_PER_INCH
        h = float(_text(ic, "scan:MaxHeight", "3508")) / UNITS_PER_INCH * MM_PER_INCH
        our_modes = []
        if any(m.startswith("RGB") for m in modes):
            our_modes.append("color")
        if any(m.startswith("Grayscale") for m in modes):
            our_modes.append("gray")
        if any(m.startswith("BlackAndWhite") for m in modes):
            our_modes.append("lineart")
        depths = [8] + ([16] if any(m in ("RGB48", "Grayscale16") for m in modes) else [])
        return Capabilities(
            sources=sources or ["Flatbed"],
            modes=our_modes or ["color"],
            resolutions=res or [75, 150, 300, 600],
            bit_depths=depths,
            max_area_mm=(w, h),
        )

    # ---------------------------------------------------------------- scan
    def _settings_xml(self, request: ScanRequest, source: str, color_mode: str, fmt: str,
                      dpi: int, area_mm, duplex: bool) -> bytes:
        x, y, w, h = area_mm
        u = UNITS_PER_INCH / MM_PER_INCH
        root = ET.Element("scan:ScanSettings", {"xmlns:scan": NS["scan"], "xmlns:pwg": NS["pwg"]})
        ET.SubElement(root, "pwg:Version").text = _text(self._caps(), "pwg:Version", "2.0")
        regions = ET.SubElement(root, "pwg:ScanRegions")
        region = ET.SubElement(regions, "pwg:ScanRegion")
        ET.SubElement(region, "pwg:ContentRegionUnits").text = "escl:ThreeHundredthsOfInches"
        ET.SubElement(region, "pwg:XOffset").text = str(int(round(x * u)))
        ET.SubElement(region, "pwg:YOffset").text = str(int(round(y * u)))
        ET.SubElement(region, "pwg:Width").text = str(int(round(w * u)))
        ET.SubElement(region, "pwg:Height").text = str(int(round(h * u)))
        ET.SubElement(root, "pwg:InputSource").text = source
        if duplex:
            ET.SubElement(root, "scan:Duplex").text = "true"
        ET.SubElement(root, "scan:ColorMode").text = color_mode
        ET.SubElement(root, "scan:XResolution").text = str(dpi)
        ET.SubElement(root, "scan:YResolution").text = str(dpi)
        ET.SubElement(root, "pwg:DocumentFormat").text = fmt
        ET.SubElement(root, "scan:DocumentFormatExt").text = fmt
        return b'<?xml version="1.0" encoding="UTF-8"?>' + ET.tostring(root)

    def _start_job(self, request: ScanRequest) -> tuple[str, float, tuple]:
        caps = self.capabilities()
        src_name = request.source or caps.sources[0]
        source = SOURCES.get(src_name, "Feeder" if "adf" in src_name.lower() else "Platen")
        duplex = "duplex" in src_name.lower()
        modes = self._color_modes(source)
        want = {"color": ["RGB24"], "gray": ["Grayscale8"], "lineart": ["BlackAndWhite1", "Grayscale8"]}[request.mode]
        if request.bit_depth == 16:
            want = {"color": ["RGB48"], "gray": ["Grayscale16"]}.get(request.mode, []) + want
        color_mode = next((m for m in want if m in modes), modes[0] if modes else want[0])
        formats = self._formats(source)
        # Prefer lossless formats; JPEG only when nothing else is offered.
        fmt = next((f for f in ("image/png", "image/tiff", "image/jpeg") if f in formats), "image/jpeg")
        dpi = caps.nearest_resolution(request.dpi)
        area = request.area_mm or (0.0, 0.0, *caps.max_area_mm)
        body = self._settings_xml(request, source, color_mode, fmt, dpi, area, duplex)
        try:
            with self._request("POST", f"{self.base}/ScanJobs", body, {"Content-Type": "text/xml"}) as r:
                location = r.headers.get("Location")
        except urllib.error.HTTPError as e:
            if e.code == 503:
                raise ScannerError("scanner is busy") from e
            if e.code == 409:
                raise ScannerError("scanner refused the job (no paper in the feeder?)") from e
            raise ScannerError(f"scanner rejected the job: HTTP {e.code}") from e
        except (urllib.error.URLError, OSError) as e:
            raise ScannerError(f"cannot reach scanner: {e}") from e
        if not location:
            raise ScannerError("scanner did not return a job location")
        job = urllib.parse.urljoin(self.base + "/", location)
        return job, float(dpi), area

    def _next_document(self, job: str, progress: ProgressFn | None,
                       cancel: threading.Event | None) -> bytes | None:
        """Fetch the next page; ``None`` when the job has no more pages."""
        url = job.rstrip("/") + "/NextDocument"
        for attempt in range(120):
            if cancel is not None and cancel.is_set():
                raise ScanCancelled("scan cancelled")
            try:
                with self._request("GET", url) as r:
                    total = int(r.headers.get("Content-Length") or 0)
                    chunks, got = [], 0
                    while True:
                        if cancel is not None and cancel.is_set():
                            raise ScanCancelled("scan cancelled")
                        chunk = r.read(256 * 1024)
                        if not chunk:
                            break
                        chunks.append(chunk)
                        got += len(chunk)
                        if progress and total:
                            progress(min(1.0, got / total))
                    return b"".join(chunks)
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None
                if e.code == 503:  # still warming up / busy: try again
                    time.sleep(0.5)
                    continue
                raise ScannerError(f"scanner error: HTTP {e.code}") from e
            except (urllib.error.URLError, OSError) as e:
                raise ScannerError(f"lost connection to scanner: {e}") from e
        raise ScannerError("scanner did not deliver the page in time")

    def _delete(self, job: str) -> None:
        try:
            self._request("DELETE", job).close()
        except Exception:
            pass

    def scan_batch(self, request: ScanRequest, progress: ProgressFn | None = None,
                   cancel: threading.Event | None = None) -> Iterator[ScanResult]:
        job, dpi, area = self._start_job(request)
        try:
            while True:
                data = self._next_document(job, progress, cancel)
                if data is None:
                    return
                yield ScanResult(decode_image(data), dpi, area)
                if "adf" not in (request.source or "").lower():
                    return
        finally:
            # Release the job: needed after a cancel or when we stop reading
            # early, harmless when the scanner has already finished it.
            self._delete(job)

    def scan(self, request: ScanRequest, progress: ProgressFn | None = None,
             cancel: threading.Event | None = None) -> ScanResult:
        for result in self.scan_batch(request, progress, cancel):
            return result
        raise ScannerError("scanner returned no page")


def _discover(timeout: float = 2.0) -> list[DeviceInfo]:
    try:
        from zeroconf import ServiceBrowser, Zeroconf
    except ImportError:
        return []
    found: dict[str, DeviceInfo] = {}

    class Listener:
        def add_service(self, zc, type_, name):
            info = zc.get_service_info(type_, name, timeout=1500)
            if not info or not info.parsed_addresses():
                return
            props = {k.decode(errors="ignore"): (v or b"").decode(errors="ignore")
                     for k, v in (info.properties or {}).items()}
            scheme = "https" if type_.startswith("_uscans") else "http"
            rs = props.get("rs", "eSCL").strip("/")
            host = info.parsed_addresses()[0]
            if ":" in host:
                host = f"[{host}]"
            url = f"{scheme}://{host}:{info.port}/{rs}"
            label = props.get("ty") or name.split(".")[0]
            found.setdefault(label, DeviceInfo(f"escl:{url}", label, "", label, "escl"))

        def update_service(self, *a):
            pass

        def remove_service(self, *a):
            pass

    zc = Zeroconf()
    try:
        listener = Listener()
        ServiceBrowser(zc, ["_uscan._tcp.local.", "_uscans._tcp.local."], listener)
        time.sleep(timeout)
    finally:
        zc.close()
    return list(found.values())


class EsclBackend(Backend):
    name = "escl"

    def available(self) -> bool:
        return True

    def list_devices(self) -> list[DeviceInfo]:
        out = _discover()
        for url in filter(None, (u.strip() for u in os.environ.get("SCANNER_ESCL", "").split(","))):
            out.append(DeviceInfo(f"escl:{url}", url, "", "", "escl"))
        return out

    def open(self, native_id: str) -> Scanner:
        url = native_id
        if not url.startswith(("http://", "https://")):
            url = f"http://{url}"
        if urllib.parse.urlparse(url).path in ("", "/"):
            url = url.rstrip("/") + "/eSCL"
        return EsclScanner(url)
