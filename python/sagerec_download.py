"""MovieLens 100K archive download and ``u.data`` extraction (ADR-001).

Fetches the official GroupLens zip, verifies the published archive MD5 when
the default URL is used, and extracts ``u.data`` under a caller-supplied
``data/raw/`` directory. Does not parse ratings, split, or handle MovieLens 1M.
"""

from __future__ import annotations

import hashlib
import io
import ssl
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

MOVIELENS_100K_URL: Final[str] = (
    "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
)
MOVIELENS_100K_MD5_URL: Final[str] = (
    "https://files.grouplens.org/datasets/movielens/ml-100k.zip.md5"
)
# Official GroupLens sidecar: "MD5 (ml-100k.zip) = 0e33842e24a9c977be4e0107933c0723"
MOVIELENS_100K_ARCHIVE_MD5: Final[str] = "0e33842e24a9c977be4e0107933c0723"
MOVIELENS_100K_LICENSE: Final[str] = (
    "GroupLens MovieLens research license; see ml-100k-README.txt in the archive"
)
_USER_AGENT: Final[str] = "SageRec/0.1 (MovieLens 100K research download)"
_DEFAULT_TIMEOUT_S: Final[float] = 60.0


class MovieLensDownloadError(RuntimeError):
    """Actionable failure while fetching or extracting MovieLens 100K."""


@dataclass(frozen=True)
class DownloadResult:
    """Local paths and provenance for one extracted 100K ratings file."""

    archive_path: Path
    udata_path: Path
    source_url: str
    archive_md5: str
    udata_sha256: str
    archive_member: str
    used_unverified_tls: bool


def md5_hex(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def format_checksum(algorithm: str, digest: str) -> str:
    return f"{algorithm}:{digest}"


def _reject_non_100k_url(url: str) -> None:
    lowered = url.lower()
    if "ml-1m" in lowered or "1m.zip" in lowered:
        raise MovieLensDownloadError(
            f"refusing {url!r}: MovieLens 1M is deferred (ADR-001). "
            f"Use the 100K archive at {MOVIELENS_100K_URL}"
        )
    host = (urlparse(url).hostname or "").lower()
    path = urlparse(url).path.lower()
    looks_official_100k = path.endswith("ml-100k.zip") or "ml-100k.zip" in path
    if host.endswith("grouplens.org") and not looks_official_100k:
        raise MovieLensDownloadError(
            f"refusing {url!r}: expected a MovieLens 100K zip "
            f"(…/ml-100k.zip), not another GroupLens edition"
        )


def _is_certificate_error(exc: BaseException | str) -> bool:
    message = str(exc).lower()
    if "certificate" in message or "ssl:" in message or "ssl error" in message:
        return True
    if not isinstance(exc, BaseException):
        return False
    if isinstance(exc, ssl.SSLError):
        return True
    cause = getattr(exc, "reason", None)
    if isinstance(cause, BaseException) and cause is not exc:
        return _is_certificate_error(cause)
    if isinstance(exc.__cause__, BaseException):
        return _is_certificate_error(exc.__cause__)
    return False


def fetch_bytes(
    url: str,
    *,
    timeout: float = _DEFAULT_TIMEOUT_S,
    tls_verify: bool = True,
) -> bytes:
    """GET ``url`` and return the response body. Does not follow to 1M editions."""
    _reject_non_100k_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    # Apply the context to the opener so HTTPS redirects (HTTP → HTTPS) honor
    # the caller's TLS setting. GroupLens HTTP URLs currently 301 to HTTPS.
    context: ssl.SSLContext | None = None
    if not tls_verify:
        context = ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            data = response.read()
    except urllib.error.HTTPError as exc:
        raise MovieLensDownloadError(
            f"HTTP {exc.code} fetching {url}: {exc.reason}. "
            "Check the documented GroupLens MovieLens 100K URL and network access."
        ) from exc
    except urllib.error.URLError as exc:
        reason = exc.reason
        if _is_certificate_error(exc):
            raise MovieLensDownloadError(
                f"TLS certificate verification failed for {url}: {reason}. "
                "files.grouplens.org has historically presented an expired "
                "certificate; retry is allowed only when an expected archive "
                "MD5 will be checked."
            ) from exc
        raise MovieLensDownloadError(
            f"network error fetching {url}: {reason}. "
            "A live download needs outbound access to files.grouplens.org."
        ) from exc
    except TimeoutError as exc:
        raise MovieLensDownloadError(
            f"timed out after {timeout}s fetching {url}"
        ) from exc
    if not data:
        raise MovieLensDownloadError(f"empty response body from {url}")
    return data


def verify_md5(data: bytes, expected_md5: str, *, label: str) -> str:
    actual = md5_hex(data)
    expected = expected_md5.strip().lower()
    if actual != expected:
        raise MovieLensDownloadError(
            f"{label} MD5 mismatch: expected {expected}, got {actual}. "
            "The file is truncated, substituted, or not the official "
            "MovieLens 100K archive."
        )
    return actual


def extract_udata(archive_bytes: bytes) -> tuple[bytes, str]:
    """Return ``(u.data bytes, zip member name)`` from a 100K zip."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
    except zipfile.BadZipFile as exc:
        raise MovieLensDownloadError(
            "downloaded bytes are not a valid zip archive. "
            "Expected the official GroupLens MovieLens 100K zip."
        ) from exc
    with archive:
        names = archive.namelist()
        matches = [
            name
            for name in names
            if Path(name).name == "u.data" and not name.endswith("/")
        ]
        if not matches:
            preview = ", ".join(names[:12]) or "(empty zip)"
            raise MovieLensDownloadError(
                "zip layout is not MovieLens 100K: no member named u.data. "
                f"Archive members include: {preview}"
            )
        preferred = next(
            (name for name in matches if name.replace("\\", "/").endswith("ml-100k/u.data")),
            matches[0],
        )
        if len(matches) > 1 and not preferred.replace("\\", "/").endswith("ml-100k/u.data"):
            raise MovieLensDownloadError(
                "zip contains multiple u.data members and none is ml-100k/u.data: "
                + ", ".join(matches)
            )
        try:
            payload = archive.read(preferred)
        except KeyError as exc:
            raise MovieLensDownloadError(
                f"failed to read zip member {preferred!r}"
            ) from exc
    if not payload.strip():
        raise MovieLensDownloadError(f"zip member {preferred!r} is empty")
    return payload, preferred


def fetch_movielens_100k_archive(
    url: str = MOVIELENS_100K_URL,
    *,
    expected_md5: str | None = MOVIELENS_100K_ARCHIVE_MD5,
    timeout: float = _DEFAULT_TIMEOUT_S,
) -> tuple[bytes, str, bool]:
    """Download the archive, verify MD5 when known, and return bytes.

    Returns ``(archive_bytes, md5_hex, used_unverified_tls)``.
    """
    _reject_non_100k_url(url)
    used_unverified = False
    try:
        archive = fetch_bytes(url, timeout=timeout, tls_verify=True)
    except MovieLensDownloadError as exc:
        if expected_md5 is None or not _is_certificate_error(exc):
            raise
        # Integrity is the published MD5; GroupLens TLS has been expired (2026).
        archive = fetch_bytes(url, timeout=timeout, tls_verify=False)
        used_unverified = True
    digest = md5_hex(archive)
    if expected_md5 is not None:
        verify_md5(archive, expected_md5, label="MovieLens 100K archive")
    elif url.rstrip("/") == MOVIELENS_100K_URL.rstrip("/"):
        raise MovieLensDownloadError(
            "refusing the official MovieLens 100K URL without an expected MD5; "
            f"pass expected_md5={MOVIELENS_100K_ARCHIVE_MD5!r}"
        )
    return archive, digest, used_unverified


def download_movielens_100k(
    dest_dir: str | Path,
    *,
    url: str = MOVIELENS_100K_URL,
    expected_md5: str | None = MOVIELENS_100K_ARCHIVE_MD5,
    timeout: float = _DEFAULT_TIMEOUT_S,
) -> DownloadResult:
    """Fetch the 100K zip, verify checksum, and write ``u.data`` under ``dest_dir``."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    archive, archive_md5, used_unverified = fetch_movielens_100k_archive(
        url, expected_md5=expected_md5, timeout=timeout
    )
    udata, member = extract_udata(archive)
    archive_path = dest / "ml-100k.zip"
    udata_dir = dest / "ml-100k"
    udata_dir.mkdir(parents=True, exist_ok=True)
    udata_path = udata_dir / "u.data"
    archive_path.write_bytes(archive)
    udata_path.write_bytes(udata)
    return DownloadResult(
        archive_path=archive_path,
        udata_path=udata_path,
        source_url=url,
        archive_md5=archive_md5,
        udata_sha256=sha256_hex(udata),
        archive_member=member,
        used_unverified_tls=used_unverified,
    )
