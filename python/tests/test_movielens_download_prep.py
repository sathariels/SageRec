"""Phase 2 MovieLens 100K download and on-disk prep tests.

Uses tiny zip/u.data fixtures and temporary directories. Default cases never
touch the network. A live-download test is skipped unless
``SAGEREC_LIVE_MOVIELENS=1`` is set.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock
from urllib.error import URLError

_PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import sagerec_dataset as dataset
import sagerec_download as download
import sagerec_prep as prep

_TINY_UDATA = (
    "9\t8\t5\t10\n"
    "9\t3\t4\t20\n"
    "9\t4\t3\t30\n"
    "5\t3\t1\t40\n"
)


def _zip_bytes(member: str, payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(member, payload)
    return buffer.getvalue()


class DownloadHelperTests(unittest.TestCase):
    def test_extract_udata_from_official_layout(self) -> None:
        archive = _zip_bytes("ml-100k/u.data", _TINY_UDATA.encode())
        payload, member = download.extract_udata(archive)
        self.assertEqual(member, "ml-100k/u.data")
        self.assertEqual(payload, _TINY_UDATA.encode())

    def test_extract_udata_rejects_missing_member(self) -> None:
        archive = _zip_bytes("ml-100k/README", b"hello")
        with self.assertRaises(download.MovieLensDownloadError) as ctx:
            download.extract_udata(archive)
        self.assertIn("u.data", str(ctx.exception))
        self.assertIn("layout", str(ctx.exception).lower())

    def test_extract_udata_rejects_corrupt_zip(self) -> None:
        with self.assertRaises(download.MovieLensDownloadError) as ctx:
            download.extract_udata(b"not-a-zip")
        self.assertIn("zip", str(ctx.exception).lower())

    def test_verify_md5_mismatch_is_actionable(self) -> None:
        with self.assertRaises(download.MovieLensDownloadError) as ctx:
            download.verify_md5(b"abc", "0" * 32, label="fixture archive")
        message = str(ctx.exception)
        self.assertIn("MD5 mismatch", message)
        self.assertIn(hashlib.md5(b"abc").hexdigest(), message)

    def test_rejects_movielens_1m_url(self) -> None:
        with self.assertRaises(download.MovieLensDownloadError) as ctx:
            download.fetch_bytes(
                "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
            )
        self.assertIn("1M", str(ctx.exception))

    def test_download_writes_raw_paths_and_checks_md5(self) -> None:
        archive = _zip_bytes("ml-100k/u.data", _TINY_UDATA.encode())
        expected = hashlib.md5(archive).hexdigest()

        def fake_fetch(url: str, *, timeout: float, tls_verify: bool) -> bytes:
            self.assertIn("ml-100k.zip", url)
            self.assertTrue(tls_verify)
            return archive

        dest = Path(self._tempdir)
        with mock.patch.object(download, "fetch_bytes", side_effect=fake_fetch):
            result = download.download_movielens_100k(
                dest,
                url=download.MOVIELENS_100K_URL,
                expected_md5=expected,
            )
        self.assertTrue(result.archive_path.is_file())
        self.assertTrue(result.udata_path.is_file())
        self.assertEqual(result.udata_path.read_bytes(), _TINY_UDATA.encode())
        self.assertEqual(result.archive_md5, expected)
        self.assertEqual(result.udata_sha256, download.sha256_hex(_TINY_UDATA.encode()))
        self.assertFalse(result.used_unverified_tls)

    def test_download_retries_unverified_tls_only_with_checksum(self) -> None:
        archive = _zip_bytes("ml-100k/u.data", _TINY_UDATA.encode())
        expected = hashlib.md5(archive).hexdigest()
        calls: list[bool] = []

        def fake_fetch(url: str, *, timeout: float, tls_verify: bool) -> bytes:
            calls.append(tls_verify)
            if tls_verify:
                raise download.MovieLensDownloadError(
                    f"TLS certificate verification failed for {url}: expired"
                )
            return archive

        dest = Path(self._tempdir)
        with mock.patch.object(download, "fetch_bytes", side_effect=fake_fetch):
            result = download.download_movielens_100k(
                dest, expected_md5=expected
            )
        self.assertEqual(calls, [True, False])
        self.assertTrue(result.used_unverified_tls)
        self.assertEqual(result.archive_md5, expected)

    def test_network_error_is_actionable(self) -> None:
        def boom(*_args: object, **_kwargs: object) -> bytes:
            raise URLError("Name or service not known")

        with mock.patch("urllib.request.urlopen", side_effect=boom):
            with self.assertRaises(download.MovieLensDownloadError) as ctx:
                download.fetch_bytes(download.MOVIELENS_100K_URL)
        self.assertIn("network error", str(ctx.exception).lower())

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._tempdir = self._tmpdir.name

    def tearDown(self) -> None:
        self._tmpdir.cleanup()


class OnDiskPrepTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._tempdir = self._tmpdir.name

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_prepare_from_path_writes_manifest_and_train_only_pairs(self) -> None:
        raw = Path(self._tempdir) / "u.data"
        raw.write_text(_TINY_UDATA, encoding="utf-8")
        processed = Path(self._tempdir) / "processed"
        prepared = dataset.prepare_movielens_100k(
            raw,
            processed,
            source_url=download.MOVIELENS_100K_URL,
            license=download.MOVIELENS_100K_LICENSE,
        )
        self.assertTrue(prepared.manifest_path.is_file())
        manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["dataset_edition"], "100k")
        self.assertEqual(manifest["split_policy_id"], prep.SPLIT_POLICY_ID)
        self.assertEqual(manifest["split_policy_version"], prep.SPLIT_POLICY_VERSION)
        self.assertEqual(manifest["min_interactions_for_eval"], 3)
        self.assertEqual(manifest["cold_start_policy"], prep.COLD_START_POLICY)
        self.assertEqual(manifest["source_url"], download.MOVIELENS_100K_URL)
        self.assertEqual(manifest["license"], download.MOVIELENS_100K_LICENSE)
        self.assertTrue(str(manifest["checksum"]).startswith("sha256:"))
        self.assertEqual(manifest["checksum"], f"sha256:{download.sha256_hex(_TINY_UDATA.encode())}")
        self.assertEqual(manifest["counts"]["users"], 2)
        self.assertEqual(manifest["counts"]["movies"], 3)
        self.assertEqual(manifest["counts"]["eligible_users"], 1)
        self.assertEqual(manifest["counts"]["cold_start_users"], 1)
        self.assertEqual(manifest["counts"]["interactions"]["total"], 4)

        train_pairs = {tuple(pair) for pair in json.loads(prepared.train_pairs_path.read_text())}
        held_out = prepared.split.positive_pair_set("validation") | prepared.split.positive_pair_set(
            "test"
        )
        self.assertTrue(held_out)
        self.assertTrue(train_pairs.isdisjoint(held_out))
        self.assertEqual(train_pairs, set(prepared.split.train_positive_pairs()))

        loaded = dataset.load_split_from_processed(processed)
        self.assertEqual(loaded.train_positive_pairs(), prepared.split.train_positive_pairs())
        self.assertEqual(loaded.eligible_user_ids, prepared.split.eligible_user_ids)
        self.assertEqual(dataset.load_manifest(processed)["checksum"], manifest["checksum"])

    def test_prepare_from_bytes_matches_path(self) -> None:
        processed_a = Path(self._tempdir) / "a"
        processed_b = Path(self._tempdir) / "b"
        from_bytes = dataset.prepare_movielens_100k(_TINY_UDATA.encode(), processed_a)
        raw = Path(self._tempdir) / "u.data"
        raw.write_text(_TINY_UDATA, encoding="utf-8")
        from_path = dataset.prepare_movielens_100k(raw, processed_b)
        self.assertEqual(from_bytes.manifest["counts"], from_path.manifest["counts"])
        self.assertEqual(from_bytes.split, from_path.split)

    def test_missing_path_and_empty_bytes_fail(self) -> None:
        missing = Path(self._tempdir) / "missing.data"
        with self.assertRaises(dataset.MovieLensPrepError) as missing_ctx:
            dataset.read_udata_bytes(missing)
        self.assertIn("does not exist", str(missing_ctx.exception))

        with self.assertRaises(dataset.MovieLensPrepError) as empty_ctx:
            dataset.read_udata_bytes(b"   ")
        self.assertIn("empty", str(empty_ctx.exception))

    def test_rejects_1m_edition_through_prep_split(self) -> None:
        # Parser/split still refuse a 1M edition label if a caller bypasses helpers.
        with self.assertRaises(ValueError) as ctx:
            prep.split_interactions(
                [prep.NormalizedInteraction(0, 0, 1)],
                dataset_edition="1m",
            )
        self.assertIn("100K", str(ctx.exception))


@unittest.skipUnless(
    os.environ.get("SAGEREC_LIVE_MOVIELENS") == "1",
    "live MovieLens download disabled (set SAGEREC_LIVE_MOVIELENS=1)",
)
class LiveMovieLensDownloadTests(unittest.TestCase):
    def test_official_archive_checksum_and_udata(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            result = download.download_movielens_100k(tmp)
            self.assertEqual(result.archive_md5, download.MOVIELENS_100K_ARCHIVE_MD5)
            self.assertTrue(result.udata_path.is_file())
            self.assertGreater(result.udata_path.stat().st_size, 1_000_000)


if __name__ == "__main__":
    unittest.main()
