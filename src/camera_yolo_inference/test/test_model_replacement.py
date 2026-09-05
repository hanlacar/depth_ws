from pathlib import Path
import tempfile
import unittest

from camera_yolo_inference.model_replacement import (
    install_model, rollback_model, sha256_file)


class ModelReplacementTest(unittest.TestCase):
    def test_success_sha_and_rollback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target, candidate = root/"best.pt", root/"candidate.pt"
            target.write_bytes(b"known-good")
            candidate.write_bytes(b"new-model")
            result = install_model(candidate, target, "20260101_010203")
            self.assertEqual(result["candidate_sha256"], sha256_file(target))
            self.assertEqual(Path(result["backup"]).read_bytes(), b"known-good")
            rollback_model(result["backup"], target)
            self.assertEqual(target.read_bytes(), b"known-good")

    def test_failed_candidate_preserves_existing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root/"best.pt"
            target.write_bytes(b"known-good")
            with self.assertRaises(FileNotFoundError):
                install_model(root/"missing.pt", target, "20260101_010203")
            self.assertEqual(target.read_bytes(), b"known-good")


if __name__ == "__main__":
    unittest.main()
