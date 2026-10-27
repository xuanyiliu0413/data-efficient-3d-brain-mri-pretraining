import csv
import tempfile
import unittest
from pathlib import Path
from scripts.check_manifests import validate

class TestManifests(unittest.TestCase):
    def test_nested_sets(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)

            for scale, count in [
                ("12p5", 2),
                ("25", 4),
                ("50", 8),
                ("100", 16),
            ]:
                with (base / f"jepa_{scale}_train.csv").open(
                    "w", newline=""
                ) as f:
                    writer = csv.writer(f)
                    writer.writerow(["img_path"])
                    for i in range(count):
                        writer.writerow([f"/data/scan_{i}.npy"])

            result = validate(
                base,
                {"12p5": 2, "25": 4, "50": 8, "100": 16}
            )
            self.assertEqual(result["100"], 16)
