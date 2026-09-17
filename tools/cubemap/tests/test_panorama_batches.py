"""Test actual shell batch discovery without requiring Blender or a GPU."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
import zipfile


LAUNCHER = (
    Path(__file__).resolve().parents[3]
    / "training/mesh2sdf/tools/sage-10k/render_panoramas_batched.sh"
)


class PanoramaBatchTests(unittest.TestCase):
    def test_regular_and_staged_zip_reach_renderer(self):
        with tempfile.TemporaryDirectory(prefix="panorama batches ") as temporary:
            root = Path(temporary)
            source = root / "source.zip"
            selected = root / "selected"
            selected.mkdir()
            for archive in (source, selected / "regular.zip"):
                with zipfile.ZipFile(archive, "w") as zf:
                    zf.writestr("layout.json", '{"rooms": []}')
            (selected / "linked.zip").symlink_to(source)
            (selected / "broken.zip").symlink_to(root / "missing.zip")
            (selected / "directory.zip").symlink_to(root, target_is_directory=True)

            # Replace only the external runtime. Exercise the real launcher,
            # including find, batch creation, and both levels of symlinks.
            bin_dir = root / "bin"
            bin_dir.mkdir()
            conda = bin_dir / "conda"
            conda.write_text(f"#!{sys.executable}\n" + textwrap.dedent("""\
                import json
                import os
                from pathlib import Path
                import sys
                import zipfile

                if '-c' in sys.argv:
                    raise SystemExit(0)
                renderer = next(i for i, arg in enumerate(sys.argv)
                                if arg.endswith('/render_panoramas.py'))
                batch = Path(sys.argv[renderer + 1])
                scenes = sorted(batch.glob('*.zip'))
                assert len(scenes) == 1
                for scene in scenes:
                    with zipfile.ZipFile(scene) as zf:
                        assert json.loads(zf.read('layout.json')) == {'rooms': []}
                    print(scene.stem, os.environ['CUDA_VISIBLE_DEVICES'])
                raise SystemExit(int(os.environ.get('MOCK_RENDER_EXIT', '0')))
                """))
            conda.chmod(0o755)
            for jobs in (1, 4):
                for status in (0, 137):
                    with self.subTest(jobs=jobs, renderer_status=status):
                        output = root / f"output-{jobs}-{status}"
                        result = subprocess.run(
                            ["bash", str(LAUNCHER), str(selected), str(output),
                             "--batch-size", "1", "--jobs", str(jobs), "--gpus", "0,1"],
                            env={**os.environ,
                                 "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
                                 "MOCK_RENDER_EXIT": str(status)},
                            text=True, capture_output=True, timeout=30,
                        )
                        if status:
                            self.assertNotEqual(result.returncode, 0)
                            self.assertIn("exited with status 137", result.stderr)
                        else:
                            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                            self.assertIn("Rendering 2 scenes in 2 batches", result.stdout)
                            logs = sorted((output / "logs-batched").glob("batch-*.log"))
                            self.assertEqual([log.read_text().strip() for log in logs], ["linked 0", "regular 1"])


if __name__ == "__main__":
    unittest.main()
