import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('guarded_queue', Path(__file__).parents[1] / 'guarded_queue.py')
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


class QueueRecoveryTests(unittest.TestCase):
    def test_pid_reuse_is_not_adopted(self):
        with patch.object(queue, 'process', return_value=('S', 'new')):
            self.assertFalse(queue.live({'pid': 123, 'start_ticks': 'old'}))

    def test_zombie_is_dead(self):
        with patch.object(queue, 'process', return_value=('Z', 'same')):
            self.assertFalse(queue.live({'pid': 123, 'start_ticks': 'same'}))

    def test_stopped_worker_is_still_adopted(self):
        with patch.object(queue, 'process', return_value=('T', 'same')):
            self.assertTrue(queue.live({'pid': 123, 'start_ticks': 'same'}))

    def test_child_survives_wrapper_failure(self):
        child = {'pid': 456, 'start_ticks': 'child', 'started': 11}
        with patch.object(queue, 'read', return_value=child), patch.object(queue, 'process', side_effect=lambda pid: ('S', 'child') if pid == 456 else None):
            self.assertTrue(queue.running('training', {'pid': 123, 'start_ticks': 'wrapper', 'started': 10}))

    def test_stale_child_is_not_adopted(self):
        with patch.object(queue, 'read', return_value={'pid': 456, 'start_ticks': 'child', 'started': 1}), patch.object(queue, 'process', return_value=None):
            self.assertFalse(queue.running('training', {'pid': 123, 'start_ticks': 'wrapper', 'started': 10}))

    def test_only_expected_successful_existing_meshes_count(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'full_list.json').write_text(json.dumps(['house/room/room_panorama_001.png', 'room 00002']))
            mesh = root / 'mesh.ply'
            mesh.touch()
            for index, record in enumerate([
                dict(sample='3dfront_house_room_00001_4', status='ok', mesh=str(mesh)),
                dict(sample='3dfront_room_00002_4', status='error', mesh=str(mesh)),
                dict(sample='3dfront_other_4', status='ok', mesh=str(mesh)),
            ]):
                directory = root / str(index)
                directory.mkdir()
                (directory / 'metrics.json').write_text(json.dumps(record))
            with patch.dict(queue.RUNS, {'3dfront': root}):
                self.assertEqual(queue.completed('3dfront'), (1, 2))


if __name__ == '__main__':
    unittest.main()
