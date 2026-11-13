import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import warnings

from PIL import Image
from eval.baselines.common import SageSample, resolve_erp_path
from eval.baselines.run import parse_args, REPO_ROOT


class PanoramaInputTests(unittest.TestCase):
    def test_scannet_default_is_prepared(self):
        with patch('sys.argv', ['run', '--model', 'pager', '--dataset-layout', 'scannetpp', '--input-projection', 'erp']):
            args = parse_args()
        self.assertEqual(args.erp_root, REPO_ROOT / 'datasets/scannetpp/panos/updated_final_test_100_prepared_erp')

    def test_aspect_warning_and_valid_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'scene' / 'panocam' / 'images'
            folder.mkdir(parents=True)
            image = folder / '0.png'
            sample = SageSample('scene 00000', 'scene', 0, root, root, root, 'scannetpp')
            Image.new('RGB', (100, 40)).save(image)
            with self.assertWarnsRegex(UserWarning, 'not 2:1.*100x40'):
                self.assertEqual(resolve_erp_path(sample, root), image)
            Image.new('RGB', (100, 50)).save(image)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                resolve_erp_path(sample, root)
            self.assertEqual(caught, [])


if __name__ == '__main__':
    unittest.main()

class AspectPreservingInputTests(unittest.TestCase):
    def test_nova3r_keeps_full_erp_ratio_and_duplicates_padded_image(self):
        import torch
        from eval.baselines.adapters.nova3r import prepare_official_inputs
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'erp.png'
            Image.new('RGB', (1024, 512), 'white').save(path)
            views, single = prepare_official_inputs([path], (518, 392), 1, preserve_aspect=True)
        self.assertTrue(single)
        image = views[0]['img'][0]
        self.assertEqual(tuple(image.shape), (3, 392, 518))
        content = (image[0] > 0).nonzero()
        self.assertEqual(int(content[:, 0].max() - content[:, 0].min() + 1), 259)
        self.assertEqual(int(content[:, 1].max() - content[:, 1].min() + 1), 518)
        self.assertTrue(torch.equal(views[0]['img'], views[1]['img']))
