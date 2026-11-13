import unittest
import torch
from training.mesh2sdf.tools.hrudf import _full_precision_meshing

class HRUDFPrecisionTests(unittest.TestCase):
    def test_field_and_classifier_matmuls_escape_outer_autocast(self):
        old=torch.get_float32_matmul_precision()
        model=torch.nn.Linear(8,4)
        x=torch.ones(2,8)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            self.assertEqual(model(x).dtype,torch.bfloat16)
            with _full_precision_meshing('cpu'):
                self.assertFalse(torch.is_autocast_enabled('cpu'))
                self.assertEqual(model(x).dtype,torch.float32)
                self.assertEqual(torch.get_float32_matmul_precision(),'highest')
            self.assertTrue(torch.is_autocast_enabled('cpu'))
            self.assertEqual(model(x).dtype,torch.bfloat16)
        self.assertEqual(torch.get_float32_matmul_precision(),old)

    def test_precision_is_restored_on_failure(self):
        old=torch.get_float32_matmul_precision()
        with self.assertRaisesRegex(RuntimeError,'extraction failure'):
            with _full_precision_meshing('cpu'):
                raise RuntimeError('extraction failure')
        self.assertEqual(torch.get_float32_matmul_precision(),old)

if __name__=='__main__':unittest.main()
