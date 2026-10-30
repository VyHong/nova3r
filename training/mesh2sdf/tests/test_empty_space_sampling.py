import unittest

import torch

from training.mesh2sdf.empty_space_sampling import append_horizontal_empty_points


class EmptySpaceSamplingTests(unittest.TestCase):
    def batch(self, low=(-0.5, -0.3, -0.5), high=(0.5, 0.8, 0.5)):
        corners = torch.tensor([
            [x, y, z] for x in (low[0], high[0])
            for y in (low[1], high[1]) for z in (low[2], high[2])
        ])
        return {
            "geo_points": torch.ones(1, 3, 9),
            "sampling_bounds_corners": corners.unsqueeze(0),
        }

    def test_uniform_complement_and_saturated_labels(self):
        torch.manual_seed(23)
        batch = self.batch()
        result = append_horizontal_empty_points(batch, 60000, 128)
        points = result["geo_points"][:, 3:]
        torch.testing.assert_close(result["geo_points"][:, :3], batch["geo_points"])
        self.assertEqual(points.shape, (1, 60000, 9))
        self.assertTrue((points[..., :3].abs() <= 2).all())
        self.assertTrue((points[..., 1] < -0.3 - 1 / 128).any())
        self.assertTrue((points[..., 1] > 0.8 + 1 / 128).any())
        low = torch.tensor([-0.5, -0.3, -0.5])
        high = torch.tensor([0.5, 0.8, 0.5])
        delta = torch.maximum(low - points[..., :3], points[..., :3] - high).clamp_min(0)
        self.assertTrue((delta.norm(dim=-1) > 0).all())
        self.assertTrue((delta.norm(dim=-1) < 1 / 128).any())
        self.assertTrue(((points[..., 3] * 128).clamp(0, 1) == 1).all())
        self.assertTrue((points[..., 4:] == 0).all())
        # Equal-area columns outside the excluded box receive equal density.
        counts = [((points[..., 0] >= x) & (points[..., 0] < x + 0.5)
                   & (points[..., 2] >= z) & (points[..., 2] < z + 0.5)).sum()
                  for x, z in [(-2, -2), (1.5, 1.5), (-0.25, 1.5), (1.5, -0.25)]]
        expected = 60000 * (0.25 * 4) / (64 - 1 * 1.1 * 1)
        self.assertTrue(all(abs(float(value) - expected) < 5 * expected ** 0.5 for value in counts))
        self.assertEqual(batch["geo_points"].shape[1], 3)

    def test_box_outside_target_and_alternative_vertical_axis(self):
        result = append_horizontal_empty_points(self.batch((3, 4, 5), (4, 5, 6)), 50, 128, 2)
        points = result["geo_points"][:, 3:]
        self.assertTrue((points[..., :3].abs() <= 2).all())

    def test_vertical_caps_fill_when_horizontal_domain_is_covered(self):
        torch.manual_seed(11)
        result = append_horizontal_empty_points(
            self.batch((-3, -0.5, -3), (3, 0.5, 3)), 4096, 128)
        points = result["geo_points"][0, 3:, :3]
        self.assertTrue((points.abs() <= 2).all())
        self.assertTrue((points[:, 1].abs() > 0.5).all())
        self.assertTrue((points[:, 1] < -0.5).any())
        self.assertTrue((points[:, 1] > 0.5).any())

    def test_mixed_batch_preserves_fallback_distances_and_features(self):
        covered = self.batch((-3, -3, -3), (3, 3, 3))
        covered["geo_points"][..., 3] = 0.002
        covered["geo_points"][..., 7] = 0
        open_domain = self.batch()
        batch = {key: torch.cat((covered[key], open_domain[key])) for key in covered}
        result = append_horizontal_empty_points(batch, 100, 128)
        extra = result["geo_points"][:, 3:]
        torch.testing.assert_close(extra[0], covered["geo_points"][0, :1].expand(100, -1))
        self.assertTrue((extra[1, :, 3] * 128 >= 1).all())
        self.assertTrue((extra[1, :, 4:] == 0).all())

    def test_rotated_sampling_box_fills_enclosing_box_wedges(self):
        torch.manual_seed(7)
        batch = self.batch()
        angle = torch.tensor(torch.pi / 4)
        rotation = torch.tensor([[angle.cos(), 0, -angle.sin()],
                                 [0, 1, 0], [angle.sin(), 0, angle.cos()]])
        original = batch["sampling_bounds_corners"].clone()
        batch["sampling_bounds_corners"] = original @ rotation.T
        points = append_horizontal_empty_points(batch, 12000, 128)["geo_points"][0, 3:, :3]
        local = points @ rotation
        lo, hi = original[0].amin(0), original[0].amax(0)
        self.assertTrue(((local < lo) | (local > hi)).any(-1).all())
        enclosing_lo = batch["sampling_bounds_corners"][0].amin(0)
        enclosing_hi = batch["sampling_bounds_corners"][0].amax(0)
        in_wedges = ((points >= enclosing_lo) & (points <= enclosing_hi)).all(-1)
        self.assertGreater(int(in_wedges.sum()), 100)

    def test_disabled_and_impossible_domain(self):
        batch = self.batch()
        self.assertIs(append_horizontal_empty_points(batch, 0, 128), batch)
        covered = self.batch((-3, -3, -3), (3, 3, 3))
        result = append_horizontal_empty_points(covered, 10, 128)
        torch.testing.assert_close(result["geo_points"][:, 3:], torch.ones(1, 10, 9))
        with self.assertRaisesRegex(ValueError, "full sampling bounds"):
            append_horizontal_empty_points({"geo_points": batch["geo_points"]}, 10, 128)


if __name__ == "__main__":
    unittest.main()
