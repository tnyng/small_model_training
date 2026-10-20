import unittest

from compare_gscore_binary import ranking


class RankingTest(unittest.TestCase):
    def test_separated_scores(self):
        metrics, curve = ranking([1, 0, 1, 0], [.9, .8, .7, .1])
        self.assertAlmostEqual(metrics["ap"], 5 / 6)
        self.assertAlmostEqual(metrics["auc"], .75)
        self.assertEqual(curve[-1]["n_triggered"], 4)

    def test_score_ties_do_not_inflate_ap(self):
        metrics, _ = ranking([1, 0, 1, 0], [.5, .5, .5, .5])
        self.assertAlmostEqual(metrics["ap"], .5)
        self.assertAlmostEqual(metrics["auc"], .5)


if __name__ == "__main__":
    unittest.main()
