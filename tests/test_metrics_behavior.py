import unittest

from term_eval.data_model import build_gold_map
from term_eval.metrics_accuracy import compute_accuracy
from term_eval.metrics_consistency import compute_consistency


class TestMetricsBehavior(unittest.TestCase):
    def test_accuracy_accepts_multiple_gold_variants(self):
        gold_records = [
            {"术语A": ["Term X", "Term Y"]},
        ]
        gold_map = build_gold_map(gold_records)
        records = [
            {
                "source_file": "s1.txt",
                "extracted_terms": {
                    "术语A": ["Term X", "Term Y", "Term Z"],
                },
            }
        ]

        f1, precision, recall = compute_accuracy(records, gold_map)
        self.assertAlmostEqual(precision, 2 / 3)
        self.assertAlmostEqual(recall, 2 / 3)
        self.assertAlmostEqual(f1, 2 / 3)

    def test_accuracy_skips_terms_not_in_gold(self):
        gold_records = [{"术语A": ["Term X"]}]
        gold_map = build_gold_map(gold_records)
        records = [
            {
                "source_file": "s1.txt",
                "extracted_terms": {
                    "术语A": ["Term X", "Wrong"],
                    "术语B": ["Anything", "Another"],
                },
            }
        ]

        f1, precision, recall = compute_accuracy(records, gold_map)
        # 术语B 不在 gold 中，整项跳过，只统计术语A 的 2 个 occurrence
        self.assertAlmostEqual(precision, 0.5)
        self.assertAlmostEqual(recall, 0.5)
        self.assertAlmostEqual(f1, 0.5)

    def test_consistency_entropy_uses_occurrences_only(self):
        records = [
            {
                "source_file": "s1.txt",
                "extracted_terms": {
                    # 术语A 出现 3 次，其中两个译法，分布 2/3 与 1/3
                    "术语A": ["Term X", "Term X", "Term Y"],
                    # 单一译法，熵应为 0
                    "术语B": ["Term B", "Term B"],
                },
            }
        ]

        consistency = compute_consistency(records)
        # H(2/3,1/3)=~0.9182958341, 与术语B(0)平均 => ~0.459147917
        self.assertAlmostEqual(consistency, 0.459147917, places=6)


if __name__ == "__main__":
    unittest.main()
