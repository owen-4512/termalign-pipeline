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


    def test_accuracy_predicted_contains_gold_scores_full(self):
        gold_map = build_gold_map([{"术语A": ["hong kong applied science and technology research institute"]}])
        records = [
            {
                "source_file": "s1.txt",
                "extracted_terms": {
                    "术语A": ["Hong Kong Applied Science and Technology Research Institute (ASTRI)"],
                },
            }
        ]
        f1, precision, recall = compute_accuracy(records, gold_map)
        self.assertAlmostEqual(precision, 1.0)
        self.assertAlmostEqual(recall, 1.0)
        self.assertAlmostEqual(f1, 1.0)

    def test_accuracy_gold_contains_predicted_scores_token_ratio(self):
        gold_map = build_gold_map([{"术语A": ["cybersecurity risk assessment"]}])
        records = [
            {
                "source_file": "s1.txt",
                "extracted_terms": {
                    "术语A": ["risk assessment"],
                },
            }
        ]
        f1, precision, recall = compute_accuracy(records, gold_map)
        self.assertAlmostEqual(precision, 2 / 3)
        self.assertAlmostEqual(recall, 2 / 3)
        self.assertAlmostEqual(f1, 2 / 3)

    def test_accuracy_predicted_covers_gold_non_contiguous_scores_full(self):
        gold_map = build_gold_map([{"银行体系稳定": ["banking stability"]}])
        records = [
            {
                "source_file": "s1.txt",
                "extracted_terms": {
                    "银行体系稳定": ["banking system stability"],
                },
            }
        ]
        f1, precision, recall = compute_accuracy(records, gold_map)
        self.assertAlmostEqual(precision, 1.0)
        self.assertAlmostEqual(recall, 1.0)
        self.assertAlmostEqual(f1, 1.0)

    def test_accuracy_predicted_partial_of_gold_scores_ratio(self):
        gold_map = build_gold_map([{"楼宇按揭业务": ["mortgage lending business"]}])
        records = [
            {
                "source_file": "s1.txt",
                "extracted_terms": {
                    "楼宇按揭业务": ["mortgage business"],
                },
            }
        ]
        f1, precision, recall = compute_accuracy(records, gold_map)
        self.assertAlmostEqual(precision, 2 / 3)
        self.assertAlmostEqual(recall, 2 / 3)
        self.assertAlmostEqual(f1, 2 / 3)

    def test_accuracy_plural_variant_scores_full_for_ratio_term(self):
        gold_map = build_gold_map([{"资本充足率": ["capital adequacy ratios"]}])
        records = [
            {
                "source_file": "s1.txt",
                "extracted_terms": {
                    "资本充足率": ["capital adequacy ratio"],
                },
            }
        ]
        f1, precision, recall = compute_accuracy(records, gold_map)
        self.assertAlmostEqual(precision, 1.0)
        self.assertAlmostEqual(recall, 1.0)
        self.assertAlmostEqual(f1, 1.0)

    def test_accuracy_plural_variant_scores_full_for_liquidity_term(self):
        gold_map = build_gold_map([{"流动性比率": ["liquidity ratios"]}])
        records = [
            {
                "source_file": "s1.txt",
                "extracted_terms": {
                    "流动性比率": ["liquidity ratio"],
                },
            }
        ]
        f1, precision, recall = compute_accuracy(records, gold_map)
        self.assertAlmostEqual(precision, 1.0)
        self.assertAlmostEqual(recall, 1.0)
        self.assertAlmostEqual(f1, 1.0)

    def test_accuracy_keeps_service_token_intact_in_canonicalization(self):
        gold_map = build_gold_map([{"银行分行服务": ["bank branch service"]}])
        records = [
            {
                "source_file": "s1.txt",
                "extracted_terms": {
                    "银行分行服务": ["bank branch services"],
                },
            }
        ]
        f1, precision, recall = compute_accuracy(records, gold_map)
        self.assertAlmostEqual(precision, 1.0)
        self.assertAlmostEqual(recall, 1.0)
        self.assertAlmostEqual(f1, 1.0)

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
