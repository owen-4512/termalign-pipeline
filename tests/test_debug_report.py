import tempfile
import unittest
from pathlib import Path

from term_eval.pipeline import run_evaluation


class TestDebugReport(unittest.TestCase):
    def test_run_evaluation_includes_debug_when_requested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "align.tsv").write_text(
                "source_file\tzh_term\ten_term\tsimilarity\tzh_source\ten_source\tzh_confidence\ten_confidence\tzh_sentence\ten_sentence\n"
                "s1.txt\t术语A\tTerm X\t1\t-\t-\t1\t1\t-\t-\n"
                "s1.txt\t术语A\tTerm Y\t1\t-\t-\t1\t1\t-\t-\n",
                encoding="utf-8",
            )
            (d / "gold.jsonl").write_text('{"术语A": ["Term X"]}\n', encoding="utf-8")
            (d / "target.txt").write_text("foo Term X bar Term Y", encoding="utf-8")

            result = run_evaluation(
                term_align_tsv=d / "align.tsv",
                gold_jsonl=d / "gold.jsonl",
                mode="simple",
                metrics=["all"],
                alpha=0.2,
                beta=0.1,
                target_txt=d / "target.txt",
                report_level="batch",
                include_debug=True,
            )

            self.assertIn("debug", result)
            self.assertIn("accuracy", result["debug"])
            self.assertIn("consistency", result["debug"])
            self.assertIn("score_summary", result["debug"])
            self.assertGreater(len(result["debug"]["accuracy"]["occurrence_details"]), 0)

    def test_accuracy_debug_contains_normalized_and_canonical_match_details(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "align.tsv").write_text(
                "source_file\tzh_term\ten_term\tsimilarity\tzh_source\ten_source\tzh_confidence\ten_confidence\tzh_sentence\ten_sentence\n"
                "s1.txt\t資本充足率\tcapital adequacy ratio\t1\t-\t-\t1\t1\t-\t-\n",
                encoding="utf-8",
            )
            (d / "gold.jsonl").write_text('{"資本充足率": ["capital adequacy ratios"]}\n', encoding="utf-8")
            (d / "target.txt").write_text("capital adequacy ratio", encoding="utf-8")

            result = run_evaluation(
                term_align_tsv=d / "align.tsv",
                gold_jsonl=d / "gold.jsonl",
                mode="simple",
                metrics=["accuracy"],
                alpha=0.2,
                beta=0.1,
                target_txt=d / "target.txt",
                report_level="batch",
                include_debug=True,
            )

            details = result["debug"]["accuracy"]["occurrence_details"]
            self.assertEqual(len(details), 1)
            item = details[0]
            self.assertEqual(item["predicted_variant_normalized"], "capital adequacy ratio")
            self.assertEqual(item["best_gold_variant_normalized"], "capital adequacy ratio")
            self.assertEqual(item["predicted_tokens_canonical"], ["capital", "adequacy", "ratio"])
            self.assertEqual(item["best_gold_tokens_canonical"], ["capital", "adequacy", "ratio"])
            self.assertEqual(item["match_rule"], "exact_normalized")
            self.assertAlmostEqual(item["score"], 1.0)

    def test_accuracy_debug_normalizes_open_ended_in_gold_and_predicted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "align.tsv").write_text(
                "source_file\tzh_term\ten_term\tsimilarity\tzh_source\ten_source\tzh_confidence\ten_confidence\tzh_sentence\ten_sentence\n"
                "s1.txt\t開放式基金型公司結構\tOpen-ended Fund Company Structure\t1\t-\t-\t1\t1\t-\t-\n",
                encoding="utf-8",
            )
            (d / "gold.jsonl").write_text(
                '{"開放式基金型公司結構": ["open ended fund company structure"]}\n',
                encoding="utf-8",
            )
            (d / "target.txt").write_text("Open-ended Fund Company Structure", encoding="utf-8")

            result = run_evaluation(
                term_align_tsv=d / "align.tsv",
                gold_jsonl=d / "gold.jsonl",
                mode="simple",
                metrics=["accuracy"],
                alpha=0.2,
                beta=0.1,
                target_txt=d / "target.txt",
                report_level="batch",
                include_debug=True,
            )

            item = result["debug"]["accuracy"]["occurrence_details"][0]
            self.assertEqual(item["predicted_variant_normalized"], "open end fund company structure")
            self.assertEqual(item["best_gold_variant_normalized"], "open end fund company structure")
            self.assertEqual(item["best_gold_tokens_canonical"], ["open", "end", "fund", "company", "structure"])
            self.assertEqual(item["match_rule"], "exact_normalized")
            self.assertAlmostEqual(item["score"], 1.0)

    def test_accuracy_debug_canonical_tokens_do_not_over_stem_service(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            d = Path(tmpdir)
            (d / "align.tsv").write_text(
                "source_file\tzh_term\ten_term\tsimilarity\tzh_source\ten_source\tzh_confidence\ten_confidence\tzh_sentence\ten_sentence\n"
                "s1.txt\t银行分行服务\tbank branch services\t1\t-\t-\t1\t1\t-\t-\n",
                encoding="utf-8",
            )
            (d / "gold.jsonl").write_text('{"银行分行服务": ["bank branch service"]}\n', encoding="utf-8")
            (d / "target.txt").write_text("bank branch services", encoding="utf-8")

            result = run_evaluation(
                term_align_tsv=d / "align.tsv",
                gold_jsonl=d / "gold.jsonl",
                mode="simple",
                metrics=["accuracy"],
                alpha=0.2,
                beta=0.1,
                target_txt=d / "target.txt",
                report_level="batch",
                include_debug=True,
            )

            item = result["debug"]["accuracy"]["occurrence_details"][0]
            self.assertEqual(item["predicted_tokens_canonical"], ["bank", "branch", "service"])
            self.assertEqual(item["best_gold_tokens_canonical"], ["bank", "branch", "service"])
            self.assertAlmostEqual(item["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
