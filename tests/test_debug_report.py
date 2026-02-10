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
            self.assertIn("distance", result["debug"])
            self.assertGreater(len(result["debug"]["accuracy"]["occurrence_details"]), 0)


if __name__ == "__main__":
    unittest.main()
