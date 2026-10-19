import json
from pathlib import Path
import tempfile
import unittest
from kv_cache.serve_summary import summarize


class SummaryTests(unittest.TestCase):
    def make_reports(self, root):
        completed=[]
        for repeat, (base,cached) in enumerate(((100,80),(200,220))):
            for policy,p95 in (("none",base),("learned",cached)):
                label=f"{policy}-{repeat}"
                row=dict(info=dict(model="tiny",dtype="float32",timing_is_instrumented=False,
                                   policy=policy,copy_mode="baseline",logits_mode="last"),
                         correctness=dict(status="passed"),concurrency=1,repeat=repeat,
                         http_p95_ms=p95,requests_per_second=10,hit_rate=.5)
                (root/(label+'.json')).write_text(json.dumps(row))
                completed.append(dict(label=label))
        (root/'manifest.json').write_text(json.dumps(dict(status="passed",completed=completed)))

    def test_paired_reductions_and_median_of_run_p95(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            self.make_reports(root)
            row=next(r for r in summarize(root)['groups'] if r['policy']=='learned')
            self.assertEqual(row['paired_p95_reduction_percent'],[20,-10])
            self.assertEqual(row['median_paired_p95_reduction_percent'],5)
            self.assertEqual(row['faster_than_no_cache_repeats'],1)
            self.assertEqual(row['median_run_p95_ms'],150)

    def test_mixed_instrumentation_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            self.make_reports(root)
            path=root/'learned-0.json'
            row=json.loads(path.read_text());row['info']['timing_is_instrumented']=True
            path.write_text(json.dumps(row))
            with self.assertRaises(ValueError):
                summarize(root)
