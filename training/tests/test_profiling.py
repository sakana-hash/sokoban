import unittest

from training.profiling import summarize_trace


class ProfileSummaryTests(unittest.TestCase):
    def test_overlapping_gpu_intervals_and_host_annotations(self):
        events = [
            {"ph": "X", "cat": "user_annotation", "name": "sokoban/environment", "ts": 0, "dur": 1000},
            {"ph": "X", "cat": "gpu_user_annotation", "name": "sokoban/environment", "ts": 0, "dur": 2000},
            {"ph": "X", "cat": "kernel", "name": "conv-a", "ts": 0, "dur": 2000},
            {"ph": "X", "cat": "kernel", "name": "conv-b", "ts": 1000, "dur": 2000},
            {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy HtoD", "ts": 2000, "dur": 2000},
            {"ph": "X", "cat": "gpu_memset", "name": "Memset", "ts": 5000, "dur": 500},
        ]
        summary = summarize_trace(events, 10.0, cuda=True)
        self.assertEqual(summary["host_ms"]["environment"], 1.0)
        self.assertEqual(summary["host_ms"]["other"], 9.0)
        self.assertEqual(summary["gpu"], {
            "kernel_ms": 3.0, "memcpy_ms": 2.0, "memset_ms": 0.5, "active_ms": 4.5, "active_pct": 45.0,
        })
        self.assertIsNone(summarize_trace(events[:2], 10.0, cuda=True)["gpu"])
        self.assertIsNone(summarize_trace(events[:1], 10.0, cuda=False)["gpu"])
