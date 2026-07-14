from __future__ import annotations

import unittest
from pathlib import Path


class ModelAdapterReliabilityTests(unittest.TestCase):
    def test_all_model_api_calls_forward_client_request_id_header(self) -> None:
        source = Path("src/agent/model_adapter.py").read_text(encoding="utf-8")

        self.assertEqual(source.count("client.chat.completions.create("), 4)
        self.assertEqual(
            source.count("extra_headers=request_headers(context)"),
            4,
        )
        self.assertIn("build_no_credentials_error", source)
        self.assertIn("request_headers", source)


if __name__ == "__main__":
    unittest.main()
