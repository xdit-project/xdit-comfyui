"""Runtime log-context behavior."""

import logging
import unittest

from xdit_comfyui.log_context import (
    RunLogContext,
    configure_xdit_logging,
    reset_run_context,
    run_logger,
    set_run_context,
)


class LogContextTest(unittest.TestCase):
    def test_run_context_prefix(self):
        ctx = RunLogContext(loader_node_id="2", sample_node_id="3", cache_key_short="abc123")
        self.assertEqual(ctx.prefix(), "[M=2 S=3 k=abc123] ")

    def test_child_loggers_receive_context_prefix(self):
        configure_xdit_logging()
        logger = run_logger()
        records = []

        class Handler(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = Handler()
        logging.getLogger("xdit.run").addHandler(handler)
        token = set_run_context(loader_node_id="9")
        try:
            logger.info("hello")
        finally:
            reset_run_context(token)
            logging.getLogger("xdit.run").removeHandler(handler)
        self.assertTrue(records)
        self.assertIn("[M=9]", records[0])
        self.assertIn("hello", records[0])

    def test_worker_out_logger_defaults_to_debug(self):
        configure_xdit_logging()
        self.assertEqual(logging.getLogger("xdit.worker.out").level, logging.DEBUG)
