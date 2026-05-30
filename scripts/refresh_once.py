"""
Standalone entry point for the GitHub Actions refresh job.
Calls run_incremental_refresh and exits non-zero on error.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nba.etl.refresh import run_incremental_refresh

status = {"status": "running", "message": "", "progress": 0.0}
run_incremental_refresh(status)

print(f"[refresh] {status['message']}")
if status.get("status") == "error":
    sys.exit(1)
