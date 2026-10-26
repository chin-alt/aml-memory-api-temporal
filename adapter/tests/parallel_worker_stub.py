"""仅用于并发测试的本地进程；不访问模型、网络或数据库。"""

import json
import os
import sys
import time
from pathlib import Path

output, markers, index, fail = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "1"
markers.mkdir(exist_ok=True)
(markers / f"started-{index}").write_text(str(os.getpid()))
deadline = time.monotonic() + 15
while len(list(markers.glob("started-*"))) < 6:
    if time.monotonic() > deadline:
        raise RuntimeError("Six workers did not overlap")
    time.sleep(0.02)
time.sleep(0.1 if fail else 0.8)
output.mkdir(parents=True)
result = {"protocol_success": not fail, "qa": {"verdict": {"correct": index % 2 == 0}},
          "qa_cutoffs": {"50": index % 2 == 0, "10": True, "20": True},
          "worker_pid": os.getpid(), "worker_marker": os.environ["TEST_WORKER_MARKER"]}
(output / "result.json").write_text(json.dumps(result), encoding="utf-8")
sys.exit(1 if fail else 0)
