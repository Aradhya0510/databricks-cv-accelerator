"""Process launching for multi-GPU training.

HF Trainer runs real DDP only when there is one process per GPU; a single
process that sees several GPUs silently falls back to DataParallel, which is
slower and changes the effective batch size.  Something therefore has to fan
out the processes, and on AI Runtime that is one of two launchers:

* **Scripts** (``databricks air run``, ``ai_runtime_task``, SSH): AI Runtime
  runs the command once per node and populates ``NUM_NODES``, ``WORLD_SIZE``,
  ``LOCAL_WORLD_SIZE``, ``MASTER_ADDR`` and ``MASTER_PORT``.  The script
  relaunches itself under ``torchrun``, built here from those variables.
* **Notebooks**: ``serverless_gpu``'s ``@distributed``, driven from
  :meth:`TrainingEngine.train`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import List, Mapping, Optional, Sequence

# Every node of one task must present the same rendezvous id.  The endpoint is
# already unique to the task, so a constant is sufficient.
_RDZV_ID = "cv-accelerator"


def build_torchrun_argv(
    script: str,
    script_args: Sequence[str],
    nproc_per_node: int,
    env: Optional[Mapping[str, str]] = None,
) -> List[str]:
    """Return the argv that runs *script* under ``torchrun`` on this node.

    Single node uses ``--standalone``.  Multi-node uses c10d rendezvous at
    ``MASTER_ADDR:MASTER_PORT``, which assigns node ranks itself — AI Runtime
    does not document a node-rank variable, and c10d does not need one.
    """
    env = os.environ if env is None else env
    nodes = int(env.get("NUM_NODES", "1"))

    argv = [sys.executable, "-m", "torch.distributed.run", f"--nproc_per_node={nproc_per_node}"]
    if nodes > 1:
        try:
            endpoint = f"{env['MASTER_ADDR']}:{env['MASTER_PORT']}"
        except KeyError as missing:
            raise RuntimeError(
                f"NUM_NODES={nodes} but {missing} is not set; AI Runtime sets "
                f"MASTER_ADDR and MASTER_PORT on multi-node tasks."
            ) from None
        argv += [
            f"--nnodes={nodes}",
            "--rdzv_backend=c10d",
            f"--rdzv_endpoint={endpoint}",
            f"--rdzv_id={_RDZV_ID}",
        ]
    else:
        argv.append("--standalone")

    return argv + [script, *script_args]


def run_torchrun(script: str, script_args: Sequence[str], nproc_per_node: int) -> int:
    """Run *script* under ``torchrun`` and return its exit code."""
    argv = build_torchrun_argv(script, script_args, nproc_per_node)
    print("Launching:", " ".join(argv), flush=True)
    return subprocess.call(argv)
