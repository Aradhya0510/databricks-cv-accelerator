import os
import socket
import logging
import torch.distributed as dist
from typing import Any, Dict

from lightning.pytorch.strategies import DDPStrategy
from pyspark.sql import SparkSession

log = logging.getLogger(__name__)

class DatabricksDDPStrategy(DDPStrategy):
    """
    A production-ready, plug-and-play DDP strategy for Databricks notebooks.
    Provides flexible port configuration, robust cleanup, and smart environment setup.
    """
    def __init__(self, master_port: str | None = None, **kwargs: Any):
        self._spark_config = self._get_spark_config_with_validation(master_port)
        super().__init__(process_group_backend="nccl", **kwargs)
        log.info("🚀 DatabricksDDPStrategy initialized.")

    def _get_spark_config_with_validation(self, master_port: str | None) -> Dict[str, str]:
        """Gets Spark config and resolves the master port with clear priority."""
        spark = SparkSession.getActiveSession()
        if spark is None:
            raise RuntimeError("Requires an active Spark session on Databricks.")

        master_addr = spark.conf.get("spark.driver.host")
        if not master_addr:
            raise RuntimeError("spark.driver.host missing; check cluster config.")

        # Priority for port: 1. Direct argument, 2. Env variable, 3. Auto-find
        resolved_port = (
            master_port
            or os.environ.get("MASTER_PORT")
            or self._find_free_port()
        )

        return {"master_addr": master_addr, "master_port": str(resolved_port)}

    def _find_free_port(self) -> int:
        """Finds an available TCP port on the host."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def _setup_env(self) -> None:
        """Sets environment variables for the distributed process group."""
        # Respects user's custom NCCL_DEBUG setting if it exists
        os.environ.setdefault("NCCL_DEBUG", "WARN")
        os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
        os.environ["MASTER_ADDR"] = self._spark_config["master_addr"]
        os.environ["MASTER_PORT"] = self._spark_config["master_port"]

    def setup_environment(self) -> None:
        """Sets up the environment for each worker process."""
        self._setup_env()
        log.info(
            f"Rank {self.local_rank} env configured: "
            f"MASTER={os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
        )
        super().setup_environment()

    def teardown(self) -> None:
        """Cleans up the process group."""
        if dist.is_initialized():
            log.info(f"Destroying process group for rank {self.global_rank}.")
            dist.destroy_process_group()
            log.info("🧹 Process group destroyed.")
        super().teardown()