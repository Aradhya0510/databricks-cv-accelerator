"""The AI Runtime integration: launching, run hand-off, and job definitions.

The job definitions are YAML the test suite never executes, so the invariants
that would otherwise surface only as a failed remote run are checked here.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest
import yaml

from src.engine.launch import build_torchrun_argv
from src.utils.manifest import read_run_manifest, write_run_manifest

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# torchrun relaunch
# ---------------------------------------------------------------------------

def test_single_node_runs_standalone():
    argv = build_torchrun_argv("jobs/train.py", ["--config_path", "c.yaml"], 8, env={})

    assert argv[:4] == [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=8"]
    assert "--standalone" in argv
    assert not any(a.startswith("--rdzv") for a in argv)
    assert argv[-3:] == ["jobs/train.py", "--config_path", "c.yaml"]


def test_multi_node_rendezvous_uses_the_ai_runtime_master():
    env = {"NUM_NODES": "2", "MASTER_ADDR": "10.0.0.5", "MASTER_PORT": "29400"}
    argv = build_torchrun_argv("jobs/train.py", [], 8, env=env)

    assert "--nnodes=2" in argv
    assert "--rdzv_backend=c10d" in argv
    assert "--rdzv_endpoint=10.0.0.5:29400" in argv
    assert "--standalone" not in argv


def test_multi_node_without_a_master_fails_clearly():
    with pytest.raises(RuntimeError, match="MASTER_ADDR"):
        build_torchrun_argv("jobs/train.py", [], 8, env={"NUM_NODES": "2"})


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------

def test_manifest_round_trips(tmp_path):
    results = tmp_path / "results"
    write_run_manifest(str(results), run_id="abc", model_uri="models:/m-1", task_type="detection")

    manifest = read_run_manifest(str(results))
    assert manifest == {"run_id": "abc", "model_uri": "models:/m-1", "task_type": "detection"}
    assert not list(results.glob("*.partial"))


def test_missing_manifest_says_what_to_do(tmp_path):
    with pytest.raises(FileNotFoundError, match="--run_id"):
        read_run_manifest(str(tmp_path))


# ---------------------------------------------------------------------------
# Job definitions
# ---------------------------------------------------------------------------

def _requirement_lines(path: Path) -> list[str]:
    return [
        line.split("#", 1)[0].strip()
        for line in path.read_text().splitlines()
        if line.split("#", 1)[0].strip()
    ]


RUNTIME_REQUIREMENTS = _requirement_lines(REPO / "requirements_runtime.txt")
BUNDLE = yaml.safe_load((REPO / "databricks.yml").read_text())
AIR_WORKLOADS = sorted((REPO / "air").glob("*.yaml"))


def _bundle_environment(job: str, key: str) -> dict:
    envs = BUNDLE["resources"]["jobs"][job]["environments"]
    return next(e["spec"] for e in envs if e["environment_key"] == key)


def test_bundle_gpu_environment_matches_requirements_runtime():
    spec = _bundle_environment("cv_training_pipeline", "gpu")
    assert spec["base_environment"] == "databricks_ai_v6"
    assert spec["dependencies"] == RUNTIME_REQUIREMENTS


@pytest.mark.parametrize("path", AIR_WORKLOADS, ids=lambda p: p.name)
def test_air_workload_environment_matches_requirements_runtime(path):
    workload = yaml.safe_load(path.read_text())
    assert workload["environment"]["version"] == "databricks_ai_v6"
    assert workload["environment"]["dependencies"] == RUNTIME_REQUIREMENTS


@pytest.mark.parametrize("path", AIR_WORKLOADS, ids=lambda p: p.name)
def test_air_workload_runs_a_script_that_ships_in_the_snapshot(path):
    workload = yaml.safe_load(path.read_text())
    snapshot = workload["code_source"]["snapshot"]
    root = (path.parent / snapshot["root_path"]).resolve()
    assert root == REPO

    script = re.search(r"\$CODE_SOURCE_PATH/(\S+)", workload["command"]).group(1)
    assert (root / script).is_file()
    assert script.split("/", 1)[0] in snapshot["include_paths"]
    assert re.fullmatch(r"[A-Za-z0-9_-]{1,100}", workload["experiment_name"])


def test_bundle_has_no_classic_compute():
    """Everything runs on AI Runtime or serverless; a cluster spec is a regression."""
    text = (REPO / "databricks.yml").read_text()
    for key in ("job_clusters", "new_cluster", "existing_cluster_id", "spark_version", "node_type_id"):
        assert key not in text


def _ai_runtime_tasks():
    for job in BUNDLE["resources"]["jobs"].values():
        for task in job["tasks"]:
            if "ai_runtime_task" in task:
                yield task


def test_ai_runtime_tasks_run_executable_scripts_from_the_code_artifact():
    artifact = BUNDLE["artifacts"]["code"]
    tasks = list(_ai_runtime_tasks())
    assert tasks

    for task in tasks:
        spec = task["ai_runtime_task"]
        assert spec["code_source_path"] == artifact["files"][0]["source"]
        (deployment,) = spec["deployments"]
        script = REPO / deployment["command_path"]
        assert script.is_file() and os.access(script, os.X_OK), script
        assert task["environment_key"] == "gpu"

    for included in artifact["include"]:
        assert (REPO / included).is_dir()


def test_pipeline_config_default_is_shared_by_scripts_and_bundle():
    env = (REPO / "jobs/air/pipeline.env").read_text()
    default = re.search(r"CV_CONFIG_PATH:-([^}]+)}", env).group(1)

    assert (REPO / default).is_file()
    assert BUNDLE["variables"]["config_path"]["default"] == default


# ---------------------------------------------------------------------------
# Resume and MLflow run ownership (need the training stack)
# ---------------------------------------------------------------------------

def test_latest_checkpoint_skips_partial_and_incomplete(tmp_path):
    pytest.importorskip("transformers")
    from src.engine.callbacks import find_latest_checkpoint

    for step in (100, 200):
        (tmp_path / f"checkpoint-{step}").mkdir()
        (tmp_path / f"checkpoint-{step}" / "trainer_state.json").write_text("{}")
    # Newer, but interrupted mid-copy or missing trainer state.
    (tmp_path / "checkpoint-300.partial").mkdir()
    (tmp_path / "checkpoint-400").mkdir()

    assert find_latest_checkpoint(str(tmp_path)) == str(tmp_path / "checkpoint-200")
    assert find_latest_checkpoint(str(tmp_path / "missing")) is None
    assert find_latest_checkpoint(None) is None


def _engine(tmp_path, **training):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from src.config.schema import PipelineConfig
    from src.engine.engine import TrainingEngine

    config = PipelineConfig(
        model={"model_name": "google/vit-base-patch16-224", "task_type": "classification"},
        data={"train_data_path": str(tmp_path), "val_data_path": str(tmp_path)},
        training=training,
        mlflow={"experiment_name": "/Users/me/exp", "run_name": "mine"},
    )
    return TrainingEngine(config)


def test_the_ai_runtime_mlflow_run_is_reused(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    import mlflow

    calls = {}
    monkeypatch.setattr(mlflow, "set_experiment", lambda name: calls.setdefault("experiment", name))
    monkeypatch.setattr(mlflow, "start_run", lambda **kw: calls.setdefault("start", kw))

    monkeypatch.setenv("MLFLOW_RUN_ID", "platform-run")
    engine._start_mlflow_run()
    assert calls == {"start": {"run_id": "platform-run"}}
    # Left set, HF Trainer's MLflowCallback would try to start it a second time.
    assert "MLFLOW_RUN_ID" not in os.environ

    calls.clear()
    engine._start_mlflow_run()
    assert calls == {"experiment": "/Users/me/exp", "start": {"run_name": "mine"}}


def test_resume_latest_starts_fresh_then_resumes(tmp_path):
    ckpts = tmp_path / "ckpts"
    engine = _engine(
        tmp_path, volume_checkpoint_dir=str(ckpts), resume_from_checkpoint="latest",
    )
    assert engine._resolve_resume_checkpoint() is None

    (ckpts / "checkpoint-50").mkdir(parents=True)
    (ckpts / "checkpoint-50" / "trainer_state.json").write_text("{}")
    assert engine._resolve_resume_checkpoint() == str(ckpts / "checkpoint-50")


def test_multi_gpu_script_relaunches_under_torchrun(tmp_path, monkeypatch):
    """jobs/train.py stages once, then hands workers a config on local paths."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import importlib.util

    volumes = tmp_path / "Volumes"
    (volumes / "main/cv/train").mkdir(parents=True)
    (volumes / "main/cv/train/x.jpg").write_text("x")
    (volumes / "main/cv/val").mkdir(parents=True)
    monkeypatch.setattr("src.utils.environment._VOLUMES_ROOT", str(volumes))

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump({
        "model": {"model_name": "google/vit-base-patch16-224", "task_type": "classification"},
        "data": {
            "train_data_path": "/Volumes/main/cv/train/",
            "val_data_path": "/Volumes/main/cv/val/",
            "local_cache_dir": str(tmp_path / "cache"),
        },
    }))

    spec = importlib.util.spec_from_file_location("train_job", REPO / "jobs/train.py")
    train_job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_job)

    launched = {}

    def fake_run_torchrun(script, args, nproc):
        launched.update(script=script, args=args, nproc=nproc)
        return 0

    monkeypatch.setattr("src.engine.launch.run_torchrun", fake_run_torchrun)
    monkeypatch.setattr("src.utils.environment.gpus_per_node", lambda: 8)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("NUM_NODES", raising=False)
    monkeypatch.setattr(sys, "argv", ["train.py", "--config_path", str(config_path)])

    assert train_job.main() == 0
    assert launched["nproc"] == 8

    from src.config.schema import load_config

    resolved = load_config(launched["args"][1])
    assert resolved.data.train_data_path == str(tmp_path / "cache/main/cv/train")
    assert (tmp_path / "cache/main/cv/train/x.jpg").exists()
