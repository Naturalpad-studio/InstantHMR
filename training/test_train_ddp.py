"""Where train_ddp.py takes its process ranks from: torchrun, Slurm, or neither.

    python -m pytest -q training/test_train_ddp.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_ddp import ranks_from_env  # noqa: E402


def test_torchrun_ranks():
    env = {"RANK": "5", "LOCAL_RANK": "1", "WORLD_SIZE": "8", "MASTER_ADDR": "node0", "MASTER_PORT": "29500"}
    assert ranks_from_env(env) == (5, 1, 8)


def test_slurm_ranks():
    env = {"SLURM_PROCID": "3", "SLURM_LOCALID": "3", "SLURM_NTASKS": "4", "SLURM_JOB_NODELIST": "n[1-2]"}
    assert ranks_from_env(env) == (3, 3, 4)


def test_single_process_without_a_launcher():
    assert ranks_from_env({}) == (0, 0, 1)


def test_one_process_launches_stay_single_process():
    # torchrun --nproc_per_node=1 and a one-task Slurm step need no process group
    assert ranks_from_env({"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"}) == (0, 0, 1)
    assert ranks_from_env({"SLURM_PROCID": "0", "SLURM_LOCALID": "0", "SLURM_NTASKS": "1"}) == (0, 0, 1)


def test_one_torchrun_process_inside_a_multi_task_slurm_job_is_single_process():
    # sbatch --ntasks=4 + torchrun --nproc_per_node=1: torchrun's world of one decides, not Slurm's 4 tasks
    env = {"SLURM_PROCID": "0", "SLURM_LOCALID": "0", "SLURM_NTASKS": "4",
           "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"}
    assert ranks_from_env(env) == (0, 0, 1)


def test_torchrun_wins_when_run_inside_a_slurm_allocation():
    # torchrun started by hand inside an salloc: Slurm's variables describe the allocation, not the ranks
    env = {"SLURM_PROCID": "0", "SLURM_LOCALID": "0", "SLURM_NTASKS": "1",
           "RANK": "2", "LOCAL_RANK": "2", "WORLD_SIZE": "4"}
    assert ranks_from_env(env) == (2, 2, 4)
