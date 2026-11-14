"""Inspect environment without logging in, installing packages, or submitting jobs."""
import json
import os
from pathlib import Path
import platform
import shutil
import socket

ROOT=Path(__file__).resolve().parents[1]
data=dict(hostname=socket.gethostname(),platform=platform.platform(),logical_cpus=os.cpu_count(),
          commands={k:shutil.which(k) for k in ["cc","make","mpicc","mpiexec","qsub","sbatch","ssh"]},
          serial_available=(ROOT/"build/serial").is_file(),
          mpi_available=(ROOT/"build/mpi").is_file(),
          hybrid_available=(ROOT/"build/hybrid").is_file(),
          allocated_job=os.environ.get("PBS_JOBID") or os.environ.get("SLURM_JOB_ID"),
          interpretation="No cluster connection or account permission was verified by this local inspection.")
print(json.dumps(data,indent=2))
