"""Collect measurements and evidence from a CSV plan, without invoking a shell.

Each measured output is compared cell-by-cell against a saved serial reference.
HPC labels require Linux and an allocated PBS/Slurm job. Local runs stay local.
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import tempfile

from oracle import read_pbm
from verify import run_case, validate_metrics

ROOT=Path(__file__).resolve().parents[1]
FIELDS=["run_id","owner","environment","hostname","job_id","repeat","variant",
        "rows","cols","steps","seed","density_ppm","ranks","threads","nodes",
        "status","kernel_seconds","total_seconds","live_cells","checksum_fnv1a64",
        "source_sha256","binary_sha256","full_grid_match","detail"]


def source_hash():
    h=hashlib.sha256()
    paths=sorted([ROOT/"Makefile"]+list(ROOT.glob("*.c"))+list((ROOT/"include").glob("*.h"))+
                 list((ROOT/"src").glob("*.c")))
    for path in paths:
        h.update(str(path.relative_to(ROOT)).encode());h.update(b"\0");h.update(path.read_bytes())
    return h.hexdigest()


def require_environment(environment):
    job_id=os.environ.get("PBS_JOBID") or os.environ.get("SLURM_JOB_ID") or ""
    if environment != "local" and (platform.system()!="Linux" or not job_id):
        raise ValueError("HPC collection requires Linux and an allocated PBS/Slurm job; use --environment local for laptop data")
    return job_id


def available(variant, config):
    if variant=="serial":
        return (ROOT/"build/serial").is_file(),"build/serial"
    spec=config.get(variant)
    if spec is None:return False,"missing variant adapter"
    missing=[p for p in spec["required_paths"] if not (ROOT/p).is_file()]
    launcher=spec["command"][0]
    if not shutil.which(launcher) and not (ROOT/launcher).is_file():missing.append(launcher)
    return not missing,", ".join(missing)


def execute(command, folder, name, threads, timeout):
    result=subprocess.run(command,cwd=str(ROOT),env=dict(os.environ,OMP_NUM_THREADS=str(threads)),
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                          universal_newlines=True,timeout=timeout)
    (folder/(name+".stdout")).write_text(result.stdout)
    (folder/(name+".stderr")).write_text(result.stderr)
    (folder/(name+".command.json")).write_text(json.dumps(dict(argv=command,cwd=str(ROOT),
             OMP_NUM_THREADS=threads,returncode=result.returncode),indent=2)+"\n")
    if result.returncode:raise ValueError("exit {}; see {} logs".format(result.returncode,name))
    return json.loads(result.stdout)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,default=ROOT/"config/experiment_plan.csv")
    p.add_argument("--owner",choices=["A","B","C","all"],default="A")
    p.add_argument("--environment",choices=["local","atlas","vanda"],required=True)
    p.add_argument("--serial-only",action="store_true")
    p.add_argument("--output",type=Path,default=ROOT/"results/benchmarks")
    p.add_argument("--timeout",type=float,default=600)
    args=p.parse_args()
    try:job_id=require_environment(args.environment)
    except ValueError as exc:p.error(str(exc))
    run_id=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    folder=args.output.resolve()/run_id;folder.mkdir(parents=True,exist_ok=False)
    logs=folder/"logs";logs.mkdir()
    config={v["name"]:v for v in json.loads((ROOT/"config/variants.json").read_text())["variants"]}
    with args.plan.open(newline="") as f:plan=list(csv.DictReader(f))
    selected=[row for row in plan if (args.owner=="all" or row["owner"]==args.owner)
              and (not args.serial_only or row["variant"]=="serial")]
    if not selected:p.error("no matching measurements in plan")
    digest=source_hash()
    env_keys=["OMP_NUM_THREADS","OMP_PROC_BIND","OMP_PLACES","PBS_JOBID",
              "PBS_NP","PBS_NUM_NODES","SLURM_JOB_ID","SLURM_JOB_NUM_NODES",
              "SLURM_CPUS_PER_TASK","SLURM_NTASKS","LOADEDMODULES"]
    manifest=dict(run_id=run_id,environment=args.environment,hostname=socket.gethostname(),
        job_id=job_id,platform=platform.platform(),machine=platform.machine(),
        logical_cpus=os.cpu_count(),source_sha256=digest,plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        selected_measurements=len(selected),environment_variables={k:os.environ[k] for k in env_keys if k in os.environ},
        timing="kernel: halo preparation/exchange + update + swap; total: allocation/init/input + kernel + diagnostics; no output I/O",
        compiler_command="Not inferred from binary. Preserve actual make output and compiler version alongside the run.",
        source_hash_scope="C sources in project root/src, headers in include, and Makefile; retain external B/C sources as well",
        limitation="local is not Atlas/Vanda; node placement and CPU binding must also be checked against the job script")
    (folder/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    for pattern in ["*.c","include/*.h","src/*.c","Makefile"]:
        for source in ROOT.glob(pattern):
            target=folder/"source_snapshot"/source.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
    records=[];references={};prechecked=set()
    with tempfile.TemporaryDirectory(prefix="life_bench_") as tmp:
        work=Path(tmp)
        for index,row in enumerate(selected):
            variant=row["variant"]
            ints={k:int(row[k]) for k in ["rows","cols","steps","seed","density_ppm","ranks","threads","repeat"]}
            h,w,n,seed,density=[ints[k] for k in ["rows","cols","steps","seed","density_ppm"]]
            ranks,threads=ints["ranks"],ints["threads"]
            record=dict(row,run_id=run_id,environment=args.environment,hostname=socket.gethostname(),
                        job_id=job_id,status="BLOCKED",source_sha256=digest,full_grid_match=False)
            ok,reason=available(variant,config)
            if not ok:
                record["detail"]="not supplied/available: "+reason;records.append(record);continue
            if variant!="serial" and "benchmark_command" not in config[variant]:
                record["detail"]="configure benchmark_command adapter";records.append(record);continue
            binary=ROOT/("build/serial" if variant=="serial" else config[variant]["required_paths"][0])
            record["binary_sha256"]=hashlib.sha256(binary.read_bytes()).hexdigest()
            try:
                check_key=(variant,ranks,threads)
                if check_key not in prechecked:
                    from oracle import random_grid
                    shape=max(17,ranks+1)
                    check=dict(name="benchmark_precheck",rows=shape,cols=19,steps=8,
                               live=random_grid(shape,19,123,500000),known=None)
                    template=([str(ROOT/"build/serial"),"--input","{input}","--steps","{steps}","--output","{output}"]
                              if variant=="serial" else config[variant]["command"])
                    checked,evidence=run_case(check,template,variant,ranks,threads,work,folder/"failures",args.timeout)
                    (logs/("precheck_{}_p{}_t{}.json".format(*check_key))).write_text(
                        json.dumps(dict(record=checked,evidence=evidence),indent=2)+"\n")
                    if checked["status"]!="PASS":raise ValueError("small-grid independent-oracle precheck failed")
                    prechecked.add(check_key)
                ref_key=(h,w,n,seed,density)
                if ref_key not in references:
                    ref_path=folder/("reference_{}x{}_s{}_seed{}.pbm".format(h,w,n,seed))
                    command=[str(ROOT/"build/serial"),"--rows",str(h),"--cols",str(w),"--steps",str(n),
                             "--seed",str(seed),"--density-ppm",str(density),"--output",str(ref_path)]
                    metrics=execute(command,logs,"reference_{}".format(len(references)),1,args.timeout)
                    ref=read_pbm(ref_path);validate_metrics(metrics,h,w,n,1,1,ref[2])
                    references[ref_key]=ref
                output=work/"measured.pbm"
                if output.exists():output.unlink()
                if variant=="serial":
                    command=[str(ROOT/"build/serial"),"--rows",str(h),"--cols",str(w),"--steps",str(n),
                             "--seed",str(seed),"--density-ppm",str(density),"--output",str(output)]
                else:
                    params=dict(ints,output=str(output))
                    command=[str(s).format(**params) for s in config[variant]["benchmark_command"]]
                metrics=execute(command,logs,"measurement_{:03d}".format(index),threads,args.timeout)
                if not isinstance(metrics,dict) or metrics.get("variant")!=variant:
                    raise ValueError("JSON must identify the executed variant")
                actual=read_pbm(output)
                validate_metrics(metrics,h,w,n,ranks,threads,actual[2])
                if actual!=references[ref_key]:raise ValueError("full-scale grid differs from serial reference")
                record.update(status="PASS",full_grid_match=True,detail="measured; full grid matches serial reference")
                for key in ["kernel_seconds","total_seconds","live_cells","checksum_fnv1a64"]:record[key]=metrics[key]
            except (OSError,ValueError,subprocess.TimeoutExpired) as exc:
                record.update(status="FAIL",detail=str(exc))
                if (work/"measured.pbm").exists():
                    shutil.copy2(work/"measured.pbm",folder/("failed_{:03d}.pbm".format(index)))
            records.append(record)
            with (folder/"raw_results.csv").open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(records)
    with (folder/"raw_results.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(records)
    counts={s:sum(r["status"]==s for r in records) for s in ["PASS","FAIL","BLOCKED"]}
    (folder/"summary.json").write_text(json.dumps(dict(counts=counts,environment=args.environment),indent=2)+"\n")
    print(json.dumps(dict(directory=str(folder),counts=counts,environment=args.environment)))
    return int(bool(counts["FAIL"]))


if __name__=="__main__":raise SystemExit(main())
