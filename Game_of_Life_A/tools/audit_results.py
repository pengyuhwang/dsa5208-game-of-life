"""Validate raw benchmark CSVs and flag variation without deleting observations."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def audit(paths):
    errors=[];warnings=[];groups={};seen=set();counts={"PASS":0,"FAIL":0,"BLOCKED":0}
    for path in paths:
        with path.open(newline="") as f:
            for index,row in enumerate(csv.DictReader(f),2):
                label="{}:{}".format(path,index)
                status=row.get("status")
                if status not in counts:errors.append(label+": invalid status");continue
                counts[status]+=1
                if status!="PASS":continue
                try:
                    values={k:int(row[k]) for k in ["rows","cols","steps","ranks","threads","repeat","seed","density_ppm","live_cells"]}
                    if any(values[k]<1 for k in ["rows","cols","ranks","threads","repeat"]):raise ValueError("nonpositive size/resource count")
                    if values["steps"]<0 or not 0<=values["density_ppm"]<=1000000:raise ValueError("invalid input parameters")
                    if not 0<=values["live_cells"]<=values["rows"]*values["cols"]:raise ValueError("invalid live-cell count")
                    kernel,total=float(row["kernel_seconds"]),float(row["total_seconds"])
                    if not all(math.isfinite(v) and v>=0 for v in [kernel,total]):raise ValueError("invalid timing")
                    if kernel>total+1e-9:raise ValueError("kernel exceeds total")
                    if row["full_grid_match"] not in ["True","true","1"]:raise ValueError("missing full-grid validation")
                    if row["environment"] not in ["local","atlas","vanda"]:raise ValueError("invalid environment")
                    if row["environment"]!="local" and not row["job_id"]:raise ValueError("HPC row missing job ID")
                    if len(row["checksum_fnv1a64"])!=16:raise ValueError("bad checksum")
                    int(row["checksum_fnv1a64"],16)
                    key=tuple(row[k] for k in ["environment","hostname","variant","rows","cols","steps","seed","density_ppm","ranks","threads","source_sha256"])
                    identity=(row["run_id"],key,row["repeat"])
                    if identity in seen:raise ValueError("duplicate measurement identity")
                    seen.add(identity)
                    groups.setdefault(key,[]).append((kernel,row["checksum_fnv1a64"],label))
                except (KeyError,ValueError) as exc:errors.append(label+": "+str(exc))
    summaries=[]
    for key,observations in groups.items():
        times=[o[0] for o in observations];mean=statistics.mean(times)
        cv=statistics.stdev(times)/mean if len(times)>1 and mean else 0
        if len({o[1] for o in observations})!=1:errors.append(str(key)+": repeated final states differ")
        if len(times)<3:warnings.append(str(key)+": fewer than 3 repetitions")
        if cv>0.2:warnings.append(str(key)+": coefficient of variation exceeds diagnostic threshold 0.20; inspect, do not discard")
        summaries.append(dict(group=list(key),n=len(times),median_kernel_seconds=statistics.median(times),
                              min_kernel_seconds=min(times),max_kernel_seconds=max(times),coefficient_of_variation=cv))
    return dict(counts=counts,errors=errors,warnings=warnings,groups=summaries,
                interpretation="Variation threshold is diagnostic only. Local data never establish HPC speedup.")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs",nargs="+",type=Path);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();result=audit(args.inputs)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(counts=result["counts"],errors=result["errors"],warnings=result["warnings"])))
    return int(bool(result["errors"]))


if __name__=="__main__":raise SystemExit(main())
