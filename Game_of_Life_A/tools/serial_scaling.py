"""Serial runtime against problem size: one C run per row, no grid output.

The fitted exponent describes how runtime grows with cell count. HPC labels need
an allocated PBS job, so laptop data can never pass for Vanda data.
"""
import argparse
import csv
import datetime
import json
import math
import os
from pathlib import Path
import platform
import socket
import statistics
import subprocess

from benchmark import source_hash

ROOT=Path(__file__).resolve().parents[1]
FIELDS=["environment","hostname","job_id","rows","cols","steps","seed","density_ppm","repeat",
        "kernel_seconds","total_seconds","live_cells","checksum_fnv1a64","ns_per_cell_update"]


def fit_exponent(points):
    xs=[math.log(c) for c,_ in points];ys=[math.log(t) for _,t in points]
    mx,my=statistics.mean(xs),statistics.mean(ys)
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes",default="256,512,1024,2048,4096,8192,16384")
    p.add_argument("--steps",type=int,default=500)
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--seed",type=int,default=20261026)
    p.add_argument("--density-ppm",type=int,default=500000)
    p.add_argument("--environment",choices=["local","vanda"],required=True)
    p.add_argument("--binary",type=Path,default=ROOT/"build/serial")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    job_id=os.environ.get("PBS_JOBID","")
    if args.environment!="local" and not job_id:p.error("Vanda data must come from an allocated PBS job")
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[];host=socket.gethostname()
    for size in [int(s) for s in args.sizes.split(",")]:
        for repeat in range(1,args.repeats+1):
            result=subprocess.run([str(args.binary),"--rows",str(size),"--cols",str(size),"--steps",str(args.steps),
                "--seed",str(args.seed),"--density-ppm",str(args.density_ppm)],stdout=subprocess.PIPE,
                universal_newlines=True,check=True)
            m=json.loads(result.stdout)
            rows.append(dict(environment=args.environment,hostname=host,job_id=job_id,rows=size,cols=size,steps=args.steps,
                seed=args.seed,density_ppm=args.density_ppm,repeat=repeat,kernel_seconds=m["kernel_seconds"],
                total_seconds=m["total_seconds"],live_cells=m["live_cells"],checksum_fnv1a64=m["checksum_fnv1a64"],
                ns_per_cell_update=m["kernel_seconds"]*1e9/(size*size*args.steps)))
            print(size,repeat,m["kernel_seconds"],flush=True)
            with (args.output/"raw_results.csv").open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
    groups=[]
    for size in sorted({r["rows"] for r in rows}):
        g=[r for r in rows if r["rows"]==size]
        if len({r["checksum_fnv1a64"] for r in g})!=1:raise SystemExit("repeated runs of {} differ".format(size))
        t=[r["kernel_seconds"] for r in g]
        groups.append(dict(size=size,cells=size*size,repeats=len(g),median_seconds=statistics.median(t),min_seconds=min(t),
            max_seconds=max(t),median_ns_per_cell_update=statistics.median(r["ns_per_cell_update"] for r in g),
            checksum_fnv1a64=g[0]["checksum_fnv1a64"],live_cells=g[0]["live_cells"]))
    summary=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),environment=args.environment,
        hostname=host,job_id=job_id,platform=platform.platform(),steps=args.steps,source_sha256=source_hash(),
        exponent_time_vs_cells=fit_exponent([(g["cells"],g["median_seconds"]) for g in groups]),groups=groups)
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(dict(exponent=summary["exponent_time_vs_cells"],sizes=len(groups))))


if __name__=="__main__":main()
