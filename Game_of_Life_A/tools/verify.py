"""Run complete-grid verification; missing external programs are BLOCKED.

No shell evaluation. JSON command arrays allow launcher-specific adapters.
Failure directories contain the input, expected grid, actual output and logs.
"""
import argparse
import csv
import datetime
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from cases import all_cases, named_cases
from oracle import evolve, fnv1a64, grid_sha256, read_pbm, write_pbm

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["variant","case","rows","cols","steps","ranks","threads",
          "status","mismatched_cells","first_mismatch","expected_sha256",
          "actual_sha256","detail"]


def validate_metrics(metrics, rows, cols, steps, ranks, threads, live=None):
    for key, expected in [("interface_version","1.0"),("rows",rows),("cols",cols),
                          ("steps",steps),("ranks",ranks),("threads",threads)]:
        if isinstance(expected,int) and type(metrics.get(key)) is not int:
            raise ValueError("metric {} must be an integer".format(key))
        if metrics.get(key) != expected:
            raise ValueError("metric {}: expected {}, got {}".format(key,expected,metrics.get(key)))
    for key in ("kernel_seconds", "total_seconds"):
        value = metrics.get(key)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid {}".format(key))
    if metrics["total_seconds"] + 1e-9 < metrics["kernel_seconds"]:
        raise ValueError("total_seconds smaller than kernel_seconds")
    if live is not None:
        if metrics.get("live_cells") != len(live):
            raise ValueError("live_cells does not match output grid")
        if metrics.get("checksum_fnv1a64") != fnv1a64(rows,cols,live):
            raise ValueError("checksum does not match output grid")


def run_case(spec, command_template, variant, ranks, threads, work, failures, timeout, expect=None):
    rows, cols, steps = spec["rows"], spec["cols"], spec["steps"]
    expected = evolve(rows,cols,spec["live"],steps)
    if spec.get("known") is not None and expected != spec["known"]:
        raise AssertionError("independent oracle disagrees with known answer: " + spec["name"])
    input_path, output_path = work / "input.pbm", work / "actual.pbm"
    write_pbm(input_path,rows,cols,spec["live"])
    if output_path.exists(): output_path.unlink()
    params = dict(input=str(input_path),output=str(output_path),steps=steps,
                  rows=rows,cols=cols,ranks=ranks,threads=threads)
    command = [str(s).format(**params) for s in command_template]
    record = dict(variant=variant,case=spec["name"],rows=rows,cols=cols,steps=steps,
                  ranks=ranks,threads=threads,status="FAIL",mismatched_cells="",
                  first_mismatch="",expected_sha256=grid_sha256(rows,cols,expected),
                  actual_sha256="",detail="")
    evidence = dict(command=command, cwd=str(ROOT),input_live=sorted(spec["live"]),
                    environment={"OMP_NUM_THREADS":str(threads)},stdout="",stderr="")
    try:
        env = dict(os.environ,OMP_NUM_THREADS=str(threads))
        result = subprocess.run(command,cwd=str(ROOT),env=env,stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,universal_newlines=True,timeout=timeout)
        evidence.update(stdout=result.stdout,stderr=result.stderr,returncode=result.returncode)
        if result.returncode:
            raise ValueError("executable exited {}".format(result.returncode))
        out_rows,out_cols,actual = read_pbm(output_path)
        if (out_rows,out_cols) != (rows,cols):
            raise ValueError("output dimensions differ")
        diff = sorted(expected ^ actual)
        record.update(mismatched_cells=len(diff),
                      first_mismatch="" if not diff else "{},{}".format(*diff[0]),
                      actual_sha256=grid_sha256(rows,cols,actual))
        if diff: raise ValueError("complete-grid mismatch")
        metrics = json.loads(result.stdout)
        if not isinstance(metrics,dict) or metrics.get("variant") != variant:
            raise ValueError("JSON must identify the executed variant")
        validate_metrics(metrics,rows,cols,steps,ranks,threads,actual)
        for key, value in (expect or {}).items():
            if metrics.get(key) != value:
                raise ValueError("metric {}: expected {}, got {}".format(key,value,metrics.get(key)))
        evidence["metrics"] = metrics
        record.update(status="PASS",detail="all cells and diagnostics match")
    except (ValueError,OSError,subprocess.TimeoutExpired) as exc:
        record["detail"] = str(exc)
        target = failures / "{}_{}_p{}_t{}".format(variant,spec["name"],ranks,threads)
        target.mkdir(parents=True,exist_ok=True)
        shutil.copy2(input_path,target / "input.pbm")
        write_pbm(target / "expected.pbm",rows,cols,expected)
        if output_path.exists(): shutil.copy2(output_path,target / "actual.pbm")
        replay = [str(s).format(**dict(params,input="input.pbm",output="reproduced.pbm"))
                  for s in command_template]
        evidence["replay_from_project_root_with_input_paths_adjusted"] = replay
        (target / "failure.json").write_text(json.dumps(dict(record=record,evidence=evidence),indent=2))
    return record,evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary",default="./build/serial")
    parser.add_argument("--output",type=Path,default=Path("results/verification"))
    parser.add_argument("--external-config",type=Path,default=ROOT / "config/variants.json")
    parser.add_argument("--skip-external",action="store_true")
    parser.add_argument("--require-all",action="store_true")
    parser.add_argument("--timeout",type=float,default=30)
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    records=[]
    with tempfile.TemporaryDirectory(prefix="life_verify_") as tmp, (output/"commands.jsonl").open("w") as log:
        work=Path(tmp)
        serial=[str(Path(args.binary).resolve()),"--input","{input}","--steps","{steps}","--output","{output}"]
        for spec in all_cases():
            record,evidence=run_case(spec,serial,"serial",1,1,work,output/"failures",args.timeout)
            records.append(record)
            log.write(json.dumps(dict(record=record,evidence=evidence))+"\n")
        if not args.skip_external:
            config=json.loads(args.external_config.read_text())
            for variant in config["variants"]:
                template=variant["command"]
                # The executable itself, rather than just mpiexec, must exist.
                required=variant["required_paths"]
                missing=[p for p in required if not (ROOT/p).is_file()]
                if not shutil.which(template[0]) and not (ROOT/template[0]).is_file():
                    missing.append(template[0])
                if missing:
                    records.append(dict(variant=variant["name"],case="external_availability",
                        status="BLOCKED",detail="not supplied/available: "+", ".join(missing)))
                    continue
                for spec in named_cases():
                    for ranks in variant["ranks"]:
                        if ranks > spec["rows"]: continue
                        for threads in variant["threads"]:
                            record,evidence=run_case(spec,template,variant["name"],ranks,threads,
                                                     work,output/"failures",args.timeout)
                            records.append(record)
                            log.write(json.dumps(dict(record=record,evidence=evidence))+"\n")
    with (output/"correctness_results.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(records)
    counts={s:sum(r["status"]==s for r in records) for s in ("PASS","FAIL","BLOCKED")}
    summary=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 counts=counts, full_grid_comparison=True, independent_oracle=True,
                 all_requested_variants_verified=not counts["FAIL"] and not counts["BLOCKED"]
                     and not args.skip_external,
                 external_verification_requested=not args.skip_external,
                 limitation="No MPI/Hybrid correctness claim unless their actual runs pass.")
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary))
    return 1 if counts["FAIL"] else (2 if args.require_all and counts["BLOCKED"] else 0)


if __name__ == "__main__":
    raise SystemExit(main())
