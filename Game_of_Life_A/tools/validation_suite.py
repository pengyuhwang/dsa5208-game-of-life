"""The test suite described in the report: each row below is one reported test.

tools/verify.py keeps the wider regression. This suite is deliberately small: a
reader can see why each test exists. --describe writes the table for the report;
--run executes it, comparing complete grids against the independent oracle.
"""
import argparse
import csv
import datetime
import filecmp
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from cases import all_cases, case, named_cases
from verify import ROOT, run_case, validate_metrics

COMMS=["nonblocking","overlap","rma"]
LARGE=dict(rows=1024,cols=768,steps=20,seed=20261026,density=500000,ranks=36)


def extra_cases():
    return [case("local_cols_one",7,4,{(0,0),(1,0),(2,3),(6,1),(3,2)},9),
            case("single_row_grid",1,9,{(0,0),(0,1),(0,4),(0,8)},5),
            case("single_col_grid",9,1,{(0,0),(1,0),(4,0),(8,0)},5)]


def report_plan():
    plan=[]
    def add(group,binary,name,purpose,decomp=None,comm=None,ranks=1,threads=1):
        plan.append(dict(id="{}{:02d}".format(group,1+sum(t["id"][0]==group for t in plan)),
                         binary=binary,case=name,purpose=purpose,decomp=decomp,comm=comm,
                         ranks=ranks,threads=threads))
    notes={"empty":"An empty grid stays empty.","single_dies":"Underpopulation kills an isolated cell.",
           "block_stable":"A still life is unchanged over 20 generations.",
           "blinker_odd":"An oscillator has the known phase after one generation.",
           "blinker_even":"The oscillator returns to its start after an even count.",
           "corner_block":"A block split across all four corners is stable, testing the torus corners.",
           "edge_blinker":"An oscillator crossing the left and right edges.",
           "glider_four":"A glider moves one row and column in four generations.",
           "glider_wrap":"A glider crosses the bottom and right edges and keeps its shape.",
           "one_cell_alive":"The smallest grid: wrapped offsets are counted with multiplicity.",
           "all_alive":"Overcrowding kills a full grid.","zero_steps":"Zero generations return the input unchanged.",
           "local_rows_one":"Four rows, later used so each rank owns a single row.",
           "nondivisible_rows":"Five rows, later used so rank block sizes differ."}
    for spec in named_cases():add("S","serial",spec["name"],notes[spec["name"]])
    add("S","serial","exhaustive_3x3","All 512 states of a 3x3 torus, one generation, reported as one test: this exhausts every neighbourhood of the rule.")
    add("S","serial","random_shapes","84 random grids in 14 shapes, including single rows and columns, up to 25 generations, reported as one test.")
    for p in [1,2,3,4,6,8,9,12]:
        add("D","mpi","glider_wrap","Row blocks at P={}: halo rows and the wrapped ring.".format(p),"1d","blocking",p)
    for p in [1,2,3,4,6,8,9,12]:
        add("D","mpi","glider_wrap","Process grid for P={}: all eight ghost directions.".format(p),"2d","blocking",p)
    add("D","mpi","local_rows_one","Each rank owns one row in 1D.","1d","blocking",4)
    add("D","mpi","local_rows_one","2x2 process grid, where north equals south and diagonal neighbours repeat.","2d","blocking",4)
    add("D","mpi","local_rows_one","4x4 process grid: every block is a single row.","2d","blocking",16)
    add("D","mpi","local_cols_one","2x2 process grid on a narrow grid.","2d","blocking",4)
    add("D","mpi","local_cols_one","4x4 process grid: every block is a single column.","2d","blocking",16)
    for p in [2,4]:add("D","mpi","nondivisible_rows","Uneven row blocks at P={}.".format(p),"1d","blocking",p)
    for p in [2,4,6]:add("D","mpi","nondivisible_rows","Uneven rows and columns at P={}.".format(p),"2d","blocking",p)
    add("D","mpi","corner_block","Corner ghosts in 1D come with the full halo row.","1d","blocking",4)
    add("D","mpi","corner_block","Corner ghosts need diagonal messages on a 2x2 process grid.","2d","blocking",4)
    add("D","mpi","corner_block","Corner ghosts on a 3x3 process grid, with no repeated neighbours.","2d","blocking",9)
    add("D","mpi","edge_blinker","An oscillator crossing a block boundary.","2d","blocking",4)
    add("D","mpi","one_cell_alive","One rank, one cell, 1D path.","1d","blocking",1)
    add("D","mpi","one_cell_alive","One rank, one cell, 2D path.","2d","blocking",1)
    add("D","mpi","all_alive","Overcrowding across block boundaries.","2d","blocking",4)
    add("D","mpi","zero_steps","Distribution and collection only, 1D: no update runs.","1d","blocking",4)
    add("D","mpi","zero_steps","Distribution and collection only, 2D: no update runs.","2d","blocking",4)
    add("D","mpi","single_row_grid","A one-row grid on a 1x3 process grid.","2d","blocking",3)
    add("D","mpi","single_col_grid","A one-column grid on a 3x1 process grid.","2d","blocking",3)
    for comm in COMMS:
        for decomp in ["1d","2d"]:
            add("C","mpi","glider_wrap","{} at P=2, where both neighbours are one rank.".format(comm),decomp,comm,2)
            add("C","mpi","glider_wrap","{} at P=4.".format(comm),decomp,comm,4)
            add("C","mpi","nondivisible_rows","{} with uneven blocks.".format(comm),decomp,comm,4)
            add("C","mpi","local_rows_one","{} when blocks are too thin for an interior.".format(comm),decomp,comm,4)
        add("C","mpi","glider_wrap","{} on a 2x4 process grid.".format(comm),"2d",comm,8)
    # The two later methods reuse the layouts above, so each gets only the cases
    # that stress what is new in it: request reuse, and the send/receive order.
    add("C","mpi","glider_wrap","persistent at P=2: both neighbours are one rank, and the two request sets alternate with the grid buffers.","1d","persistent",2)
    add("C","mpi","nondivisible_rows","persistent with uneven row blocks and an odd number of generations.","1d","persistent",4)
    add("C","mpi","local_rows_one","persistent on a 2x2 process grid, where neighbours repeat and blocks have no interior.","2d","persistent",4)
    add("C","mpi","glider_wrap","persistent on a 2x4 process grid: the column buffers have their own requests.","2d","persistent",8)
    add("C","mpi","glider_wrap","sendrecv2 at P=2: one rank sends first and the other receives first.","1d","sendrecv2",2)
    add("C","mpi","glider_wrap","sendrecv2 on an odd ring (P=3), where two neighbouring ranks both send first.","1d","sendrecv2",3)
    add("C","mpi","glider_wrap","sendrecv2 on a 3x3 process grid: odd cycles along rows, columns and diagonals.","2d","sendrecv2",9)
    add("C","mpi","nondivisible_rows","sendrecv2 on a 2x3 process grid with uneven blocks.","2d","sendrecv2",6)
    add("H","hybrid","glider_wrap","Threads over row blocks.","1d","blocking",2,2)
    add("H","hybrid","glider_wrap","Threads over 2D blocks.","2d","blocking",4,2)
    add("H","hybrid","glider_wrap","Interior rows computed by threads while messages are in flight.","2d","overlap",2,4)
    add("H","hybrid","glider_wrap","Threads between RMA synchronisation points.","2d","rma",4,2)
    add("H","hybrid","nondivisible_rows","More threads than local rows.","1d","blocking",2,4)
    add("L","mpi","large_random","Larger random grid, row blocks, byte-compared with the serial output.","1d","blocking",LARGE["ranks"])
    for comm in ["blocking"]+COMMS:
        add("L","mpi","large_random","Larger random grid on a 6x6 process grid using {}, byte-compared with the serial output.".format(comm),"2d",comm,LARGE["ranks"])
    return plan


def describe(path):
    lines=["# Validation tests in the report\n",
           "Generated by `python3 tools/validation_suite.py --describe`. {} tests.\n".format(len(report_plan())),
           "S = serial rule, D = decomposition, C = communication method, H = threads, L = larger grid.",
           "Every test except L compares the complete final grid with an independent Python oracle.\n",
           "| ID | Program | Case | Decomposition | Communication | Ranks x threads | Why it exists |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for t in report_plan():
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(t["id"],t["binary"],t["case"],t["decomp"] or "-",
                     t["comm"] or "-","{} x {}".format(t["ranks"],t["threads"]),t["purpose"]))
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")


def template(binary,launcher,test):
    if binary=="serial":return [str(ROOT/"build/serial"),"--input","{input}","--steps","{steps}","--output","{output}"]
    return launcher+["-np","{ranks}",str(ROOT/"build"/binary),"--input","{input}","--steps","{steps}",
                     "--output","{output}","--decomp",test["decomp"],"--comm",test["comm"]]


def run_large(test,launcher,work,reference,timeout):
    output=work/"large.pbm"
    command=launcher+["-np",str(test["ranks"]),str(ROOT/"build"/test["binary"]),
        "--rows",str(LARGE["rows"]),"--cols",str(LARGE["cols"]),"--steps",str(LARGE["steps"]),
        "--seed",str(LARGE["seed"]),"--density-ppm",str(LARGE["density"]),"--output",str(output),
        "--decomp",test["decomp"],"--comm",test["comm"]]
    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,
                          timeout=timeout,env=dict(os.environ,OMP_NUM_THREADS="1"))
    if result.returncode:return "FAIL","exit {}: {}".format(result.returncode,result.stderr[-300:])
    metrics=json.loads(result.stdout)
    validate_metrics(metrics,LARGE["rows"],LARGE["cols"],LARGE["steps"],test["ranks"],1)
    if metrics.get("decomp")!=test["decomp"] or metrics.get("comm")!=test["comm"]:
        return "FAIL","decomp/comm echo mismatch"
    if metrics["checksum_fnv1a64"]!=reference["checksum_fnv1a64"]:return "FAIL","checksum differs from serial"
    if not filecmp.cmp(str(output),str(work/"large_serial.pbm"),shallow=False):return "FAIL","output bytes differ from serial"
    return "PASS","checksum and complete output bytes match serial"


def run(args):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    launcher=args.launcher.split()
    specs={c["name"]:c for c in named_cases()+extra_cases()}
    rows=[];counts={"PASS":0,"FAIL":0,"BLOCKED":0}
    with tempfile.TemporaryDirectory(prefix="life_report_") as tmp:
        work=Path(tmp)
        families={"exhaustive_3x3":[c for c in all_cases() if c["name"].startswith("exhaustive_")],
                  "random_shapes":[c for c in all_cases() if c["name"].startswith("random_")]}
        reference=None
        for test in report_plan():
            status,detail,runs="BLOCKED","",1
            binary=ROOT/"build"/test["binary"]
            if test["binary"]!="serial" and not binary.is_file():
                detail="not available: build/"+test["binary"]
            elif test["case"]=="large_random":
                if reference is None:
                    result=subprocess.run([str(ROOT/"build/serial"),"--rows",str(LARGE["rows"]),"--cols",str(LARGE["cols"]),
                        "--steps",str(LARGE["steps"]),"--seed",str(LARGE["seed"]),"--density-ppm",str(LARGE["density"]),
                        "--output",str(work/"large_serial.pbm")],stdout=subprocess.PIPE,universal_newlines=True,check=True)
                    reference=json.loads(result.stdout)
                status,detail=run_large(test,launcher,work,reference,args.timeout)
            else:
                specs_now=families.get(test["case"]) or [specs[test["case"]]]
                runs=len(specs_now);failed=[]
                expect=None if test["binary"]=="serial" else dict(decomp=test["decomp"],comm=test["comm"])
                for spec in specs_now:
                    record,_=run_case(spec,template(test["binary"],launcher,test),test["binary"],test["ranks"],
                                      test["threads"],work,output/"failures",args.timeout,expect=expect)
                    if record["status"]!="PASS":failed.append(spec["name"]+": "+record["detail"])
                status="FAIL" if failed else "PASS"
                detail="; ".join(failed[:3]) or "complete grid and JSON match the oracle"
            counts[status]+=1
            rows.append(dict(test,status=status,runs=runs,detail=detail))
            print(test["id"],status,detail[:100],flush=True)
    fields=["id","binary","case","decomp","comm","ranks","threads","runs","status","purpose","detail"]
    with (output/"validation_report.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    summary=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),counts=counts,tests=len(rows),
                 environment=args.environment,job_id=os.environ.get("PBS_JOBID",""),launcher=args.launcher)
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary))
    return 1 if counts["FAIL"] else (2 if args.require_all and counts["BLOCKED"] else 0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe",type=Path)
    parser.add_argument("--run",action="store_true")
    parser.add_argument("--output",type=Path,default=ROOT/"results/validation_report")
    parser.add_argument("--launcher",default="mpirun")
    parser.add_argument("--environment",choices=["local","vanda"],default="local")
    parser.add_argument("--require-all",action="store_true")
    parser.add_argument("--timeout",type=float,default=120)
    args=parser.parse_args()
    if args.describe:describe(args.describe)
    return run(args) if args.run else 0


if __name__=="__main__":raise SystemExit(main())
