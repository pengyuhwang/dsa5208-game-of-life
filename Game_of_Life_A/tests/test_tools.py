import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from oracle import evolve, random_grid, read_pbm, write_pbm
from verify import validate_metrics
from verify import run_case
from audit_results import audit
from benchmark import require_environment
from validation_suite import report_plan
from unittest.mock import patch


def run_text(command,**options):
    return subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                          universal_newlines=True,**options)


def build_decomp_test():
    command=[os.environ.get("CC","cc"),"-std=c11","-Wall","-Wextra","-Wpedantic","-Wconversion",
             "-Wshadow","-fsanitize=address,undefined","-g","-Iinclude","tests/test_decomp.c",
             "src/life.c","src/life_decomp.c","-o","build/test_decomp"]
    result=run_text(command,cwd=str(ROOT))
    if result.returncode:raise AssertionError(result.stderr)
    return str(ROOT/"build/test_decomp")


def model_layout(h,w,p):
    """Independent 2D layout model: search factor pairs instead of walking from sqrt(p)."""
    pairs=sorted(((a,p//a) for a in range(1,p+1) if p%a==0 and a*a<=p),key=lambda q:q[1]-q[0])
    dims=None
    for a,b in pairs:
        dims=(a,b) if a<=h and b<=w else (b,a) if b<=h and a<=w else None
        if dims:break
    if not dims:return ["invalid"]
    R,C=dims;lines=["dims {} {}".format(R,C)]
    def block(total,parts,i):
        sizes=[total//parts+(k<total%parts) for k in range(parts)]
        return sizes[i],sum(sizes[:i])
    for rank in range(p):
        r,c=divmod(rank,C);rows,ro=block(h,R,r);cols,co=block(w,C,c)
        at=lambda dr,dc:((r+dr)%R)*C+(c+dc)%C
        nb=[at(-1,0),at(1,0),at(0,-1),at(0,1),at(-1,-1),at(-1,1),at(1,-1),at(1,1)]
        lines.append(" ".join(map(str,[rows,cols,ro,co]+nb)))
    return lines


class SerialContractTests(unittest.TestCase):
    def run_serial(self,args):
        return run_text([str(ROOT/"build/serial")]+args,timeout=15)

    def test_stateless_seed_and_density(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"grid.pbm"
            for seed in [0,1,42,(1<<64)-1]:
                for density in [0,1,123456,500000,999999,1000000]:
                    result=self.run_serial(["--rows","7","--cols","11","--steps","0",
                        "--seed",str(seed),"--density-ppm",str(density),"--output",str(out)])
                    self.assertEqual(result.returncode,0,result.stderr)
                    self.assertEqual(read_pbm(out),(7,11,random_grid(7,11,seed,density)))

    def test_seeded_evolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"grid.pbm"
            result=self.run_serial(["--rows","17","--cols","19","--steps","31",
                                   "--seed","18446744073709551615","--output",str(out)])
            self.assertEqual(result.returncode,0,result.stderr)
            expected=evolve(17,19,random_grid(17,19,(1<<64)-1,500000),31)
            self.assertEqual(read_pbm(out),(17,19,expected))

    def test_bad_cli(self):
        cases=[["--rows","0"],["--cols","-1"],["--steps","-1"],
               ["--seed","18446744073709551616"],["--density-ppm","1000001"],
               ["--rows","18446744073709551615"],["--rows","3.5"],
               ["--cols"],["--unknown"],["--steps","1","--steps","2"],
               ["--seed","+1"],["--seed"," 1"],["--rows","2x"],
               ["--output","/nonexistent-life-test-dir/output.pbm"]]
        for args in cases:
            with self.subTest(args=args):
                result=self.run_serial(args)
                self.assertNotEqual(result.returncode,0)
                self.assertTrue(result.stderr)
                self.assertFalse(result.stdout)

    def test_bad_pbm(self):
        cases=["P1", "P1\n1", "P4\n1 1\n0", "P1\n0 2\n", "P1\n2 2\n0 1 0",
               "P1\n1 1\n2", "P1\n1 1\n0 1", "P1\n-1 1\n0",
               "P1\n"+"9"*80+" 1\n0", "P1\n1 1\n00", "P1\n1 1\n"]
        with tempfile.TemporaryDirectory() as tmp:
            inp=Path(tmp)/"bad.pbm"
            for text in cases:
                with self.subTest(text=text):
                    inp.write_text(text)
                    result=self.run_serial(["--input",str(inp)])
                    self.assertNotEqual(result.returncode,0)
                    self.assertTrue(result.stderr)

    def test_pbm_comments_and_dimension_agreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp=Path(tmp)/"input.pbm";out=Path(tmp)/"out.pbm"
            inp.write_text("P1 #type\n3 #width\n1\n0 1#pixel\n0 #end\n")
            result=self.run_serial(["--input",str(inp),"--steps","0","--output",str(out)])
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(read_pbm(out),(1,3,{(0,1)}))
            for extra in [["--rows","2"],["--cols","4"],["--seed","1"],
                          ["--density-ppm","0"]]:
                result=self.run_serial(["--input",str(inp)]+extra)
                self.assertNotEqual(result.returncode,0)
            inp.write_text("P1\n1 1\n0#no-final-newline")
            result=self.run_serial(["--input",str(inp),"--steps","0","--output",str(out)])
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(read_pbm(out),(1,1,set()))

    def test_same_live_count_is_not_equality(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/"a.pbm";b=Path(tmp)/"b.pbm"
            write_pbm(a,4,5,{(0,0)});write_pbm(b,4,5,{(0,1)})
            result=run_text([sys.executable,str(ROOT/"tools/compare_grids.py"),str(a),str(b)])
            self.assertEqual(result.returncode,1)
            self.assertEqual(json.loads(result.stdout)["mismatched_cells"],2)

    def test_invalid_timing_rejected(self):
        metrics=dict(interface_version="1.0",rows=3,cols=3,steps=1,ranks=1,threads=1,
                     kernel_seconds=1.0,total_seconds=1.5)
        validate_metrics(metrics,3,3,1,1,1)
        for bad in [float("nan"),float("inf"),-1,True,"1"]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    validate_metrics(dict(metrics,kernel_seconds=bad),3,3,1,1,1)
        with self.assertRaises(ValueError):
            validate_metrics(dict(metrics,total_seconds=0.1),3,3,1,1,1)
        with self.assertRaises(ValueError):
            validate_metrics(dict(metrics,ranks=True),3,3,1,1,1)

    def test_failure_and_timeout_evidence(self):
        spec=dict(name="deliberately_broken_backend",rows=3,cols=5,steps=1,
                  live={(1,2)},known=set())
        commands=[[sys.executable,"-c","raise SystemExit(7)"],
                  [sys.executable,"-c","import time; time.sleep(2)"]]
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp)
            for i,command in enumerate(commands):
                failures=work/("failure_"+str(i))
                record,evidence=run_case(spec,command,"serial",1,1,work,failures,0.1)
                self.assertEqual(record["status"],"FAIL")
                outputs=list(failures.rglob("failure.json"))
                self.assertEqual(len(outputs),1)
                self.assertTrue((outputs[0].parent/"input.pbm").is_file())
                self.assertTrue((outputs[0].parent/"expected.pbm").is_file())

    def test_hpc_label_requires_real_environment(self):
        with patch("benchmark.platform.system",return_value="Darwin"):
            with self.assertRaises(ValueError):require_environment("vanda")

    def test_auditor_rejects_invalid_measurements(self):
        import csv
        row=dict(run_id="unit-test-fixture",owner="A",environment="local",hostname="fixture",
                 job_id="",repeat="1",variant="serial",rows="3",cols="5",steps="2",
                 seed="1",density_ppm="500000",ranks="1",threads="1",nodes="1",status="PASS",
                 kernel_seconds="-1",total_seconds="1",live_cells="0",checksum_fnv1a64="0"*16,
                 source_sha256="fixture",binary_sha256="fixture",full_grid_match="True",detail="fixture only")
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"fixture.csv"
            with path.open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerow(row)
            self.assertTrue(audit([path])["errors"])

    def test_core_slab_contract(self):
        command=[os.environ.get("CC","cc"),"-std=c11","-Wall","-Wextra","-Wpedantic",
                 "-fsanitize=address,undefined","-g","-Iinclude","tests/test_core.c",
                 "src/life.c","-o","build/test_core"]
        result=run_text(command,cwd=str(ROOT))
        self.assertEqual(result.returncode,0,result.stderr)
        result=run_text([str(ROOT/"build/test_core")],timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_2d_layout_contract(self):
        binary=build_decomp_test()
        result=run_text([binary],timeout=60)
        self.assertEqual(result.returncode,0,result.stderr)
        for h,w,p in [(1,9,3),(9,1,3),(2,2,4),(3,3,9),(4,7,16),(7,4,16),(5,7,6),(12,13,8),
                      (12,13,12),(97,53,8),(2,2,5),(1,1,2),(256,256,256),(1024,1024,72)]:
            with self.subTest(h=h,w=w,p=p):
                result=run_text([binary,"--probe",str(h),str(w),str(p),"2d"],timeout=15)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertEqual(result.stdout.split("\n")[:-1],model_layout(h,w,p))

    def test_report_suite_is_small_and_described(self):
        plan=report_plan()
        self.assertLess(len(plan),100)
        self.assertEqual(len({t["id"] for t in plan}),len(plan))
        for test in plan:
            self.assertTrue(test["purpose"])
            if test["binary"]!="serial":
                self.assertIn(test["decomp"],("1d","2d"))
                self.assertLessEqual(test["ranks"]*test["threads"],36)
        by_comm={t["comm"] for t in plan if t["binary"]!="serial"}
        self.assertEqual(by_comm,{"blocking","nonblocking","overlap","rma","persistent","sendrecv2"})

    def test_unexpected_json_field_is_a_failure(self):
        spec=dict(name="json_echo",rows=3,cols=5,steps=1,live={(1,2)},known=set())
        command=[str(ROOT/"build/serial"),"--input","{input}","--steps","{steps}","--output","{output}"]
        with tempfile.TemporaryDirectory() as tmp:
            work=Path(tmp)
            record,_=run_case(spec,command,"serial",1,1,work,work/"failures",15,expect={"decomp":"2d"})
            self.assertEqual(record["status"],"FAIL")
            self.assertIn("decomp",record["detail"])


if __name__=="__main__": unittest.main()
