"""Compare complete PBM grids. Exit 0=identical, 1=different, 2=invalid input."""
import argparse
import json
from pathlib import Path
from oracle import read_pbm


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("expected",type=Path);p.add_argument("actual",type=Path)
    args=p.parse_args()
    try:
        er,ec,e=read_pbm(args.expected);ar,ac,a=read_pbm(args.actual)
        if (er,ec)!=(ar,ac):
            print(json.dumps(dict(equal=False,reason="dimensions differ",expected=[er,ec],actual=[ar,ac])))
            return 1
        differences=sorted(e^a)
        print(json.dumps(dict(equal=not differences,mismatched_cells=len(differences),
                              first_mismatch=differences[0] if differences else None,
                              expected_live=len(e),actual_live=len(a))))
        return int(bool(differences))
    except (OSError,ValueError) as exc:
        print(json.dumps(dict(error=str(exc))));return 2


if __name__=="__main__": raise SystemExit(main())
