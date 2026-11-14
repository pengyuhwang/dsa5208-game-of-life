"""Make two reproducible simulation figures from real serial C outputs."""
import json
import hashlib
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw, ImageFont
from oracle import evolve, read_pbm, write_pbm
from cases import named_cases
from benchmark import source_hash

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"figures"
FONT=Path("/System/Library/Fonts/Supplemental/Arial.ttf")


def font(size):
    if FONT.exists():return ImageFont.truetype(str(FONT),size)
    try:return ImageFont.truetype("DejaVuSans.ttf",size)
    except OSError:return ImageFont.load_default()


def panel(draw,x,y,rows,cols,live,title,cell=30):
    draw.text((x,y),title,font=font(24),fill="#142a3b")
    y+=42
    for r in range(rows):
        for c in range(cols):
            draw.rectangle((x+c*cell,y+r*cell,x+(c+1)*cell,y+(r+1)*cell),
                           fill="#147d92" if (r,c) in live else "#f4f7fa",outline="#c7d2dc",width=1)
    draw.rectangle((x,y,x+cols*cell,y+rows*cell),outline="#142a3b",width=2)


def state(spec,step,name,records):
    inp=OUT/"pbm"/(name+"_input.pbm")
    target=OUT/"pbm"/(name+"_t{}.pbm".format(step))
    write_pbm(inp,spec["rows"],spec["cols"],spec["live"])
    command=[str(ROOT/"build/serial"),"--input",str(inp),"--steps",str(step),"--output",str(target)]
    run=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,check=True)
    rows,cols,live=read_pbm(target)
    assert live==evolve(rows,cols,spec["live"],step)
    records.append(dict(command=command,metrics=json.loads(run.stdout),grid=str(target.relative_to(ROOT)),
                        source_sha256=source_hash(),binary_sha256=hashlib.sha256((ROOT/"build/serial").read_bytes()).hexdigest()))
    return rows,cols,live


def main():
    OUT.mkdir(exist_ok=True);records=[]
    specs={c["name"]:c for c in named_cases()}
    im=Image.new("RGB",(1440,900),"white");draw=ImageDraw.Draw(im)
    draw.text((64,36),"Known patterns verify the update rule",font=font(38),fill="#142a3b")
    draw.text((64,92),"Actual serial C output | teal = alive | periodic boundaries",font=font(23),fill="#526474")
    for row,name in enumerate(["block_stable","blinker_odd"]):
        for col,step in enumerate([0,1,2]):
            h,w,live=state(specs[name],step,name,records)
            panel(draw,64+col*460,160+row*340,h,w,live,
                  ("Stable block" if row==0 else "Blinker")+" | t = {}".format(step),cell=29)
    draw.text((64,846),"Expected: block unchanged; blinker returns to its initial state after two steps.",font=font(23),fill="#526474")
    im.save(OUT/"01_known_patterns.png")
    im=Image.new("RGB",(1440,1080),"white");draw=ImageDraw.Draw(im)
    draw.text((64,36),"A glider crosses periodic boundaries",font=font(38),fill="#142a3b")
    draw.text((64,92),"Actual serial C output | 12 rows x 13 columns | opposite edges are connected",font=font(23),fill="#526474")
    spec=specs["glider_wrap"]
    for i,step in enumerate([0,4,8,12]):
        h,w,live=state(spec,step,"glider_wrap",records)
        panel(draw,120+(i%2)*700,155+(i//2)*435,h,w,live,"t = {}".format(step),cell=27)
    draw.text((64,1030),"After every four steps, the glider shifts one row and one column, modulo the grid size.",font=font(23),fill="#526474")
    im.save(OUT/"02_periodic_glider.png")
    (OUT/"provenance.json").write_text(json.dumps(records,indent=2)+"\n")
    print("Created 2 figures from 10 C runs, each checked against the independent oracle")


if __name__=="__main__":main()
