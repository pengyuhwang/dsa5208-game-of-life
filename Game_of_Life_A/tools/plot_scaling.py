"""Draw the serial scaling figure from recorded results, with logarithmic axes."""
import json
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
SURFACE,INK,MUTED,GRID,SERIES="#fcfcfb","#0b0b0b","#52514e","#e3e2de","#2a78d6"
FONTS=["/System/Library/Fonts/Supplemental/Arial.ttf","DejaVuSans.ttf"]


def font(size):
    for name in FONTS:
        try:return ImageFont.truetype(name,size)
        except OSError:pass
    return ImageFont.load_default()


def nice_ticks(low,high):
    ticks=[]
    for exponent in range(math.floor(math.log10(low))-1,math.ceil(math.log10(high))+1):
        for lead in (1,2,5):
            value=lead*10**exponent
            if low<=value<=high:ticks.append(value)
    return ticks if len(ticks)<=7 else [t for t in ticks if str(t).lstrip("0.").startswith("1")]


def label(value):
    return "{:g}".format(value)


def panel(draw,box,title,ylabel,sizes,cells,values,annotation,reference=None,pad=1.5):
    x0,y0,x1,y1=box
    low,high=min(values)/pad,max(values)*pad
    if reference:low,high=min(low,min(reference)/1.2),max(high,max(reference)*1.2)
    X=lambda c:x0+(math.log(c)-math.log(cells[0]))/(math.log(cells[-1])-math.log(cells[0]))*(x1-x0)
    Y=lambda v:y1-(math.log(v)-math.log(low))/(math.log(high)-math.log(low))*(y1-y0)
    draw.text((x0-96,y0-104),title,font=font(32),fill=INK)
    draw.text((x0-96,y0-60),ylabel,font=font(24),fill=MUTED)
    for tick in nice_ticks(low,high):
        draw.line((x0,Y(tick),x1,Y(tick)),fill=GRID,width=2)
        w=draw.textlength(label(tick),font=font(24))
        draw.text((x0-w-14,Y(tick)-14),label(tick),font=font(24),fill=MUTED)
    for size,c in zip(sizes,cells):
        w=draw.textlength(str(size),font=font(24))
        draw.line((X(c),y1,X(c),y1+8),fill=MUTED,width=2)
        draw.text((X(c)-w/2,y1+14),str(size),font=font(24),fill=MUTED)
    text="Grid side length (log scale)"
    draw.text(((x0+x1)/2-draw.textlength(text,font=font(24))/2,y1+50),text,font=font(24),fill=MUTED)
    if reference:
        points=[(X(c),Y(v)) for c,v in zip(cells,reference)]
        for a,b in zip(points,points[1:]):
            n=int(math.hypot(b[0]-a[0],b[1]-a[1])//18)
            for i in range(0,max(n,1),2):
                draw.line((a[0]+(b[0]-a[0])*i/max(n,1),a[1]+(b[1]-a[1])*i/max(n,1),
                           a[0]+(b[0]-a[0])*(i+1)/max(n,1),a[1]+(b[1]-a[1])*(i+1)/max(n,1)),fill=MUTED,width=2)
    points=[(X(c),Y(v)) for c,v in zip(cells,values)]
    draw.line(points,fill=SERIES,width=6,joint="curve")
    for x,y in points:
        draw.ellipse((x-13,y-13,x+13,y+13),fill=SURFACE)
        draw.ellipse((x-9,y-9,x+9,y+9),fill=SERIES)
    w=draw.textlength(annotation,font=font(24))
    draw.rectangle((x1-w-14,y1-54,x1,y1-14),fill=SURFACE)
    draw.text((x1-w-6,y1-48),annotation,font=font(24),fill=INK)


def draw_figure(summary_path,out_path):
    s=json.loads(Path(summary_path).read_text())
    g=s["groups"];sizes=[x["size"] for x in g];cells=[x["cells"] for x in g]
    times=[x["median_seconds"] for x in g];per=[x["median_ns_per_cell_update"] for x in g]
    image=Image.new("RGB",(2200,960),SURFACE);draw=ImageDraw.Draw(image)
    ideal=[times[0]*c/cells[0] for c in cells]
    panel(draw,(150,190,1030,800),"Serial kernel time","seconds, {} generations (log scale)".format(s["steps"]),sizes,cells,times,
          "Fitted: time grows as cells^{:.2f}. Dashed: exactly linear".format(s["exponent_time_vs_cells"]),ideal)
    panel(draw,(1270,190,2150,800),"Cost of one cell update","nanoseconds (log scale)",sizes,cells,per,
          "A flat line means a constant cost per cell",pad=2.2)
    source="Median of {} repeats on {}, one core.".format(g[0]["repeats"],"Vanda" if s["environment"]=="vanda" else "a local laptop (development only)")
    draw.text((54,904),source,font=font(24),fill=MUTED)
    Path(out_path).parent.mkdir(parents=True,exist_ok=True);image.save(out_path)


if __name__=="__main__":
    draw_figure(ROOT/"results/serial_scaling/summary.json",ROOT/"figures/03_serial_scaling.png")
