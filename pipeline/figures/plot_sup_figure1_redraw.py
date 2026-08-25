#!/usr/bin/env python3
"""
Supplementary Figure 1 — GO/InterPro domain enrichment of the high-confidence
positive vs negative benchmark, computed against the whole reviewed human
proteome (one-sided Fisher exact, Benjamini-Hochberg FDR).

Self-contained (no external skill dependency). Journal-standard styling:
Nature double-column (183 mm), Okabe-Ito colorblind-safe palette, Arial
(falls back to DejaVu Sans if Arial is unavailable), mathtext subscripts.

Inputs  (results/sup_fig1_enrichment/):
    supfig1a_go_enrichment_proteomeBG.csv
    supfig1b_domain_enrichment_proteomeBG.csv
    ipr_names.json                      # InterPro accession -> short name
Outputs (same dir):
    Sup_figure1_redraw.{pdf,png,svg}
"""
import csv, json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec

BASE = Path(__file__).resolve().parents[2] / "results" / "sup_fig1_enrichment"

plt.rcParams.update({
    "font.family":"sans-serif",
    "font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "mathtext.default":"regular","mathtext.fontset":"dejavusans",
    "svg.fonttype":"none","pdf.fonttype":42,"ps.fonttype":42,
    "figure.dpi":300,"savefig.dpi":600,"savefig.bbox":"tight",
    "axes.titleweight":"bold","axes.linewidth":0.8,
    "font.size":8,"axes.labelsize":8,"xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":7,
})
CAT={"GO:BP":"#009E73","GO:CC":"#0072B2","GO:MF":"#E69F00"}  # Okabe-Ito green/blue/orange
DOMC="#CC79A7"  # Okabe-Ito reddish-purple
THR=-math.log10(0.05)
GRID=dict(linestyle=(0,(3,1.5)),color="#b0b0b0",linewidth=0.6,alpha=0.6)
def load(f): return list(csv.DictReader(open(BASE/f)))
def nlog(r): return -math.log10(max(float(r["fdr_bh"]),1e-300))

# ---- panel a: diverging GO (top 5 per category per direction) ----
go=load("supfig1a_go_enrichment_proteomeBG.csv")
def pick(src,d,n=5):
    s=[r for r in go if r["source"]==src and r["direction"]==d and float(r["fdr_bh"])<0.05]
    s.sort(key=lambda r:(float(r["fdr_bh"]),float(r["p_value"]))); return s[:n]
def nm(r): return r["term_name"].lstrip("; ").strip()
rows=[]
for src in ["GO:BP","GO:CC","GO:MF"]:
    blk=[(nm(r),nlog(r),CAT[src]) for r in pick(src,"positive")]
    blk+=[(nm(r),-nlog(r),CAT[src]) for r in pick(src,"negative")]
    blk.sort(key=lambda x:x[1],reverse=True); rows+=blk

fig=plt.figure(figsize=(7.2,8.6))
gs=GridSpec(2,1,height_ratios=[2.55,1.0],hspace=0.16,figure=fig)
axa=fig.add_subplot(gs[0]); y=np.arange(len(rows))
axa.barh(y,[r[1] for r in rows],color=[r[2] for r in rows],edgecolor="black",linewidth=0.4,height=0.7)
axa.axvline(0,color="black",linewidth=0.8); axa.axvline(THR,**GRID); axa.axvline(-THR,**GRID)
axa.set_yticks(y); axa.set_yticklabels([r[0] for r in rows]); axa.invert_yaxis()
axa.set_xlabel(r"Enrichment score ($\mathregular{-log_{10}}$ FDR)")
mx=max(abs(r[1]) for r in rows)*1.08; axa.set_xlim(-mx,mx)
axa.xaxis.set_major_formatter(mt.FuncFormatter(lambda v,_:f"{abs(v):.0f}"))
axa.text(0.99,1.008,"Positive (RLBP)",transform=axa.transAxes,ha="right",va="bottom",fontweight="bold",fontsize=8)
axa.text(0.01,1.008,"Negative (Non-RLBP)",transform=axa.transAxes,ha="left",va="bottom",fontweight="bold",fontsize=8)
axa.grid(axis="x",**GRID); axa.set_axisbelow(True)
for sp in ["top","right"]: axa.spines[sp].set_visible(False)
axa.legend(handles=[Patch(facecolor=CAT[s],edgecolor="black",linewidth=0.4,label=s.replace("GO:","")) for s in ["GO:BP","GO:CC","GO:MF"]],
           title="GO category",loc="lower left",frameon=False)
axa.text(0.0,1.04,"a",transform=axa.transAxes,fontsize=11,fontweight="bold",va="bottom")

# ---- panel b: InterPro domains (positive, top 10) ----
ipr_names=json.load(open(BASE/"ipr_names.json"))
dom=load("supfig1b_domain_enrichment_proteomeBG.csv")
ipr=[r for r in dom if r["source"]=="INTERPRO" and r["direction"]=="positive" and float(r["fdr_bh"])<0.05]
ipr.sort(key=lambda r:(float(r["fdr_bh"]),float(r["p_value"]))); ipr=ipr[:10]
axb=fig.add_subplot(gs[1])
vb=[nlog(r) for r in ipr]; lb=[ipr_names.get(r["term_id"],r["term_id"]) for r in ipr]
o=np.argsort(vb)[::-1]; vb=[vb[i] for i in o]; lb=[lb[i] for i in o]
axb.barh(np.arange(len(ipr)),vb,color=DOMC,edgecolor="black",linewidth=0.4,height=0.68)
axb.axvline(THR,**GRID)
axb.set_yticks(np.arange(len(ipr))); axb.set_yticklabels(lb); axb.invert_yaxis()
axb.set_xlabel(r"Enrichment score ($\mathregular{-log_{10}}$ FDR)")
axb.grid(axis="x",**GRID); axb.set_axisbelow(True)
for sp in ["top","right"]: axb.spines[sp].set_visible(False)
axb.text(0.0,1.04,"b",transform=axb.transAxes,fontsize=11,fontweight="bold",va="bottom")

for ext in ("pdf","png","svg"):
    fig.savefig(BASE/f"Sup_figure1_redraw.{ext}")
print("Saved Sup_figure1_redraw.{pdf,png,svg} ->", BASE)
