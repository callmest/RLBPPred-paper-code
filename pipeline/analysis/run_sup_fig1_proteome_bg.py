#!/usr/bin/env python3
"""Sup Fig 1 enrichment with WHOLE reviewed human proteome as background.
positive direction: FG=positives(376) vs BG=proteome
negative direction: FG=negatives(9109) vs BG=proteome
Reuses annotation cache; fetches only missing IDs."""
import csv, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import go_domain_analysis_uniprot as g

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results"/"sup_fig1_enrichment"

def fa(p): return [l[1:].strip().split()[0] for l in open(p) if l.startswith(">")]
pos=fa(ROOT/"data/idr_analysis/known_376.fasta")
neg=fa(ROOT/"data/idr_analysis/negative_9109.fasta")
proteome=[l.strip() for l in open(OUT/"proteome_ids.txt") if l.strip()]
# ensure pos/neg are inside the background population
proteome=sorted(set(proteome)|set(pos)|set(neg))
print(f"pos={len(pos)} neg={len(neg)} proteome_bg={len(proteome)}",flush=True)

cache_f=OUT/"annotations_cache.json"
raw=json.load(open(cache_f))
ann={k:{s:set(v) for s,v in d.items()} for k,d in raw["ann"].items()}
go_names=dict(raw["go_names"])
print(f"cache has {len(ann)} proteins",flush=True)
missing=[p for p in proteome if p not in ann]
print(f"missing to fetch: {len(missing)}",flush=True)
if missing:
    ann2,gn2=g.fetch_uniprot_annotations(missing)
    ann.update(ann2); go_names.update(gn2)
    json.dump({"ann":{k:{s:sorted(v) for s,v in d.items()} for k,d in ann.items()},
               "go_names":go_names}, open(OUT/"annotations_cache_proteome.json","w"))
    print(f"now {len(ann)} proteins cached",flush=True)

GO=["GO:BP","GO:MF","GO:CC"]; DOM=["INTERPRO","PFAM"]
def run(fg,sources):
    r=g.enrich(fg,proteome,ann,sources,go_names)
    return sorted(r,key=lambda x:(x["source"],x["fdr_bh"],x["p_value"]))
def write(path,rows):
    cols=["direction","source","term_id","term_name","fg_hits","fg_size","bg_hits","bg_size",
          "fold_enrichment","p_value","fdr_bh"]
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore");w.writeheader()
        for r in rows:w.writerow(r)
allgo=[];alldom=[]
for direction,fg in [("positive",pos),("negative",neg)]:
    go=run(fg,GO);dom=run(fg,DOM)
    for r in go:r["direction"]=direction
    for r in dom:r["direction"]=direction
    allgo+=go;alldom+=dom
write(OUT/"supfig1a_go_enrichment_proteomeBG.csv",allgo)
write(OUT/"supfig1b_domain_enrichment_proteomeBG.csv",alldom)
print("WROTE proteomeBG csvs",flush=True)
def top(rows,d,s,n=6):
    sub=[r for r in rows if r["direction"]==d and r["source"]==s and r["fdr_bh"]<=0.05]
    sub=sorted(sub,key=lambda r:(r["fdr_bh"],r["p_value"]))[:n]
    return [(r["term_name"].lstrip('; ').strip(),round(r["fold_enrichment"],2),f'{r["fdr_bh"]:.2e}') for r in sub]
print("\n===== VERIFICATION (proteome BG) =====")
for d in ["positive","negative"]:
    for s in GO:
        print(f"[{d}|{s}]")
        for t in top(allgo,d,s):print("   ",t)
