"""Recompute current GO tests only from explicitly supplied frozen annotations.

This opt-in upstream command is separate from figure replay. It never downloads
annotations, propagates GO ancestors, or silently treats unreturned proteins as
GO-negative. The public frozen result CSV supports replay without these inputs.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def bh_adjust(values):
    order=sorted(range(len(values)),key=values.__getitem__)
    adjusted=[1.0]*len(values);previous=1.0
    for i in range(len(order)-1,-1,-1):
        index=order[i];previous=min(previous,values[index]*len(order)/(i+1))
        adjusted[index]=min(1.0,previous)
    return adjusted


def enrich_frozen(annotations,candidates,returned,min_pool_count=5):
    """One-sided Fisher, complete eligible family, BH separately by aspect."""
    from scipy.stats import fisher_exact
    candidates=set(candidates);returned=set(returned)
    if not candidates<=returned:raise ValueError('Candidate annotations missing from returned universe')
    background=returned-candidates
    if not candidates or not background:raise ValueError('Both foreground and background are required')
    terms=defaultdict(set);names={}
    for row in annotations:
        if row['protein_id'] not in returned:continue
        if row['aspect'] not in ['BP','CC','MF']:raise ValueError('GO aspect must be BP, CC or MF')
        key=(row['aspect'],row['term_id']);terms[key].add(row['protein_id']);names[key]=row['term_name']
    results=[]
    for (aspect,term),members in sorted(terms.items()):
        if len(members)<min_pool_count:continue
        fg=members&candidates;bg=members&background
        odds,p=fisher_exact([[len(fg),len(candidates)-len(fg)],[len(bg),len(background)-len(bg)]],alternative='greater')
        results.append({'group':'new401','aspect':aspect,'term_id':term,'term_name':names[(aspect,term)],
                        'n':len(candidates),'hits':len(fg),'percent':100*len(fg)/len(candidates),
                        'bg_n':len(background),'bg_hits':len(bg),'bg_percent':100*len(bg)/len(background),
                        'odds_ratio':float(odds),'p_value':float(p),'hit_ids':';'.join(sorted(fg))})
    for aspect in ['BP','CC','MF']:
        rows=[r for r in results if r['aspect']==aspect]
        for row,q in zip(rows,bh_adjust([r['p_value'] for r in rows])):row['q_bh']=q
    return sorted(results,key=lambda r:(['BP','CC','MF'].index(r['aspect']),r['q_bh'],r['p_value'],r['term_id']))


def read_ids(path):
    ids=[line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(ids)!=len(set(ids)):raise ValueError(f'Duplicate IDs in {path.name}')
    return set(ids)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ['annotations','candidate-ids','untrained-ids','returned-ids','output-dir']:
        parser.add_argument('--'+arg,type=Path,required=True)
    parser.add_argument('--snapshot-id',required=True,help='Frozen annotation snapshot identifier; no live fetching')
    args=parser.parse_args()
    if args.output_dir.exists():raise FileExistsError('Use a new output directory')
    candidates=read_ids(args.candidate_ids);pool=read_ids(args.untrained_ids);returned=read_ids(args.returned_ids)
    if (len(candidates),len(pool),len(returned))!=(401,5640,5638):raise ValueError('Expected401/5640/5638 input counts')
    if not candidates<=returned<=pool:raise ValueError('Invalid candidate/returned/pool membership')
    with args.annotations.open() as stream:annotations=list(csv.DictReader(stream))
    rows=enrich_frozen(annotations,candidates,returned)
    args.output_dir.mkdir(parents=True)
    with (args.output_dir/'GO_enrichment.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    provenance={'snapshot':args.snapshot_id,'method':'direct GO; no propagation; pool>=5; one-sided Fisher; BH within aspect',
                'candidate_n':401,'returned_pool_n':5638,'comparison_n':5237,'unreturned_ids':sorted(pool-returned),
                'inputs':{key:hashlib.sha256(getattr(args,key).read_bytes()).hexdigest() for key in ['annotations','candidate_ids','untrained_ids','returned_ids']},
                'tested':len(rows),'significant':sum(r['q_bh']<.05 for r in rows)}
    (args.output_dir/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__=='__main__':main()
