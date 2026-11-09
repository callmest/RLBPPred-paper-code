"""Current frozen-data quantitative panel replay; never runs upstream analyses.

Contract: retain current panel identities, scientific meaning, palette and exact
data, while exporting editable Python reconstructions rather than promising
byte-identical Illustrator layouts. Manual schematics/images are not recreated.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle, Patch
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from pipeline.analysis.current_frozen_analysis import motif_composition
from scripts.reproduce_figures_legacy import plot_upset

ROOT=Path(__file__).resolve().parents[2]
GROUPS=['Novel predicted RLBPs','Known RLBPs','Non-RLBPs']
COLORS=['#BD6251','#6886AD','#AAB1B8']
METHOD_COLORS=['#9ECAE1','#3182BD','#A1D99B','#31A354','#FDAE6B','#E6550D']
REPS=['ESM-C 300M','ESM-C 600M','PST T33','PST T33-SO','SaProt AF2','SaProt PDB']
GO_COLORS={'BP':'#009E73','CC':'#0072B2','MF':'#E69F00'}
MOTIF_COLORS={'Cys-containing':'#B57F24','Lys/Arg-rich':'#4E79A7','Other sequences':'#9AA5B1'}
GROUP_NAMES={'sig_pos_ids':'High-confidence positives','significant_positive':'High-confidence positives','potential_positive':'Potential positives','untrained_proteome':'Untrained proteome','neg_ids':'High-confidence negatives','negative':'High-confidence negatives'}
MANUAL={'Fig1c':'dataset-construction schematic','Fig2a':'representation schematic','Fig2b':'balanced-ensemble schematic',**{f'Fig4{p}':'experimental image or structure render' for p in 'abcdef'},'Fig6':'database screenshot; supplied query records are not a replacement screenshot'}

def load(d,name):
    return pd.read_csv(d/f'{name}.csv',float_precision='round_trip')

def axis(title,w=6.5,h=4.5):
    fig,ax=plt.subplots(figsize=(w,h),layout='constrained');ax.set_title(title,loc='left',weight='bold',pad=12)
    return fig,ax

def corona(d):
    df=load(d,'Figure1a');depth=df.iloc[:,1:].astype(int).sum(axis=1);union=int(depth.gt(0).sum())
    fig,ax=axis('Overlap between five human datasets',4.4,4.4)
    for k,r,c in zip(range(2,6),[1,.81,.62,.43],['#4E79A7','#F28E2B','#59A14F','#B07AA1']):
        frac=depth.eq(k).sum()/union
        ax.add_patch(Wedge((0,0),r,0,360,width=.13,color='#E7E7E7'))
        ax.add_patch(Wedge((0,0),r,90-frac*360,90,width=.13,color=c))
        ax.text(-.025,r-.065,f'{k} datasets',ha='right',va='center',size=7)
        theta=math.radians(90-frac*180);ax.text((r+.08)*math.cos(theta),(r+.08)*math.sin(theta),f'{frac*100:.3g}%',ha='center',va='center',size=7)
    ax.text(0,.045,'Union',ha='center',weight='bold');ax.text(0,-.10,f'n = {union:,}',ha='center',size=7)
    ax.text(0,-1.21,'Mutually exclusive overlap depths; denominator = five-study union.',ha='center',size=6)
    ax.set(xlim=(-1.35,1.35),ylim=(-1.3,1.25),aspect='equal');ax.axis('off');return fig

def venn(d):
    df=load(d,'Figure1b');cols=['GO-related','Domain-related','RloopBase'];bits=df[cols].astype(int)
    counts=Counter(''.join(map(str,row)) for row in bits.itertuples(index=False,name=None));counts.pop('000',None)
    fig,ax=axis('Complementarity of exclusion sources',4.7,4.7)
    for xy,color in zip([(-.55,.31),(.55,.31),(0,-.60)],['#76B7B2','#E15759','#EDC948']):ax.add_patch(Circle(xy,.98,facecolor=color,alpha=.42,edgecolor='#555',lw=.6))
    pos={'100':(-1,.48),'010':(1,.48),'001':(0,-1.13),'110':(0,.67),'101':(-.56,-.47),'011':(.56,-.47),'111':(0,-.1)}
    for bit,xy in pos.items():ax.text(*xy,f'{counts[bit]:,}',ha='center',va='center',weight='bold')
    for xy,label,col in zip([(-.9,1.4),(.9,1.4),(0,-1.8)],['GO-related','Domain-related','R-loopBase'],cols):ax.text(*xy,f'{label}\nn = {int(bits[col].sum()):,}',ha='center',va='center',size=8)
    ax.text(0,-2.12,f'Union n = {sum(counts.values()):,}; circle areas are not proportional.',ha='center',size=6)
    ax.set(xlim=(-1.7,1.7),ylim=(-2.3,1.65),aspect='equal');ax.axis('off');return fig

def select_benchmark_go(data):
    """Original figure rule: stable q,p top5 per aspect/direction; signed sort."""
    blocks=[]
    for source in ['GO:BP','GO:CC','GO:MF']:
        pieces=[]
        for direction in ['positive','negative']:
            part=data.loc[data.source.eq(source)&data.direction.eq(direction)&data.fdr_bh.lt(.05)]
            pieces.append(part.sort_values(['fdr_bh','p_value'],kind='stable').head(5))
        block=pd.concat(pieces).copy()
        block['score']=-np.log10(block.fdr_bh.clip(lower=1e-300))*np.where(block.direction.eq('positive'),1,-1)
        blocks.append(block.sort_values('score',ascending=False,kind='stable'))
    return pd.concat(blocks,ignore_index=True)

def benchmark_go(d):
    df=select_benchmark_go(load(d,'Figure1d'));df['aspect']=df.source.str.replace('GO:','',regex=False)
    fig,ax=axis('GO enrichment of the stringent benchmark',7.2,max(6,len(df)*.16))
    ax.barh(np.arange(len(df)),df.score,color=df.aspect.map(GO_COLORS));ax.set_yticks(np.arange(len(df)),df.term_name.str.lstrip('; ').str.strip(),fontsize=6)
    ax.invert_yaxis();ax.axvline(0,color='#333',lw=.6);ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{abs(v):g}'))
    limit=df.score.abs().max()*1.08;ax.set_xlim(-limit,limit)
    for sign in [-1,1]:ax.axvline(sign*-math.log10(.05),color='#AAA',ls=':',lw=.6)
    ax.legend(handles=[Patch(facecolor=c,label=a) for a,c in GO_COLORS.items()],title='GO category',loc='lower left',fontsize=6)
    ax.set_xlabel('Negative set ← −log10(BH FDR) → Positive set');return fig

def auprc(d):
    df=load(d,'Figure2c');fig,ax=axis('Validation AUPRC',10,4.6);x=np.arange(len(df));width=.13
    for i,col in enumerate(df.columns[1:]):
        bars=ax.bar(x+(i-2.5)*width,df[col],width,color=METHOD_COLORS[i],label=col)
        ax.bar_label(bars,labels=[f'{v:.3f}' for v in df[col]],fontsize=5,rotation=90,padding=2)
    labels=df.iloc[:,0].replace({'Logistic regression':'LR','Random forest':'RF','Gaussian naive Bayes':'GNB'})
    values=df.iloc[:,1:].to_numpy(float);row,col=np.unravel_index(values.argmax(),values.shape)
    ax.scatter(row+(col-2.5)*width,values[row,col]+.065,marker='*',s=95,c='#F3B52B',zorder=5)
    ax.set_xticks(x,labels,rotation=25,ha='right');ax.set(ylabel='AUPRC',ylim=(0,1.12));ax.legend(ncols=3,fontsize=7,loc='upper center',bbox_to_anchor=(.5,1.16));return fig

def roc(d,name):
    df=load(d,name);fig,ax=axis('Held-out validation ROC',5.4,4.8)
    for label,g in df.groupby('label',sort=False):
        color='#E15759' if label.startswith(('Filtered Dataset','RLBPPred')) else '#4E79A7'
        ax.plot(g.fpr,g.tpr,color=color,lw=1.5,label=f'{label}\nAUROC = {g.auroc.iloc[0]:.4f}')
    ax.plot([0,1],[0,1],ls='--',color='#AAA',lw=.7);ax.set(xlabel='False-positive rate',ylabel='True-positive rate',xlim=(0,1),ylim=(0,1.02));ax.legend(fontsize=6,loc='lower right');return fig

def probability(d,name,value):
    df=load(d,name);fig,ax=axis('Group-level predictions',6.2,4.7)
    bars=ax.bar(np.arange(len(df)),df[value],color=['#3C8F63','#8CB375','#8D969C','#C56955'][:len(df)])
    ax.bar_label(bars,labels=[f'{v:.3g}\n(n={int(n):,})' for v,n in zip(df[value],df.sample_size)],padding=3,fontsize=7)
    ax.set_xticks(np.arange(len(df)),[GROUP_NAMES.get(x,x) for x in df.dataset],rotation=20,ha='right',fontsize=7)
    rate=value=='percentage_positive';ax.set(ylabel='Called as RLBP at probability ≥ 0.5 (%)' if rate else 'Mean predicted probability',ylim=(0,115 if rate else 1.15));return fig

def grouped(d,name,category,value,title):
    df=load(d,name);fig,ax=axis(title,7.2,4.7);groups=[g for g in GROUPS if g in set(df.Group)]
    if category=='Group':
        bars=ax.bar(np.arange(len(groups)),[float(df.loc[df.Group.eq(g),value].iloc[0]) for g in groups],color=COLORS)
        ax.bar_label(bars,fmt='%.3g',padding=3,fontsize=7);ax.set_xticks(np.arange(len(groups)),groups,rotation=15,ha='right')
    else:
        categories=list(dict.fromkeys(df[category]));width=.8/len(groups);x=np.arange(len(categories))
        for i,g in enumerate(groups):
            sub=df.loc[df.Group.eq(g)].set_index(category).reindex(categories);bars=ax.bar(x-.4+width/2+i*width,sub[value],width,color=COLORS[i],label=g)
            ncol=next(c for c in ['Count','Overlap count'] if c in sub);ax.bar_label(bars,labels=[str(int(v)) for v in sub[ncol]],padding=2,fontsize=7)
        ax.set_xticks(x,categories,rotation=20,ha='right');ax.legend(fontsize=6)
    ax.set_ylabel('Proteins (%)');ax.margins(y=.18);return fig

def disorder(d):
    df=load(d,'Figure5c_raw');tests=load(d,'Figure5c_stats')
    arrays=[pd.to_numeric(df.loc[df.group_label.eq(g),'disorder_percent'],errors='coerce').dropna().to_numpy() for g in GROUPS]
    fig,ax=axis('MobiDB-lite disorder',5.4,5.2);parts=ax.violinplot(arrays,showmedians=False,showextrema=False,bw_method=.22)
    for body,color in zip(parts['bodies'],COLORS):body.set_facecolor(color);body.set_edgecolor('#333');body.set_alpha(.72)
    boxes=ax.boxplot(arrays,positions=[1,2,3],widths=.18,patch_artist=True,showfliers=False)
    for patch,color in zip(boxes['boxes'],COLORS):patch.set_facecolor(color)
    ax.set_xticks([1,2,3],[f'{g}\nn = {len(a):,}' for g,a in zip(GROUPS,arrays)],fontsize=7);ax.set(ylabel='Disordered residues (%)',ylim=(-3,127))
    for (left,right),height,q in zip([(1,2),(1,3),(2,3)],[105,114,123],pd.to_numeric(tests['BH-adjusted q'])):
        label='ns' if q>=.05 else ('****' if q<1e-4 else f'q={q:.2g}')
        ax.plot([left,left,right,right],[height-1,height,height,height-1],color='#333',lw=.6);ax.text((left+right)/2,height+.4,label,ha='center',fontsize=7)
    return fig

def motifs(d):
    df=load(d,'Figure5d');df=df.loc[pd.to_numeric(df['E-value']).le(1.2)].sort_values('Rank',ascending=False)
    fig,ax=axis('Full-length sequence motifs in 401 candidates',7.2,5);labels=df['Consensus label'];values=-np.log10(pd.to_numeric(df['E-value']))
    ax.barh(labels,values,color=[MOTIF_COLORS[motif_composition(s)] for s in labels])
    for y,v,n in zip(range(len(df)),values,df['Unique candidate proteins (n)']):ax.text(v+.06,y,f'n = {int(n)}',va='center',fontsize=7)
    ax.axvline(-math.log10(.05),ls='--',color='#AD3B35',lw=.8);ax.set_xlabel('−log10(E-value); dashed line: E = 0.05');ax.margins(x=.17);return fig

def umap(d,rep,subset):
    df=load(d,'SupplementaryFigure2_3_umap');df=df.loc[df.Representation.eq(rep)&df.Subset.eq(subset)]
    if df.empty:raise ValueError(f'Missing UMAP {rep}/{subset}')
    fig,ax=axis(f'{rep}: '+('full benchmark' if subset=='full' else 'balanced subsample'),4.5,4.3)
    for label,color in [('negative','#4E79A7'),('positive','#E15759')]:
        g=df.loc[df.Label.eq(label)];ax.scatter(g.UMAP_1,g.UMAP_2,s=2 if subset=='full' else 5,c=color,alpha=.6,label=f'{label} (n={len(g):,})',rasterized=True)
    ax.set(xlabel='UMAP 1',ylabel='UMAP 2');ax.legend(markerscale=3,fontsize=6);return fig

def heatmap(d,metric):
    df=load(d,'SupplementaryFigure4');rows=list(dict.fromkeys(df.Classifier));table=df.pivot(index='Classifier',columns='Representation',values=metric).reindex(index=rows,columns=REPS)
    fig,ax=axis(metric.replace('F1','F1 score'),6.5,5.8);a=table.to_numpy(float);im=ax.imshow(a,cmap='Blues',aspect='auto',vmin=np.nanmin(a),vmax=np.nanmax(a))
    for i,j in np.ndindex(a.shape):
        ax.text(j,i,f'{a[i,j]:.3f}',ha='center',va='center',fontsize=6,color='white' if a[i,j]>(np.nanmin(a)+np.nanmax(a))/2 else '#222')
        if a[i,j]==np.nanmax(a):ax.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,edgecolor='#F28E2B',lw=1.5))
    ax.set_xticks(range(6),REPS,rotation=35,ha='right',fontsize=7);ax.set_yticks(range(len(rows)),rows,fontsize=7);fig.colorbar(im,ax=ax,shrink=.6);return fig

def robustness(d,part):
    df=load(d,{'a':'SupplementaryFigure5a','b':'SupplementaryFigure5b','c':'SupplementaryFigure5cd','d':'SupplementaryFigure5cd'}[part]);fig,ax=axis('Independent sampling-configuration analysis',5.6,4.4)
    if part in 'ab':
        key='neg_ratio' if part=='a' else 'n_rounds'
        for metric,color in [('auroc','#4E79A7'),('auprc','#E15759')]:ax.plot(df[key],df[metric],marker='o',color=color,label=metric.upper())
        ax.set_xlabel('Negative samples per positive' if part=='a' else 'Ensemble rounds');ax.legend()
    else:
        metric='auprc' if part=='c' else 'auroc';bars=ax.bar(np.arange(len(df)),df[metric],color=['#4E79A7','#76B7B2','#F28E2B']);ax.bar_label(bars,fmt='%.4f',padding=3,fontsize=7)
        ax.set_xticks(np.arange(len(df)),df.strategy.str.replace('_',' '),rotation=15,ha='right');ax.set_ylabel(metric.upper());ax.margins(y=.12)
    return fig

def confusion(d):
    df=load(d,'SupplementaryFigure6a');labels=['Non-RLBP','RLBP'];counts=df.pivot(index='true_label',columns='predicted_label',values='count').reindex(index=labels,columns=labels);perc=df.pivot(index='true_label',columns='predicted_label',values='row_percentage').reindex(index=labels,columns=labels)
    fig,ax=axis('Validation calls at probability ≥ 0.5',4.5,4.2);ax.imshow(perc,cmap='Blues',vmin=0,vmax=100)
    for i,j in np.ndindex((2,2)):ax.text(j,i,f'{int(counts.iloc[i,j]):,}\n({perc.iloc[i,j]:.2f}%)',ha='center',va='center',color='white' if perc.iloc[i,j]>50 else '#222')
    ax.set_xticks([0,1],labels);ax.set_yticks([0,1],labels);ax.set(xlabel='Predicted class',ylabel='True class');return fig

def threshold(d):
    df=load(d,'SupplementaryFigure6b')
    if not df.heldout_negative_n.eq(1821).all():raise ValueError('FPR denominator must be held-out1,821 negatives')
    chosen=df.loc[df.threshold.eq(.98)]
    if len(chosen)!=1:raise ValueError('Expected one frozen0.98 row')
    row=chosen.iloc[0]
    fig,ax=axis('Empirical held-out FPR and candidate count',6.3,4.5);twin=ax.twinx();ax.plot(df.threshold,df.heldout_negative_fpr_percent,color='#E15759');twin.plot(df.threshold,df.untrained_candidate_count,color='#4E79A7')
    ax.axvline(.98,color='#333',ls='--',lw=.8);ax.axhline(.1,color='#E15759',ls=':',lw=.7);ax.set(xlabel='Probability threshold',ylabel='Empirical FPR (%)',ylim=(0,None));twin.set_ylabel('Candidate proteins')
    ax.text(.98,.97,f'0.98\nFPR {int(row.heldout_negative_pass_count)}/{int(row.heldout_negative_n):,} = {row.heldout_negative_fpr_percent:.4f}%\n{int(row.untrained_candidate_count)} candidates',transform=ax.get_xaxis_transform(),ha='right',va='top',fontsize=7);return fig

def epifactors(d):
    df=load(d,'SupplementaryFigure7b');categories=list(dict.fromkeys(df.Category));groups=list(dict.fromkeys(df.Group));fig,ax=axis('EpiFactors category enrichment',7.2,3.9)
    for i,g in enumerate(groups):
        sub=df.loc[df.Group.eq(g)].set_index('Category').reindex(categories);points=ax.scatter(np.arange(len(categories)),[i]*len(categories),s=sub.Count*22,c=sub['-log10(P)'],cmap='YlOrRd',vmin=0,vmax=df['-log10(P)'].max(),edgecolors='#555')
        for x,count,odds in zip(range(len(categories)),sub.Count,sub['Odds ratio']):ax.text(x,i,f'{int(count)}\nOR {odds:.3g}',ha='center',va='center',fontsize=6)
    fig.colorbar(points,ax=ax,label='−log10(P)',shrink=.78,pad=.03)
    ax.set_xticks(range(len(categories)),categories,rotation=15,ha='right');ax.set_yticks(range(len(groups)),groups);ax.margins(y=.4)
    ax.set_xlabel('Bubble area is proportional to annotated protein count.\nLabels show count and odds ratio (OR).',fontsize=7)
    return fig

def motif_groups(d):
    df=load(d,'SupplementaryFigure7c');fig,ax=axis('Motif sequence-composition groups',6.2,3.6);total='Total displayed motifs (E ≤ 1.2)';sig='Significant motifs (E < 0.05)'
    ax.barh(df['Composition group'],df[total],color='#D6DADC',label='All displayed motifs');ax.barh(df['Composition group'],df[sig],color=[MOTIF_COLORS[g] for g in df['Composition group']],label='E < 0.05')
    for i,row in df.iterrows():ax.text(row[total]+.12,i,f'{int(row[sig])}/{int(row[total])}',va='center')
    ax.set_xlabel('Motif count (descriptive groups, not validated functions)');ax.margins(x=.18);ax.legend(fontsize=6);return fig

def aa(d):
    df=load(d,'SupplementaryFigure7d');order=list(dict.fromkeys(df['Amino acid']));angles=np.linspace(0,2*np.pi,20,endpoint=False);closed=np.r_[angles,angles[0]]
    fig,ax=plt.subplots(figsize=(6.5,6),subplot_kw={'projection':'polar'},layout='constrained')
    for g,color in zip(GROUPS,COLORS):
        sub=df.loc[df['Protein group'].eq(g)].set_index('Amino acid').reindex(order);vals=sub['Pooled percentage (%)'].to_numpy(float)
        if not np.isclose(vals.sum(),100,atol=1e-8):raise ValueError('Pooled AA percentages must sum to100')
        ax.plot(closed,np.r_[vals,vals[0]],color=color,label=f'{g} (n={int(sub["Proteins (n)"].iloc[0]):,})',lw=1.3)
    ax.set_xticks(angles,order);ax.set_title('Pooled amino-acid composition (%)',pad=24);ax.legend(loc='upper right',bbox_to_anchor=(1.25,-.03),fontsize=7);return fig

def go(d,aspect):
    df=load(d,'SupplementaryFigure8_all_GO');selected=df.plot_selected.astype(str).str.lower().isin(['true','1']);sub=df.loc[df.aspect.eq(aspect)&selected].sort_values(['q_bh','p_value','term_id'])
    fig,ax=axis({'BP':'Biological process','CC':'Cellular component','MF':'Molecular function'}[aspect],7.2,4.7)
    ax.barh(sub.term_name,-np.log10(sub.q_bh),color=GO_COLORS[aspect]);ax.invert_yaxis();ax.tick_params(axis='y',labelsize=7);ax.set_xlabel('−log10(BH-adjusted P); 401 candidates vs 5,237 pool controls');return fig

PLOTS={'Fig1a':corona,'Fig1b':venn,'Fig1d':benchmark_go,'Fig2c':auprc,
       'Fig3a':lambda d:roc(d,'Figure3a'),'Fig3b':lambda d:roc(d,'Figure3b'),'Fig3c':lambda d:probability(d,'Figure3c','mean_probability'),'Fig3d':lambda d:probability(d,'Figure3d','mean_probability'),
       'Fig5a':lambda d:grouped(d,'Figure5a','Category','Percentage (%)','EpiFactors functional categories'),'Fig5b':lambda d:grouped(d,'Figure5b','Database','Overlap (%)','Phase-separation database annotations'),'Fig5c':disorder,'Fig5d':motifs,
       'SupFig1a':lambda d:plot_upset(load(d,'SupplementaryFigure1a'),'Protein ID',['Human Proteome','All Positive','GO-related','Domain-related','RloopBase'],'Negative-set construction categories'),
       **{f'SupFig4{p}':(lambda d,m=m:heatmap(d,m)) for p,m in zip('abcde',['AUROC','F1','MCC','Precision','Recall'])},
       **{f'SupFig5{p}':(lambda d,p=p:robustness(d,p)) for p in 'abcd'},'SupFig6a':confusion,'SupFig6b':threshold,
       'SupFig6c':lambda d:probability(d,'SupplementaryFigure6c','percentage_positive'),'SupFig6d':lambda d:probability(d,'SupplementaryFigure6d','percentage_positive'),
       'SupFig7a':lambda d:grouped(d,'SupplementaryFigure7a','Group','Percentage (%)','Overall EpiFactors annotation'),'SupFig7b':epifactors,'SupFig7c':motif_groups,'SupFig7d':aa,
       **{f'SupFig8{p}':(lambda d,a=a:go(d,a)) for p,a in zip('abc',['BP','CC','MF'])}}
for number,reps in [(2,['ESM-C 300M','ESM-C 600M','PST T33']),(3,['SaProt PDB','PST T33-SO','SaProt AF2'])]:
    for i,rep in enumerate(reps):
        for j,subset in enumerate(['full','sampled']):PLOTS[f'SupFig{number}{chr(97+2*i+j)}']=lambda d,r=rep,s=subset:umap(d,r,s)

def validate_output_dir(path, root=ROOT):
    """Read-only preflight: source trees and nonempty outputs are never writable."""
    destination=Path(path).resolve()
    repository=Path(root).resolve()
    # Inside the repository, only the designated outputs tree is an output area.
    # Resolving paths first also prevents a symlink from bypassing this boundary.
    if destination.is_relative_to(repository) and not destination.is_relative_to(repository/'outputs'):
        raise ValueError('Refusing to write into the repository source tree; use outputs/ or an external empty directory')
    if destination.exists():
        if not destination.is_dir():
            raise ValueError('Output path exists and is not a directory')
        if any(destination.iterdir()):
            raise ValueError('Output directory is not empty; choose a new isolated directory')
    return destination

def main():
    parser=argparse.ArgumentParser(description=__doc__);choice=parser.add_mutually_exclusive_group(required=True);choice.add_argument('--all',action='store_true');choice.add_argument('--figure',help='Current panel, e.g. Fig5d or SupFig8a')
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data/panels/current');parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/figures/current');parser.add_argument('--dpi',type=int,default=300);args=parser.parse_args()
    panels=sorted(PLOTS) if args.all else [args.figure]
    invalid=[panel for panel in panels if panel not in PLOTS and panel not in MANUAL]
    if invalid:parser.error(f'Unknown current panel {invalid[0]}; choices: '+', '.join(sorted(PLOTS)))
    if args.dpi<=0:parser.error('--dpi must be positive')
    try:args.output_dir=validate_output_dir(args.output_dir)
    except ValueError as exc:parser.error(str(exc))
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,'axes.titlesize':9,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42})
    args.output_dir.mkdir(parents=True,exist_ok=True)
    report={'mode':'frozen quantitative panel replay','generated':[],'manual_panels_not_generated':MANUAL,'data_files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.data_dir.glob('*.csv'))},'upstream_analysis_executed':False}
    for panel in panels:
        if panel in MANUAL:print(f'{panel}: manual asset, not generated ({MANUAL[panel]})');continue
        fig=PLOTS[panel](args.data_dir)
        for ext in ['svg','pdf','png']:fig.savefig(args.output_dir/f'{panel}.{ext}',dpi=args.dpi,bbox_inches='tight',facecolor='white')
        plt.close(fig);report['generated'].append(panel);print(f'Generated {panel}')
    (args.output_dir/'replay_report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
