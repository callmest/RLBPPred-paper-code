"""Offline tests: no training, network, external STREME invocation or official exports."""
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.analysis.current_frozen_analysis import (
    motif_composition, motif_summary, parse_streme_text, parse_streme_xml, verify_frozen_go,
)
from pipeline.analysis.run_fullseq_motif_discovery import run_streme
from pipeline.analysis.current_go_enrichment import bh_adjust, enrich_frozen


class CurrentAnalysisTests(unittest.TestCase):
    def test_composition_priority(self):
        self.assertEqual(motif_composition('CKKK'), 'Cys-containing')
        self.assertEqual(motif_composition('GKQ'), 'Lys/Arg-rich')
        self.assertEqual(motif_composition('GTGKTL'), 'Other sequences')
        with self.assertRaises(ValueError): motif_composition('')

    def test_text_parser_with_whitespace(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'streme.txt'
            path.write_text('MOTIF 1-AKKRKL STREME-1\nletter-probability matrix: alength= 20 w= 6 nsites= 63 E= 8.5e-005\n')
            motif = parse_streme_text(path)[0]
            self.assertEqual(motif['consensus'], 'AKKRKL')
            self.assertEqual((motif['width'], motif['nsites']), (6, 63))
            self.assertEqual(motif['e_value'], .000085)

    def test_invalid_text_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'streme.txt';path.write_text('not STREME')
            with self.assertRaises(ValueError):parse_streme_text(path)

    def test_xml_uses_unique_proteins_not_sites(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'streme.xml'
            path.write_text('<STREME><motifs><motif id="1-AKKRKL" width="6" npassing="63" total_sites="109" test_evalue="8.5e-5"/></motifs></STREME>')
            self.assertEqual(parse_streme_xml(path)[0]['unique_candidate_proteins'],63)

    def test_frozen_motif_sources_agree(self):
        source=ROOT/'data/frozen_analysis/streme401'
        xml=parse_streme_xml(source/'streme.xml');text=parse_streme_text(source/'streme.txt')
        self.assertEqual(len(xml),13)
        for a,b in zip(xml,text):
            self.assertEqual((a['consensus'],a['e_value'],a['unique_candidate_proteins']),
                             (b['consensus'],b['e_value'],b['nsites']))
        summary=motif_summary(xml)
        self.assertEqual([summary[x] for x in ['reported','significant','displayed']],[13,7,11])
        self.assertEqual(summary['groups'],{'Cys-containing':{'significant':3,'displayed':3},'Lys/Arg-rich':{'significant':4,'displayed':6},'Other sequences':{'significant':0,'displayed':2}})

    def test_streme_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):run_streme(Path('unused-primary'),Path('unused-control'),Path(tmp))

    def test_historical_scripts_require_opt_in(self):
        scripts=['analysis/prepare_idr_sequences.py','analysis/run_exp1_3b_llps_overlap.py',
                 'analysis/run_exp2_1_chromatin.py','analysis/run_exp1_4_mobidb_disorder.py',
                 'analysis/plot_aa_composition_radar.py','analysis/go_domain_analysis_uniprot.py',
                 'figures/plot_exp1_2c_fullseq_motif_pub.py','figures/plot_exp1_3b_llps_overlap_pub.py',
                 'figures/plot_exp2_1_chromatin_pub.py']
        for script in scripts:
            result=subprocess.run([sys.executable,str(ROOT/'pipeline'/script)],capture_output=True,text=True,timeout=10)
            self.assertNotEqual(result.returncode,0,script)
            self.assertIn('Historical 342-candidate workflow',result.stderr,script)

    def test_current_go_family(self):
        report=verify_frozen_go(ROOT/'data/panels/current/SupplementaryFigure8_all_GO.csv')
        self.assertEqual((report['tested'],report['significant']),(3082,88))
        self.assertEqual(report['by_aspect']['BP'],{'tested':1908,'significant':33})

    def test_bh_adjust(self):
        self.assertEqual(bh_adjust([.01,.04,.03]),[.03,.04,.04])

    @unittest.skipUnless(importlib.util.find_spec('scipy'), 'upstream GO dependency not installed')
    def test_go_includes_zero_foreground_terms(self):
        annotations=[{'protein_id':p,'aspect':'BP','term_id':'GO:1','term_name':'foreground'} for p in ['a','b']]
        annotations += [{'protein_id':p,'aspect':'BP','term_id':'GO:2','term_name':'control-only'} for p in ['c','d']]
        rows=enrich_frozen(annotations,{'a','b'},{'a','b','c','d'},min_pool_count=2)
        self.assertEqual(len(rows),2)
        control=next(r for r in rows if r['term_id']=='GO:2')
        self.assertEqual(control['hits'],0)
        self.assertEqual(control['p_value'],1)

    def test_threshold_denominator_and_inclusive_call(self):
        with (ROOT/'data/panels/current/SupplementaryFigure6b.csv').open() as f:rows=list(csv.DictReader(f))
        selected=[r for r in rows if float(r['threshold'])==.98]
        self.assertEqual(len(selected),1)
        row=selected[0]
        self.assertEqual((int(row['heldout_negative_pass_count']),int(row['heldout_negative_n']),int(row['untrained_candidate_count'])),(1,1821,401))
        self.assertAlmostEqual(float(row['heldout_negative_fpr_percent']),100/1821)

    def test_disorder_missing_is_not_zero(self):
        with (ROOT/'data/panels/current/Figure5c_raw.csv').open() as f:rows=list(csv.DictReader(f))
        neg=[r for r in rows if r['group_label']=='Non-RLBPs']
        self.assertEqual(len(neg),9109)
        self.assertEqual(sum(r['disorder_percent']=='' for r in neg),50)
        self.assertEqual(sum(r['group_label']=='Novel predicted RLBPs' for r in rows),401)

    def test_pooled_aa_counts(self):
        with (ROOT/'data/panels/current/SupplementaryFigure7d.csv').open() as f:rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),60)
        expected={'Novel predicted RLBPs':(401,240972),'Known RLBPs':(376,234811),'Non-RLBPs':(9109,4356393)}
        for group,(n,total) in expected.items():
            selected=[r for r in rows if r['Protein group']==group]
            self.assertEqual(len(selected),20)
            self.assertEqual(sum(int(r['Residue count']) for r in selected),total)
            self.assertTrue(all(int(r['Proteins (n)'])==n for r in selected))
            self.assertAlmostEqual(sum(float(r['Pooled percentage (%)']) for r in selected),100)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'plotting dependency not installed')
    def test_current_plot_targets(self):
        from pipeline.current_figures.replay import PLOTS,MANUAL
        self.assertIn('Fig2c',PLOTS);self.assertNotIn('Fig2b',PLOTS)
        self.assertIn('SupFig8c',PLOTS);self.assertNotIn('Fig3e',PLOTS)
        self.assertIn('Fig4f',MANUAL)
        self.assertEqual(len([k for k in PLOTS if k.startswith('SupFig4')]),5)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'plotting dependency not installed')
    def test_benchmark_go_original_selection(self):
        from pipeline.current_figures.replay import load,select_benchmark_go
        selected=select_benchmark_go(load(ROOT/'data/panels/current','Figure1d'))
        self.assertEqual(len(selected),30)
        self.assertEqual(set(selected.groupby(['source','direction']).size()),{5})
        self.assertTrue((selected.loc[selected.direction.eq('positive'),'score']>0).all())
        self.assertTrue((selected.loc[selected.direction.eq('negative'),'score']<0).all())

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'plotting dependency not installed')
    def test_replay_rejects_source_directories(self):
        from pipeline.current_figures.replay import validate_output_dir
        for name in ['data/panels/current','configs','pipeline','scripts','tests','docs','']:
            with self.assertRaises(ValueError,msg=name):validate_output_dir(ROOT/name)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'plotting dependency not installed')
    def test_replay_rejects_nonempty_output_unchanged(self):
        from pipeline.current_figures.replay import validate_output_dir
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);existing=out/'old.svg';existing.write_text('preserve this exact result')
            with self.assertRaises(ValueError):validate_output_dir(out)
            self.assertEqual(existing.read_text(),'preserve this exact result')
            self.assertEqual(list(out.iterdir()),[existing])

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'plotting dependency not installed')
    def test_replay_rejects_symlink_into_source_tree(self):
        from pipeline.current_figures.replay import validate_output_dir
        with tempfile.TemporaryDirectory() as tmp:
            link=Path(tmp)/'source_link';link.symlink_to(ROOT/'data',target_is_directory=True)
            with self.assertRaises(ValueError):validate_output_dir(link/'not-created')

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'plotting dependency not installed')
    def test_replay_invalid_options_have_no_output_side_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            for flags in [['--figure','not-a-panel'],['--figure','Fig5d','--dpi','0'],['--figure','Fig5d','--dpi','-1']]:
                out=Path(tmp)/'must-not-exist'
                result=subprocess.run([sys.executable,str(ROOT/'scripts/reproduce_figures.py'),*flags,'--output-dir',str(out)],capture_output=True,text=True,timeout=15)
                self.assertNotEqual(result.returncode,0)
                self.assertFalse(out.exists())

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'plotting dependency not installed')
    def test_replay_allows_empty_external_output_without_creating_it(self):
        from pipeline.current_figures.replay import validate_output_dir
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'new'/'isolated'
            self.assertEqual(validate_output_dir(out),out.resolve())
            self.assertFalse(out.exists())

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'plotting dependency not installed')
    def test_epifactors_encodings_are_labeled(self):
        from pipeline.current_figures.replay import epifactors
        import matplotlib.pyplot as plt
        fig=epifactors(ROOT/'data/panels/current')
        try:
            self.assertEqual(len(fig.axes),2)
            self.assertEqual(fig.axes[1].get_ylabel(),'−log10(P)')
            self.assertIn('Bubble area is proportional',fig.axes[0].get_xlabel())
            self.assertIn('count and odds ratio',fig.axes[0].get_xlabel())
        finally:plt.close(fig)


if __name__=='__main__':unittest.main()
