#!/usr/bin/env python3
"""Validate current frozen strict401 evidence; no training, network or source writes."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics_from_scores(labels, scores):
    """Threshold metrics and tie-aware ROC/AP, independent of stored metrics."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s, kind="stable")
    sy, ss = y[order], s[order]
    ends = np.r_[np.flatnonzero(np.diff(ss)), len(ss) - 1]
    tp = np.cumsum(sy)[ends]
    fp = (ends + 1) - tp
    recall = tp / y.sum()
    precision = tp / (ends + 1)
    x = np.r_[0.0, fp / (len(y) - y.sum())]
    z = np.r_[0.0, recall]
    auroc = np.sum(np.diff(x) * (z[1:] + z[:-1]) / 2)
    ap = np.sum(np.diff(np.r_[0.0, recall]) * precision)
    calls = s >= .5
    tn = int(((y == 0) & ~calls).sum()); fp = int(((y == 0) & calls).sum())
    fn = int(((y == 1) & ~calls).sum()); tp = int(((y == 1) & calls).sum())
    return dict(auroc=float(auroc), auprc=float(ap), f1=2*tp/(2*tp+fp+fn),
                precision=tp/(tp+fp), recall=tp/(tp+fn),
                mcc=(tp*tn-fp*fn)/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)),
                accuracy=(tp+tn)/len(y), confusion_matrix=[[tn, fp], [fn, tp]])


def run_checks(root, source_workbook=None):
    root = Path(root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    data = root / "data/panels/current"
    frozen = root / "data/frozen_model"
    results = []

    def check(name, ok, observed=None, expected=None):
        results.append(dict(check=name, passed=bool(ok), observed=observed, expected=expected))

    def close(a, b, tol=1e-12):
        return bool(np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=tol, atol=tol, equal_nan=True))

    def csv(name):
        return pd.read_csv(data / (name + ".csv"), float_precision="round_trip")

    def test_all():
        required = ["configs/panel_map.json", "configs/paper.yaml", "data/panels/current/manifest.json",
                    "data/frozen_model/manifest.json", "pipeline/modeling/strict_runtime.py",
                    "scripts/reproduce_figures.py", "scripts/export_source_data.py",
                    "docs/reproduce_model.md", "docs/reproduce_figures.md"]
        for name in required:
            check("required:" + name, (root / name).is_file())

        manifest = json.loads((data / "manifest.json").read_text())
        mapping = json.loads((root / "configs/panel_map.json").read_text())
        check("current panel-map hash", sha(root / "configs/panel_map.json") == manifest["panel_map_sha256"])
        check("source workbook identity", manifest["workbook_sha256"] == mapping["workbook_sha256"])
        if source_workbook:
            check("provided source workbook hash", sha(source_workbook) == manifest["workbook_sha256"])
        check("no index or historical-sheet export", not manifest["index_sheet_exported"] and not manifest["historical_sheets_exported"])
        entries = manifest["datasets"]
        byid = {item["id"]: item for item in entries}
        check("unique current dataset IDs", len(byid) == len(entries))
        names = [item["file"] for item in entries]
        check("unique flat current filenames", len(names) == len(set(names)) and all(Path(x).name == x for x in names))
        check("no unmanifested current files", set(names) | {"manifest.json"} == {p.name for p in data.iterdir() if p.is_file()})
        expected = {r["id"]: r for r in mapping["datasets"] + mapping["external_files"]}
        check("complete current dataset coverage", set(byid) == set(expected))
        for item in entries:
            name = item["file"]
            check("hash:" + name, sha(data / name) == item["sha256"])
            spec = expected[item["id"]]
            check("panel mapping:" + name, item["panels"] == spec["panels"])
            if name.endswith(".csv"):
                table = pd.read_csv(data / name, float_precision="round_trip")
                check("schema/rows:" + name, list(table.columns) == item["columns"] and len(table) == item["rows"] == spec["expected_rows"])
            if "sheet" in spec:
                check("workbook block:" + name, item["source_sheet"] == spec["sheet"] and item["source_range"] == spec["range"] and item["source_workbook_sha256"] == manifest["workbook_sha256"])

        runtime = load_module(root / "pipeline/modeling/strict_runtime.py", "strict_runtime_validation")
        verified = runtime.validate_release(frozen)
        check("strict split/sampling/score/candidate provenance", verified["pass"])
        model_manifest = json.loads((frozen / "manifest.json").read_text())
        provenance = json.loads((frozen / "provenance.json").read_text())
        audit = model_manifest["artifacts"]["provenance.json"]
        sanitized = audit["sanitization"]
        unchanged = {k: v for k, v in provenance.items() if k not in ("source", "output")}
        canonical = json.dumps(unchanged, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        check("public provenance logical paths", provenance["source"] == "source-project/RLBP" and provenance["output"] == "experiments/strict_saprot_mlp_reproduction_20261013")
        check("public provenance original hash retained", audit["original_sha256"] == "7e3b8d1e38c6384dc1d6a36b6a7048492591711e899efaebe9bb3fe71ea0f51f" and audit["original_bytes"] == 1480 and sanitized["changed_fields"] == ["source", "output"])
        check("public provenance non-path fields preserved", hashlib.sha256(canonical).hexdigest() == sanitized["preserved_fields_canonical_sha256"] == "6e21675314c90b644925b530c8761a008cb21b52e74e155dff71b1afcae3eda8")
        validation = pd.read_csv(frozen / "val_predictions.csv", float_precision="round_trip")
        human = pd.read_csv(frozen / "human_all_predictions.csv", float_precision="round_trip")
        mouse = pd.read_csv(frozen / "mouse_all_predictions.csv", float_precision="round_trip")
        measured = metrics_from_scores(validation.label, validation.probability)
        metrics = json.loads((frozen / "validation_metrics.json").read_text())
        for key, value in measured.items():
            check("recomputed strict metric:" + key, close(value, metrics[key]), value, metrics[key])
        config = yaml.safe_load((root / "configs/paper.yaml").read_text())
        sc = config["screening"]
        check("current threshold/FPR configuration", sc["adopted_threshold"] == .98 and sc["operator"] == ">=" and sc["target_empirical_fpr"] == .001 and sc["false_positives"] == 1 and sc["heldout_negative_n"] == 1821 and sc["high_confidence_candidates"] == 401)
        check("classification distinct from discovery", config["ensemble"]["classification_threshold"] == .5 and config["ensemble"]["classification_operator"] == ">=")
        check("config metrics", close([config["reported_performance"]["auroc"], config["reported_performance"]["auprc"]], [metrics["auroc"], metrics["auprc"]]))

        expected_inputs = {"recurrent_positive_ids.txt":407, "significant_positive_ids.txt":376,
                           "negative_ids.txt":9109, "potential_positive_ids.txt":1879,
                           "untrained_proteome_ids.txt":5649, "high_confidence_candidate_ids.txt":401}
        for name, n in expected_inputs.items():
            ids = (root / "data/model_inputs" / name).read_text().split()
            check("input IDs:" + name, len(ids) == len(set(ids)) == n)
        for stem, label in [("potential_positive","potential_positive"),("untrained_proteome","untrained_proteome")]:
            inputs = set((root / "data/model_inputs" / (stem + "_ids.txt")).read_text().split())
            scored = set(human.loc[human.dataset == label, "protein_id"])
            check("input versus scored coverage:" + label, scored <= inputs and len(inputs-scored) == (35 if label == "potential_positive" else 9))

        membership = csv("Figure1a")
        cols = ["Cristini","Wang","Mosler","Yan","Shinriki"]
        depth = membership[cols].sum(axis=1)
        check("five-study overlap", len(membership) == 1430 and (depth >= 2).sum() == 407 and (depth == 5).sum() == 4)
        check("five study sizes", membership[cols].sum().tolist() == [462,364,308,434,430])
        exclusions = csv("Figure1b")
        excols = ["GO-related","Domain-related","RloopBase"]
        # R-loopBase has 1,182 nonblank source rows but 1,181 unique IDs.
        check("unique exclusion-source sizes", exclusions[excols].sum().tolist() == [7050,1785,1181])
        exdepth = exclusions[excols].sum(axis=1)
        check("exclusion union and overlap", (exdepth > 0).sum() == 7674 and (exdepth == 1).sum() == 5693 and (exdepth == 3).sum() == 361)
        check("Sup1 includes reviewed proteome", "Human Proteome" in csv("SupplementaryFigure1a") and len(csv("SupplementaryFigure1a")) == 20435)

        performance = csv("SupplementaryTable2")
        best = performance[(performance.Representation == "SaProt AF2") & (performance.Classifier == "MLP")]
        keys = ["AUROC","AUPRC","F1","Precision","Recall","MCC","ACC"]
        vals = [metrics[k] for k in ["auroc","auprc","f1","precision","recall","mcc","accuracy"]]
        check("final MLP metric row", len(best) == 1 and close(best[keys].values[0], vals))
        check("Figure2c current AUPRC", close(csv("Figure2c").set_index("Classifier").loc["MLP","SaProt AF2"], metrics["auprc"]))
        legacy = pd.read_csv(root / "data/panels/SuppFigure2b.csv")
        others = ~((performance.Representation == "SaProt AF2") & (performance.Classifier == "MLP"))
        check("other 59 model records preserved", performance.loc[others].reset_index(drop=True).equals(legacy.loc[others].reset_index(drop=True)))
        for current, old in [("SupplementaryFigure4a","SuppFigure2c_ratio"),("SupplementaryFigure4b","SuppFigure2c_rounds"),("SupplementaryFigure4cd","SuppFigure2d")]:
            previous = pd.read_csv(root / "data/panels" / (old + ".csv"), float_precision="round_trip")
            check("independent robustness preserved:" + current, csv(current).equals(previous))
        for panel, control in [("Figure3a",.9001),("Figure3b",.968)]:
            curves = csv(panel)
            check("strict ROC:" + panel, any(close(value, metrics["auroc"]) for value in curves.auroc.unique()))
            old = pd.read_csv(root / "data/panels" / (panel + ".csv"), float_precision="round_trip")
            a = curves.loc[np.isclose(curves.auroc, control), ["auroc","fpr","tpr"]].to_numpy()
            b = old.loc[np.isclose(old.auroc, control), ["auroc","fpr","tpr"]].to_numpy()
            check("independent ROC control preserved:" + panel, a.shape == b.shape and close(a,b))

        labelmaps = [
            (human, "Figure3c", "SupplementaryFigure5c", ["sig_pos_ids","potential_positive","untrained_proteome","neg_ids"]),
            (mouse, "Figure3d", "SupplementaryFigure5d", ["significant_positive","potential_positive","negative"])]
        distribution = load_module(root / "pipeline/current_figures/score_distributions.py", "distribution_validation")
        for scores, distribution_panel, rate_panel, labels in labelmaps:
            bins, rates = csv(distribution_panel), csv(rate_panel)
            distribution.validate_bins(bins, distribution_panel)
            group_ns = [int((scores.dataset == name).sum()) for name in labels]
            group_rates = [(scores.loc[scores.dataset == name,"probability"] >= .5).mean()*100 for name in labels]
            for name, n in zip(labels, group_ns):
                block = bins.loc[bins.dataset == name].sort_values("bin_left")
                counts, _ = np.histogram(scores.loc[scores.dataset == name, "probability"], bins=np.arange(21)/20)
                check("distribution bins/counts:" + distribution_panel + ":" + name,
                      block.protein_count.tolist() == counts.tolist() and block.n.eq(n).all()
                      and close(block.within_group_percentage, counts*100/n))
                xx, yy = distribution.pchip(block.bin_center, block.within_group_percentage)
                check("PCHIP exact knots/no extrapolation:" + distribution_panel + ":" + name,
                      len(xx) == 381 and close(xx[::20],block.bin_center)
                      and np.array_equal(yy[::20],block.within_group_percentage.to_numpy())
                      and close([xx[0],xx[-1]],[.025,.975]))
            check("group calls >=0.5:" + rate_panel, close(rates.percentage_positive,group_rates) and rates.sample_size.tolist() == group_ns)
        from pipeline.current_figures.replay import PLOTS, OPTIONAL_PLOTS
        check("current figure registry", len(PLOTS) == 40 and "SupFig8a" not in PLOTS
              and len([p for p in PLOTS if p.startswith("SupFig4")]) == 4
              and set(OPTIONAL_PLOTS) == {"SuppTable2_"+m for m in ["AUROC","F1","MCC","Precision","Recall"]})
        check("held-out confusion data", csv("SupplementaryFigure5a")["count"].tolist() == [1732,89,4,71])
        scan = csv("SupplementaryFigure5b")
        negatives = validation.loc[validation.label == 0,"probability"].to_numpy()
        pool = human.loc[human.dataset == "untrained_proteome","probability"].to_numpy()
        check("threshold grid", close(scan.threshold,np.linspace(.5,1,101)))
        check("threshold denominators", set(scan.heldout_negative_n) == {1821} and set(scan.untrained_scored_n) == {5640})
        fp = [(negatives >= t).sum() for t in scan.threshold]
        counts = [(pool >= t).sum() for t in scan.threshold]
        check("all threshold passes recomputed", scan.heldout_negative_pass_count.tolist() == fp and scan.untrained_candidate_count.tolist() == counts and close(scan.heldout_negative_fpr_percent,np.array(fp)/1821*100))
        row = scan.loc[np.isclose(scan.threshold,.98)].iloc[0]
        check("operating point 0.98", row.heldout_negative_pass_count == 1 and row.untrained_candidate_count == 401 and row.heldout_negative_fpr_percent <= .1)

        epi = csv("Figure5a")
        epi_candidates = epi.loc[epi.Group == "Novel predicted RLBPs"]
        check("401 chromatin category counts", epi_candidates.Count.tolist() == [24,10,8,10,11] and set(epi_candidates["Group total"]) == {401})
        check("401 chromatin percentages", close(epi_candidates["Percentage (%)"],epi_candidates.Count/401*100))
        llps = csv("Figure5b")
        llps_candidates = llps.loc[llps.Group == "Novel predicted RLBPs"]
        check("401 LLPS counts and union", llps_candidates["Overlap count"].tolist() == [27,172,8,2,183] and set(llps_candidates["Group total"]) == {401})
        check("401 LLPS percentages", close(llps_candidates["Overlap (%)"],llps_candidates["Overlap count"]/401*100))
        epi_union = csv("SupplementaryFigure6a").iloc[0]
        check("401 EpiFactors union", epi_union["EpiFactors-annotated"] == 63 and epi_union["Group total"] == 401)

        disorder = csv("Figure5c_raw")
        valid = pd.to_numeric(disorder.disorder_percent, errors="coerce").notna()
        observed = disorder.loc[valid].groupby("group_label").size().sort_values().tolist()
        check("disorder valid coverage", observed == [376,401,9059] and len(disorder) == 9886 and int((~valid).sum()) == 50)
        motifs = csv("Figure5d")
        e = motifs["E-value"].to_numpy()
        check("STREME reported/significant/displayed", len(e) == 13 and (e < .05).sum() == 7 and (e <= 1.2).sum() == 11)
        check("STREME exact display flags", motifs["Displayed (E ≤ 1.2)"].eq("Yes").tolist() == (e <= 1.2).tolist())
        comp = csv("SupplementaryFigure6c")
        check("motif composition groups", comp.iloc[:,1:].values.tolist() == [[3,3],[4,6],[0,2]])
        streme_dir = root / "data/frozen_analysis/streme401"
        streme_meta = json.loads((streme_dir / "manifest.json").read_text())
        for name, item in streme_meta["outputs"].items():
            check("frozen STREME hash:" + name, sha(streme_dir / name) == item["sha256"])
        offline = load_module(root / "pipeline/analysis/current_frozen_analysis.py", "frozen_analysis_audit")
        xml = offline.parse_streme_xml(streme_dir / "streme.xml")
        txt = offline.parse_streme_text(streme_dir / "streme.txt")
        check("STREME text/XML agreement", len(xml) == len(txt) == 13 and all((a["consensus"],a["width"],a["e_value"],a["unique_candidate_proteins"]) == (b["consensus"],b["width"],b["e_value"],b["nsites"]) for a,b in zip(xml,txt)))
        check("STREME raw/panel E-values", close([r["e_value"] for r in xml],e))
        check("STREME independent composition", offline.motif_summary(xml)["groups"] == {"Cys-containing":{"significant":3,"displayed":3},"Lys/Arg-rich":{"significant":4,"displayed":6},"Other sequences":{"significant":0,"displayed":2}})
        aa = csv("SupplementaryFigure6d")
        check("AA composition count/sums", len(aa) == 60 and close(aa.groupby("Protein group")["Pooled percentage (%)"].sum(), [100]*3))
        check("AA residue denominators", sorted(aa["Standard residues (total)"].unique()) == [234811,240972,4356393])
        go = csv("SupplementaryFigure7_all_GO")
        check("complete GO family and subsets", len(go) == 3082 and (go.q_bh < .05).sum() == 88 and go.plot_selected.sum() == 18)
        check("GO foreground/background", set(go.n) == {401} and set(go.bg_n) == {5237})
        check("GO aspect tests", go.groupby("aspect").size().to_dict() == {"BP":1908,"CC":508,"MF":666})
        # Recompute aspect-wise BH directly from the frozen raw p-values.
        bh_ok = True
        for _, group in go.groupby("aspect"):
            p = group.p_value.to_numpy()
            order = np.argsort(p, kind="stable")
            adjusted = np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
            bh_ok &= close(group.q_bh.to_numpy()[order],np.minimum(adjusted,1))
        check("GO aspect-wise BH correction", bh_ok)

        query = csv("Figure6_query")
        meta = json.loads((data / "Figure6_metadata.json").read_text())
        completion_dir = root / "data/database_completion"
        completion_meta = json.loads((completion_dir / "manifest.json").read_text())
        completion_path = completion_dir / completion_meta["file"]
        completion = pd.read_csv(completion_path, float_precision="round_trip")
        check("database completion file identity", sha(completion_path) == completion_meta["sha256"]
              == meta["source_prediction_hashes"]["strict_missing_scores_20261110/predictions.csv"]
              and completion_path.stat().st_size == completion_meta["bytes"])
        check("database completion disjoint exact coverage", len(completion) == 3106
              and completion.protein_id.nunique() == 3106
              and not set(completion.protein_id) & set(human.protein_id))
        check("database completion fixed model", completion_meta["source_checkpoint_sha256"]
              == "72875c9414f9f3705b9437ce2c2b088257810b737464ab2fb332a0df7406aa41"
              and completion_meta["model_retrained"] is False)
        completed_human = pd.concat([human[["protein_id","probability"]],
                                     completion[["protein_id","probability"]]], ignore_index=True)
        check("database complete human score coverage", len(completed_human) == 20075
              and completed_human.protein_id.nunique() == 20075
              and completed_human.probability.between(0,1).all()
              and int(completed_human.probability.ge(.98).sum()) == 1344
              and int(completion.probability.ge(.98).sum()) == 35)
        check("database query identities", query.gene_name.tolist() == ["TMA7B","NUDT4B","IGKV3-7","IGKV1D-42","IGLV4-69","IGLV8-61"])
        complete_lookup = completed_human.set_index("protein_id").probability
        check("database query full-precision scores", all(
            r.species == "Human" and r.uniprot_id in complete_lookup.index
            and float(r.confidence) == float(complete_lookup.loc[r.uniprot_id])
            for r in query.itertuples()))
        check("database original frozen membership labels", all(
            r.evidence_group == (human.set_index("protein_id").dataset.loc[r.uniprot_id]
                                 if r.uniprot_id in set(human.protein_id) else "not_scored")
            for r in query.itertuples()))
        check("database default-view metadata", meta["query"]["keyword"] == ""
              and meta["query"]["species"] == "all" and meta["query"]["minimum_prediction_score"] == 0
              and meta["query"]["sort"] == "id_asc" and meta["matched_records"] == 37077)
        check("database query contains no novel candidates", query.novel_candidate_401.astype(str).str.lower().eq("true").sum() == 0 and meta["visible_novel_candidate_count"] == 0)
        check("database coverage metadata", meta["coverage"]["total_annotation_records"] == 37077
              and meta["coverage"]["strict_scored_records"] == 37077
              and meta["coverage"]["human_strict_scored"] == 20075
              and meta["coverage"]["mouse_strict_scored"] == 17002
              and meta["coverage"]["unscored_human_records"] == 0
              and meta["human_all_groups_score_ge_0_98"] == 1344
              and meta["human_untrained_pool_score_ge_0_98"] == 401)
        check("database strict score provenance", meta["source_prediction_hashes"]["human_all_predictions.csv"] == sha(frozen/"human_all_predictions.csv") and meta["source_prediction_hashes"]["mouse_all_predictions.csv"] == sha(frozen/"mouse_all_predictions.csv"))
        check("database runtime snapshot hashes", meta["runtime_json_sha256"] == completion_meta["runtime_json_sha256"]
              and meta["runtime_csv_sha256"] == completion_meta["runtime_csv_sha256"])
        check("current data profile excludes historical panels", not any(p.name in {"Figure3e.csv","SuppFigure1c.csv","Figure index.csv"} for p in data.iterdir()))
        return verified

    try:
        strict = test_all()
    except Exception as error:
        check("validation exception", False, f"{type(error).__name__}: {error}")
        strict = None
    passed = sum(x["passed"] for x in results)
    return dict(profile="strict401-manuscript-v12", passed=passed, total=len(results),
                checks=results, strict_validation=strict,
                source_workbook_rechecked=bool(source_workbook),
                limitations=["No retraining or unbundled checkpoint inference performed.",
                             "No assertion about remote publication or experimental reproduction."])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path, help="Optional report; omit for read-only stdout.")
    parser.add_argument("--source-workbook", type=Path, help="Optional actual workbook to recheck its frozen hash.")
    args = parser.parse_args(argv)
    report = run_checks(args.root, args.source_workbook)
    for item in report["checks"]:
        print(f"[{'PASS' if item['passed'] else 'FAIL'}] {item['check']}")
        if not item["passed"]:
            print("  ", item.get("observed"), "expected", item.get("expected"))
    if args.report:
        protected = [(args.root / "data").resolve(), (args.root / "configs").resolve()]
        target = args.report.resolve()
        if any(target == x or x in target.parents for x in protected):
            raise ValueError("Validation report cannot overwrite source data/configuration.")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"{report['passed']}/{report['total']} current-profile checks passed.")
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
