import csv
import json
from pathlib import Path

ORDER = ["source", "rtn3_g128", "rtn4_g128", "awq3_g128", "awq4_g128", "gptq3_g128"]

def compute_deltas(source_exact: float, exact: float, source_ppl: float, ppl: float) -> dict:
    return {"delta_exact_rate": exact-source_exact,
            "relative_exact_retention": None if source_exact == 0 else exact/source_exact,
            "delta_ppl_abs": ppl-source_ppl,
            "delta_ppl_pct": (ppl-source_ppl)/source_ppl*100}

def collect_rows(root: Path) -> list[dict]:
    rows = []
    for setting in ORDER:
        directory = root / setting
        try:
            imf = json.loads((directory / "imf_summary.json").read_text())
            ppl = json.loads((directory / "ppl.json").read_text())
            meta_path = directory / "metadata.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        except FileNotFoundError as exc:
            raise ValueError("results must contain all six settings") from exc
        ppl_value=ppl["ppl"]; ppl_value=ppl_value.get("wikitext2") if isinstance(ppl_value,dict) else ppl_value
        rows.append({"setting": setting, "quantizer": meta.get("quantizer", "source"),
                     "bits": meta.get("bits"), "group_size": meta.get("group_size"),
                     "native_payload_success": imf.get("payload_success"), "native_payload_rate": imf.get("payload_rate"),
                     "false_verification_rate": imf.get("false_verification_rate"), "exact_success": imf["exact_success"],
                     "exact_rate": imf["exact_rate"], "mean_sequence_target_nll": imf["mean_sequence_target_nll"],
                     "mean_target_logprob": imf["mean_target_logprob"], "wikitext2_ppl": ppl_value,
                     "checkpoint_path": meta.get("checkpoint_path"), "quantizer_commit": meta.get("quantizer_commit"),
                     "calibration_sha256": meta.get("calibration_sha256"), "eval_ppl_sha256": ppl.get("eval_ppl_sha256")})
    source = rows[0]
    for row in rows:
        row.update(compute_deltas(source["exact_rate"], row["exact_rate"], source["wikitext2_ppl"], row["wikitext2_ppl"]))
        source_payload = source["native_payload_rate"]
        row["relative_payload_retention"] = (
            None if source_payload in (None, 0) or row["native_payload_rate"] is None
            else row["native_payload_rate"] / source_payload
        )
    return rows

def write_results(root: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
