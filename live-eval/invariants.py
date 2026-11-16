"""Run invariants. Any violation marks the run INVALID; its numbers must not be reported as results."""
from __future__ import annotations

import config


def check(summary: dict, spend: dict, manual_checked: bool = False) -> dict:
    arms = summary.get("arms", {})
    violations, warnings = [], []
    o, i = arms.get("oracle"), arms.get("iso")
    if o and i and o["n"] and i["n"]:
        d = i["acc"] - o["acc"]
        if d > config.INV_OVERALL_TOL:
            violations.append(f"iso beats oracle overall by {d:+.3f} (> {config.INV_OVERALL_TOL}): oracle arm is suspect")
        for t, it in i["by_type"].items():
            ot = o["by_type"].get(t)
            if ot and it["n"] >= config.INV_TYPE_MIN_N and it["acc"] - ot["acc"] > config.INV_TYPE_TOL:
                violations.append(f"iso beats oracle on {t} by {it['acc'] - ot['acc']:+.3f} (n={it['n']})")
    for name, a in arms.items():
        if not a["n"]:
            continue
        if a["acc"] > config.INV_TOO_GOOD and not manual_checked:
            violations.append(f"{name} accuracy {a['acc']:.3f} > {config.INV_TOO_GOOD}: check 10 rows by hand, then rerun summary with --manual-checked")
        if a["parse_fail"] / a["n"] > config.INV_MAX_PARSE_FAIL:
            violations.append(f"{name} judge parse failures {a['parse_fail']}/{a['n']}")
        if a["errors"] / a["n"] > config.INV_MAX_ERRORS:
            violations.append(f"{name} errors {a['errors']}/{a['n']}")
        if a.get("truncated", 0):
            warnings.append(f"{name}: {a['truncated']} questions had context truncated by the budget")
    if spend.get("stopped"):
        violations.append("spend cap reached: run incomplete")
    if spend.get("incomplete_rows"):
        violations.append(f"{spend['incomplete_rows']} rows not run yet: resume before reading results")
    return {"valid": not violations, "violations": violations, "warnings": warnings}
