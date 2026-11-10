"""Load and check the root-cause taxonomy in eval/taxonomy.toml.

Usage:
  python -m eval.taxonomy        check the file and print its codes

Prit writes the taxonomy. This module only checks its shape, so the labels,
the tagger prompt and the report all read the same codes:

- every cause has code, name, definition, signals, not_this, rule, examples
- codes are unique snake_case, and third_party, other and unclear exist
- 6 to 8 root causes, not counting third_party, other and unclear
- precedence lists every code exactly once
- rule is a table keyed by the exact product names, each value a Reg E or
  Reg Z provision such as "1005.11" or "1026.12(b)"; third_party, other and
  unclear have rule = {}
- examples are refs from the reading sample (eval/taxonomy_sample.json);
  every code except other and unclear has at least one
"""
import json
import os
import re
import sys
import tomllib

from ingest import sources

TAXONOMY_PATH = "eval/taxonomy.toml"
READING_PATH = "eval/taxonomy_sample.json"

NOT_ROOT_CAUSES = ("third_party", "other", "unclear")
MAY_LACK_EXAMPLES = ("other", "unclear")
MIN_CAUSES, MAX_CAUSES = 6, 8
TEXT_FIELDS = ("name", "definition", "signals", "not_this")
CODE = re.compile(r"^[a-z][a-z_]*[a-z]$")
PROVISION = re.compile(r"^10(05|26)\.\d+(\([a-z0-9]+\))*$")


class TaxonomyError(Exception):
    pass


def reading_refs(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return {it["ref"] for it in json.load(f)["items"]}


def problems(tax, refs):
    """Return a list of problems, empty when the taxonomy is well formed."""
    out = []
    for key in ("version", "written", "primary_rule"):
        if not str(tax.get(key, "")).strip():
            out.append(f"top-level {key} is empty")
    causes = tax.get("cause", [])
    codes = [c.get("code", "") for c in causes]
    for i, c in enumerate(causes, 1):
        code = c.get("code", "")
        where = f"cause {i} ({code or 'no code'})"
        if not CODE.match(code):
            out.append(f"{where}: code must be snake_case letters")
        for field in TEXT_FIELDS:
            if not str(c.get(field, "")).strip():
                out.append(f"{where}: {field} is empty")
        rule = c.get("rule")
        if not isinstance(rule, dict):
            out.append(f"{where}: rule must be a table, e.g. "
                       '{ "Credit card" = "1026.13" } or {}')
        else:
            for product, provision in rule.items():
                if product not in sources.PRODUCTS:
                    out.append(f"{where}: rule key {product!r} is not a product name "
                               f"({' or '.join(sources.PRODUCTS)})")
                if not isinstance(provision, str) or not PROVISION.match(provision):
                    out.append(f"{where}: rule value {provision!r} is not a provision "
                               "like 1005.11 or 1026.12(b)")
            if code in NOT_ROOT_CAUSES and rule:
                out.append(f"{where}: rule must be {{}} for {code}")
        examples = c.get("examples", [])
        if not isinstance(examples, list):
            out.append(f"{where}: examples must be a list")
            examples = []
        if not examples and code not in MAY_LACK_EXAMPLES:
            out.append(f"{where}: needs at least one example ref")
        if refs is not None:
            unknown = [r for r in examples if r not in refs]
            if unknown:
                out.append(f"{where}: examples not in the reading sample: {unknown}")
    dupes = sorted({c for c in codes if codes.count(c) > 1})
    if dupes:
        out.append(f"codes used more than once: {dupes}")
    for code in NOT_ROOT_CAUSES:
        if code not in codes:
            out.append(f"code {code} is missing")
    n = len([c for c in codes if c not in NOT_ROOT_CAUSES])
    if not MIN_CAUSES <= n <= MAX_CAUSES:
        out.append(f"{n} root causes; expected {MIN_CAUSES} to {MAX_CAUSES} "
                   f"(not counting {', '.join(NOT_ROOT_CAUSES)})")
    precedence = tax.get("precedence", [])
    if sorted(precedence) != sorted(set(codes)) or len(precedence) != len(set(precedence)):
        missing = sorted(set(codes) - set(precedence))
        extra = sorted(set(precedence) - set(codes))
        twice = sorted({p for p in precedence if precedence.count(p) > 1})
        out.append(f"precedence must list every code once (missing {missing}, "
                   f"unknown {extra}, repeated {twice})")
    return out


def load(path=None, reading_path=None):
    """Return the parsed taxonomy, or raise TaxonomyError listing every problem."""
    path = path or TAXONOMY_PATH
    reading_path = reading_path or READING_PATH
    if not os.path.exists(path):
        raise TaxonomyError(f"{path} not found")
    with open(path, "rb") as f:
        try:
            tax = tomllib.load(f)
        except tomllib.TOMLDecodeError as err:
            raise TaxonomyError(f"{path} is not valid TOML: {err}") from err
    found = problems(tax, reading_refs(reading_path))
    if found:
        raise TaxonomyError(f"{path} has {len(found)} problem(s):\n  " + "\n  ".join(found))
    tax["codes"] = [c["code"] for c in tax["cause"]]
    tax["by_code"] = {c["code"]: c for c in tax["cause"]}
    return tax


def main(argv):
    if argv:
        sys.exit("usage: python -m eval.taxonomy")
    try:
        tax = load()
    except TaxonomyError as err:
        sys.exit(f"ERROR {err}")
    roots = [c for c in tax["codes"] if c not in NOT_ROOT_CAUSES]
    print(f"{TAXONOMY_PATH} version {tax['version']}: {len(roots)} root causes "
          f"plus {', '.join(NOT_ROOT_CAUSES)}")
    for code in tax["precedence"]:
        c = tax["by_code"][code]
        rule = ", ".join(f"{p}: {v}" for p, v in c["rule"].items()) or "none"
        print(f"  {code:<20} {c['name']:<40} rule {rule}")


if __name__ == "__main__":
    main(sys.argv[1:])
