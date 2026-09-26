#!/usr/bin/env bash
# End to end: dataset -> run -> target -> evaluators -> results -> compare -> gate -> report.
# Run from this directory:  bash run.sh      (uses ./quickstart.db; delete it to start over)
set -u
export EVALKIT_DB_PATH=./quickstart.db
export PYTHONPATH=.            # so `targets:candidate` can be imported (trusted, operator code)

evalkit dataset import arithmetic cases.jsonl                       # -> arithmetic@1
BASE=$(evalkit runs create arithmetic --config baseline.toml --name baseline | python -c 'import json,sys; print(json.load(sys.stdin)["id"])')
evalkit runs execute "$BASE" > /dev/null                            # 60 cases x 2 evaluators
evalkit runs tag "$BASE" main                                       # a baseline is "tag:main"

CAND=$(evalkit runs create arithmetic --config candidate.toml --name candidate | python -c 'import json,sys; print(json.load(sys.stdin)["id"])')
evalkit runs execute "$CAND" --target targets:candidate > /dev/null

evalkit runs show "$CAND" --summary | python -m json.tool | head -40   # counts, coverage, metrics with CIs
evalkit runs failures "$CAND"                                          # [] : nothing failed

evalkit compare "$CAND" --baseline tag:main --gates gates.toml > compare.json
echo "gate exit code: $?  (0 pass, 3 regression, 4 inconclusive under --strict)"

evalkit report "$CAND" --compare tag:main --gates gates.toml --out report.html --force
echo "open report.html"
