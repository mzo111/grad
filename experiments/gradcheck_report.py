"""Run every gradcheck in the test suite and report how many ran and the worst error.

The suite asserts each finite-difference check is under ``tests.gradcheck.TOL``, which
establishes a bound but never says how much room is left under it. This runs the whole
suite, collects the error from every ``check_grad`` call, and reports the count, the
maximum, and which test produced it — so the figure quoted in the README has a command
behind it rather than a claim.

It reports what it measures. If the worst error moves, that is a result about the engine or
the platform, not something to tune away.

Usage, from the repo root:
    python -m experiments.gradcheck_report [-n 10]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from tests.gradcheck import OBSERVED, TOL  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--top", type=int, default=10, help="how many worst cases to list")
    ap.add_argument("--tests", default="tests", help="path passed to pytest")
    args = ap.parse_args()

    # -p no:cacheprovider keeps a reporting run from rewriting .pytest_cache.
    code = pytest.main(["-q", "--no-header", "-p", "no:cacheprovider", args.tests])
    if code != 0:
        print(f"\npytest exited {code}; the suite must pass before the report means anything")
        return int(code)
    if not OBSERVED:
        print("\nno check_grad calls were recorded — the reporter is not wired to the suite")
        return 1

    errors = sorted(OBSERVED, key=lambda row: row[1], reverse=True)
    worst_test, worst = errors[0]
    print()
    print(f"gradcheck calls:  {len(errors)}")
    print(f"tolerance:        {TOL:.0e}   (central differences, eps 1e-6, random cotangent)")
    print(f"worst error:      {worst:.3e}   ({TOL / worst:.0f}x inside the tolerance)")
    print(f"worst case:       {worst_test}")
    print()
    print(f"top {min(args.top, len(errors))} by error:")
    for test, err in errors[: args.top]:
        print(f"  {err:.3e}  {test}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
