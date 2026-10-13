from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .case_discovery import select_cases
from .config import ValidationConfig
from .environment import load_project_environment
from .llm_agent import RunBudget
from .pipeline import ValidationPipeline
from .reporting import write_reports

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate archived OpenFOAM cases without modifying them.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "flow_fields")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "cfd_validation_agent")
    parser.add_argument("--case-ids", nargs="+", help="Analyze only these explicit case IDs.")
    parser.add_argument("--limit", type=int, help="Analyze at most this many cases in sorted order.")
    parser.add_argument("--no-llm", action="store_true", help="Run deterministic validation only.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare, but never send, LLM requests.")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument("--all-cases", action="store_true",
                        help="Explicitly select every discovered case (deterministic runs are allowed).")
    parser.add_argument("--full-dataset-llm", action="store_true",
                        help="Explicitly permit live LLM analysis without --case-ids or --limit.")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    if args.max_cost_usd is not None and args.max_cost_usd < 0:
        raise SystemExit("--max-cost-usd cannot be negative")
    live_llm = not args.no_llm and not args.dry_run
    if args.all_cases and (args.case_ids or args.limit is not None):
        raise SystemExit("--all-cases cannot be combined with --case-ids or --limit")
    unrestricted = not args.case_ids and args.limit is None
    if live_llm and unrestricted and not args.full_dataset_llm:
        raise SystemExit("Live full-dataset LLM processing requires --full-dataset-llm")
    if unrestricted and not args.all_cases and not args.full_dataset_llm:
        args.limit = 5
    return args


def main(argv: list[str] | None = None) -> int:
    load_project_environment(PROJECT_ROOT)
    args = parse_args(argv)
    selected = select_cases(args.dataset.resolve(), args.case_ids, args.limit)
    live_llm = not args.no_llm and not args.dry_run
    config = ValidationConfig()
    config = replace(config, agent=replace(config.agent, model=args.model))
    upper_cost = sum(
        config.agent.max_output_tokens * config.agent.output_cost_per_million / 1_000_000
        + config.agent.max_prompt_chars / 4 * config.agent.input_cost_per_million / 1_000_000
        for _ in selected
    )
    print(f"Selected cases: {len(selected)}")
    print(f"LLM mode: {'live' if live_llm else 'dry-run' if args.dry_run else 'disabled'}")
    print(f"Maximum initial estimated API cost: ${upper_cost:.4f} before cache hits")
    if args.max_cost_usd is not None:
        print(f"Run-level API budget: ${args.max_cost_usd:.4f}")
    pipeline = ValidationPipeline(config)
    budget = RunBudget(args.max_cost_usd)
    cache_dir = args.output.resolve() / ".llm_cache"
    results = []
    for index, case_dir in enumerate(selected, 1):
        print(f"[{index}/{len(selected)}] {case_dir.name}")
        results.append(pipeline.analyze(case_dir, llm=live_llm, dry_run=args.dry_run,
                                        cache_dir=cache_dir, budget=budget))
    summary = write_reports(results, args.output.resolve())
    print(f"PASS={summary.pass_count} FAIL={summary.fail_count} REVIEW={summary.review_count}")
    print(f"Reports: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
