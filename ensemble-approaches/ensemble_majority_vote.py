#!/usr/bin/env python3
"""
Universal Prompt-Ensemble Majority Vote Implementation
Supports any combination of judges for flexible ensemble configurations.
"""

import json
import asyncio
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import Counter
import argparse

# Import existing modules from your project
from judge_evaluator import (
    JudgeEvaluator,
    ChatEvent,
    JudgeResult,
    load_judge_definition,
    load_events,
)


@dataclass
class EnsembleResult:
    """Result of ensemble evaluation"""

    correlation_id: str
    ensemble_name: str
    final_answer: str
    individual_answers: Dict[str, str]  # judge_name -> answer
    vote_counts: Dict[str, int]  # answer -> count
    confidence_score: float  # agreement ratio
    tie_broken: bool  # whether tie-breaking was used
    tie_break_method: str  # how tie was broken
    individual_analyses: Dict[str, str]  # judge_name -> analysis
    timestamp: str
    error: Optional[str] = None


class UniversalEnsembleMajorityVote:
    """
    Universal ensemble implementation that works with any combination of judges.
    Supports multiple tie-breaking strategies and flexible judge selection.
    """

    def __init__(self, api_key: str, base_url: str = None):
        """Initialize with Mistral client"""
        self.evaluator = JudgeEvaluator(api_key, base_url)
        self.judges = []
        self.ensemble_name = "Universal Ensemble"

        # Tie-breaking strategies
        self.tie_break_strategies = {
            "hierarchy": self._tie_break_hierarchy,
            "random": self._tie_break_random,
            "abstain": self._tie_break_abstain,
            "first": self._tie_break_first,
        }

    def load_judges_from_list(
        self,
        judge_files: List[str],
        judges_dir: str = "judges",
        use_folders: bool = True,
    ):
        """Load judges from a list of filenames"""
        self.judges = []
        loaded_names = []

        for judge_file in judge_files:
            try:
                judge = load_judge_definition(judge_file, use_folders)
                self.judges.append(judge)
                loaded_names.append(judge.name)
                print(f"[OK] Loaded judge: {judge.name}")
            except Exception as e:
                print(f"[X] Failed to load {judge_file}: {e}")
                raise

        self.ensemble_name = f"{len(self.judges)}-Judge Ensemble"
        print(f" Created ensemble with {len(self.judges)} judges")
        return loaded_names

    def load_judges_from_pattern(
        self, pattern: str, judges_dir: str = "judges", use_folders: bool = True
    ):
        """Load all judges matching a pattern (e.g., 'directive_strength_*.json')"""
        if use_folders:
            search_dir = Path(judges_dir)
        else:
            search_dir = Path(".")

        judge_files = list(search_dir.glob(pattern))

        if not judge_files:
            raise ValueError(f"No judge files found matching pattern: {pattern}")

        # Convert to relative paths for load_judge_definition
        relative_files = [f.name if use_folders else str(f) for f in judge_files]
        return self.load_judges_from_list(relative_files, judges_dir, use_folders)

    def load_all_judges_in_directory(
        self, judges_dir: str = "judges", use_folders: bool = True
    ):
        """Load all .json files from judges directory"""
        return self.load_judges_from_pattern("*.json", judges_dir, use_folders)

    def set_tie_break_hierarchy(self, judge_names_priority_order: List[str]):
        """Set custom tie-breaking hierarchy (higher index = higher priority)"""
        self.tie_breaking_hierarchy = judge_names_priority_order

    def _majority_vote(
        self, individual_results: List[JudgeResult], tie_break_method: str = "hierarchy"
    ) -> Tuple[str, Dict[str, int], float, bool, str]:
        """
        Perform majority voting with configurable tie-breaking

        Returns:
            final_answer: str
            vote_counts: Dict[str, int]
            confidence_score: float (0.0 to 1.0)
            tie_broken: bool
            tie_break_method_used: str
        """
        # Count votes from successful results
        answers = []
        successful_results = []

        for result in individual_results:
            if result.answer and not result.error:
                answers.append(result.answer)
                successful_results.append(result)

        if not answers:
            return "", {}, 0.0, False, "no_valid_answers"

        vote_counts = Counter(answers)
        max_votes = max(vote_counts.values())
        winners = [
            answer for answer, count in vote_counts.items() if count == max_votes
        ]

        # Calculate confidence as agreement ratio
        confidence_score = max_votes / len(answers)

        # Handle ties
        tie_broken = len(winners) > 1
        tie_break_method_used = tie_break_method if tie_broken else "no_tie"

        if tie_broken:
            if tie_break_method in self.tie_break_strategies:
                final_answer = self.tie_break_strategies[tie_break_method](
                    winners, successful_results
                )
                print(
                    f" Tie detected ({winners}) - using {tie_break_method}: {final_answer}"
                )
            else:
                final_answer = self._tie_break_first(winners, successful_results)
                tie_break_method_used = "first"
                print(f" Unknown tie-break method, using first: {final_answer}")
        else:
            final_answer = winners[0]

        return (
            final_answer,
            dict(vote_counts),
            confidence_score,
            tie_broken,
            tie_break_method_used,
        )

    def _tie_break_hierarchy(
        self, tied_answers: List[str], results: List[JudgeResult]
    ) -> str:
        """Break ties using judge hierarchy (requires set_tie_break_hierarchy to be called)"""
        if not hasattr(self, "tie_breaking_hierarchy"):
            print("[!]  No hierarchy set, falling back to first answer")
            return tied_answers[0]

        # Create mapping of judge name to answer
        judge_to_answer = {}
        for result in results:
            if result.answer in tied_answers:
                judge_to_answer[result.judge_name] = result.answer

        # Find highest priority judge that voted for a tied answer
        for judge_name in reversed(self.tie_breaking_hierarchy):
            if judge_name in judge_to_answer:
                return judge_to_answer[judge_name]

        # Fallback: return first tied answer
        return tied_answers[0]

    def _tie_break_random(
        self, tied_answers: List[str], results: List[JudgeResult]
    ) -> str:
        """Break ties randomly"""
        import random

        return random.choice(tied_answers)

    def _tie_break_abstain(
        self, tied_answers: List[str], results: List[JudgeResult]
    ) -> str:
        """Abstain from decision when there's a tie"""
        return "TIE_ABSTAIN"

    def _tie_break_first(
        self, tied_answers: List[str], results: List[JudgeResult]
    ) -> str:
        """Break ties by taking first answer"""
        return tied_answers[0]

    async def evaluate_ensemble(
        self, event: ChatEvent, tie_break_method: str = "hierarchy"
    ) -> EnsembleResult:
        """Evaluate a single event using ensemble majority vote"""
        try:
            # Get predictions from all judges
            individual_results = await asyncio.gather(*[
                self.evaluator.evaluate_event(judge, event) for judge in self.judges
            ])

            # Perform majority vote
            final_answer, vote_counts, confidence_score, tie_broken, tie_method = (
                self._majority_vote(individual_results, tie_break_method)
            )

            # Prepare individual answers and analyses
            individual_answers = {}
            individual_analyses = {}

            for result in individual_results:
                judge_key = self._get_judge_short_name(result.judge_name)
                individual_answers[judge_key] = (
                    result.answer if not result.error else "ERROR"
                )
                individual_analyses[judge_key] = (
                    result.analysis if not result.error else result.error
                )

            return EnsembleResult(
                correlation_id=event.correlation_id or "unknown",
                ensemble_name=self.ensemble_name,
                final_answer=final_answer,
                individual_answers=individual_answers,
                vote_counts=vote_counts,
                confidence_score=confidence_score,
                tie_broken=tie_broken,
                tie_break_method=tie_method,
                individual_analyses=individual_analyses,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            return EnsembleResult(
                correlation_id=event.correlation_id or "unknown",
                ensemble_name=self.ensemble_name,
                final_answer="",
                individual_answers={},
                vote_counts={},
                confidence_score=0.0,
                tie_broken=False,
                tie_break_method="error",
                individual_analyses={},
                timestamp=datetime.now().isoformat(),
                error=str(e),
            )

    def _get_judge_short_name(self, full_name: str) -> str:
        """Extract short name from full judge name for cleaner output"""
        # Extract key identifying words
        words = full_name.split()
        if "CoT" in full_name and "Role" in full_name and "Few" in full_name:
            return "CoT+Role+FS"
        elif "CoT" in full_name or "Chain-of-Thought" in full_name:
            return "CoT"
        elif "Role" in full_name:
            return "Role"
        elif "Few" in full_name or "Few-shot" in full_name:
            return "Few-shot"
        elif "Rubric" in full_name:
            return "Rubric"
        elif "Closed" in full_name:
            return "Closed"
        elif "Open" in full_name:
            return "Open"
        else:
            # Fallback: take last meaningful word
            return words[-2] if len(words) > 2 else full_name

    async def evaluate_batch(
        self,
        events: List[ChatEvent],
        tie_break_method: str = "hierarchy",
        max_concurrent: int = 3,
    ) -> List[EnsembleResult]:
        """Evaluate multiple events with concurrency control"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_evaluate(event):
            async with semaphore:
                return await self.evaluate_ensemble(event, tie_break_method)

        print(f" Starting ensemble evaluation of {len(events)} events...")
        print(
            f"   Using {len(self.judges)} judges with {tie_break_method} tie-breaking"
        )
        results = await asyncio.gather(*[bounded_evaluate(event) for event in events])
        return results


def save_ensemble_results(
    results: List[EnsembleResult], output_file: str, use_folders: bool = False
):
    """Save ensemble results to JSON file"""
    if use_folders:
        ensembles_dir = Path("ensembles")
        ensembles_dir.mkdir(exist_ok=True)

        output_path = Path(output_file)
        if len(output_path.parts) == 1:
            output_file = ensembles_dir / output_file

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    results_data = [asdict(result) for result in results]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    print(f"Ensemble results saved to {output_file}")


def print_ensemble_summary(results: List[EnsembleResult]):
    """Print summary of ensemble evaluation"""
    successful = sum(1 for r in results if not r.error)
    failed = len(results) - successful

    print(f"\n{'=' * 60}")
    print(f"ENSEMBLE EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total events evaluated: {len(results)}")
    print(f"Successful evaluations: {successful}")
    print(f"Failed evaluations: {failed}")

    if successful > 0:
        # Answer distribution
        answer_counts = Counter()
        confidence_scores = []
        tie_counts = 0
        tie_methods = Counter()

        for result in results:
            if not result.error and result.final_answer:
                answer_counts[result.final_answer] += 1
                confidence_scores.append(result.confidence_score)
                if result.tie_broken:
                    tie_counts += 1
                    tie_methods[result.tie_break_method] += 1

        print(f"\nAnswer distribution:")
        for answer, count in sorted(answer_counts.items()):
            percentage = (count / successful) * 100
            print(f"  {answer}: {count} ({percentage:.1f}%)")

        # Confidence analysis
        avg_confidence = (
            sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        )
        print(f"\nEnsemble confidence:")
        print(f"  Average agreement: {avg_confidence:.3f}")
        print(f"  Ties broken: {tie_counts} ({tie_counts / successful * 100:.1f}%)")

        if tie_methods:
            print(f"  Tie-break methods used:")
            for method, count in tie_methods.items():
                print(f"    {method}: {count}")

        # Individual judge performance summary
        judge_accuracy = Counter()
        total_votes = Counter()

        for result in results:
            if not result.error:
                for judge_name, answer in result.individual_answers.items():
                    if answer != "ERROR":
                        total_votes[judge_name] += 1
                        if answer == result.final_answer:
                            judge_accuracy[judge_name] += 1

        print(f"\nIndividual judge agreement with ensemble:")
        for judge_name in sorted(total_votes.keys()):
            if total_votes[judge_name] > 0:
                agreement = judge_accuracy[judge_name] / total_votes[judge_name]
                print(f"  {judge_name}: {agreement:.3f}")

    print(f"{'=' * 60}")


def load_dataset_events(
    dataset_file: str, use_folders: bool = False
) -> List[ChatEvent]:
    """Load events from directive_strength_dataset.json format"""
    if use_folders:
        events_path = Path(dataset_file)
        if len(events_path.parts) == 1:
            events_dir = Path("events")
            if (events_dir / dataset_file).exists():
                dataset_file = events_dir / dataset_file

    with open(dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    events = []

    # Handle your dataset format
    if isinstance(data, list):
        for i, item in enumerate(data):
            # Extract correlation_id - try multiple possible field names
            correlation_id = (
                item.get("correlation_id")
                or item.get("correlationId")
                or item.get("id")
                or item.get("conversation_id")
                or f"event_{i}"  # fallback
            )

            # Convert dataset format to ChatEvent
            event = ChatEvent(
                correlation_id=correlation_id,
                correlationId=correlation_id,  # Ensure both fields are set
                messages=item.get("messages", []),
                model=item.get("model", "unknown"),
                metadata=item.get("metadata", {}),
                extraFields=item.get("extraFields", {}),
            )
            events.append(event)
    else:
        raise ValueError(f"Unexpected dataset format in {dataset_file}")

    print(f" Loaded {len(events)} events with correlation IDs:")
    for i, event in enumerate(events[:3]):  # Show first 3 IDs
        print(f"   {i + 1}. {event.correlation_id}")
    if len(events) > 3:
        print(f"   ... and {len(events) - 3} more")

    return events


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Universal Prompt-Ensemble Majority Vote evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use specific judge files
  %(prog)s --judges directive_strength_cot_prompt_judge.json directive_strength_role_prompt_judge.json --events directive_strength_dataset.json --output ensemble_results.json --use-folders
  
  # Use all directive strength judges
  %(prog)s --judge-pattern "directive_strength_*.json" --events directive_strength_dataset.json --output ensemble_results.json --use-folders
  
  # Use all judges in directory
  %(prog)s --all-judges --events directive_strength_dataset.json --output ensemble_results.json --use-folders
  
  # Custom tie-breaking
  %(prog)s --judges judge1.json judge2.json judge3.json --events events.json --tie-break random --output results.json
        """,
    )

    # Judge selection options (mutually exclusive)
    judge_group = parser.add_mutually_exclusive_group(required=True)
    judge_group.add_argument(
        "--judges", nargs="+", help="List of specific judge files to use"
    )
    judge_group.add_argument(
        "--judge-pattern",
        help="Pattern to match judge files (e.g., 'directive_strength_*.json')",
    )
    judge_group.add_argument(
        "--all-judges", action="store_true", help="Use all judges in judges directory"
    )

    parser.add_argument(
        "--events",
        default="directive_strength_dataset.json",
        help="Path to events JSON file (default: directive_strength_dataset.json)",
    )
    parser.add_argument(
        "--output", required=True, help="Path to output results JSON file"
    )
    parser.add_argument(
        "--use-folders", action="store_true", help="Use organized folder structure"
    )
    parser.add_argument(
        "--api-key", help="Mistral API key (or set MISTRAL_API_KEY env var)"
    )
    parser.add_argument("--base-url", help="Custom API base URL")
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Max concurrent evaluations (default: 3)",
    )
    parser.add_argument(
        "--tie-break",
        choices=["hierarchy", "random", "abstain", "first"],
        default="hierarchy",
        help="Tie-breaking method (default: hierarchy)",
    )
    parser.add_argument(
        "--hierarchy", nargs="*", help="Judge names in priority order for tie-breaking"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    api_key = args.api_key or os.getenv("MISTRAL_API_KEY")
    base_url = args.base_url or os.getenv("MISTRAL_BASE_URL")

    if not api_key:
        print("[X] Error: API key required!")
        print("   Use --api-key argument or set MISTRAL_API_KEY environment variable")
        return 1

    try:
        if args.verbose:
            print(f"[...] Loading events from {args.events}")
        events = load_events(
            args.events, args.use_folders
        )  # Use standard load_events instead

        ensemble = UniversalEnsembleMajorityVote(api_key, base_url)

        # Load judges based on selection method
        if args.judges:
            judge_names = ensemble.load_judges_from_list(
                args.judges, use_folders=args.use_folders
            )
        elif args.judge_pattern:
            judge_names = ensemble.load_judges_from_pattern(
                args.judge_pattern, use_folders=args.use_folders
            )
        elif args.all_judges:
            judge_names = ensemble.load_all_judges_in_directory(
                use_folders=args.use_folders
            )

        # Set up tie-breaking hierarchy if provided
        if args.hierarchy:
            ensemble.set_tie_break_hierarchy(args.hierarchy)
            print(f" Set tie-break hierarchy: {args.hierarchy}")
        elif args.tie_break == "hierarchy":
            # Default hierarchy for directive strength judges
            default_hierarchy = [
                "Direct Command Judge Chain-of-Thought Prompt",
                "Direct Command Judge Role Prompt",
                "Direct Command Judge Few-Shots Prompt",
                "Direct Command Judge CoT Role FewShots Prompt",
            ]
            # Only use judges that were actually loaded
            valid_hierarchy = [
                name for name in default_hierarchy if name in judge_names
            ]
            if valid_hierarchy:
                ensemble.set_tie_break_hierarchy(valid_hierarchy)
                print(f" Using default hierarchy: {valid_hierarchy}")

        print(f" Ensemble: {len(judge_names)}-judge majority vote")
        print(f" Events to evaluate: {len(events)}")
        print(f" Max concurrent: {args.max_concurrent}")
        print(f" Tie-break method: {args.tie_break}")

        # Run ensemble evaluation
        start_time = datetime.now()
        results = await ensemble.evaluate_batch(
            events, args.tie_break, args.max_concurrent
        )
        end_time = datetime.now()

        duration = (end_time - start_time).total_seconds()

        print_ensemble_summary(results)
        print(f" Total time: {duration:.2f} seconds")
        print(f" Average time per event: {duration / len(events):.2f} seconds")
        print(
            f" API calls made: {len(events) * len(judge_names)} ({len(judge_names)} per event)"
        )

        save_ensemble_results(results, args.output, args.use_folders)

        return 0

    except Exception as e:
        print(f"[X] Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
