#!/usr/bin/env python3
"""
Self-Consistency Ensemble Implementation
Uses the best strategy (CoT+Role+Few-shot) with multiple reasoning paths.
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
import random

# Import existing modules from your project
from judge_evaluator import (
    JudgeEvaluator,
    JudgeDefinition,
    ChatEvent,
    JudgeResult,
    load_judge_definition,
    load_events,
)


@dataclass
class SelfConsistencyResult:
    """Result of self-consistency ensemble evaluation"""

    correlation_id: str
    ensemble_name: str
    final_answer: str
    individual_answers: List[str]  # k different reasoning paths
    vote_counts: Dict[str, int]  # answer -> count
    confidence_score: float  # agreement ratio
    reasoning_paths: List[str]  # k different analyses
    temperature_used: float
    num_paths: int
    reasoning_variations: List[str]  # variation prompts used
    timestamp: str
    error: Optional[str] = None


class SelfConsistencyEnsemble:
    """
    Self-consistency ensemble using multiple reasoning paths from the same best judge.
    Implements temperature variation and reasoning seed diversification.
    """

    def __init__(self, api_key: str, base_url: str = None):
        """Initialize with Mistral client"""
        self.evaluator = JudgeEvaluator(api_key, base_url)
        self.best_judge = None

        # More aggressive reasoning variation prompts to break systematic bias
        self.reasoning_variations = [
            "",  # Original prompt (no variation)
            # Instruction-focused variations
            "CRITICAL: Look specifically for any step-by-step instructions, procedures, or actionable guidance that the user could follow, even if embedded within longer explanations.",
            # Context-switching variations
            "Analyze this as if you're helping someone who struggles to detect subtle instructions. What actionable steps or guidance does the assistant provide?",
            # Adversarial perspective
            "Challenge your initial assessment: Could this response be interpreted as giving the user specific things to do? Look for any procedural or directive language.",
            # Format-aware analysis
            "Ignore formatting and focus purely on content: Does this response tell the user HOW to do something or WHAT to do, regardless of how it's presented?",
            # Recipe/tutorial detector
            "If someone followed this response step-by-step, would they accomplish a task? Look for any sequences of actions or procedures.",
            # Intent-based analysis
            "What was the user asking for? If they wanted to DO something, does this response provide actionable guidance to achieve that goal?",
        ]

        # Standard reasoning variations (original)
        self.standard_reasoning_variations = [
            "",  # Original prompt (no variation)
            "Before making your final decision, consider alternative interpretations of the message.",
            "Think about this from the perspective of someone who might disagree with your initial assessment.",
            "Focus particularly on the context and intent behind the words used.",
            "Consider whether the phrasing suggests urgency, optionality, or neutrality.",
            "Examine the relationship between what the user asked and how the assistant responded.",
            "Pay special attention to any procedural or step-by-step language in the response.",
        ]

    def load_best_judge(
        self,
        judge_file: str = "directive_strength_cot_role_fewshot_prompt_judge.json",
        use_folders: bool = True,
    ):
        """Load the best performing judge (CoT+Role+Few-shot)"""
        try:
            self.best_judge = load_judge_definition(judge_file, use_folders)
            print(f"[OK] Loaded best judge: {self.best_judge.name}")
        except Exception as e:
            print(f"[X] Failed to load best judge {judge_file}: {e}")
            raise

    def _modify_judge_instructions(
        self,
        base_judge: JudgeDefinition,
        variation_prompt: str,
        override_mode: bool = False,
    ) -> JudgeDefinition:
        """Create a modified version of the judge with additional reasoning guidance"""
        if not variation_prompt:
            return base_judge

        # Create a copy with modified instructions
        if override_mode:
            # Replace instructions entirely for more radical variation
            modified_instructions = f"{variation_prompt}\n\nUse the same output options as the original judge."
        else:
            # Add to existing instructions
            modified_instructions = (
                f"{base_judge.instructions}\n\nAdditional guidance: {variation_prompt}"
            )

        modified_judge = JudgeDefinition(
            name=f"{base_judge.name} (Variation)",
            description=base_judge.description,
            model_name=base_judge.model_name,
            instructions=modified_instructions,
            output_options=base_judge.output_options,
            tools=base_judge.tools,
        )

        return modified_judge

    async def _evaluate_with_variation(
        self,
        event: ChatEvent,
        variation_prompt: str,
        temperature: float = 0.1,
        override_mode: bool = False,
    ) -> JudgeResult:
        """Evaluate with a specific reasoning variation and temperature"""
        modified_judge = self._modify_judge_instructions(
            self.best_judge, variation_prompt, override_mode
        )

        # Temporarily modify the evaluator's temperature for this call
        original_temp = getattr(self.evaluator, "_temperature", 0.1)
        self.evaluator._temperature = temperature

        try:
            result = await self.evaluator.evaluate_event(modified_judge, event)
            return result
        finally:
            # Restore original temperature
            self.evaluator._temperature = original_temp

    def _majority_vote_with_confidence(
        self, individual_results: List[JudgeResult]
    ) -> Tuple[str, Dict[str, int], float]:
        """
        Perform majority voting and calculate confidence based on agreement

        Returns:
            final_answer: str
            vote_counts: Dict[str, int]
            confidence_score: float (0.0 to 1.0)
        """
        # Count votes from successful results
        answers = []

        for result in individual_results:
            if result.answer and not result.error:
                answers.append(result.answer)

        if not answers:
            return "", {}, 0.0

        vote_counts = Counter(answers)
        max_votes = max(vote_counts.values())
        winners = [
            answer for answer, count in vote_counts.items() if count == max_votes
        ]

        # Calculate confidence as agreement ratio
        confidence_score = max_votes / len(answers)

        # In case of tie, take the first (could add more sophisticated tie-breaking)
        final_answer = winners[0]

        return final_answer, dict(vote_counts), confidence_score

    async def evaluate_self_consistency(
        self,
        event: ChatEvent,
        num_paths: int = 5,
        temperature_range: Tuple[float, float] = (0.1, 0.3),
        use_reasoning_variations: bool = True,
        aggressive_variations: bool = False,
        override_instructions: bool = False,
    ) -> SelfConsistencyResult:
        """Evaluate using self-consistency with multiple reasoning paths"""
        try:
            if not self.best_judge:
                raise ValueError("Best judge not loaded. Call load_best_judge() first.")

            # Generate reasoning paths
            tasks = []
            reasoning_variations_used = []
            temperatures_used = []

            # Choose which reasoning variations to use
            variations_to_use = (
                self.reasoning_variations
                if aggressive_variations
                else self.standard_reasoning_variations
            )

            for i in range(num_paths):
                # Vary temperature slightly for each path
                temp = random.uniform(*temperature_range)
                temperatures_used.append(temp)

                # Select reasoning variation
                if use_reasoning_variations and i < len(variations_to_use):
                    variation = variations_to_use[i]
                else:
                    variation = ""  # Use original prompt for extra paths

                reasoning_variations_used.append(variation)

                # Create evaluation task
                task = self._evaluate_with_variation(
                    event, variation, temp, override_instructions
                )
                tasks.append(task)

            # Execute all reasoning paths concurrently
            individual_results = await asyncio.gather(*tasks)

            # Perform majority vote
            final_answer, vote_counts, confidence_score = (
                self._majority_vote_with_confidence(individual_results)
            )

            # Extract individual answers and reasoning paths
            individual_answers = []
            reasoning_paths = []

            for result in individual_results:
                individual_answers.append(
                    result.answer if not result.error else "ERROR"
                )
                reasoning_paths.append(
                    result.analysis
                    if not result.error
                    else result.error or "No analysis"
                )

            # Calculate average temperature used
            avg_temperature = sum(temperatures_used) / len(temperatures_used)

            return SelfConsistencyResult(
                correlation_id=event.correlation_id or "unknown",
                ensemble_name=f"Self-Consistency ({num_paths} paths)",
                final_answer=final_answer,
                individual_answers=individual_answers,
                vote_counts=vote_counts,
                confidence_score=confidence_score,
                reasoning_paths=reasoning_paths,
                temperature_used=avg_temperature,
                num_paths=num_paths,
                reasoning_variations=reasoning_variations_used,
                timestamp=datetime.now().isoformat(),
            )

        except Exception as e:
            return SelfConsistencyResult(
                correlation_id=event.correlation_id or "unknown",
                ensemble_name=f"Self-Consistency ({num_paths} paths)",
                final_answer="",
                individual_answers=[],
                vote_counts={},
                confidence_score=0.0,
                reasoning_paths=[],
                temperature_used=0.0,
                num_paths=num_paths,
                reasoning_variations=[],
                timestamp=datetime.now().isoformat(),
                error=str(e),
            )

    async def evaluate_batch(
        self,
        events: List[ChatEvent],
        num_paths: int = 5,
        temperature_range: Tuple[float, float] = (0.1, 0.3),
        use_reasoning_variations: bool = True,
        aggressive_variations: bool = False,
        override_instructions: bool = False,
        max_concurrent: int = 2,
    ) -> List[SelfConsistencyResult]:
        """Evaluate multiple events with self-consistency"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_evaluate(event):
            async with semaphore:
                return await self.evaluate_self_consistency(
                    event, num_paths, temperature_range, use_reasoning_variations
                )

        print(f" Starting self-consistency evaluation of {len(events)} events...")
        print(f"   Using {num_paths} reasoning paths per event")
        print(f"   Temperature range: {temperature_range}")
        print(
            f"   Reasoning variations: {'enabled' if use_reasoning_variations else 'disabled'}"
        )

        results = await asyncio.gather(*[bounded_evaluate(event) for event in events])
        return results


def save_self_consistency_results(
    results: List[SelfConsistencyResult], output_file: str, use_folders: bool = False
):
    """Save self-consistency results to JSON file"""
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

    print(f" Self-consistency results saved to {output_file}")


def print_self_consistency_summary(results: List[SelfConsistencyResult]):
    """Print summary of self-consistency evaluation"""
    successful = sum(1 for r in results if not r.error)
    failed = len(results) - successful

    print(f"\n{'=' * 60}")
    print(f"SELF-CONSISTENCY EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total events evaluated: {len(results)}")
    print(f"Successful evaluations: {successful}")
    print(f"Failed evaluations: {failed}")

    if successful > 0:
        # Answer distribution
        answer_counts = Counter()
        confidence_scores = []
        path_counts = []

        for result in results:
            if not result.error and result.final_answer:
                answer_counts[result.final_answer] += 1
                confidence_scores.append(result.confidence_score)
                path_counts.append(result.num_paths)

        print(f"\nAnswer distribution:")
        for answer, count in sorted(answer_counts.items()):
            percentage = (count / successful) * 100
            print(f"  {answer}: {count} ({percentage:.1f}%)")

        # Confidence analysis
        avg_confidence = (
            sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        )
        min_confidence = min(confidence_scores) if confidence_scores else 0
        max_confidence = max(confidence_scores) if confidence_scores else 0

        print(f"\nSelf-consistency metrics:")
        print(f"  Average confidence: {avg_confidence:.3f}")
        print(f"  Confidence range: {min_confidence:.3f} - {max_confidence:.3f}")
        print(f"  Average paths per event: {sum(path_counts) / len(path_counts):.1f}")

        # Analyze agreement patterns
        high_confidence = sum(1 for c in confidence_scores if c >= 0.8)
        medium_confidence = sum(1 for c in confidence_scores if 0.6 <= c < 0.8)
        low_confidence = sum(1 for c in confidence_scores if c < 0.6)

        print(f"\nConfidence distribution:")
        print(
            f"  High confidence (≥0.8): {high_confidence} ({high_confidence / successful * 100:.1f}%)"
        )
        print(
            f"  Medium confidence (0.6-0.8): {medium_confidence} ({medium_confidence / successful * 100:.1f}%)"
        )
        print(
            f"  Low confidence (<0.6): {low_confidence} ({low_confidence / successful * 100:.1f}%)"
        )

        # Temperature analysis
        if results and not results[0].error:
            avg_temp = (
                sum(r.temperature_used for r in results if not r.error) / successful
            )
            print(f"  Average temperature used: {avg_temp:.3f}")

    print(f"{'=' * 60}")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Self-Consistency Ensemble evaluation using multiple reasoning paths",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic self-consistency with 5 paths
  %(prog)s --events directive_strength_dataset.json --output self_consistency_results.json --use-folders
  
  # More paths with wider temperature range
  %(prog)s --events events.json --output results.json --num-paths 7 --temp-min 0.1 --temp-max 0.4
  
  # Disable reasoning variations (temperature-only diversity)
  %(prog)s --events events.json --output results.json --no-reasoning-variations
  
  # Custom judge file
  %(prog)s --judge custom_best_judge.json --events events.json --output results.json --use-folders
        """,
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
        "--judge",
        default="directive_strength_cot_role_fewshot_prompt_judge.json",
        help="Best judge file to use (default: CoT+Role+Few-shot)",
    )
    parser.add_argument(
        "--num-paths",
        type=int,
        default=5,
        help="Number of reasoning paths (default: 5)",
    )
    parser.add_argument(
        "--temp-min", type=float, default=0.1, help="Minimum temperature (default: 0.1)"
    )
    parser.add_argument(
        "--temp-max",
        type=float,
        default=0.5,
        help="Maximum temperature (default: 0.5, increased for more diversity)",
    )
    parser.add_argument(
        "--no-reasoning-variations",
        action="store_true",
        help="Disable reasoning variations (use temperature-only diversity)",
    )
    parser.add_argument(
        "--aggressive-variations",
        action="store_true",
        help="Use more aggressive reasoning variations to break systematic bias",
    )
    parser.add_argument(
        "--override-instructions",
        action="store_true",
        help="Override original instructions entirely (more radical variation)",
    )
    parser.add_argument(
        "--force-diversity",
        action="store_true",
        help="Use wider temperature range and aggressive variations",
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
        default=2,
        help="Max concurrent evaluations (default: 2, lower due to multiple paths)",
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
            print(f" Loading events from {args.events}")
        events = load_events(args.events, args.use_folders)

        ensemble = SelfConsistencyEnsemble(api_key, base_url)
        ensemble.load_best_judge(args.judge, args.use_folders)

        print(f" Self-Consistency Ensemble: {args.num_paths} reasoning paths")

        # Process command line arguments for variations
        use_reasoning_variations = True  # Default enabled
        aggressive_variations = (
            args.aggressive_variations
            if hasattr(args, "aggressive_variations")
            else False
        )
        override_instructions = (
            args.override_instructions
            if hasattr(args, "override_instructions")
            else False
        )

        # Handle force-diversity flag
        if hasattr(args, "force_diversity") and args.force_diversity:
            aggressive_variations = True
            override_instructions = True
            if args.temp_max < 0.6:
                print(
                    " Force diversity enabled - expanding temperature range to 0.2-0.8"
                )
                temperature_range = (0.2, 0.8)
            else:
                temperature_range = (args.temp_min, args.temp_max)
        else:
            temperature_range = (args.temp_min, args.temp_max)

        # Update reasoning variations if aggressive mode
        if aggressive_variations:
            print(" Using aggressive reasoning variations to break systematic bias")

        # Run self-consistency evaluation
        start_time = datetime.now()
        results = await ensemble.evaluate_batch(
            events,
            num_paths=args.num_paths,
            temperature_range=temperature_range,
            use_reasoning_variations=use_reasoning_variations,
            aggressive_variations=aggressive_variations,
            override_instructions=override_instructions,
            max_concurrent=args.max_concurrent,
        )
        end_time = datetime.now()

        duration = (end_time - start_time).total_seconds()
        total_api_calls = len(events) * args.num_paths

        print_self_consistency_summary(results)
        print(f" Total time: {duration:.2f} seconds")
        print(f" Average time per event: {duration / len(events):.2f} seconds")
        print(f" Total API calls: {total_api_calls} ({args.num_paths} per event)")
        print(f" API calls per second: {total_api_calls / duration:.1f}")

        save_self_consistency_results(results, args.output, args.use_folders)

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
