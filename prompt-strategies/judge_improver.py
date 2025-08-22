#!/usr/bin/env python3
"""
Judge Prompt Improver - Concise Open-ended Reasoning Enhancement
Improves judges with minimal but effective instructions focused on better thinking.
"""

import json
import asyncio
import os
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import argparse

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from mistralai import Mistral


@dataclass
class JudgeDefinition:
    name: str
    description: str
    model_name: str
    instructions: str
    output_options: List[Dict[str, str]]
    tools: List[str] = None

    def __post_init__(self):
        if self.tools is None:
            self.tools = []


@dataclass
class AnnotatedExample:
    correlation_id: str
    judge_name: str
    answer: str
    analysis: str
    timestamp: str
    error: str
    expected_output: str


@dataclass
class JudgeAnalysis:
    judge_name: str
    total_examples: int
    accuracy: float
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Any]
    error_analysis: Dict[str, Any]
    failure_patterns: List[Dict[str, Any]]
    improvement_suggestions: List[str]
    examples_by_category: Dict[str, List[Dict[str, Any]]]
    timestamp: str


class JudgeImprover:
    """Concise judge enhancement using open-ended reasoning"""

    def __init__(self, api_key: str, base_url: str = None):
        if not api_key:
            raise ValueError("API key required")
        self.client = Mistral(api_key=api_key, server_url=base_url)

    def load_data(
        self,
        annotated_path: str,
        judge_path: str,
        analysis_path: str,
        use_folders: bool = False,
    ):
        """Load all required data files"""

        def resolve_path(file_path: str, folder: str):
            if use_folders:
                path = Path(file_path)
                if len(path.parts) == 1:
                    folder_path = Path(folder) / file_path
                    if folder_path.exists():
                        return str(folder_path)
            return file_path

        # Load annotated examples
        annotated_file = resolve_path(annotated_path, "annotated")
        with open(annotated_file, "r") as f:
            annotated_data = json.load(f)
        annotated_examples = [AnnotatedExample(**item) for item in annotated_data]

        # Load judge definition
        judge_file = resolve_path(judge_path, "judges")
        with open(judge_file, "r") as f:
            judge_data = json.load(f)
        judge_def = JudgeDefinition(**judge_data)

        # Load analysis
        analysis_file = resolve_path(analysis_path, "analyses")
        with open(analysis_file, "r") as f:
            analysis_data = json.load(f)
        analysis = JudgeAnalysis(**analysis_data)

        return annotated_examples, judge_def, analysis

    def get_key_examples(self, examples: List[AnnotatedExample], max_examples: int = 4):
        """Select most informative examples"""
        key_examples = []

        # Get diverse error cases
        errors = [
            ex for ex in examples if ex.answer != ex.expected_output and not ex.error
        ][:2]
        # Get good examples
        correct = [
            ex for ex in examples if ex.answer == ex.expected_output and not ex.error
        ][:2]

        for ex in errors + correct:
            key_examples.append({
                "predicted": ex.answer,
                "expected": ex.expected_output,
                "reasoning": ex.analysis[:200] + "..."
                if len(ex.analysis) > 200
                else ex.analysis,
            })

        return key_examples[:max_examples]

    async def enhance_judge(
        self,
        judge_def: JudgeDefinition,
        analysis: JudgeAnalysis,
        key_examples: List[Dict[str, Any]],
    ) -> JudgeDefinition:
        """Generate enhanced judge with open-ended reasoning"""

        system_prompt = """You enhance AI judges by improving their reasoning ability. Focus on:

1. Clear thinking frameworks that facilitate good judgment
2. Concise but effective guidance 
3. Better output option descriptions for clarity
4. Flexibility to handle edge cases thoughtfully

Keep instructions minimal but powerful. Help the judge think better, not follow more rules."""

        user_prompt = f"""
CURRENT JUDGE:
{json.dumps(asdict(judge_def), indent=2)}

PERFORMANCE: {analysis.accuracy:.1%} accuracy on {analysis.total_examples} examples

KEY PATTERNS FROM EXAMPLES:
{json.dumps(key_examples, indent=2)}

MAIN ISSUES:
{chr(10).join(f"- {p['error_type']}: {p['frequency']} cases" for p in analysis.failure_patterns[:3])}

SUGGESTIONS:
{chr(10).join(f"- {s}" for s in analysis.improvement_suggestions)}

Enhance this judge to think more clearly and consistently. Focus on:
- What should it consider when making judgments?
- How can it better handle nuanced cases?
- What thinking process would improve accuracy?

Keep the same output options but you may improve descriptions if requires.

Return JSON:
{{
    "instructions": "enhanced instructions - concise but powerful",
    "output_options": [enhanced options if requires changes],
    "rationale": "brief explanation of changes"
}}
"""

        try:
            response = await self.client.chat.complete_async(
                model="mistral-small-latest",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )

            result = json.loads(response.choices[0].message.content)

            enhanced = JudgeDefinition(
                name=f"{judge_def.name} (Enhanced)",
                description=f"{judge_def.description}\n\nEnhanced for better reasoning and clarity.",
                model_name=judge_def.model_name,
                instructions=result["instructions"],
                output_options=result["output_options"],
                tools=judge_def.tools,
            )

            return enhanced, result.get("rationale", "")

        except Exception as e:
            print(f"Warning: Enhancement failed, using fallback: {e}")
            return self._fallback_enhancement(judge_def), "Fallback enhancement applied"

    def _fallback_enhancement(self, judge_def: JudgeDefinition) -> JudgeDefinition:
        """Simple fallback enhancement"""
        enhanced_instructions = f"""{judge_def.instructions}

Think step by step:
1. What is the core requirement being evaluated?
2. How well does the response meet this requirement?
3. Are there important contextual factors to consider?

Consider both what was achieved and what was intended."""

        return JudgeDefinition(
            name=f"{judge_def.name} (Enhanced)",
            description=f"{judge_def.description} Enhanced for clearer reasoning.",
            model_name=judge_def.model_name,
            instructions=enhanced_instructions,
            output_options=judge_def.output_options,
            tools=judge_def.tools,
        )

    def save_enhanced_judge(
        self,
        enhanced_judge: JudgeDefinition,
        original_path: str,
        use_folders: bool = False,
    ):
        """Save enhanced judge"""
        if use_folders:
            judges_dir = Path("judges")
            judges_dir.mkdir(exist_ok=True)
            filename = f"{Path(original_path).stem}_enhanced.json"
            output_path = judges_dir / filename
        else:
            output_path = (
                Path(original_path).parent / f"{Path(original_path).stem}_enhanced.json"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(asdict(enhanced_judge), f, indent=2)

        return str(output_path)

    def print_summary(
        self,
        original: JudgeDefinition,
        enhanced: JudgeDefinition,
        analysis: JudgeAnalysis,
        rationale: str,
    ):
        """Print concise summary"""
        print(f"\n{'=' * 50}")
        print(f"JUDGE ENHANCEMENT")
        print(f"{'=' * 50}")
        print(f"Judge: {original.name}")
        print(f"Accuracy: {analysis.accuracy:.1%} ({analysis.total_examples} examples)")
        print(
            f"Instructions: {len(original.instructions.split())} → {len(enhanced.instructions.split())} words"
        )
        print(f"Categories: {len(enhanced.output_options)}")
        print(f"\nEnhancement: {rationale}")
        print(f"{'=' * 50}")


async def main():
    parser = argparse.ArgumentParser(
        description="Concise judge enhancement with open-ended reasoning"
    )
    parser.add_argument("--annotated", required=True, help="Annotated examples file")
    parser.add_argument("--judge", required=True, help="Judge definition file")
    parser.add_argument("--analysis", required=True, help="Analysis results file")
    parser.add_argument(
        "--use-folders", action="store_true", help="Use folder structure"
    )
    parser.add_argument("--api-key", help="Mistral API key")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    try:
        api_key = args.api_key or os.getenv("MISTRAL_API_KEY")
        if not api_key:
            print("[X] API key required (--api-key or MISTRAL_API_KEY)")
            return 1

        improver = JudgeImprover(api_key)

        if args.verbose:
            print("[...] Loading data...")

        annotated_examples, judge_def, analysis = improver.load_data(
            args.annotated, args.judge, args.analysis, args.use_folders
        )

        print(f" Enhancing: {judge_def.name}")
        print(f" Current: {analysis.accuracy:.1%} accuracy")

        # Get key examples
        key_examples = improver.get_key_examples(annotated_examples)

        # Enhance judge
        enhanced_judge, rationale = await improver.enhance_judge(
            judge_def, analysis, key_examples
        )

        # Save result
        output_path = improver.save_enhanced_judge(
            enhanced_judge, args.judge, args.use_folders
        )

        # Summary
        improver.print_summary(judge_def, enhanced_judge, analysis, rationale)

        print(f"[OK] Enhanced judge saved: {output_path}")
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
