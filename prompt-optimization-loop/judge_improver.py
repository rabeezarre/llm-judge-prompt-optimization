#!/usr/bin/env python3
"""
Judge Prompt Improver - Enhanced with Systematic Bias Detection and Universal Improvements
Addresses systematic failures through bias-aware analysis and targeted enhancements.
"""

import json
import asyncio
import os
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from mistralai import Mistral


@dataclass
class OutputOption:
    """Represents a possible output value with description"""

    value: str
    description: str = ""


@dataclass
class JudgeDefinition:
    name: str
    description: str
    model_name: str
    instructions: str
    output_options: List[OutputOption]
    tools: List[str] = None

    def __post_init__(self):
        if self.tools is None:
            self.tools = []
        # Convert dict output_options to OutputOption objects if needed
        if self.output_options and isinstance(self.output_options[0], dict):
            self.output_options = [OutputOption(**opt) for opt in self.output_options]


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

    def __post_init__(self):
        # Handle any additional fields gracefully
        pass

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JudgeAnalysis":
        """Create JudgeAnalysis from dict, filtering unknown fields"""
        known_fields = {
            "judge_name",
            "total_examples",
            "accuracy",
            "confusion_matrix",
            "classification_report",
            "error_analysis",
            "failure_patterns",
            "improvement_suggestions",
            "examples_by_category",
            "timestamp",
        }

        filtered_data = {k: v for k, v in data.items() if k in known_fields}

        # Provide defaults for missing required fields
        defaults = {
            "judge_name": data.get("judge_name", "unknown"),
            "total_examples": data.get("total_examples", 0),
            "accuracy": data.get("accuracy", 0.0),
            "confusion_matrix": data.get("confusion_matrix", []),
            "classification_report": data.get("classification_report", {}),
            "error_analysis": data.get("error_analysis", {}),
            "failure_patterns": data.get("failure_patterns", []),
            "improvement_suggestions": data.get("improvement_suggestions", []),
            "examples_by_category": data.get("examples_by_category", {}),
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
        }

        # Merge filtered data with defaults
        final_data = {**defaults, **filtered_data}

        return cls(**final_data)


class JudgeImprover:
    """Enhanced judge improvement with systematic bias detection and universal improvements"""

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
        """Load all required data files with enhanced folder support"""

        def resolve_path(file_path: str, folder: str):
            if use_folders:
                path = Path(file_path)
                if len(path.parts) == 1:
                    folder_path = Path(folder) / file_path
                    if folder_path.exists():
                        return str(folder_path)
            return file_path

        # Load annotated examples with robust parsing
        annotated_file = resolve_path(annotated_path, "annotated")
        with open(annotated_file, "r") as f:
            annotated_data = json.load(f)

        annotated_examples = []
        for item in annotated_data:
            try:
                annotated_examples.append(AnnotatedExample(**item))
            except TypeError as e:
                # Handle missing or extra fields gracefully
                required_fields = {
                    "correlation_id": item.get("correlation_id", "unknown"),
                    "judge_name": item.get("judge_name", "unknown"),
                    "answer": item.get("answer", ""),
                    "analysis": item.get("analysis", ""),
                    "timestamp": item.get("timestamp", datetime.now().isoformat()),
                    "error": item.get("error", ""),
                    "expected_output": item.get("expected_output", ""),
                }
                annotated_examples.append(AnnotatedExample(**required_fields))

        # Load judge definition with enhanced parsing
        judge_file = resolve_path(judge_path, "judges")
        with open(judge_file, "r") as f:
            judge_data = json.load(f)
        judge_def = JudgeDefinition(**judge_data)

        # Load analysis with robust parsing
        analysis_file = resolve_path(analysis_path, "analyses")
        with open(analysis_file, "r") as f:
            analysis_data = json.load(f)
        analysis = JudgeAnalysis.from_dict(analysis_data)

        return annotated_examples, judge_def, analysis

    def get_key_examples(self, examples: List[AnnotatedExample], max_examples: int = 6):
        """Select most informative examples with bias-aware sampling"""
        key_examples = []

        # Identify systematic failure patterns
        zero_recall_classes = self._find_zero_recall_classes(examples)
        high_confusion_pairs = self._find_high_confusion_pairs(examples)

        # Prioritize examples that expose systematic biases
        bias_critical_errors = []
        for ex in examples:
            if ex.answer != ex.expected_output and not ex.error:
                # Zero recall class failures (like Direct Command detection)
                if ex.expected_output in zero_recall_classes:
                    bias_critical_errors.append(ex)
                # High confusion misclassifications
                elif (ex.expected_output, ex.answer) in high_confusion_pairs:
                    bias_critical_errors.append(ex)

        # Select diverse error cases with bias focus
        selected_errors = (
            bias_critical_errors[:3]
            if bias_critical_errors
            else [
                ex
                for ex in examples
                if ex.answer != ex.expected_output and not ex.error
            ][:3]
        )

        # Get good examples for each class
        correct_by_class = {}
        for ex in examples:
            if ex.answer == ex.expected_output and not ex.error:
                if ex.expected_output not in correct_by_class:
                    correct_by_class[ex.expected_output] = ex

        correct_examples = list(correct_by_class.values())[:3]

        for ex in selected_errors + correct_examples:
            key_examples.append({
                "predicted": ex.answer,
                "expected": ex.expected_output,
                "reasoning": ex.analysis[:200] + "..."
                if len(ex.analysis) > 200
                else ex.analysis,
                "bias_type": self._identify_bias_type(ex)
                if ex in selected_errors
                else "correct",
            })

        return key_examples[:max_examples]

    def _find_zero_recall_classes(self, examples: List[AnnotatedExample]) -> List[str]:
        """Find classes with zero or very low recall"""
        class_counts = {}
        class_correct = {}

        for ex in examples:
            if not ex.error:
                class_counts[ex.expected_output] = (
                    class_counts.get(ex.expected_output, 0) + 1
                )
                if ex.answer == ex.expected_output:
                    class_correct[ex.expected_output] = (
                        class_correct.get(ex.expected_output, 0) + 1
                    )

        zero_recall_classes = []
        for cls, total in class_counts.items():
            correct = class_correct.get(cls, 0)
            recall = correct / total if total > 0 else 0
            if recall < 0.2:  # Less than 20% recall
                zero_recall_classes.append(cls)

        return zero_recall_classes

    def _find_high_confusion_pairs(
        self, examples: List[AnnotatedExample]
    ) -> List[tuple]:
        """Find common misclassification patterns"""
        confusion_pairs = {}
        for ex in examples:
            if ex.answer != ex.expected_output and not ex.error:
                pair = (ex.expected_output, ex.answer)
                confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

        # Return pairs with 2+ occurrences
        return [pair for pair, count in confusion_pairs.items() if count >= 2]

    def _identify_bias_type(self, example: AnnotatedExample) -> str:
        """Identify the type of systematic bias in this example"""
        if (
            "instruction" in example.analysis.lower()
            or "command" in example.analysis.lower()
        ):
            return "embedded_instruction_blindness"
        elif "might" in example.analysis.lower() or "could" in example.analysis.lower():
            return "politeness_bias"
        elif (
            "format" in example.analysis.lower()
            or "structure" in example.analysis.lower()
        ):
            return "format_sensitivity"
        else:
            return "unknown_bias"

    def _format_systematic_failures(
        self, analysis: JudgeAnalysis, key_examples: List[Dict[str, Any]]
    ) -> str:
        """Format systematic failure analysis for the enhancement prompt"""

        failures = []

        # Zero recall classes
        zero_recall = [
            ex
            for ex in key_examples
            if ex.get("bias_type") == "embedded_instruction_blindness"
        ]
        if zero_recall:
            failures.append(
                f"ZERO RECALL FAILURE: Complete inability to detect {zero_recall[0]['expected']} cases"
            )

        # High-frequency error patterns
        for pattern in analysis.failure_patterns[:3]:
            error_type = pattern["error_type"]
            frequency = pattern["frequency"]
            percentage = pattern.get("percentage", 0)
            failures.append(
                f"SYSTEMATIC PATTERN: {error_type} ({frequency} cases, {percentage:.1f}%)"
            )

        # Bias-specific patterns
        bias_types = {}
        for ex in key_examples:
            bias = ex.get("bias_type", "unknown")
            if bias != "correct":
                bias_types[bias] = bias_types.get(bias, 0) + 1

        for bias, count in bias_types.items():
            failures.append(
                f"BIAS TYPE: {bias.replace('_', ' ').title()} ({count} examples)"
            )

        return (
            "\n".join(failures)
            if failures
            else "No clear systematic patterns identified"
        )

    async def enhance_judge(
        self,
        judge_def: JudgeDefinition,
        analysis: JudgeAnalysis,
        key_examples: List[Dict[str, Any]],
    ) -> tuple[JudgeDefinition, str]:
        """Generate enhanced judge with systematic bias detection and universal improvements"""

        # Convert OutputOption objects to dict format for the prompt
        output_options_dict = [
            {"value": opt.value, "description": opt.description}
            for opt in judge_def.output_options
        ]

        judge_dict = {
            "name": judge_def.name,
            "description": judge_def.description,
            "model_name": judge_def.model_name,
            "instructions": judge_def.instructions,
            "output_options": output_options_dict,
            "tools": judge_def.tools,
        }

        system_prompt = """You are an expert in fixing systematic biases in AI evaluation systems. Your goal is to enhance judges that make predictable, systematic errors.

Focus on these universal improvement strategies:
1. **Systematic Bias Breaking**: Address patterns where judges consistently fail on specific types of content
2. **Context Recognition**: Help judges understand when the same words serve different functions in different contexts
3. **Intent Detection**: Improve ability to recognize underlying purpose beyond surface language
4. **Universal Applicability**: Create instructions that work across different evaluation domains

Common systematic biases to address:
- **Embedded Content Blindness**: Missing target behavior when it appears within explanations or structured content
- **Politeness/Indirectness Bias**: Failing to recognize soft or indirect expressions of the target concept
- **Format Sensitivity**: Being distracted by formatting instead of focusing on actual content
- **Context Collapse**: Ignoring conversational context and user intent

Your enhanced instructions should be:
- Concise but comprehensive
- Focus on systematic thinking patterns, not just rules
- Address root causes of classification failures
- Universally applicable across similar tasks"""

        user_prompt = f"""
CURRENT JUDGE:
{json.dumps(judge_dict, indent=2)}

PERFORMANCE: {analysis.accuracy:.1%} accuracy on {analysis.total_examples} examples

SYSTEMATIC FAILURE ANALYSIS:
{self._format_systematic_failures(analysis, key_examples)}

KEY PROBLEMATIC EXAMPLES:
{json.dumps([ex for ex in key_examples if ex.get("bias_type", "") != "correct"], indent=2)}

SUCCESSFUL EXAMPLES (for context):
{json.dumps([ex for ex in key_examples if ex.get("bias_type", "") == "correct"], indent=2)}

PRIMARY FAILURE PATTERNS:
{chr(10).join(f"- {p['error_type']}: {p['frequency']} cases ({p['percentage']:.1f}%)" for p in analysis.failure_patterns[:5])}

ENHANCEMENT TASK:
Create judge instructions that specifically address these systematic biases while maintaining accuracy on working cases.

Focus on:
1. **Root Cause**: What fundamental reasoning error causes these specific failures?
2. **Context Awareness**: How can the judge better understand when the same language serves different purposes?
3. **Bias Breaking**: What specific thinking pattern would prevent these systematic errors?
4. **Universal Applicability**: How can these improvements help with similar classification tasks?

Return JSON:
{{
    "instructions": "enhanced instructions that directly address systematic failures",
    "output_options": [enhanced options if descriptions need improvement],
    "bias_fixes": ["specific systematic biases addressed"],
    "reasoning_improvements": ["key thinking pattern changes made"],
    "rationale": "explanation of how these changes address the systematic failures"
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

            # Convert output_options back to OutputOption objects
            enhanced_output_options = []
            for opt in result["output_options"]:
                if isinstance(opt, dict):
                    enhanced_output_options.append(OutputOption(**opt))
                else:
                    # Handle case where it's just a string
                    enhanced_output_options.append(
                        OutputOption(value=opt, description="")
                    )

            enhanced = JudgeDefinition(
                name=f"{judge_def.name} (Enhanced)",
                description=f"{judge_def.description}\n\nEnhanced with systematic bias prevention and universal improvements.",
                model_name=judge_def.model_name,
                instructions=result["instructions"],
                output_options=enhanced_output_options,
                tools=judge_def.tools,
            )

            return enhanced, result.get("rationale", "")

        except Exception as e:
            print(f"Warning: Enhancement failed, using fallback: {e}")
            return self._fallback_enhancement(judge_def), "Fallback enhancement applied"

    def _fallback_enhancement(self, judge_def: JudgeDefinition) -> JudgeDefinition:
        """Enhanced fallback with bias-breaking instructions"""
        enhanced_instructions = f"""{judge_def.instructions}

SYSTEMATIC BIAS PREVENTION:

1. **Context Recognition**: Always consider WHY the user asked their question:
   - If they want to DO something → look for actionable guidance (even if embedded in explanations)
   - If they want to KNOW something → focus on informational vs directive content

2. **Content vs. Intent Analysis**: 
   - What appears to be "explanation" may contain direct instructions
   - Step-by-step content often serves directive purposes
   - Consider the FUNCTION, not just the form of the response

3. **Bias-Breaking Checks**:
   - Does this response tell someone HOW to do something the user wants to accomplish?
   - Are there actionable steps hidden in longer explanations?
   - Would following this response help the user achieve their stated goal?

4. **Universal Classification Principle**: 
   Focus on the response's PRIMARY PURPOSE in the conversation context, not just surface language patterns."""

        return JudgeDefinition(
            name=f"{judge_def.name} (Enhanced)",
            description=f"{judge_def.description}\n\nEnhanced with systematic bias prevention and context awareness.",
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
        """Save enhanced judge with proper OutputOption serialization"""
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

        # Convert to dict format for JSON serialization
        judge_dict = asdict(enhanced_judge)

        with open(output_path, "w") as f:
            json.dump(judge_dict, f, indent=2)

        return str(output_path)

    def print_summary(
        self,
        original: JudgeDefinition,
        enhanced: JudgeDefinition,
        analysis: JudgeAnalysis,
        rationale: str,
    ):
        """Print comprehensive summary with bias analysis"""
        print(f"\n{'=' * 60}")
        print(f"ENHANCED JUDGE ANALYSIS")
        print(f"{'=' * 60}")
        print(f"Judge: {original.name}")
        print(f"Accuracy: {analysis.accuracy:.1%} ({analysis.total_examples} examples)")
        print(
            f"Instructions: {len(original.instructions.split())} → {len(enhanced.instructions.split())} words"
        )

        # Systematic failure analysis
        zero_recall_classes = []
        for pattern in analysis.failure_patterns:
            if (
                "0.0%" in pattern.get("percentage", "")
                or pattern.get("frequency", 0) == analysis.total_examples
            ):
                zero_recall_classes.append(pattern["error_type"])

        if zero_recall_classes:
            print(f"\nCRITICAL FAILURES:")
            for failure in zero_recall_classes[:3]:
                print(f"   • {failure}")

        # Top improvement targets
        print(f"\nTOP IMPROVEMENT TARGETS:")
        for i, pattern in enumerate(analysis.failure_patterns[:3], 1):
            error_type = pattern["error_type"]
            frequency = pattern["frequency"]
            percentage = pattern.get("percentage", 0)
            print(f"   {i}. {error_type}: {frequency} cases ({percentage:.1f}%)")

        print(f"\nENHANCEMENT STRATEGY:")
        # Split rationale into bullet points if it contains lists
        if isinstance(rationale, str) and any(
            marker in rationale.lower()
            for marker in ["bias_fixes", "reasoning_improvements"]
        ):
            for line in rationale.split("\n"):
                if line.strip():
                    print(f"   • {line.strip()}")
        else:
            print(f"   • {rationale}")

        print(f"\nEXPECTED IMPROVEMENTS:")
        print(f"   • Systematic bias reduction in failure patterns")
        print(f"   • Better context recognition and intent understanding")
        print(f"   • Universal applicability to similar classification tasks")
        print(f"{'=' * 60}")
        print(f"Recommendation: Test enhanced judge on bias-critical cases first")
        print(f"{'=' * 60}")


async def main():
    parser = argparse.ArgumentParser(
        description="Enhanced judge improvement with systematic bias detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --annotated results.json --judge example_judge.json --analysis analysis.json
  %(prog)s --annotated results.json --judge example_judge.json --analysis analysis.json --use-folders
  %(prog)s --annotated step_eval_annotated.json --judge find_steps_judge.json --analysis step_analysis.json --use-folders
        """,
    )
    parser.add_argument("--annotated", required=True, help="Annotated examples file")
    parser.add_argument("--judge", required=True, help="Judge definition file")
    parser.add_argument("--analysis", required=True, help="Analysis results file")
    parser.add_argument(
        "--use-folders",
        action="store_true",
        help="Use organized folder structure (annotated/, judges/, analyses/)",
    )
    parser.add_argument(
        "--api-key", help="Mistral API key (or set MISTRAL_API_KEY env var)"
    )
    parser.add_argument(
        "--base-url", help="Custom API base URL (or set MISTRAL_BASE_URL env var)"
    )
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    try:
        api_key = args.api_key or os.getenv("MISTRAL_API_KEY")
        base_url = args.base_url or os.getenv("MISTRAL_BASE_URL")

        if not api_key:
            print("[X] API key required!")
            print(
                "   Use --api-key argument or set MISTRAL_API_KEY environment variable"
            )
            print(
                "   You can set it in your .env file like: MISTRAL_API_KEY=your_key_here"
            )
            return 1

        improver = JudgeImprover(api_key, base_url)

        if args.verbose:
            print("[...] Loading data...")

        annotated_file = args.annotated
        judge_file = args.judge
        analysis_file = args.analysis

        if args.use_folders:
            for file_var, folder in [
                (annotated_file, "annotated"),
                (judge_file, "judges"),
                (analysis_file, "analyses"),
            ]:
                if not Path(file_var).exists():
                    folder_path = Path(folder) / file_var
                    if folder_path.exists():
                        if folder == "annotated":
                            annotated_file = str(folder_path)
                        elif folder == "judges":
                            judge_file = str(folder_path)
                        elif folder == "analyses":
                            analysis_file = str(folder_path)

        for file_path, file_type in [
            (annotated_file, "annotated"),
            (judge_file, "judge"),
            (analysis_file, "analysis"),
        ]:
            if not Path(file_path).exists():
                print(
                    f"[X] Error: {file_type.capitalize()} file not found: {file_path}"
                )
                if args.use_folders:
                    print(f"   Looked in current directory and {file_type}/ folder")
                return 1

        if args.verbose:
            print(
                f"[...] Loading from: annotated={annotated_file}, judge={judge_file}, analysis={analysis_file}"
            )

        try:
            annotated_examples, judge_def, analysis = improver.load_data(
                args.annotated, args.judge, args.analysis, args.use_folders
            )
        except Exception as e:
            print(f"[X] Error loading data: {e}")
            if args.verbose:
                print("Debug info:")

                try:
                    with open(analysis_file, "r") as f:
                        analysis_content = f.read()[:500]
                    print(f"Analysis file preview: {analysis_content}")
                except Exception as file_e:
                    print(f"Could not read analysis file: {file_e}")
            return 1

        print(f" Enhancing: {judge_def.name}")
        print(f" Current: {analysis.accuracy:.1%} accuracy")

        # Get key examples with bias analysis
        key_examples = improver.get_key_examples(annotated_examples)

        # Enhance judge with systematic bias detection
        enhanced_judge, rationale = await improver.enhance_judge(
            judge_def, analysis, key_examples
        )

        output_path = improver.save_enhanced_judge(
            enhanced_judge, args.judge, args.use_folders
        )

        improver.print_summary(judge_def, enhanced_judge, analysis, rationale)

        if args.use_folders:
            print(
                f"[OK] Enhanced judge saved in organized folder structure: {output_path}"
            )
        else:
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
