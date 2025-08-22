#!/usr/bin/env python3
"""
Judge Analyzer with Confusion Matrix
Analyzes judge performance against expected outputs and provides improvement suggestions.
"""

import json
import asyncio
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict
import argparse

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Using environment variables directly.")

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import (
        confusion_matrix,
        classification_report,
        accuracy_score,
        precision_recall_fscore_support,
    )

    HAS_ML_LIBS = True
except ImportError:
    print(
        "Warning: ML libraries not installed. Install with: pip install numpy matplotlib seaborn scikit-learn"
    )
    HAS_ML_LIBS = False

from mistralai import Mistral


@dataclass
class JudgeDefinition:
    """Judge configuration for evaluating events"""

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
    """Represents an annotated example with expected output"""

    correlation_id: str
    judge_name: str
    answer: str
    analysis: str
    timestamp: str
    error: Optional[str]
    expected_output: str


@dataclass
class JudgeAnalysis:
    """Complete analysis of judge performance"""

    judge_name: str
    total_examples: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Any]
    error_analysis: Dict[str, Any]
    failure_patterns: List[Dict[str, Any]]
    improvement_suggestions: List[str]
    examples_by_category: Dict[str, List[Dict[str, Any]]]
    timestamp: str


class JudgeAnalyzer:
    """Analyzes judge performance and suggests improvements"""

    def __init__(self, api_key: str = None, base_url: str = None):
        """Initialize with optional Mistral client for AI analysis"""
        self.client = None
        if api_key:
            self.client = Mistral(api_key=api_key, server_url=base_url)

    def load_annotated_data(
        self, file_path: str, use_folders: bool = False
    ) -> List[AnnotatedExample]:
        """Load annotated examples from JSON file"""
        if use_folders:
            annotated_path = Path(file_path)
            if len(annotated_path.parts) == 1:  # Just a filename
                annotated_dir = Path("annotated")
                if (annotated_dir / file_path).exists():
                    file_path = annotated_dir / file_path

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        examples = []

        # Handle both formats - direct AnnotatedExample and evaluation results with expected_output
        for item in data:
            if "judge_name" in item and "expected_output" in item and "answer" in item:
                # Direct AnnotatedExample format
                examples.append(AnnotatedExample(**item))
            else:
                # Evaluation result format with expected_output field
                # Extract fields from evaluation result
                correlation_id = (
                    item.get("correlation_id")
                    or item.get("correlationId")
                    or item.get("corelation_id", "unknown")
                )

                # Create AnnotatedExample from evaluation result
                annotated_example = AnnotatedExample(
                    correlation_id=correlation_id,
                    judge_name=item.get("judge_name", "Unknown Judge"),
                    answer=item.get("answer", ""),  # Judge's prediction
                    analysis=item.get("analysis", ""),
                    timestamp=item.get("timestamp", datetime.now().isoformat()),
                    error=item.get("error"),
                    expected_output=item.get(
                        "expected_output", ""
                    ),  # Ground truth from human annotation
                )
                examples.append(annotated_example)

        return examples

    def load_judge_definition(
        self, file_path: str, use_folders: bool = False
    ) -> JudgeDefinition:
        """Load judge definition from JSON file"""
        if use_folders:
            judge_path = Path(file_path)
            if len(judge_path.parts) == 1:  # Just a filename
                judges_dir = Path("judges")
                if (judges_dir / file_path).exists():
                    file_path = judges_dir / file_path

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JudgeDefinition(**data)

    def create_confusion_matrix(
        self, y_true: List[str], y_pred: List[str], labels: List[str]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Create confusion matrix and compute metrics"""
        if not HAS_ML_LIBS:
            # Simple fallback implementation
            matrix = {}
            for true_label in labels:
                matrix[true_label] = {}
                for pred_label in labels:
                    matrix[true_label][pred_label] = 0

            for true, pred in zip(y_true, y_pred):
                if true in matrix and pred in matrix[true]:
                    matrix[true][pred] += 1

            # Convert to list format
            matrix_list = []
            for true_label in labels:
                row = []
                for pred_label in labels:
                    row.append(matrix[true_label].get(pred_label, 0))
                matrix_list.append(row)

            accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

            metrics = {
                "accuracy": accuracy,
                "macro_precision": 0,
                "macro_recall": 0,
                "macro_f1": 0,
                "weighted_precision": 0,
                "weighted_recall": 0,
                "weighted_f1": 0,
                "matrix": matrix_list,
                "labels": labels,
            }
            return np.array(matrix_list) if HAS_ML_LIBS else matrix_list, metrics

        # Full implementation with sklearn
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        accuracy = accuracy_score(y_true, y_pred)

        # Calculate precision, recall, F1 scores
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )

        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        )

        weighted_precision, weighted_recall, weighted_f1, _ = (
            precision_recall_fscore_support(
                y_true, y_pred, labels=labels, average="weighted", zero_division=0
            )
        )

        # Classification report
        report = classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        )

        metrics = {
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1,
            "classification_report": report,
            "matrix": cm.tolist(),
            "labels": labels,
        }

        return cm, metrics

    def plot_confusion_matrix(
        self,
        cm,
        labels: List[str],
        judge_name: str,
        output_dir: Path,
        metrics: Dict[str, Any],
    ):
        """Plot and save confusion matrix visualization with enhanced metrics"""
        if not HAS_ML_LIBS:
            print(
                "Warning: Cannot create confusion matrix plot without matplotlib/seaborn"
            )
            return None

        plt.figure(figsize=(12, 10))

        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"label": "Count"},
        )

        plt.title(f"Confusion Matrix: {judge_name}", fontsize=16, fontweight="bold")
        plt.xlabel("Judge Predictions", fontsize=12)  # clarify what's being predicted
        plt.ylabel("Human Ground Truth", fontsize=12)  # clarify what's true
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        # Add metrics text box with enhanced info
        metrics_text = f"""Accuracy: {metrics["accuracy"]:.3f}
Macro F1: {metrics["macro_f1"]:.3f}
Macro Precision: {metrics["macro_precision"]:.3f}
Macro Recall: {metrics["macro_recall"]:.3f}
Weighted F1: {metrics["weighted_f1"]:.3f}"""

        plt.text(
            0.02,
            0.98,
            metrics_text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        plt.tight_layout()

        # Save plot
        plot_path = (
            output_dir / f"{judge_name.lower().replace(' ', '_')}_confusion_matrix.png"
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Saved confusion matrix PNG to: {plot_path.resolve()}")

        plt.close()

        return str(plot_path)

    def analyze_errors(self, examples: List[AnnotatedExample]) -> Dict[str, Any]:
        """Analyze error patterns and types"""
        errors = []
        correct_predictions = []

        for example in examples:
            if example.error:
                errors.append({
                    "correlation_id": example.correlation_id,
                    "error": example.error,
                    "expected": example.expected_output,
                })
            elif example.answer != example.expected_output:
                errors.append({
                    "correlation_id": example.correlation_id,
                    "predicted": example.answer,
                    "expected": example.expected_output,
                    "analysis": example.analysis,
                })
            else:
                correct_predictions.append(example)

        # Group errors by type
        error_types = defaultdict(list)
        prediction_errors = defaultdict(list)

        for error in errors:
            if "error" in error:
                # System/technical errors
                error_type = (
                    error["error"].split(":")[0]
                    if ":" in error["error"]
                    else error["error"]
                )
                error_types[error_type].append(error)
            else:
                # Prediction errors
                key = f"{error['expected']} -> {error['predicted']}"
                prediction_errors[key].append(error)

        return {
            "total_errors": len(errors),
            "system_errors": dict(error_types),
            "prediction_errors": dict(prediction_errors),
            "correct_predictions": len(correct_predictions),
        }

    def identify_failure_patterns(
        self, examples: List[AnnotatedExample]
    ) -> List[Dict[str, Any]]:
        """Identify common failure patterns"""
        patterns = []

        # Group by prediction error types
        error_groups = defaultdict(list)
        for example in examples:
            if not example.error and example.answer != example.expected_output:
                key = f"{example.expected_output} -> {example.answer}"
                error_groups[key].append(example)

        # Analyze each error type
        for error_type, error_examples in error_groups.items():
            if len(error_examples) >= 2:  # Only patterns with multiple occurrences
                analyses = [ex.analysis for ex in error_examples if ex.analysis]

                pattern = {
                    "error_type": error_type,
                    "frequency": len(error_examples),
                    "percentage": len(error_examples) / len(examples) * 100,
                    "example_ids": [
                        ex.correlation_id for ex in error_examples[:3]
                    ],  # First 3 examples
                    "common_themes": self._extract_common_themes(analyses),
                }
                patterns.append(pattern)

        # Sort by frequency
        patterns.sort(key=lambda x: x["frequency"], reverse=True)
        return patterns

    def _extract_common_themes(self, analyses: List[str]) -> List[str]:
        """Extract common themes from analysis texts"""
        if not analyses:
            return []

        # Simple keyword extraction
        common_words = []
        word_counts = defaultdict(int)

        for analysis in analyses:
            words = analysis.lower().split()
            for word in words:
                if (
                    len(word) > 4 and word.isalpha()
                ):  # Filter short and non-alphabetic words
                    word_counts[word] += 1

        # Get most common meaningful words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        common_words = [word for word, count in sorted_words[:10] if count > 1]

        return common_words

    async def generate_ai_improvements(
        self, analysis_data: Dict[str, Any], judge_def: JudgeDefinition
    ) -> List[str]:
        """Use AI to generate improvement suggestions"""
        if not self.client:
            return self._generate_rule_based_improvements(analysis_data, judge_def)

        system_prompt = """You are an expert in AI judge evaluation systems. Analyze why the judge's predictions don't match the human ground truth annotations.

CRITICAL: The "answer" field is the judge's PREDICTION, and "expected_output" is the human GROUND TRUTH. Focus on why the judge is making incorrect predictions compared to human judgment.

Focus on:
1. Why the judge misclassifies cases that humans label differently
2. What patterns in judge reasoning lead to wrong predictions
3. How to improve judge instructions to match human judgment better
4. Specific failure modes where judge disagrees with human annotators

Provide 5-7 concrete suggestions for why predictions fail and how to fix them."""

        context = f"""
Judge Definition:
Name: {judge_def.name}
Instructions: {judge_def.instructions}
Output Options: {[opt["value"] + ": " + opt.get("description", "") for opt in judge_def.output_options]}

Performance Against Human Ground Truth:
- Accuracy: {analysis_data.get("accuracy", 0):.3f} (how often judge predictions match human annotations)
- Macro F1: {analysis_data.get("macro_f1", 0):.3f}
- Macro Precision: {analysis_data.get("macro_precision", 0):.3f}
- Macro Recall: {analysis_data.get("macro_recall", 0):.3f}
- Total Examples: {analysis_data.get("total_examples", 0)}

Common Prediction Failures (Human Ground Truth -> Judge Prediction):
{json.dumps(analysis_data.get("failure_patterns", []), indent=2)}

Error Analysis (Judge Predictions vs Expected Human Labels):
{json.dumps(analysis_data.get("error_analysis", {}), indent=2)}

REMEMBER: "answer" = judge prediction, "expected_output" = human ground truth
Analyze why the judge disagrees with human judgment and how to align them better.
"""

        try:
            response = await self.client.chat.complete_async(
                model="mistral-small-latest",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context},
                ],
                temperature=0.1,
            )

            content = response.choices[0].message.content

            # Extract suggestions (simple parsing)
            suggestions = []
            lines = content.split("\n")
            for line in lines:
                line = line.strip()
                if line and (
                    line.startswith("-") or line.startswith("•") or line[0].isdigit()
                ):
                    suggestions.append(line.lstrip("-•0123456789. "))

            return suggestions if suggestions else [content]

        except Exception as e:
            print(f"Warning: AI improvement generation failed: {e}")
            return self._generate_rule_based_improvements(analysis_data, judge_def)

    def _generate_rule_based_improvements(
        self, analysis_data: Dict[str, Any], judge_def: JudgeDefinition
    ) -> List[str]:
        """Generate rule-based improvement suggestions"""
        suggestions = []

        accuracy = analysis_data.get("accuracy", 0)
        macro_f1 = analysis_data.get("macro_f1", 0)
        failure_patterns = analysis_data.get("failure_patterns", [])

        # Low accuracy suggestions
        if accuracy < 0.7:
            suggestions.append(
                "Judge predictions often disagree with human judgment - revise instructions for better alignment"
            )
            suggestions.append(
                "Add examples of human-labeled cases to guide judge toward human-like evaluation"
            )

        # Low F1 score suggestions
        if macro_f1 < 0.6:
            suggestions.append(
                f"Low F1 score ({macro_f1:.3f}) indicates poor balance of precision and recall - review decision boundaries"
            )

        if accuracy < 0.5:
            suggestions.append(
                "Judge is frequently disagreeing with humans - fundamental rethinking of evaluation criteria needed"
            )

        # Pattern-based suggestions for multilingual/general judges
        for pattern in failure_patterns[:3]:  # Top 3 patterns
            error_type = pattern["error_type"]
            human_label, judge_prediction = error_type.split(" -> ")
            suggestions.append(
                f"Judge often predicts '{judge_prediction}' when humans label '{human_label}' - review criteria for this distinction"
            )

        # Error analysis suggestions
        error_analysis = analysis_data.get("error_analysis", {})
        if error_analysis.get("system_errors"):
            suggestions.append(
                "Address technical issues causing system errors in judge evaluation"
            )

        # Output options suggestions
        if len(judge_def.output_options) == 2:
            suggestions.append(
                "Consider adding intermediate category for cases where judge and humans frequently disagree"
            )

        # Generic improvements if no specific patterns found
        if not suggestions:
            suggestions.extend([
                "Study cases where judge predictions differ from human annotations",
                "Include human-annotated examples in judge instructions",
                "Clarify evaluation criteria based on human judgment patterns",
                "Test judge on human-labeled examples before deployment",
            ])

        return suggestions

    async def analyze_judge(
        self,
        annotated_examples: List[AnnotatedExample],
        judge_def: JudgeDefinition,
        output_dir: Path,
    ) -> JudgeAnalysis:
        """Perform complete judge analysis"""

        # Filter out examples with errors for accuracy calculation
        valid_examples = [ex for ex in annotated_examples if not ex.error]

        if not valid_examples:
            raise ValueError("No valid examples found for analysis")

        # Extract predictions and ground truth
        y_true = [ex.expected_output for ex in valid_examples]  # Human ground truth
        y_pred = [ex.answer for ex in valid_examples]  # Judge predictions

        # Get unique labels
        all_labels = sorted(list(set(y_true + y_pred)))

        # Create confusion matrix
        cm, metrics = self.create_confusion_matrix(y_true, y_pred, all_labels)

        # Plot confusion matrix
        plot_path = None
        if HAS_ML_LIBS:
            plot_path = self.plot_confusion_matrix(
                cm, all_labels, judge_def.name, output_dir, metrics
            )

        # Analyze errors
        error_analysis = self.analyze_errors(annotated_examples)

        # Identify failure patterns
        failure_patterns = self.identify_failure_patterns(annotated_examples)

        # Generate improvement suggestions
        analysis_data = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "total_examples": len(annotated_examples),
            "failure_patterns": failure_patterns,
            "error_analysis": error_analysis,
        }

        improvement_suggestions = await self.generate_ai_improvements(
            analysis_data, judge_def
        )

        # Categorize examples
        examples_by_category = self._categorize_examples(annotated_examples)

        return JudgeAnalysis(
            judge_name=judge_def.name,
            total_examples=len(annotated_examples),
            accuracy=metrics["accuracy"],
            macro_precision=metrics["macro_precision"],
            macro_recall=metrics["macro_recall"],
            macro_f1=metrics["macro_f1"],
            weighted_precision=metrics["weighted_precision"],
            weighted_recall=metrics["weighted_recall"],
            weighted_f1=metrics["weighted_f1"],
            confusion_matrix=metrics["matrix"],
            classification_report=metrics.get("classification_report", {}),
            error_analysis=error_analysis,
            failure_patterns=failure_patterns,
            improvement_suggestions=improvement_suggestions,
            examples_by_category=examples_by_category,
            timestamp=datetime.now().isoformat(),
        )

    def _categorize_examples(
        self, examples: List[AnnotatedExample]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize examples by prediction outcome"""
        categories = {
            "correct_predictions": [],
            "incorrect_predictions": [],
            "system_errors": [],
        }

        for example in examples:
            example_data = {
                "correlation_id": example.correlation_id,
                "predicted": example.answer,
                "expected": example.expected_output,
                "analysis": example.analysis,
            }

            if example.error:
                example_data["error"] = example.error
                categories["system_errors"].append(example_data)
            elif example.answer == example.expected_output:
                categories["correct_predictions"].append(example_data)
            else:
                categories["incorrect_predictions"].append(example_data)

        return categories

    def save_analysis(
        self, analysis: JudgeAnalysis, output_file: str, use_folders: bool = False
    ):
        """Save analysis results to JSON file"""
        if use_folders:
            # Ensure analyses folder exists
            analyses_dir = Path("analyses")
            analyses_dir.mkdir(exist_ok=True)

            # If output_file is just a filename, put it in analyses folder
            output_path = Path(output_file)
            if len(output_path.parts) == 1:  # Just a filename
                output_file = analyses_dir / output_file

        # Ensure parent directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and save
        analysis_dict = asdict(analysis)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(analysis_dict, f, indent=2, ensure_ascii=False)

        print(f"Analysis saved to {output_file}")

    def print_summary(self, analysis: JudgeAnalysis):
        """Print analysis summary - ENHANCED: includes F1/Precision/Recall metrics"""
        print(f"\n{'=' * 60}")
        print(f"JUDGE ANALYSIS SUMMARY: {analysis.judge_name}")
        print(f"{'=' * 60}")
        print(f"Total Examples: {analysis.total_examples}")
        print(f"Accuracy: {analysis.accuracy:.3f} ({analysis.accuracy * 100:.1f}%)")
        print(f"  (How often judge predictions match human ground truth)")

        print(f"\nMacro-Averaged Metrics:")
        print(f"  Precision: {analysis.macro_precision:.3f}")
        print(f"  Recall: {analysis.macro_recall:.3f}")
        print(f"  F1-Score: {analysis.macro_f1:.3f}")

        print(f"\nWeighted-Averaged Metrics:")
        print(f"  Precision: {analysis.weighted_precision:.3f}")
        print(f"  Recall: {analysis.weighted_recall:.3f}")
        print(f"  F1-Score: {analysis.weighted_f1:.3f}")

        if analysis.classification_report:
            print(f"\nPer-Class Metrics (Judge Predictions vs Human Labels):")
            for label, metrics in analysis.classification_report.items():
                if isinstance(metrics, dict) and "precision" in metrics:
                    print(f"  {label}:")
                    print(f"    Precision: {metrics['precision']:.3f}")
                    print(f"    Recall: {metrics['recall']:.3f}")
                    print(f"    F1-Score: {metrics['f1-score']:.3f}")

        print(f"\nError Analysis (Judge vs Human Ground Truth):")
        error_analysis = analysis.error_analysis
        print(f"  Total Prediction Errors: {error_analysis['total_errors']}")
        print(f"  Correct Predictions: {error_analysis['correct_predictions']}")

        if error_analysis["system_errors"]:
            print(f"  System Errors: {len(error_analysis['system_errors'])}")

        if error_analysis["prediction_errors"]:
            print(
                f"  Prediction Disagreements (Human Ground Truth -> Judge Prediction):"
            )
            for error_type, examples in list(
                error_analysis["prediction_errors"].items()
            )[:5]:
                print(f"    {error_type}: {len(examples)} cases")

        print(f"\nTop Failure Patterns (Where Judge Disagrees with Humans):")
        for i, pattern in enumerate(analysis.failure_patterns[:3], 1):
            parts = pattern["error_type"].split(" -> ")
            if len(parts) == 2:
                human_label, judge_prediction = parts
                print(
                    f"  {i}. Human said '{human_label}' but Judge said '{judge_prediction}': {pattern['frequency']} cases ({pattern['percentage']:.1f}%)"
                )
            else:
                print(
                    f"  {i}. {pattern['error_type']}: {pattern['frequency']} cases ({pattern['percentage']:.1f}%)"
                )

        print(f"\nSuggestions to Align Judge with Human Judgment:")
        for i, suggestion in enumerate(analysis.improvement_suggestions, 1):
            print(f"  {i}. {suggestion}")

        print(f"{'=' * 60}")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Analyze judge performance with confusion matrix and improvement suggestions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --annotated multilingual_evaluation_annotated.json --judge multilingual_judge.json --output multilingual_judge_analysis.json --use-folders
  %(prog)s --annotated instruction_annotated.json --judge instruction_following_judge.json --output instruction_judge_analysis.json --use-folders
        """,
    )

    parser.add_argument(
        "--annotated", required=True, help="Path to annotated examples JSON file"
    )
    parser.add_argument(
        "--judge", required=True, help="Path to judge definition JSON file"
    )
    parser.add_argument(
        "--output", required=True, help="Path to output analysis JSON file"
    )
    parser.add_argument(
        "--use-folders", action="store_true", help="Use organized folder structure"
    )
    parser.add_argument(
        "--api-key", help="Mistral API key for AI improvement suggestions"
    )
    parser.add_argument("--base-url", help="Custom API base URL")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    try:
        api_key = args.api_key or os.getenv("MISTRAL_API_KEY")
        base_url = args.base_url or os.getenv("MISTRAL_BASE_URL")

        # Initialize analyzer
        analyzer = JudgeAnalyzer(api_key, base_url)

        # Load data
        if args.verbose:
            print(f"[...] Loading annotated examples from {args.annotated}")
        annotated_examples = analyzer.load_annotated_data(
            args.annotated, args.use_folders
        )

        if args.verbose:
            print(f"️  Loading judge definition from {args.judge}")
        judge_def = analyzer.load_judge_definition(args.judge, args.use_folders)

        print(f" Analyzing judge: {judge_def.name}")
        print(f" Total annotated examples: {len(annotated_examples)}")

        # Determine output directory
        output_path = Path(args.output)
        if args.use_folders:
            analyses_dir = Path("analyses")
            analyses_dir.mkdir(exist_ok=True)
            if len(output_path.parts) == 1:  # Just filename
                output_dir = analyses_dir
            else:
                output_dir = output_path.parent
        else:
            output_dir = output_path.parent or Path(".")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Perform analysis
        print(f"\n Starting judge analysis...")
        start_time = datetime.now()

        analysis = await analyzer.analyze_judge(
            annotated_examples, judge_def, output_dir
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Print summary
        analyzer.print_summary(analysis)
        print(f" Analysis time: {duration:.2f} seconds")

        # Save results
        analyzer.save_analysis(analysis, args.output, args.use_folders)

        if HAS_ML_LIBS:
            print(f" Confusion matrix plot saved in {output_dir}")
        else:
            print(f" Install matplotlib/seaborn for confusion matrix visualization")

        print(f"[OK] Analysis complete!")

        return 0

    except FileNotFoundError as e:
        print(f"[X] Error: File not found - {e}")
        return 1
    except json.JSONDecodeError as e:
        print(f"[X] Error: Invalid JSON format - {e}")
        return 1
    except Exception as e:
        print(f"[X] Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
