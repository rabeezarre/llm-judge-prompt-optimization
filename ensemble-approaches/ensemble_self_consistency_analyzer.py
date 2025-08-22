#!/usr/bin/env python3
"""
Self-Consistency Ensemble Analyzer
Analyzes self-consistency results with focus on reasoning path diversity and confidence.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import Counter
import argparse

try:
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    HAS_ML_LIBS = True
except ImportError:
    HAS_ML_LIBS = False
    print(
        "[!]  ML libraries not available. Install with: pip install numpy matplotlib seaborn scikit-learn"
    )


@dataclass
class SelfConsistencyAnalysis:
    """Analysis results for self-consistency ensemble performance"""

    ensemble_name: str
    total_samples: int
    accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    weighted_f1: float
    per_class_metrics: Dict[str, Dict[str, float]]

    # Self-consistency specific metrics
    average_confidence: float
    confidence_distribution: Dict[str, int]  # high/medium/low counts
    reasoning_path_analysis: Dict[str, Any]
    temperature_analysis: Dict[str, float]
    improvement_over_single_path: float
    consistency_vs_accuracy_correlation: float

    # Error analysis
    confusion_matrix: List[List[int]]
    error_patterns: Dict[str, int]
    low_confidence_errors: List[Dict[str, Any]]
    high_confidence_errors: List[Dict[str, Any]]


class SelfConsistencyAnalyzer:
    """Analyzer for self-consistency ensemble results"""

    def __init__(self, labels: Optional[List[str]] = None):
        self.labels = labels or ["Direct Command", "Suggestion", "No Instruction"]

    def load_self_consistency_results(
        self, file_path: str, use_folders: bool = False
    ) -> List[Dict[str, Any]]:
        """Load self-consistency results from JSON file"""
        if use_folders:
            ensemble_path = Path(file_path)
            if len(ensemble_path.parts) == 1:
                ensembles_dir = Path("ensembles")
                if (ensembles_dir / file_path).exists():
                    file_path = ensembles_dir / file_path

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_ground_truth_annotations(
        self, file_path: str, use_folders: bool = False
    ) -> Dict[str, str]:
        """Load ground truth annotations"""
        if use_folders:
            annotations_path = Path(file_path)
            if len(annotations_path.parts) == 1:
                annotations_dir = Path("annotations")
                if (annotations_dir / file_path).exists():
                    file_path = annotations_dir / file_path

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        ground_truth = {}
        for item in data:
            corr_id = item.get("correlation_id") or item.get("correlationId", "unknown")
            ground_truth[corr_id] = item.get("expected_output", "")

        return ground_truth

    def load_single_path_baseline(
        self, baseline_file: str, use_folders: bool = False
    ) -> float:
        """Load single-path baseline accuracy for comparison"""
        if use_folders:
            analyses_path = Path(baseline_file)
            if len(analyses_path.parts) == 1:
                analyses_dir = Path("analyses")
                if (analyses_dir / baseline_file).exists():
                    baseline_file = analyses_dir / baseline_file

        try:
            with open(baseline_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("accuracy", 0.0)
        except Exception as e:
            print(f"[!]  Could not load baseline accuracy from {baseline_file}: {e}")
            return 0.0

    def analyze_self_consistency(
        self,
        sc_results: List[Dict[str, Any]],
        ground_truth: Dict[str, str],
        single_path_baseline: Optional[float] = None,
    ) -> SelfConsistencyAnalysis:
        """Perform comprehensive self-consistency analysis"""

        # Filter valid results
        valid_results = []
        y_true = []
        y_pred = []

        for result in sc_results:
            if result.get("error"):
                continue

            corr_id = result["correlation_id"]
            if corr_id not in ground_truth:
                continue

            valid_results.append(result)
            y_true.append(ground_truth[corr_id])
            y_pred.append(result["final_answer"])

        if not valid_results:
            raise ValueError("No valid results found for analysis")

        # Basic performance metrics
        accuracy = accuracy_score(y_true, y_pred)

        if HAS_ML_LIBS:
            report = classification_report(
                y_true, y_pred, labels=self.labels, output_dict=True, zero_division=0
            )
            cm = confusion_matrix(y_true, y_pred, labels=self.labels)

            macro_f1 = report["macro avg"]["f1-score"]
            macro_precision = report["macro avg"]["precision"]
            macro_recall = report["macro avg"]["recall"]
            weighted_f1 = report["weighted avg"]["f1-score"]

            per_class_metrics = {
                label: {
                    "precision": report.get(label, {}).get("precision", 0.0),
                    "recall": report.get(label, {}).get("recall", 0.0),
                    "f1-score": report.get(label, {}).get("f1-score", 0.0),
                    "support": report.get(label, {}).get("support", 0),
                }
                for label in self.labels
            }
        else:
            macro_f1 = macro_precision = macro_recall = weighted_f1 = 0.0
            per_class_metrics = {}
            cm = [[0] * len(self.labels) for _ in range(len(self.labels))]

        # Self-consistency specific analysis
        confidence_analysis = self._analyze_confidence(valid_results)
        reasoning_analysis = self._analyze_reasoning_paths(valid_results)
        temperature_analysis = self._analyze_temperature_effects(valid_results)

        # Improvement calculation
        improvement = 0.0
        if single_path_baseline and single_path_baseline > 0:
            improvement = (
                (accuracy - single_path_baseline) / single_path_baseline
            ) * 100

        # Confidence vs accuracy correlation
        correlation = self._calculate_confidence_accuracy_correlation(
            valid_results, ground_truth
        )

        # Error analysis
        error_patterns = self._analyze_error_patterns(y_true, y_pred)
        low_conf_errors, high_conf_errors = self._analyze_confidence_errors(
            valid_results, ground_truth
        )

        return SelfConsistencyAnalysis(
            ensemble_name=valid_results[0].get(
                "ensemble_name", "Self-Consistency Ensemble"
            ),
            total_samples=len(valid_results),
            accuracy=accuracy,
            macro_f1=macro_f1,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            weighted_f1=weighted_f1,
            per_class_metrics=per_class_metrics,
            average_confidence=confidence_analysis["average"],
            confidence_distribution=confidence_analysis["distribution"],
            reasoning_path_analysis=reasoning_analysis,
            temperature_analysis=temperature_analysis,
            improvement_over_single_path=improvement,
            consistency_vs_accuracy_correlation=correlation,
            confusion_matrix=cm.tolist() if HAS_ML_LIBS else cm,
            error_patterns=error_patterns,
            low_confidence_errors=low_conf_errors,
            high_confidence_errors=high_conf_errors,
        )

    def _analyze_confidence(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze confidence score distribution"""
        confidences = [r["confidence_score"] for r in results]

        # Calculate distribution
        high_conf = sum(1 for c in confidences if c >= 0.8)
        medium_conf = sum(1 for c in confidences if 0.6 <= c < 0.8)
        low_conf = sum(1 for c in confidences if c < 0.6)

        return {
            "average": sum(confidences) / len(confidences),
            "min": min(confidences),
            "max": max(confidences),
            "distribution": {
                "high (≥0.8)": high_conf,
                "medium (0.6-0.8)": medium_conf,
                "low (<0.6)": low_conf,
            },
        }

    def _analyze_reasoning_paths(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze reasoning path diversity and agreement patterns"""
        path_counts = [r["num_paths"] for r in results]

        # Analyze vote patterns
        unanimous_decisions = 0  # All paths agree
        majority_decisions = 0  # Clear majority
        tied_decisions = 0  # No clear winner

        vote_diversity = []  # Number of different answers per sample

        for result in results:
            vote_counts = result.get("vote_counts", {})
            if not vote_counts:
                continue

            num_different_answers = len(vote_counts)
            vote_diversity.append(num_different_answers)

            max_votes = max(vote_counts.values())
            total_votes = sum(vote_counts.values())

            if max_votes == total_votes:
                unanimous_decisions += 1
            elif max_votes > total_votes / 2:
                majority_decisions += 1
            else:
                tied_decisions += 1

        return {
            "average_paths": sum(path_counts) / len(path_counts),
            "unanimous_rate": unanimous_decisions / len(results),
            "majority_rate": majority_decisions / len(results),
            "tie_rate": tied_decisions / len(results),
            "average_answer_diversity": sum(vote_diversity) / len(vote_diversity)
            if vote_diversity
            else 0,
            "max_answer_diversity": max(vote_diversity) if vote_diversity else 0,
        }

    def _analyze_temperature_effects(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Analyze temperature usage and effects"""
        temperatures = [
            r.get("temperature_used", 0.0) for r in results if r.get("temperature_used")
        ]

        if not temperatures:
            return {"average": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}

        return {
            "average": sum(temperatures) / len(temperatures),
            "min": min(temperatures),
            "max": max(temperatures),
            "std": np.std(temperatures) if HAS_ML_LIBS else 0.0,
        }

    def _calculate_confidence_accuracy_correlation(
        self, results: List[Dict[str, Any]], ground_truth: Dict[str, str]
    ) -> float:
        """Calculate correlation between confidence and correctness"""
        confidences = []
        correctness = []

        for result in results:
            corr_id = result["correlation_id"]
            if corr_id in ground_truth:
                confidences.append(result["confidence_score"])
                is_correct = result["final_answer"] == ground_truth[corr_id]
                correctness.append(1.0 if is_correct else 0.0)

        if len(confidences) < 2:
            return 0.0

        if HAS_ML_LIBS:
            correlation = np.corrcoef(confidences, correctness)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        else:
            # Simple correlation calculation
            n = len(confidences)
            mean_conf = sum(confidences) / n
            mean_corr = sum(correctness) / n

            numerator = sum(
                (c - mean_conf) * (r - mean_corr)
                for c, r in zip(confidences, correctness)
            )
            denom_conf = sum((c - mean_conf) ** 2 for c in confidences) ** 0.5
            denom_corr = sum((r - mean_corr) ** 2 for r in correctness) ** 0.5

            if denom_conf == 0 or denom_corr == 0:
                return 0.0
            return numerator / (denom_conf * denom_corr)

    def _analyze_error_patterns(
        self, y_true: List[str], y_pred: List[str]
    ) -> Dict[str, int]:
        """Analyze common error patterns"""
        error_patterns = Counter()

        for true_label, pred_label in zip(y_true, y_pred):
            if true_label != pred_label:
                error_patterns[f"{true_label} → {pred_label}"] += 1

        return dict(error_patterns)

    def _analyze_confidence_errors(
        self, results: List[Dict[str, Any]], ground_truth: Dict[str, str]
    ) -> tuple:
        """Analyze errors by confidence level"""
        low_conf_errors = []
        high_conf_errors = []

        for result in results:
            corr_id = result["correlation_id"]
            if corr_id not in ground_truth:
                continue

            true_label = ground_truth[corr_id]
            pred_label = result["final_answer"]
            confidence = result["confidence_score"]

            if true_label != pred_label:
                error_info = {
                    "correlation_id": corr_id,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "confidence": confidence,
                    "vote_counts": result.get("vote_counts", {}),
                    "num_paths": result.get("num_paths", 0),
                }

                if confidence < 0.6:
                    low_conf_errors.append(error_info)
                elif confidence >= 0.8:
                    high_conf_errors.append(error_info)

        return low_conf_errors, high_conf_errors

    def plot_confidence_analysis(
        self, analysis: SelfConsistencyAnalysis, output_dir: Path
    ):
        """Plot confidence distribution and correlation analysis"""
        if not HAS_ML_LIBS:
            print("[!]  Cannot create plots without matplotlib/seaborn")
            return None

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Confidence distribution
        conf_dist = analysis.confidence_distribution
        categories = list(conf_dist.keys())
        values = list(conf_dist.values())

        ax1.bar(categories, values, color=["red", "orange", "green"])
        ax1.set_title("Confidence Distribution")
        ax1.set_ylabel("Number of Samples")
        ax1.set_xlabel("Confidence Level")

        # Reasoning path analysis
        reasoning = analysis.reasoning_path_analysis
        path_metrics = ["unanimous_rate", "majority_rate", "tie_rate"]
        path_values = [reasoning.get(metric, 0) for metric in path_metrics]
        path_labels = ["Unanimous", "Majority", "Tie"]

        ax2.pie(path_values, labels=path_labels, autopct="%1.1f%%", startangle=90)
        ax2.set_title("Decision Pattern Distribution")

        # Confusion matrix
        sns.heatmap(
            analysis.confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.labels,
            yticklabels=self.labels,
            ax=ax3,
        )
        ax3.set_title("Confusion Matrix")
        ax3.set_xlabel("Predicted")
        ax3.set_ylabel("Actual")

        # Error patterns
        if analysis.error_patterns:
            error_items = list(analysis.error_patterns.items())[:5]  # Top 5
            patterns, counts = zip(*error_items)

            ax4.barh(patterns, counts)
            ax4.set_title("Top Error Patterns")
            ax4.set_xlabel("Count")
        else:
            ax4.text(
                0.5,
                0.5,
                "No errors to display",
                ha="center",
                va="center",
                transform=ax4.transAxes,
            )
            ax4.set_title("Error Patterns")

        plt.tight_layout()

        # Save plot
        plot_path = output_dir / f"self_consistency_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Self-consistency analysis plot saved to: {plot_path}")
        return str(plot_path)

    def print_analysis_report(self, analysis: SelfConsistencyAnalysis):
        """Print comprehensive self-consistency analysis report"""
        print(f"\n{'=' * 70}")
        print(f"SELF-CONSISTENCY ANALYSIS REPORT: {analysis.ensemble_name}")
        print(f"{'=' * 70}")

        # Basic performance
        print(f" Basic Performance:")
        print(f"   Total Samples: {analysis.total_samples}")
        print(f"   Accuracy: {analysis.accuracy:.3f} ({analysis.accuracy * 100:.1f}%)")
        print(f"   Macro-F1: {analysis.macro_f1:.3f}")
        print(f"   Macro-Precision: {analysis.macro_precision:.3f}")
        print(f"   Macro-Recall: {analysis.macro_recall:.3f}")
        print(f"   Weighted-F1: {analysis.weighted_f1:.3f}")

        # Self-consistency improvements
        print(f"\n Self-Consistency Metrics:")
        print(f"   Average Confidence: {analysis.average_confidence:.3f}")
        print(
            f"   Improvement over Single Path: {analysis.improvement_over_single_path:+.1f}%"
        )
        print(
            f"   Confidence-Accuracy Correlation: {analysis.consistency_vs_accuracy_correlation:.3f}"
        )

        # Confidence distribution
        print(f"\n Confidence Distribution:")
        for level, count in analysis.confidence_distribution.items():
            percentage = (count / analysis.total_samples) * 100
            print(f"   {level}: {count} ({percentage:.1f}%)")

        # Reasoning path analysis
        reasoning = analysis.reasoning_path_analysis
        print(f"\n Reasoning Path Analysis:")
        print(f"   Average Paths per Sample: {reasoning['average_paths']:.1f}")
        print(f"   Unanimous Decisions: {reasoning['unanimous_rate'] * 100:.1f}%")
        print(f"   Majority Decisions: {reasoning['majority_rate'] * 100:.1f}%")
        print(f"   Tied Decisions: {reasoning['tie_rate'] * 100:.1f}%")
        print(
            f"   Average Answer Diversity: {reasoning['average_answer_diversity']:.1f}"
        )

        # Temperature analysis
        temp = analysis.temperature_analysis
        print(f"\n️  Temperature Analysis:")
        print(f"   Average Temperature: {temp['average']:.3f}")
        print(f"   Temperature Range: {temp['min']:.3f} - {temp['max']:.3f}")
        if temp["std"] > 0:
            print(f"   Temperature Std Dev: {temp['std']:.3f}")

        # Per-class performance
        if analysis.per_class_metrics:
            print(f"\n Per-Class Performance:")
            for label, metrics in analysis.per_class_metrics.items():
                print(f"   {label}:")
                print(f"     Precision: {metrics['precision']:.3f}")
                print(f"     Recall: {metrics['recall']:.3f}")
                print(f"     F1-Score: {metrics['f1-score']:.3f}")
                if "support" in metrics:
                    print(f"     Support: {metrics['support']}")

        # Error analysis
        if analysis.error_patterns:
            print(f"\n[X] Top Error Patterns:")
            for pattern, count in sorted(
                analysis.error_patterns.items(), key=lambda x: x[1], reverse=True
            )[:5]:
                percentage = count / analysis.total_samples * 100
                print(f"   {pattern}: {count} ({percentage:.1f}%)")

        # Confidence-based error analysis
        if analysis.low_confidence_errors or analysis.high_confidence_errors:
            print(f"\n Confidence-Based Error Analysis:")
            print(f"   Low Confidence Errors: {len(analysis.low_confidence_errors)}")
            print(f"   High Confidence Errors: {len(analysis.high_confidence_errors)}")

            if analysis.high_confidence_errors:
                print(f"\n   High Confidence Errors (overconfident mistakes):")
                for i, error in enumerate(analysis.high_confidence_errors[:3]):
                    print(
                        f"     {i + 1}. {error['true_label']} → {error['predicted_label']} "
                        f"(conf: {error['confidence']:.3f}, votes: {error['vote_counts']})"
                    )

        print(f"{'=' * 70}")

    def save_analysis(self, analysis: SelfConsistencyAnalysis, output_file: str):
        """Save analysis to JSON file"""
        from dataclasses import asdict

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(asdict(analysis), f, indent=2, ensure_ascii=False)

        print(f"Self-consistency analysis saved to {output_file}")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Analyze self-consistency ensemble results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  %(prog)s --sc-results self_consistency_results.json --ground-truth directive_strength_annotations.json --output sc_analysis.json
  
  # With baseline comparison and plots
  %(prog)s --sc-results results.json --ground-truth annotations.json --baseline single_path_analysis.json --plot --use-folders
  
  # Custom labels
  %(prog)s --sc-results results.json --ground-truth annotations.json --labels "Class A" "Class B" "Class C"
        """,
    )

    parser.add_argument(
        "--sc-results", required=True, help="Path to self-consistency results JSON file"
    )
    parser.add_argument(
        "--ground-truth",
        default="directive_strength_annotations.json",
        help="Path to ground truth annotations JSON file",
    )
    parser.add_argument("--output", help="Path to save analysis JSON file")
    parser.add_argument(
        "--baseline", help="Single-path baseline analysis file for comparison"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["Direct Command", "Suggestion", "No Instruction"],
        help="Class labels for the task",
    )
    parser.add_argument("--plot", action="store_true", help="Generate analysis plots")
    parser.add_argument(
        "--use-folders", action="store_true", help="Use organized folder structure"
    )

    args = parser.parse_args()

    try:
        # Initialize analyzer
        analyzer = SelfConsistencyAnalyzer(args.labels)

        # Load data
        print(f"[...] Loading self-consistency results from {args.sc_results}")
        sc_results = analyzer.load_self_consistency_results(
            args.sc_results, args.use_folders
        )

        print(f"[...] Loading ground truth from {args.ground_truth}")
        ground_truth = analyzer.load_ground_truth_annotations(
            args.ground_truth, args.use_folders
        )

        # Load baseline if provided
        single_path_baseline = None
        if args.baseline:
            print(f"[...] Loading single-path baseline from {args.baseline}")
            single_path_baseline = analyzer.load_single_path_baseline(
                args.baseline, args.use_folders
            )

        # Analyze
        print(f" Analyzing self-consistency performance...")
        analysis = analyzer.analyze_self_consistency(
            sc_results, ground_truth, single_path_baseline
        )

        # Print report
        analyzer.print_analysis_report(analysis)

        # Generate plots if requested
        if args.plot:
            if args.use_folders:
                output_dir = Path("analyses")
                output_dir.mkdir(exist_ok=True)
            else:
                output_dir = Path(".")
            analyzer.plot_confidence_analysis(analysis, output_dir)

        # Save analysis if requested
        if args.output:
            if args.use_folders:
                analyses_dir = Path("analyses")
                analyses_dir.mkdir(exist_ok=True)

                output_path = Path(args.output)
                if len(output_path.parts) == 1:
                    args.output = analyses_dir / args.output

            analyzer.save_analysis(analysis, args.output)

        return 0

    except Exception as e:
        print(f"[X] Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import asyncio

    exit_code = asyncio.run(main())
    exit(exit_code)
