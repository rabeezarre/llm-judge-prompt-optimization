#!/usr/bin/env python3
"""
Universal Ensemble Performance Analyzer
Analyzes any ensemble results against ground truth with configurable metrics.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import Counter, defaultdict
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
class EnsembleAnalysis:
    """Analysis results for ensemble performance"""

    ensemble_name: str
    total_samples: int
    accuracy: float
    macro_f1: float
    macro_precision: float
    macro_recall: float
    weighted_f1: float
    per_class_metrics: Dict[str, Dict[str, float]]

    # Ensemble-specific metrics
    agreement_rate: float
    improvement_over_best_individual: float
    error_recovery_rate: float
    computational_cost_ratio: float
    abstention_rate: float
    tie_break_rate: float

    # Individual judge performance within ensemble
    individual_accuracies: Dict[str, float]
    judge_agreement_with_ensemble: Dict[str, float]

    # Error analysis
    confusion_matrix: List[List[int]]
    error_patterns: Dict[str, int]
    tie_break_analysis: Dict[str, int]


class UniversalEnsembleAnalyzer:
    """Universal analyzer for any ensemble evaluation results"""

    def __init__(self, labels: Optional[List[str]] = None):
        # Default labels for directive strength task
        self.labels = labels or ["Direct Command", "Suggestion", "No Instruction"]

    def set_labels(self, labels: List[str]):
        """Set custom labels for different tasks"""
        self.labels = labels

    def load_ensemble_results(
        self, file_path: str, use_folders: bool = False
    ) -> List[Dict[str, Any]]:
        """Load ensemble results from JSON file"""
        if use_folders:
            # Check if file exists in ensembles folder
            ensemble_path = Path(file_path)
            if len(ensemble_path.parts) == 1:  # Just a filename
                ensembles_dir = Path("ensembles")
                if (ensembles_dir / file_path).exists():
                    file_path = ensembles_dir / file_path

        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_ground_truth_annotations(
        self, file_path: str, use_folders: bool = False
    ) -> Dict[str, str]:
        """Load ground truth from directive_strength_annotations.json format"""
        if use_folders:
            # Check if file exists in annotations folder
            annotations_path = Path(file_path)
            if len(annotations_path.parts) == 1:  # Just a filename
                annotations_dir = Path("annotations")
                if (annotations_dir / file_path).exists():
                    file_path = annotations_dir / file_path

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert to correlation_id -> expected_output mapping
        ground_truth = {}
        for item in data:
            corr_id = item.get("correlation_id") or item.get("correlationId", "unknown")
            ground_truth[corr_id] = item.get("expected_output", "")

        return ground_truth

    def load_individual_baselines(self, baseline_files: List[str]) -> Dict[str, float]:
        """Load individual judge baseline accuracies for comparison"""
        baselines = {}

        for baseline_file in baseline_files:
            try:
                with open(baseline_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Extract judge name and accuracy from analysis file
                judge_name = data.get("judge_name", Path(baseline_file).stem)
                accuracy = data.get("accuracy", 0.0)
                baselines[judge_name] = accuracy

            except Exception as e:
                print(f"[!]  Could not load baseline from {baseline_file}: {e}")

        return baselines

    def analyze_ensemble(
        self,
        ensemble_results: List[Dict[str, Any]],
        ground_truth: Dict[str, str],
        individual_baselines: Optional[Dict[str, float]] = None,
    ) -> EnsembleAnalysis:
        """Perform comprehensive ensemble analysis"""

        # Filter successful results and align with ground truth
        valid_results = []
        y_true = []
        y_pred = []

        for result in ensemble_results:
            if result.get("error"):
                continue

            corr_id = result["correlation_id"]
            if corr_id not in ground_truth:
                continue

            final_answer = result["final_answer"]
            if final_answer == "TIE_ABSTAIN":
                continue  # Skip abstentions for accuracy calculation

            valid_results.append(result)
            y_true.append(ground_truth[corr_id])
            y_pred.append(final_answer)

        if not valid_results:
            raise ValueError("No valid results found for analysis")

        # Basic performance metrics
        accuracy = accuracy_score(y_true, y_pred)

        if HAS_ML_LIBS:
            # Detailed metrics using sklearn
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
            # Fallback calculations
            macro_f1 = macro_precision = macro_recall = weighted_f1 = 0.0
            per_class_metrics = {}
            cm = [[0] * len(self.labels) for _ in range(len(self.labels))]

        # Ensemble-specific metrics
        agreement_rate = self._calculate_agreement_rate(valid_results)
        tie_break_rate = self._calculate_tie_break_rate(ensemble_results)
        abstention_rate = self._calculate_abstention_rate(ensemble_results)
        tie_break_analysis = self._analyze_tie_break_methods(ensemble_results)

        # Individual judge analysis
        individual_accuracies = self._analyze_individual_judges(
            valid_results, ground_truth
        )
        judge_agreement = self._calculate_judge_agreement(valid_results)

        # Determine number of judges for cost calculation
        sample_result = valid_results[0] if valid_results else {}
        num_judges = len(sample_result.get("individual_answers", {}))

        # Improvement calculation
        if individual_baselines:
            best_individual_accuracy = max(individual_baselines.values())
        else:
            best_individual_accuracy = (
                max(individual_accuracies.values()) if individual_accuracies else 0
            )

        improvement = (
            (accuracy - best_individual_accuracy) / best_individual_accuracy * 100
            if best_individual_accuracy > 0
            else 0
        )

        # Error analysis
        error_patterns = self._analyze_error_patterns(y_true, y_pred)

        return EnsembleAnalysis(
            ensemble_name=ensemble_results[0].get("ensemble_name", "Unknown Ensemble")
            if ensemble_results
            else "Empty Ensemble",
            total_samples=len(valid_results),
            accuracy=accuracy,
            macro_f1=macro_f1,
            macro_precision=macro_precision,
            macro_recall=macro_recall,
            weighted_f1=weighted_f1,
            per_class_metrics=per_class_metrics,
            agreement_rate=agreement_rate,
            improvement_over_best_individual=improvement,
            error_recovery_rate=0.0,  # Would need detailed individual results to calculate
            computational_cost_ratio=float(num_judges),
            abstention_rate=abstention_rate,
            tie_break_rate=tie_break_rate,
            individual_accuracies=individual_accuracies,
            judge_agreement_with_ensemble=judge_agreement,
            confusion_matrix=cm.tolist() if HAS_ML_LIBS else cm,
            error_patterns=error_patterns,
            tie_break_analysis=tie_break_analysis,
        )

    def _calculate_agreement_rate(self, results: List[Dict[str, Any]]) -> float:
        """Calculate rate of full agreement among judges"""
        agreement_count = 0
        total_count = 0

        for result in results:
            individual_answers = result.get("individual_answers", {})
            answers = [ans for ans in individual_answers.values() if ans != "ERROR"]

            if len(answers) > 1:
                total_count += 1
                if len(set(answers)) == 1:  # All judges agree
                    agreement_count += 1

        return agreement_count / total_count if total_count > 0 else 0.0

    def _calculate_tie_break_rate(self, results: List[Dict[str, Any]]) -> float:
        """Calculate rate of tie-breaking"""
        tie_breaks = sum(1 for result in results if result.get("tie_broken", False))
        return tie_breaks / len(results) if results else 0.0

    def _calculate_abstention_rate(self, results: List[Dict[str, Any]]) -> float:
        """Calculate rate of abstentions (TIE_ABSTAIN)"""
        abstentions = sum(
            1 for result in results if result.get("final_answer") == "TIE_ABSTAIN"
        )
        return abstentions / len(results) if results else 0.0

    def _analyze_tie_break_methods(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Analyze which tie-breaking methods were used"""
        tie_methods = Counter()
        for result in results:
            if result.get("tie_broken", False):
                method = result.get("tie_break_method", "unknown")
                tie_methods[method] += 1
        return dict(tie_methods)

    def _analyze_individual_judges(
        self, results: List[Dict[str, Any]], ground_truth: Dict[str, str]
    ) -> Dict[str, float]:
        """Analyze individual judge performance"""
        judge_accuracies = {}
        judge_correct = defaultdict(int)
        judge_total = defaultdict(int)

        for result in results:
            corr_id = result["correlation_id"]
            true_label = ground_truth[corr_id]

            for judge_name, answer in result.get("individual_answers", {}).items():
                if answer != "ERROR":
                    judge_total[judge_name] += 1
                    if answer == true_label:
                        judge_correct[judge_name] += 1

        for judge_name in judge_total:
            if judge_total[judge_name] > 0:
                judge_accuracies[judge_name] = (
                    judge_correct[judge_name] / judge_total[judge_name]
                )

        return judge_accuracies

    def _calculate_judge_agreement(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate how often each judge agrees with ensemble decision"""
        judge_agreement = {}
        judge_matches = defaultdict(int)
        judge_total = defaultdict(int)

        for result in results:
            ensemble_answer = result["final_answer"]

            for judge_name, answer in result.get("individual_answers", {}).items():
                if answer != "ERROR":
                    judge_total[judge_name] += 1
                    if answer == ensemble_answer:
                        judge_matches[judge_name] += 1

        for judge_name in judge_total:
            if judge_total[judge_name] > 0:
                judge_agreement[judge_name] = (
                    judge_matches[judge_name] / judge_total[judge_name]
                )

        return judge_agreement

    def _analyze_error_patterns(
        self, y_true: List[str], y_pred: List[str]
    ) -> Dict[str, int]:
        """Analyze common error patterns"""
        error_patterns = Counter()

        for true_label, pred_label in zip(y_true, y_pred):
            if true_label != pred_label:
                error_patterns[f"{true_label} → {pred_label}"] += 1

        return dict(error_patterns)

    def plot_confusion_matrix(self, analysis: EnsembleAnalysis, output_dir: Path):
        """Plot and save confusion matrix"""
        if not HAS_ML_LIBS:
            print("[!]  Cannot create confusion matrix plot without matplotlib/seaborn")
            return None

        plt.figure(figsize=(10, 8))

        # Create heatmap
        sns.heatmap(
            analysis.confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.labels,
            yticklabels=self.labels,
            cbar_kws={"label": "Count"},
        )

        plt.title(
            f"Confusion Matrix: {analysis.ensemble_name}",
            fontsize=16,
            fontweight="bold",
        )
        plt.xlabel("Ensemble Predictions", fontsize=12)
        plt.ylabel("Human Ground Truth", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        # Add metrics text box
        metrics_text = f"""Accuracy: {analysis.accuracy:.3f}
Macro F1: {analysis.macro_f1:.3f}
Agreement Rate: {analysis.agreement_rate:.3f}
Improvement: {analysis.improvement_over_best_individual:+.1f}%"""

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
            output_dir
            / f"{analysis.ensemble_name.lower().replace(' ', '_')}_confusion_matrix.png"
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Confusion matrix saved to: {plot_path}")
        return str(plot_path)

    def print_analysis_report(self, analysis: EnsembleAnalysis):
        """Print comprehensive analysis report"""
        print(f"\n{'=' * 70}")
        print(f"ENSEMBLE ANALYSIS REPORT: {analysis.ensemble_name}")
        print(f"{'=' * 70}")

        # Basic performance
        print(f" Basic Performance:")
        print(f"   Total Samples: {analysis.total_samples}")
        print(f"   Accuracy: {analysis.accuracy:.3f} ({analysis.accuracy * 100:.1f}%)")
        print(f"   Macro-F1: {analysis.macro_f1:.3f}")
        print(f"   Macro-Precision: {analysis.macro_precision:.3f}")
        print(f"   Macro-Recall: {analysis.macro_recall:.3f}")
        print(f"   Weighted-F1: {analysis.weighted_f1:.3f}")

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

        # Ensemble-specific metrics
        print(f"\n Ensemble Metrics:")
        print(
            f"   Agreement Rate: {analysis.agreement_rate:.3f} ({analysis.agreement_rate * 100:.1f}%)"
        )
        print(
            f"   Tie-Break Rate: {analysis.tie_break_rate:.3f} ({analysis.tie_break_rate * 100:.1f}%)"
        )
        print(
            f"   Abstention Rate: {analysis.abstention_rate:.3f} ({analysis.abstention_rate * 100:.1f}%)"
        )
        print(f"   Computational Cost: {analysis.computational_cost_ratio:.1f}x")
        print(
            f"   Improvement over Best Individual: {analysis.improvement_over_best_individual:+.1f}%"
        )

        # Tie-break analysis
        if analysis.tie_break_analysis:
            print(f"\n Tie-Break Methods Used:")
            for method, count in analysis.tie_break_analysis.items():
                percentage = count / analysis.total_samples * 100
                print(f"   {method}: {count} ({percentage:.1f}%)")

        # Individual judge performance
        print(f"\n‍️ Individual Judge Performance:")
        for judge_name, accuracy in analysis.individual_accuracies.items():
            agreement = analysis.judge_agreement_with_ensemble.get(judge_name, 0)
            print(f"   {judge_name}:")
            print(f"     Accuracy: {accuracy:.3f}")
            print(f"     Agreement with Ensemble: {agreement:.3f}")

        # Error analysis
        if analysis.error_patterns:
            print(f"\n[X] Top Error Patterns:")
            for pattern, count in sorted(
                analysis.error_patterns.items(), key=lambda x: x[1], reverse=True
            )[:5]:
                percentage = count / analysis.total_samples * 100
                print(f"   {pattern}: {count} ({percentage:.1f}%)")

        print(f"{'=' * 70}")

    def save_analysis(self, analysis: EnsembleAnalysis, output_file: str):
        """Save analysis to JSON file"""
        from dataclasses import asdict

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(asdict(analysis), f, indent=2, ensure_ascii=False)

        print(f"Analysis saved to {output_file}")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Universal ensemble evaluation analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with directive strength annotations
  %(prog)s --ensemble-results ensemble_results.json --ground-truth directive_strength_annotations.json --output analysis.json
  
  # Analysis with custom labels
  %(prog)s --ensemble-results results.json --ground-truth annotations.json --labels "Class A" "Class B" "Class C" --output analysis.json
  
  # Include baseline comparisons
  %(prog)s --ensemble-results results.json --ground-truth annotations.json --baselines baseline1.json baseline2.json --output analysis.json
  
  # Generate confusion matrix plot
  %(prog)s --ensemble-results results.json --ground-truth annotations.json --plot --use-folders
        """,
    )

    parser.add_argument(
        "--ensemble-results", required=True, help="Path to ensemble results JSON file"
    )
    parser.add_argument(
        "--ground-truth",
        default="directive_strength_annotations.json",
        help="Path to ground truth annotations JSON file (default: directive_strength_annotations.json)",
    )
    parser.add_argument("--output", help="Path to save analysis JSON file")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["Direct Command", "Suggestion", "No Instruction"],
        help="Class labels for the task",
    )
    parser.add_argument(
        "--baselines",
        nargs="*",
        help="Individual judge baseline analysis files for comparison",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Generate confusion matrix plot"
    )
    parser.add_argument(
        "--use-folders", action="store_true", help="Use organized folder structure"
    )

    args = parser.parse_args()

    try:
        analyzer = UniversalEnsembleAnalyzer(args.labels)

        print(f"[...] Loading ensemble results from {args.ensemble_results}")
        ensemble_results = analyzer.load_ensemble_results(
            args.ensemble_results, args.use_folders
        )

        print(f"[...] Loading ground truth from {args.ground_truth}")
        ground_truth = analyzer.load_ground_truth_annotations(
            args.ground_truth, args.use_folders
        )

        # Load baselines if provided
        individual_baselines = None
        if args.baselines:
            print(f"[...] Loading individual baselines...")
            individual_baselines = analyzer.load_individual_baselines(args.baselines)

        # Analyze
        print(f" Analyzing ensemble performance...")
        analysis = analyzer.analyze_ensemble(
            ensemble_results, ground_truth, individual_baselines
        )

        analyzer.print_analysis_report(analysis)

        # Generate confusion matrix plot if requested
        if args.plot:
            if args.use_folders:
                output_dir = Path("analyses")
                output_dir.mkdir(exist_ok=True)
            else:
                output_dir = Path(".")
            analyzer.plot_confusion_matrix(analysis, output_dir)

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
