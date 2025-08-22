#!/usr/bin/env python3
"""
Concatenate Evaluations with Human Annotations
Merges evaluation results with human annotations and saves to annotated folder.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Union
from datetime import datetime


def load_evaluations(file_path: str, use_folders: bool = False) -> List[Dict[str, Any]]:
    """Load evaluations from JSON file - handles both single results and arrays"""
    if use_folders:
        evaluations_path = Path(file_path)
        if len(evaluations_path.parts) == 1:
            evaluations_dir = Path("evaluations")
            if (evaluations_dir / file_path).exists():
                file_path = evaluations_dir / file_path

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return [data]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(
            f"Invalid JSON format: expected dict or list, got {type(data)}"
        )


def load_annotations(file_path: str, use_folders: bool = False) -> List[Dict[str, Any]]:
    """Load annotations from JSON file"""
    if use_folders:
        annotations_path = Path(file_path)
        if len(annotations_path.parts) == 1:
            markings_dir = Path("markings")
            if (markings_dir / file_path).exists():
                file_path = markings_dir / file_path

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def concatenate_evaluations_annotations(
    evaluations: List[Dict[str, Any]], annotations: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Concatenate evaluations with their annotations"""
    # Create lookup dict for annotations by correlation_id
    annotations_by_id = {}
    for annotation in annotations:
        correlation_id = annotation["correlation_id"]

        if annotation["expected_output"] != "SKIPPED":
            annotations_by_id[correlation_id] = annotation["expected_output"]

    # Add expected_output to evaluations
    annotated_evaluations = []
    for evaluation in evaluations:
        correlation_id = evaluation.get("correlation_id") or evaluation.get(
            "correlationId"
        )

        if correlation_id in annotations_by_id:
            # Create new evaluation with expected_output
            annotated_evaluation = evaluation.copy()
            annotated_evaluation["expected_output"] = annotations_by_id[correlation_id]
            annotated_evaluations.append(annotated_evaluation)

    return annotated_evaluations


def save_annotated_evaluations(
    annotated_evaluations: List[Dict[str, Any]],
    output_file: str,
    use_folders: bool = False,
):
    """Save annotated evaluations to file"""
    if use_folders:
        annotated_dir = Path("annotated")
        annotated_dir.mkdir(exist_ok=True)

        output_path = Path(output_file)
        if len(output_path.parts) == 1:
            output_file = annotated_dir / output_file

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(annotated_evaluations, f, indent=2, ensure_ascii=False)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Concatenate evaluation results with human annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --evaluations multilingual_evaluation_results.json --annotations annotations_20250725_130227.json --output multilingual_evaluation_annotated.json --use-folders
  %(prog)s --evaluations results.json --annotations my_annotations.json --output result.json
        """,
    )

    parser.add_argument(
        "--evaluations", required=True, help="Path to evaluation results JSON file"
    )
    parser.add_argument(
        "--annotations", required=True, help="Path to annotations JSON file"
    )
    parser.add_argument(
        "--output", required=True, help="Output file for annotated evaluations"
    )
    parser.add_argument(
        "--use-folders",
        action="store_true",
        help="Use organized folder structure (evaluations/, markings/, annotated/)",
    )

    args = parser.parse_args()

    evaluations_file = args.evaluations
    annotations_file = args.annotations

    if args.use_folders:
        if not Path(evaluations_file).exists():
            evaluations_path = Path("evaluations") / evaluations_file
            if evaluations_path.exists():
                evaluations_file = str(evaluations_path)

        if not Path(annotations_file).exists():
            annotations_path = Path("markings") / annotations_file
            if annotations_path.exists():
                annotations_file = str(annotations_path)

    if not Path(evaluations_file).exists():
        print(f"[X] Error: Evaluations file not found: {evaluations_file}")
        return 1

    if not Path(annotations_file).exists():
        print(f"[X] Error: Annotations file not found: {annotations_file}")
        return 1

    try:
        # Load data
        print(f"[...] Loading evaluations from {evaluations_file}")
        evaluations = load_evaluations(evaluations_file, args.use_folders)

        print(f"[...] Loading annotations from {annotations_file}")
        annotations = load_annotations(annotations_file, args.use_folders)

        print(
            f" Concatenating {len(evaluations)} evaluations with {len(annotations)} annotations"
        )

        # Concatenate
        annotated_evaluations = concatenate_evaluations_annotations(
            evaluations, annotations
        )

        print(
            f"[OK] Successfully matched {len(annotated_evaluations)} evaluations with annotations"
        )

        # Check for unmatched
        unmatched = len(evaluations) - len(annotated_evaluations)
        if unmatched > 0:
            print(f"[!]  Warning: {unmatched} evaluations had no matching annotations")

        skipped_annotations = sum(
            1 for a in annotations if a["expected_output"] == "SKIPPED"
        )
        if skipped_annotations > 0:
            print(f"⏭️  Skipped annotations: {skipped_annotations}")

        # Save results
        save_annotated_evaluations(annotated_evaluations, args.output, args.use_folders)

        if args.use_folders:
            print(f"Saved annotated evaluations to annotated/{Path(args.output).name}")
        else:
            print(f"Saved annotated evaluations to {args.output}")

        # Print summary
        if annotated_evaluations:
            output_counts = {}
            for evaluation in annotated_evaluations:
                expected = evaluation["expected_output"]
                output_counts[expected] = output_counts.get(expected, 0) + 1

            print(f"\nExpected output distribution:")
            for output, count in sorted(output_counts.items()):
                percentage = (count / len(annotated_evaluations)) * 100
                print(f"  {output}: {count} ({percentage:.1f}%)")

        return 0

    except Exception as e:
        print(f"[X] Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
