#!/bin/bash

python human_annotator.py \
  --judge directive_strength_open_prompt_judge.json \
  --events directive_strength_dataset.json \
  --output directive_strength_annotations.json \
  --use-folders

python judge_evaluator.py \
  --judge directive_strength_open_prompt_judge.json \
  --events directive_strength_dataset.json \
  --output directive_strength_open_prompt_evaluation.json \
  --use-folders

python concatenate_annotations.py \
  --evaluations directive_strength_open_prompt_evaluation.json \
  --annotations annotations/directive_strength_annotations.json \
  --output directive_strength_open_prompt_annotated.json \
  --use-folders

python judge_analyzer.py \
  --annotated directive_strength_open_prompt_annotated.json \
  --judge directive_strength_open_prompt_judge.json \
  --output directive_strength_open_prompt_analysis.json \
  --use-folders

========================================================================================

python judge_evaluator.py --judge directive_strength_open_prompt_judge_enhanced.json --events directive_strength_dataset.json --output directive_strength_open_prompt_enhanced_evaluation.json --use-folders
python judge_evaluator.py --judge directive_strength_closed_prompt_judge_enhanced.json --events directive_strength_dataset.json --output directive_strength_closed_prompt_enhanced_evaluation.json --use-folders
python judge_evaluator.py --judge directive_strength_cot_prompt_judge_enhanced.json --events directive_strength_dataset.json --output directive_strength_cot_prompt_enhanced_evaluation.json --use-folders
python judge_evaluator.py --judge directive_strength_role_prompt_judge_enhanced.json --events directive_strength_dataset.json --output directive_strength_role_prompt_enhanced_evaluation.json --use-folders
python judge_evaluator.py --judge directive_strength_rubric_prompt_judge_enhanced.json --events directive_strength_dataset.json --output directive_strength_rubric_prompt_enhanced_evaluation.json --use-folders
python judge_evaluator.py --judge directive_strength_fewshots_prompt_judge_enhanced.json --events directive_strength_dataset.json --output directive_strength_fewshots_prompt_enhanced_evaluation.json --use-folders
python judge_evaluator.py --judge directive_strength_cot_role_fewshot_prompt_judge_enhanced.json --events directive_strength_dataset.json --output directive_strength_cot_role_fewshot_prompt_enhanced_evaluation.json --use-folders

python concatenate_annotations.py --evaluations directive_strength_open_prompt_enhanced_evaluation.json --annotations annotations/directive_strength_annotations.json --output directive_strength_open_prompt_enhanced_annotated.json --use-folders
python concatenate_annotations.py --evaluations directive_strength_closed_prompt_enhanced_evaluation.json --annotations annotations/directive_strength_annotations.json --output directive_strength_closed_prompt_enhanced_annotated.json --use-folders
python concatenate_annotations.py --evaluations directive_strength_cot_prompt_enhanced_evaluation.json --annotations annotations/directive_strength_annotations.json --output directive_strength_cot_prompt_enhanced_annotated.json --use-folders
python concatenate_annotations.py --evaluations directive_strength_role_prompt_enhanced_evaluation.json --annotations annotations/directive_strength_annotations.json --output directive_strength_role_prompt_enhanced_annotated.json --use-folders
python concatenate_annotations.py --evaluations directive_strength_rubric_prompt_enhanced_evaluation.json --annotations annotations/directive_strength_annotations.json --output directive_strength_rubric_prompt_enhanced_annotated.json --use-folders
python concatenate_annotations.py --evaluations directive_strength_fewshots_prompt_enhanced_evaluation.json --annotations annotations/directive_strength_annotations.json --output directive_strength_fewshots_prompt_enhanced_annotated.json --use-folders
python concatenate_annotations.py --evaluations directive_strength_cot_role_fewshot_prompt_enhanced_evaluation.json --annotations annotations/directive_strength_annotations.json --output directive_strength_cot_role_fewshot_prompt_enhanced_annotated.json --use-folders

python judge_analyzer.py --annotated directive_strength_open_prompt_enhanced_annotated.json --judge directive_strength_open_prompt_judge_enhanced.json --output directive_strength_open_prompt_enhanced_analysis.json --use-folders
python judge_analyzer.py --annotated directive_strength_closed_prompt_enhanced_annotated.json --judge directive_strength_closed_prompt_judge_enhanced.json --output directive_strength_closed_prompt_enhanced_analysis.json --use-folders
python judge_analyzer.py --annotated directive_strength_cot_prompt_enhanced_annotated.json --judge directive_strength_cot_prompt_judge_enhanced.json --output directive_strength_cot_prompt_enhanced_analysis.json --use-folders
python judge_analyzer.py --annotated directive_strength_role_prompt_enhanced_annotated.json --judge directive_strength_role_prompt_judge_enhanced.json --output directive_strength_role_prompt_enhanced_analysis.json --use-folders
python judge_analyzer.py --annotated directive_strength_rubric_prompt_enhanced_annotated.json --judge directive_strength_rubric_prompt_judge_enhanced.json --output directive_strength_rubric_prompt_enhanced_analysis.json --use-folders
python judge_analyzer.py --annotated directive_strength_fewshots_prompt_enhanced_annotated.json --judge directive_strength_fewshots_prompt_judge_enhanced.json --output directive_strength_fewshots_prompt_enhanced_analysis.json --use-folders
python judge_analyzer.py --annotated directive_strength_cot_role_fewshot_prompt_enhanced_annotated.json --judge directive_strength_cot_role_fewshot_prompt_judge_enhanced.json --output directive_strength_cot_role_fewshot_prompt_enhanced_analysis.json --use-folders

python judge_improver.py --annotated directive_strength_closed_prompt_judge_annotated.json --judge directive_strength_closed_prompt_judge.json \ --analysis directive_strength_closed_prompt_judge_analysis.json \ --use-folders
python judge_improver.py --annotated directive_strength_cot_prompt_judge_annotated.json        --judge directive_strength_cot_prompt_judge.json        --analysis directive_strength_cot_prompt_judge_analysis.json        --use-folders
python judge_improver.py --annotated directive_strength_cot_role_fewshot_prompt_judge_annotated.json --judge directive_strength_cot_role_fewshot_prompt_judge.json --analysis directive_strength_cot_role_fewshot_prompt_judge_analysis.json --use-folders
python judge_improver.py --annotated directive_strength_fewshots_prompt_judge_annotated.json   --judge directive_strength_fewshots_prompt_judge.json   --analysis directive_strength_fewshots_prompt_judge_analysis.json   --use-folders
python judge_improver.py --annotated directive_strength_open_prompt_judge_annotated.json       --judge directive_strength_open_prompt_judge.json       --analysis directive_strength_open_prompt_judge_analysis.json       --use-folders
python judge_improver.py --annotated directive_strength_role_prompt_judge_annotated.json       --judge directive_strength_role_prompt_judge.json       --analysis directive_strength_role_prompt_judge_analysis.json       --use-folders
python judge_improver.py --annotated directive_strength_rubric_prompt_judge_annotated.json     --judge directive_strength_rubric_prompt_judge.json     --analysis directive_strength_rubric_prompt_judge_analysis.json     --use-folders
