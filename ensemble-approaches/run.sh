#!/bin/bash

# Run majority vote ensemble
python ensemble_majority_vote.py \
    --judges directive_strength_cot_prompt_judge.json directive_strength_role_prompt_judge.json directive_strength_fewshots_prompt_judge.json directive_strength_cot_role_fewshot_prompt_judge.json \
    --events directive_strength_dataset.json \
    --output directive_strength_ensemble_majority_vote_results.json \
    --tie-break hierarchy \
    --hierarchy "Direct Command Judge Chain-of-Thought Prompt" "Direct Command Judge Role Prompt" "Direct Command Judge Few-Shots Prompt" "Direct Command Judge CoT Role FewShots Prompt" \
    --use-folders

# Analyze majority vote results
python ensemble_majority_vote_analyzer.py \
    --ensemble-results directive_strength_ensemble_majority_vote_results.json \
    --ground-truth directive_strength_annotations.json \
    --output directive_strength_ensemble_majority_vote_analysis.json \
    --plot \
    --use-folders

========================================================================================

# Run self-consistency ensemble
python ensemble_self_consistency.py \
    --events directive_strength_dataset.json \
    --output directive_strength_ensemble_self_consistency_results.json \
    --use-folders

# Analyze self-consistency results
python ensemble_self_consistency_analyzer.py \
    --sc-results directive_strength_ensemble_self_consistency_results.json \
    --ground-truth directive_strength_annotations.json \
    --output directive_strength_ensemble_self_consistency_analysis.json \
    --plot \
    --use-folders