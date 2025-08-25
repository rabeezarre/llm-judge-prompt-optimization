# Prompt Strategies

This folder contains code and experiments for evaluating **prompting strategies in LLM-as-a-Judge systems**. The task studied here is **directive-strength classification**, where assistant messages are categorized as *Direct Command*, *Suggestion*, or *No Instruction*.  

---

## Background

LLM-as-a-Judge refers to using large language models to automatically evaluate outputs of other models. This reduces the need for human annotation but introduces reliability issues. In particular, judges can show **systematic biases** and **consistent failure patterns**.  

The experiments here test seven prompting strategies on 50 multi-turn conversations sampled from Mistral’s production distribution. All evaluations were run with the `mistral-small` model at **temperature = 0** for reproducibility.  

---

## Prompt Strategies Tested

| Strategy                  | Accuracy | Macro-P | Macro-R | Macro-F1 | Weighted-F1 |
|----------------------------|----------|---------|---------|----------|-------------|
| **CoT + Role + Few-shot** | **0.760** | 0.453   | 0.472   | 0.461    | 0.722       |
| Few-shot                   | 0.740    | **0.591** | 0.415   | 0.437    | 0.704       |
| Role prompting             | 0.680    | 0.509   | 0.459   | 0.481    | 0.693       |
| Chain-of-Thought (CoT)     | 0.680    | 0.523   | 0.411   | 0.444    | 0.683       |
| Open-ended                 | 0.600    | 0.504   | 0.470   | 0.479    | 0.656       |
| Rubric-based               | 0.560    | 0.481   | 0.428   | 0.448    | 0.621       |
| Closed                     | 0.500    | 0.485   | 0.400   | 0.429    | 0.580       |

---

## Key Findings

- Simple strategies (Open, Rubric, Closed) perform poorly and often miss context.  
- Enhanced strategies (CoT, Role, Few-shot) improve reliability, with the combined method performing best.
- **The composite CoT + Role + Few-shot strategy** achieved the best overall accuracy (76%).  
- **Embedded instruction blindness** is a consistent failure: judges fail to detect commands embedded in explanations or structured content.  
- Deterministic evaluation with **temperature = 0** makes results stable and reproducible.  

---

## Repository Structure

```
prompt-strategies/
├── analyses/               # Performance reports (JSON, metrics, confusion matrices)
├── annotated/              # Final merged annotation files
├── annotations/            # Raw human annotations
├── evaluations/            # Model outputs per strategy
├── events/                 # Evaluation event logs
├── judges/                 # Judge definitions (prompt templates in JSON)
├── concatenate_annotations.py # Merge annotations from multiple annotators (human and judge)
├── human_annotator.py      # Tool for manual annotation
├── judge_analyzer.py       # Compute metrics, error analysis, confusion matrices
├── judge_evaluator.py      # Run judge prompts on dataset
├── judge_improver.py       # Improve prompts
├── run.sh                  # Commands
├── requirements.txt        # Python dependencies
└── README.md               # (this file)
```
---

## Setup

1. Create and activate virtual environment:

```bash
python3.12 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
pip install numpy matplotlib seaborn scikit-learn
```

3. Set up your API key in `.env` file:

```bash
MISTRAL_API_KEY=your_actual_api_key_here
```

---

## Running Experiments

The pipeline has four steps: **annotate → evaluate → merge → analyze**.

### 1. Annotate Conversations
Manual labeling of assistant messages.  

```bash
python human_annotator.py \
  --judge directive_strength_open_prompt_judge.json \
  --events directive_strength_dataset.json \
  --output directive_strength_annotations.json \
  --use-folders
```

### 2. Run Evaluation
Model classifies the dataset using a chosen judge.  

```bash
python judge_evaluator.py \
  --judge directive_strength_open_prompt_judge.json \
  --events directive_strength_dataset.json \
  --output directive_strength_open_prompt_evaluation.json \
  --use-folders
```

### 3. Merge Annotations and Evaluations
Combine human and model labels for comparison.  

```bash
python concatenate_annotations.py \
  --evaluations directive_strength_open_prompt_evaluation.json \
  --annotations annotations/directive_strength_annotations.json \
  --output directive_strength_open_prompt_annotated.json \
  --use-folders
```

### 4. Analyze Results
Compute metrics, confusion matrices, and error patterns.  

```bash
python judge_analyzer.py \
  --annotated directive_strength_open_prompt_annotated.json \
  --judge directive_strength_open_prompt_judge.json \
  --output directive_strength_open_prompt_analysis.json \
  --use-folders
```

---

## Generalization

This framework is **universal**:  
- Any judge JSON in `judges/` can be used.  
- Any dataset of events can be substituted.  
- The same workflow applies to other evaluation tasks (factuality, safety, coherence, etc.).  
