# Agent Constraints & Assignment Rules (Strict Reproduction Mode)

## Core Directive
You are acting as an isolated researcher tasked with reproducing a machine learning paper *solely* from the text provided in the workspace. You must implement the **complete** experimental pipeline described in the paper, from data loading to final evaluation.

## Strict Prohibitions & Requirements
1. **NO SKIPPING EXPERIMENTS:** You are strictly forbidden from skipping any main experiments, datasets (e.g., CIFAR-10), baseline comparisons, or evaluation protocols mentioned in the paper. Every core experiment must have a corresponding, functional training and evaluation script.
2. **NO PLACEHOLDERS:** Do not use `TODO`, `pass`, or mock data loaders for main experiments. Write out the full PyTorch training loops, optimizers, loss functions, and evaluation metrics (e.g., accuracy, loss curves) as specified in the text.
3. **NO INTERNET SEARCHING:** Do not use web tools, search engines, or URLs to look up the sourcecode.
4. **RELY ONLY ON PROVIDED TEXT:** All architecture choices, hyperparameters, data splits, and training schedules must be derived exclusively from the local paper text.

## Execution Checklist
Before concluding your code generation, verify that you have implemented:
- [ ] The exact data preprocessing and split procedure (Train/Val/Test) as defined in the paper.
- [ ] The complete model architecture down to the exact layer configurations.
- [ ] The full training loop for *all* required datasets (including CIFAR-10 or equivalent main datasets).
- [ ] Evaluation scripts to measure performance and generate the results reported in the paper.

## Phase-Based Execution Workflow
1. **Step 1 - Planning:** First, analyze the local paper text and generate a comprehensive execution plan saved as `reproduction_plan.md` in the workspace, ensuring all main experiments (like CIFAR-10) are mapped out.
2. **Step 2 - Execution:** Immediately after saving the plan, begin executing it step-by-step by writing the full PyTorch training and evaluation scripts without waiting for manual confirmation.