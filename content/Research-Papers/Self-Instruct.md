---
tags:
  - flashcards
source: 
summary: generate instructions, input, and output samples from a pretrained LLM to finetune it for instruction following
aliases:
  - Aligning Language Models with Self-Generated Instructions
---
A framework for improving the instruction-following capabilities of pretrained language models by bootstrapping off their own generations. Our pipeline generates instructions, input, and output samples from a language model, then filters invalid or similar ones before using them to finetune the original model.

Their pipeline consists of 4 parts:
1) generating task instructions
2) determining if the instruction represents a classification task
3) instance generation with either an input-first or output-first approach
4) filtering low-quality data.
### Instruction Data
They start with 175 seed tasks that look like:
```json
{"id": "seed_task_0", "name": "breakfast_suggestion", "instruction": "Is there anything I can eat for a breakfast that doesn't include eggs, yet includes protein, and has roughly 700-1000 calories?", "instances": [{"input": "", "output": "Yes, you can have 1 oatmeal banana protein shake and 4 strips of bacon. The oatmeal banana protein shake may contain 1/2 cup oatmeal, 60 grams whey protein powder, 1/2 medium banana, 1tbsp flaxseed oil and 1/2 cup watter, totalling about 550 calories. The 4 strips of bacon contains about 200 calories."}], "is_classification": false}
{"id": "seed_task_1", "name": "antonym_relation", "instruction": "What is the relation between the given pairs?", "instances": [{"input": "Night : Day :: Right : Left", "output": "The relation between the given pairs is that they are opposites."}], "is_classification": false}
```

Each seed task has a instruction, an optional additional input, and an output. An example of this is the instruction "write an essay about school safety" which requires no additional output and the model can respond directly to. You could also have the instruction "write an essay about the following topic" and provide the input "schools safety" as the input. They allow instructions w/ and w/o additional input.

### Instruction Generation
They initiate the task pool with 175 tasks (1 instruction and 1 instance for each task. For every step, we sample 8 task instructions from this pool as in-context examples. Of the 8 instructions, 6 are from the human-written tasks, and 2 are from the model-generated tasks in previous steps to promote diversity.

### Classification Task Identification
They then identify whether the generated instruction represents a classification task or not. They prompt the LM in a few-shot way to determine this, using 12 classification instructions and 19 non-classification instructions from the seed tasks.

### Classification Task Identification
Given the instruction and task type, they then generate instances for each instruction independently. This requires figuring out what the target task is, figure out what additional input fields are needed + generate then, and based on the inputs, complete the task with an output.

They used in-context examples of instruction-input-output pairs to get the LM to generate more. This can be done with an input-first or output-first approach.

> [!NOTE]
> They used an output-first approach for classification tasks and an input-first approach for the other tasks.

**Input-First Approach**
Ask an LM to come up with the input fields first based on the instruction, and then produce the corresponding output.

However, they found found that this approach can generate inputs biased toward one label, especially for classification tasks (e.g., for grammar error detection, it usually generates grammatical input).

**Output-First Approach**
You first generate the possible class labels, and then condition the input generation on each class label.

# Prompting Templates
Prompt used for generating new instructions. 8 existing instructions are randomly sampled from the task pool for in-context demonstration. The model is allowed to generate instructions for new tasks, until it stops its generation, reaches its length limit or generates “Task 16” tokens:
![[screenshot 2023-10-11_10_07_35@2x.png]]

Prompt used for classifying whether a task instruction is a classification task or not:
![[screenshot 2023-10-11_10_10_45@2x.png]]

Prompt used for the input-first approach of instance generation. The model is prompted to generate the instance first, and then generate the corresponding output. For instructions that don’t require additional input, the output is allowed to be generated directly:
![[screenshot 2023-10-11_10_11_19@2x.png]]

Prompt used for the output-first approach of instance generation. The model is prompted to generate the class label first, and then generate the corresponding input. This prompt is used for generating the instances for classification tasks:
![[screenshot 2023-10-11_10_12_36@2x.png]]
