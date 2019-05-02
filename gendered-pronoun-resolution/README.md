# 22nd place solution

Here we describe our solution, and then share thorough instructions on how to reproduce the solution in a fresh pip environment.
If you're interested in our afterthoughts and what didn't work for us - [here](https://www.kaggle.com/c/gendered-pronoun-resolution/discussion/90431) is the corresponding Kaggle post. 

## 1. Our final solution
Our approach was similar to those already described. 
In a nutshell, the pipeline includes:
 - BERT embeddings
 - hand-crafted features
 - several MLPs

### 1.1. BERT embeddings
We concatenated embeddings for A, B, and Pronoun taken from Cased and Uncased large BERT models - 3 layers (-4, -5, -6 turned out to work best). For each BERT model this yield 9216-dimensional output: 3 (layers) x 3 (entity) x 1024 (BERT embedding size). 

### 1.2 Hand-crafted features
We ended up with 69 features of different nature (**mistake:** turns out we should've put more effort on BERT finetuning):
 - Neuralcoref, Stanford NLP and e2e-coref model predictions
 - Predictions of MLP trained with ELMo embeddings
 - Syntactic roles of A, B, and Pronoun (subject, direct object, attribute etc)
 - Positional and frequency-based (distancies between A, B, Pronoun and derivations, whether they all are in the same sentence or Pronoun is in the folowing one etc.)
 - Dependency tree-based (from [this](https://www.kaggle.com/negedng/extracting-features-from-spacy-dependency) Kernel)
 - Named entities predicted for A and B
 - GAP heuristics (from [this](https://www.kaggle.com/sattree/2-reproducing-gap-results) Kernel)

### 1.3. Models
The final combined model was a dense classification head built upon output from 5 other models:
 - Two MLPs (like in Matei's [kernel](https://www.kaggle.com/mateiionita/taming-the-bert-a-baseline)) - separate for Cased and Uncased BERTs, both taking 9216-d input and outputing 112-d vectors
 - Two Siamese models with distances between Pronoun and A-embeddings, Pronoun and B-embeddings as inputs and shared weights
 - One more MLP taking 69-d feature vectors as an input  

Final predictions were clipped with 0.02 threshold (turns out it's better without clipping).

## 2. Instructions to reproduce the solution from scratch

**Prerequisites:** 
- Python 3.6.7
- CUDA 9.0 (in case of other CUDA versions installed, modify tensorflow version in `requirements.txt`)
- (optionally) virtualenv - to run the script in a fresh environment (otherwise, check requirements.txt to see what is going to be installed).

**Steps:** 
  1\. Create a fresh pip environment
 - `virtualenv stage1_submission_from_scratch`
 - `source stage1_submission_from_scratch/bin/activate`
 
  2\. Run preparatory script which installs all necessary dependencies (from `requirements.txt`) and downloads BERT ([uncased_L-24_H-1024_A-16](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip) and [cased_L-24_H-1024_A-16](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip)) and [ELMo](https://github.com/allenai/allennlp) models
 - `(sh preparation.sh > pip_preparation.log 2>&1 &)` 
 
  3\. The `run.py` script creates features for `gap-test` + `gap-validation` and predicts for `gap-development`. Reproduces 0.33200 Public LB loss (6th most recent team's submission).
 - `(python3 run.py > run_stage1.log 2>&1 &)`
 
  4\. Deactivate the environment
 - `deactivate`

Training logs are also provided:
 - `pip_preparation.log`
 - `run_stage1.log`
 
Running times (256 Gb RAM, Quadro P6000, 24 GiB video memory):
 - Preprocessing: 2300s (dev) and 3200s (stage_1)
 - Training: 825s (dev) and 1200s (stage_1)

