# 22nd place solution and some afterthoughts

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

## 2. What didn't work for us
### 2.1. Augmentation
If A is the right reference, then substitute B with all other named entities in the sentence.

*Example: [A] John entered the room and saw [B] Mary. [P] She looked so perfect laughing with Jerry and Noah. Btw, Jerry and Noah are Clara's and Julia's best friends".*

True label is B ('She' refers to Mary). I used to augment with 'She', 'Mary' and all other noun phrases in the sentence (according to Spacy POS tagging):

A | B | Pronoun
Jerry - Mary - She (B is true)
Noah - Mary - She (B is true)
Clara - Mary - She (B is true)
Julia - Mary - She (B is true)

Thus the dataset was increased 9x but new instances were much simpler to classify than original ones (~0.15 CV loss, 0.6 test loss). We noticed that the model trained on augmented data tended to make more confident predictions, hence many good answers but some big misses as well. Decided to use it as one more input for stacking.

### 2.2. Stacking
Though in the beginning blending helped a lot, when CV loss was ~0.33, for some reason we were not able to have any profit from stacking. We applied it mostly to OOFs built with augmentation (training folds being augmented, validation one - not), maybe we did smth wrong.

### 2.3. BERT finetuning
Training loss would quickly drop to zero, but test loss would be ~0.6. Now we realize that we should've made more attempts. 

## 3. Afterthoughts
 - Even though it's very dissappointing to miss the golden zone, still these emotions will fade away and the competition is the best one (that I personally took part in), that's why [thanks](https://www.kaggle.com/c/gendered-pronoun-resolution/discussion/87660#latest-507277) as early as 3 weeks before the end
 - Blind LB played a trick wit us. Have to admit we were a bit over-confident about our running relative position. 
 - Some takeaways for us are prototyping all components (including stacking) on the early stage, and putting more efforts on "killer features"

