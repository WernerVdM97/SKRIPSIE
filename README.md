# Neural networks for language model
smoothing

Smoothing is critical to ensure a language model performs accurately even in the event of
data sparseness.. In this report, we investigate if it is possible to smooth a language model
with a neural network as well as determine if the neural network can address the over-
estimation of events that sometimes occurs using current methods. Background material
regarding current smoothing techniques such as Good-Turing smoothing is discussed. For
comparison, we develop a baseline language model with Good-Turing smoothing. Two
neural networks were developed, the first capable of predicting a discounted probability
for a trigram, trained on the Good-Turing estimates. The second neural network was
developed through an alternative training approach to directly calculate the estimates for
a trigram. Both neural networks were capable of achieving the same perplexity as the
baseline model. Through further analysis we show how the second neural network can
prevent over-estimation.

## This repository includes:

- A report of the entire project, https://github.com/WernerVdM97/SKRIPSIE/blob/master/project%20report.pdf
- All code implemented.

## Self-written code for:
 - The First Neural Network
    - Feature extraction given a file containing n-grams with their counts
    - Training of a neural network to produce a discounted probability for each n-gram
    - Rewriting these probabilities to an existing unsmoothed language model in ARPA format
  
- And Second Network Network
  - Feature extraction given a file containing n-grams with their counts
  - Training of a neural network to produce a "true" MLE for each n-gram
  - Rewriting these probabilities and discounting them to an existing unsmoothed language model in ARPA format
 
- Finally the scripts written to apply these networks to an ARPA file.

### Other files included are examples of basic interactions with neural networks and language modelling.
