## Debugging

> I'm getting a shape mismatch when I try to train a Keras model with
> softmax output on 10 bins. My input is (N, 1280) ESM embeddings and
> my target is (N, 10) bin probabilities. The error says expected shape
> (None, 1) but got (None, 10). What am I doing wrong?

> My focal loss function returns NaN during training. The predictions
> are from a softmax layer so they should be in [0,1]. How do I add
> numerical stability?

## Experiment Structure

> I want to search over neural network architectures for a regression
> task. I have about 3000 training samples and 1280-dimensional
> embeddings. What's a reasonable range of hidden layer sizes and
> dropout rates to sweep? I don't want to overfit.

> I'm comparing single-seed models right now but the results vary a lot
> between runs. Would a multi-seed ensemble help, and how many seeds is
> typical?

## Verification

> Can you check if this MSE value of 0.9323 matches what you'd get from
> the predictions in the JSON file? The JSON has per-seed predictions
> and I averaged them.

> I'm reporting Spearman correlations for 4 models on a design task.
> Here are the values from my summary table and here are the raw
> per-gene results from the JSON. Do they match?
