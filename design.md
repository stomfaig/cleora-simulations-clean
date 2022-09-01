# Redsign notes.

I need a few things to be modified to make this thing functional again. I hope that this will be the last iteration, because I've redesigned this whole thing at least 5 times.

## 1 Adding support for complex representations.

The main thing here is that for continous propagation it can happen that some representations will become complex valued, hence this should be supported by the link prediction algorithm. Do to this, we simply concatenate the real parts of a vector with the complex parts.
The main issue with this approach is that purely real representtions will be concatenated with lots of zeros. **To check if this is actually a problem, some experiments have to be done** (See 2.3 below).


## 2 Modifying the evaluation structure.

1. At the moment running anything is a pain, since everything is dumped into the main. This should be moved, with all the necessary information piped in, and most importantly, all task specific calculations should be in one place. The main issue with this of course will be computational cost. [DONE]
2. Also, a unified syntax shoud be created, all metrics would provide a *setup function* that takes all arguments that are constant throughout the evaluation, and return a function that can be used for actual testing. [.]
3. Third, a big flaw currently for accurate comparison is the fact that the test setups are different in every case. Although the training edges are the same, but the fake edges are changeing from evaluation to evaluation, making it impossible to accurately compare embeddings. [DONE]
4. Removing the *vecotorise* decorator, to make changes more easily comprehensible. [DONE]
5. Build some sort of logging tool, so that evaluating and printing can be done in one go, and without tons of duplicate code. [.]


## 3 Adding features for converting any matrix into a continous matrix structure.

For any matrix based on their eigendecomposition a class is created, that calculated any (even non-integer) powers of the matrix.