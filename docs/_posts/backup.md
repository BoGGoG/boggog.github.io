---
layout: post
title: "SYMBA Conclusions"
date: 2022-09-20 9:12
categories: machine learning feynman physics symba
---

This is the concluding post to the GCoC [SYMBA project]({% post_url 2022-07-14-Introduction-Feynman-Amplitudes-Project %}).
I doees not mean that I won't be posting about it any more, but it's for the official ending of theh GSoC project.

## Introduction

In quantum field theory (QFT) so called "Feynman diagrams" arise naturally when calculating things in a perturbative manner.
For example, the [scattering-matrix](https://en.wikipedia.org/wiki/S-matrix) (S-matrix) describes the scattering of particles in QFT.
Using [Wick's theorem](https://en.wikipedia.org/wiki/Wick%27s_theorem) to gete the perturbative expansion,
each order can be represented as a sum of Feynman diagrams.
While the expansion is divergent (as described in [one of my earlier posts]({% post_url 2022-07-14-Introduction-Feynman-Amplitudes-Project %})),
they are still incredibly useful and the asymptotic character of the series will only reveal itself directly at orders that
probably will never be accessible.

So what can one do with Feynman diagrams?
Well, each diagram stands for a specific integral that can be constructed using the [Feynman rules](https://en.wikipedia.org/wiki/Feynman_diagram#Feynman_rules)
for the specific theory you are working with.
Performing the integral will give you the _amplitude_ $$\mathcal{M}$$ of the diagram.
This amplitude is complex valued and usually contains lots of Lorentz indices $$\alpha, \beta, \gamma$$, $$\gamma$$-matrices $$\gamma^i_{\mu\nu}$$ and "basis vectors" $$u(p, \sigma)$$.
Squaring the amplutude involves calculating traces over $$\gamma$$-matrices, spin-sums and other tricks,
so it is far from trivial.
With the squared amplitude however, one can then calculate measurable quantities.
For a scattering process $$2\to 2$$ this would be be _differential [scattering cross section](https://en.wikipedia.org/wiki/Cross_section_(physics))_ $$\mathrm{d}\sigma(1 + 2 \to 1' + 2' + ... + n')$$, see for example [here](https://arxiv.org/pdf/1602.04182.pdf):

$$
\frac{\mathrm{d}\sigma_\mathrm{CM}}{\mathrm{d}\Omega} = \frac{1}{64 \pi^2 (E_1 + E_2)^2} \frac{|p_3|}{|p_1|} \bar{|\mathcal{M}|^2}\,,
$$

where $$E_i$$ and $$p_i$$ are the energies and momenta of the particles.
Similar expressions can be calculated for $$2\to n$$ processes.
For $$1\to n$$ processes, also called _decays_, the respective quantity is not called scattering cross section,
but _decay width_ and usually denoted by $$\Gamma(1\to 1' + 2' + ... + n')$$.

Here I am only using two theories: [Quantum Electro Dynamics](https://en.wikipedia.org/wiki/Quantum_electrodynamics) (QED) and
[Quantum Chromo Dynamics](https://en.wikipedia.org/wiki/Quantum_chromodynamics) (QCD).
While QED describes the electromagnetic interaction between particles, QCD describes the strong force
e.g. holding together neutrons and protons in the nucleus of an atom.
This is actually only a secondary effect, QCD describes the strong interaction between
quarks and gluon that are present in neutrons and protons.
While in our best model, the [Standard Model](https://en.wikipedia.org/wiki/Standard_Model),
QED appears in an extended form called [Electroweak Theory](https://en.wikipedia.org/wiki/Electroweak_interaction) where the
weak interaction e.g. responsible for the radioactive beta decay, is unified with electromagnetism. 

The goal of this project is to teach the squaring of the amplitudes to a neural network.
The motivation is three-fold:
- first, to show that it can be done and to explore how it is best done.
That it can be done has already been shown in [this paper](https://arxiv.org/pdf/2206.08901.pdf), but I want to explore a different
encoding of the expressions called _prefix notation_.
- second, to increase the speed of the calculations. Current computer programs for the calculations can take a very long time already for tree level calculations.
This of course depends on the optimization of the program and the machine learning model.
- third, it will be interesing to see how the model performs on expressions where current computer programs will never finish in a reasonable time, e.g. loop calculations or $$2\to n$$ calculations with $$n>6$$. 


## Data Generation
In order to generate amplitudes and squared amplitudes [MARTY](https://marty.in2p3.fr/) (A **M**odern **AR**tificial **T**eoreteical ph**Y**sicist)
is used.
MARTY is written in C++ and can calculate Feynman diagrams in any theory using symbolic computations.
Implementing a new theory in MARTY however is very complicated,
thus I am only using QED and QCD data.
MARTY is built for the symbolic calculation of diagrams, but not for their export.
The intended use is to build a C++ library which can then be used numerically.
The export of symbolic amplitudes and squared amplitudes can be achived with some tricks,
but I had troubles looping over particles.
Thus the data generation workflow is the following:
- there is a [C++ script](https://github.com/BoGGoG/SYMBA-Prefix/blob/main/data-generation-marty/QED/QED_AllParticles_IO.cpp) for the calculation and export of amplitudes and squared amplitudes.
This script can be called from the command line and takes as input the names of the in- and out-particles as well as the file names where the amplitudes and squared amplitudes should be saved.
The script already exports the amplitudes in some form of prefix notation or more precicely in some
form of abstract syntax tree.
It was easier to do it this way than write a parser in python for them in Python.
- in a separate [Python script](https://github.com/BoGGoG/SYMBA-Prefix/blob/main/data-generation-marty/QED/QED_loop_insertions_parallel.py)
all combinations of in- and out-particles are calculated and separate processes are spawned calling the C++ script.
In order to avoid [race conditions](https://en.wikipedia.org/wiki/Race_condition) with parallelization, each process writes to its own file.
This results in a lot of files (28224 for QED 2 to 3 processes), which is probably not good for the SSD and would not be allowed on a cluster,
but it's the best I could do.
The squared amplitudes are in a format that can be read by sympy in Python, thus no other steps are done in MARTY/C++.

The generated diagrams are then preprocessed.


### Preprocessing

#### Amplitudes
The amplitudes are rather difficult to handle.
The preprocessing is done in [this script](https://github.com/BoGGoG/SYMBA-Prefix/blob/main/data-preprocessing/2022-08-14-QED-DataPreparation/source/read_amplitudes.py).
They contain lots of indices like in the $$\gamma$$-matrices.
E.g. $$\gamma_{\tau, \gamma, \delta}$$ reads ```gamma_{ %\tau_109,%gam_161,%del_161}```.
There are also "basis function", e.g. for the photon $$A_{l, \tau}(p_3)^*$$ is represented as
```A_{l_3,+%\tau_109}(p_3)^(*)```.
The following is done in [this script](https://github.com/BoGGoG/SYMBA-Prefix/blob/8b6ca5d9f4416c2e4a5c4804abec359064555ba5/data-preprocessing/2022-08-14-QED-DataPreparation/scripts/DataPreparation_parallel.py#L78-L82) to the amplitudes:
- since they are already exported in a nested list structure that represents an abstract syntax tree,
the expressions are first converted to a tree and then to a flat list in prefix notation.
- Then the subscripts are fixed. In the expressions above, ```%```  means that a subscript is summed over.
We simply remove this information and let the model learn that repeated indices are summed over.
Next, the subscripts are converted to a more notation.
It does not matter what indices are called if they are summed over, so we convert all greek indices
to ```alpha_i``` where I enumerates them in a given expression.
The same is done for roman indices. In QCD capital roman indices are introduces which get their own category.
- Finally, the indices need to somehow be encoded.
This is done using the prefix philosopy: ```gamma_{alpha_1, alpha_2, alpha_3}``` is converted to
```gamma alpha_1 alpha_2 alpha_3```.
This works as long as each symbol like `gamma` always has the same number of indices.
The basis vectors are endoded similarly: 
```e_{i_1, alpha_1}(p_2)_u``` is the basis vector for an electron and becomes ```ee i_1 alpha_1 (p_1)_u```.
The `ee` is chosen because `e` is Euler's number in sympy.
For the complex conjugate $$e^*$$ a new symbol is introduced for each basis function, e.g. ```ee^(*)```.

#### Squared Amplitudes 

For the code see [this script](https://github.com/BoGGoG/SYMBA-Prefix/blob/main/data-preprocessing/2022-08-14-QED-DataPreparation/scripts/DataPreparation_parallel.py) for QED. 
This step includes simplification of the squared amplitudes and conversion to hybrid prefix notation.
The squared amplitudes are simplified using the [`factor` function in sympy](https://docs.sympy.org/latest/tutorials/intro-tutorial/simplification.html).
Finally, we also combine tokens that frequently appear together to a single token.
This is $$m_i^j$$ for $$j=2, 4$$ and $$m_i^j s_{kl}$$.
It is not stated in the MARTY documentation, but I think $$s_{kl}=-2p_k\cdot p_l$$, where
$$p_k$$ is the 4-momentum of particle $$k$$.

The simplification in sympy can sometimes take a very long time or not terminate at all.
I got told that the problem might arise because unlucky random numbers.
Thus, there is a timout for the simplification.
It turns out that parallelization together with timeouts is not an easy task in Python.
For my solution see the blog post on [parallel processing in Python with timeout and batches]({% post_url 2022-08-09-Parallel-Processing-Python-Timeout%}).


### Encoding of Expressions

This is the fundamentally new part of my work.
In the [existing work on SYMBA](https://arxiv.org/abs/2206.08901) the strings of the (infix) expressions are encoded using
some tensorflow functions I think.
Since expressions are actually trees, I think a tree2tree model should perform best.
However, [this paper on symbolic math with deep learning](https://arxiv.org/abs/1912.01412) by people from Facebook, mentions that seq2seq models
are good at working with tree structure data.
They cite a paper on [Grammar as a foreign language](https://arxiv.org/abs/1412.7449) and I don't see the direct
connection.
Nevertheless, I will leave tree2tree or graph2graph models for future work and focus on seq2seq models here,
albeit with prefix notation as also used in the paper mentioned above (the on symbolic math with deep learning),
but not in the existing work on SYMBA. 

A mathematical expression is fundamentally a tree.
Let's look a the equation $$a + b*\sin(c)$$.
I chose this example because it contains unary and binary operators as we will see. 
If we now see all operators like $$+$$ as acting on arguments, kind of like you would write 
a function in a programming language, then we can for example write
$$a+b$$ as `add a b`.
This way we can see above equation as a tree where the nodes are functions and the
leaves are atomic expressions like numbers or variables.
For above expression we get the tree:

<p align="center">
  <img src="/figures/expression_tree.png"> 
</p>

This tree also naturally leads to prefix notation.
Perform a depth-first tree traversal and write down everything you encounter.
In this example we get:

```add a mul b sin c```

Now a major problem is this:
The squared amplitudes are large sums.
What happens if we have a sum of many terms?
Say $$a+b+c+d$$.
In prefix notation this would be ```add a add b add c d```.
See the problem? So many `add` tokens.
The computation complexity of transformers scales quadratically with the sequence length.

First: Why do we have to add all those extra `add` tokens?
The thing is that in order for expressions to be well-defined, every operator needs to have a
fixed number of arguments.
We cannot write ```add a b c d```.
We can easily see the problem in this example:
The two expressions

$$ 2 \cdot 3 \cdot (a + b + c) $$ and     
$$ 2 \cdot 3 \cdot (a + b) \cdot c $$    

would be     

`mul ( 2 3 add ( a b c ) )`   
`mul ( 2 3 add ( a b ) c )`

respectively, where I have added parentheses for clarity.
If we leave out the parentheses, the two different infix expressions have the same prefix notation.
Why not keep parentheses you ask?
Because this would give even more tokens, since we would also have them for unary and binary operators.

**Hybrid prefix notation**:
The way I do it right now is a hybrid between prefix with parentheses and without parentheses.
For `add` and `mul` there also exists a new variant, `add(` and `mul(` that needs a closing parenthese.
Now, $$a+b$$ still stays `add a b`, but $$a+b+c$$ becomes `add( a b c )`.
Note that for 3 terms in the sum this has the same amount of tokens as `add a add b c`, but for more
there are no extra tokens added other than the actual terms.
How do I know that this is well-defined?
Basically because I have implemented functions for sympy <-> hybrid prefix and the tests work.
I have tested it on 1000 QCD amplitudes and on some random expressions
like ```8*g**4*(2*m**4 - m**2*(s+d)**2)```.
Right now `exp` and `sin` are not working, but they never appear in squared amplitudes.


## Transformer Model

The model I have used is a simple transformer and can be seen in [this script](https://github.com/BoGGoG/SYMBA-Prefix/blob/main/models/QED/QED_transformer/2022_08_24_QED_Transformer.ipynb).
The transformer architecture has conquered the deep learning landscape quickly after the introduction
in the famous paper called ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).
As the name of the paper indicates, transformers are based on the attention mechanism.
They are usually sequence2sequence models with an encoder-decoder structure, although other variants
definitely exist.
Now they not only dominate in language models, but also in computer vision [reference needed].

The code is mostly an adapted version of [keras's english-to-spanish translation](https://keras.io/examples/nlp/neural_machine_translation_with_transformer/) tutorial.
I have to admit that I am not satisfied this, but the other parts have simply taken so long that not enough
time was left for a better model.
The transformer has an embedding dimension of 256, a latent dimension of 2048 and 8 heads.
Usually for language models the embedding dimension is much higher, but I thought since we only have a
rather small number of unique "words" as opposed to the thousands of different words in language translation,
a smaller embedding dimension would maybe make sense.
For tokenization Tensorflow's [`TextVectorization`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization) function is used, which basically makes a dictionary of all words and enumerates them based on frequency of appearance.

## Training

asdf

## Inference

Inference is done token by token.
We start with the amplitude and only a `[START]` token for the squared amplitude.
Then we predict one token after the other, e.g. after the first prediction we would have
`[START] add` and after the second we would have `[START] add 8` and so on.
For each predicted token there is a probability given by the model.
Say for the first token the probabilities are `"8": 80%`, `"16": 19%` and so on.
Then one chooses the "8", because it has the highest probability.
We stop once the `[END]` token is predicted.

### Beam Search
For a good introduction to beam search see [this blogpost](https://www.width.ai/post/what-is-beam-search)
or [the "Dive into Deep Learning" book](https://d2l.ai/chapter_recurrent-modern/beam-search.html).
The idea behind beam search is the following:
Instead of choosing the token with the highest probability, choose the top 2 or 3 and
"evolve" them separately.
It might happen that the first token had a lower probability, but the consecutive tokens
have so much higher probabilities, that the total probability (just multiplicate all consecutive probabilities or add their log probabilities) of the predicted sequence is higher than only always going the "greedy" way.
Of course, the exact strategies for beam search can vary. 
A full calculation of all "strains" is usually not computationaly viable,
but maybe for the first few steps on can take the top 2 or 3 choices.

Actually I think our use case is a bit different than in a real natural language processing task.
Usually in languages there is not one correct solution, but many.
Thus, the probabilities can often look like [50%, 20%, 10%, ...].
in our case however, the model usually has a pretty high confidence like 99.9%.
I propose a different strategy:
Every time the model is not sure for a token, say <90%, also take the one with the second highest probability.


### Estimation of Uncertainties
I still have to do this, but my idea is the following:
For a predicted expression, one can get an extimation of how certain the model is, by
multiplying the probabilities for all the predicted tokens.
Using test data, I then will compare the certainty with the actual accuracy.

## Evaluation

### Viable Evaluation Measures
There are different ways one can measure the accuracy of a natural language model.
One possibility is to purely focus on the next token and calculate the next-token-accuracy
or the categorical cross-entropy.
This is also what the model is trained on.

For machine translation the [Bilingual Evaluation Understudy (BLEU)](https://aclanthology.org/P02-1040.pdf)
is usually used.
An extension is [METEOR](https://en.wikipedia.org/wiki/METEOR).
Other scores are the [ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric))
or also [Perplexity](https://en.wikipedia.org/wiki/Perplexity).

Of course, next-token-accuaracy is the easiest one to calculate.
Also, I most other scores are tuned for natural language processing and while our problem
of squaring amplitudes is formulated the same way as natural language tasks, namely as a
seq2seq problem, I think there are differences that justify a different evaluation.
First, we have one correct solution.
There can be different orderings or ways of writing it, but there is only one solution.
Parsing the prediction in sympy and simplifying should get rid of the differences
between two solutions that are written in a different way, e.g. $$a+b$$ and $$b+a$$
or $$a(b+c)$$ and $$ab + bc$$.
Of course here we already have a first test: Can the sequence be parsed into a sympy expression.
Since the predictions are in hybrid prefix format, the first step is to use the 
[`hybrid_prefix_to_sympy` function](https://github.com/BoGGoG/SYMBA-Prefix/blob/8b6ca5d9f4416c2e4a5c4804abec359064555ba5/sympy-prefix/source/SympyPrefix.py#L498) that I wrote.




### Results


## Interpretation and Conclusion

asdf





