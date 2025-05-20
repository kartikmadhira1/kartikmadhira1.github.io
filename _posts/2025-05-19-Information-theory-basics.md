---
title:  "Information theory basics."
mathjax: true
layout: post
categories: media
---


Quite a lot of times in deep learning models we across Information, probabilities, data distributions and cross-entropies. So what exactly is information? What does entropy mean in cross-entropy and why do we use it in our loss functions?


Information Theory is on such domain wherein all these ideas have been nurtured and adapted into the Neural networks domain. The blog here is my version of series of really well written blog post [here](https://machinelearningmastery.com/cross-entropy-for-machine-learning/). Do check it out.


### What is Information?

Information and probability are tighly related. Acc. to information theory, Information of an event is the negative-log of the probability of an event. I know its confusing. Why log and why the probability of an event?

$$
I(x) = -\log_2 p(x)
$$

since this is base of 2, the information will be stored in bits.

Acc. to the theory:

1. High probability of an event -> Low information.
2. Low probability of an event -> High information.


"The basic intuition behind information theory is that learning that an unlikely event has occurred is more informative than learning that a likely event has occurred." - [Deep Learning](https://www.google.com/books/edition/Deep_Learning/omivDQAAQBAJ?hl=en&gbpv=1&printsec=frontcover)


Why do we use log? Well here's graph to show the graph of -log(x),  $\forall x \in [0,1]$


![-log2 output](_data/post3/logoutput.png)


Clearly, the functions is trying to map lower probabilities to higher information scores. Hence, the -log(x).

### What is Entropy?


Entropy of a random variable X with a probability distribution P(X) for all events k is, 


$$
H(X) = -\sum_k p(k)*log(p(k))
$$


Information theory is rooted in data compression algorithms and one of the core principles in compression is how to get the minimal loss in quality compressing data X with the least amount "bits" as possible. In short if a certain target data with distribution P is to be converted to an estimate Q, we need to mimic the distribution P with the least footprint.


### What is Cross-Entropy?

If the distribution in P are skewed, then we can assume that P could be replicated by Q with only knowing the highly probable events which dominate the distribution. Hence to encode or "compress" Q we need only a few bits to represent P. On the other hand, if all the events in P are equally probable, then the reconstruction in Q will require all the information in P to make. E.g a dice rolled, coin toss etc. 

For the target distribution P and an approximation Q, what is the number of additional bits required to represent P using Q? If Q is a bad approximation, a lot more additional bits or information would be needed. To put in other words,  this information can also be a measure of the loss in information. 

For P to be represented by Q,

$$
H(P,Q) = -\sum_k P(k)*log(Q(k))
$$
