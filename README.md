# LAProp (Algorithm Implementation)

Recently I was working my way through a lot of deep learning books, and the subject of optimization came up, which I always found interesting. 
While looking for a different optimization algorithm (AMSProp), 
I stumbled upon a very interesting paper written by Liu Ziyin, Zhikang T.Wang, and Masahito Ueda called "LaProp: Separating Momentum and Adaptivity in Adam". 
Having implemented Adam from scratch earlier, I noticed the similarities and thought why not add this algorithm to the library that I was creating for myself. Fun exercise, and a very interesting paper!

Link to the paper: [LaProp: Separating Momentum and Adaptivity in Adam](https://arxiv.org/pdf/2002.04839.pdf)


### ToDo:

- implementation in TensorFlow
- run the comparison a few more times


The graph below shows the performance of both optimization algorithms on a MNIST dataset. This is the first "comparison graph". I used a
network with two hidden layers, with ReLU activation function. Because I used a DL library that I coded for myself, its performance was rather poor, but
I got the results. I had to reshape the MNIST dataset, so that I could pass the data through dense layers (no CNN's were used here this time). Ideally, I'd like
to implement this optimization algorithm in TensorFlow and see how well it performs then. 

[AdamvsLAProp](https://user-images.githubusercontent.com/19962689/107440173-7aa3ee80-6b33-11eb-8f00-fee1138d9b28.png)!
