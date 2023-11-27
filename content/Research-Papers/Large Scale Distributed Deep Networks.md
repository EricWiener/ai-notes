---
tags: [flashcards]
source: 
aliases: [ASGD]
summary: Method for training large scale distributed networks.
---

# Training Deep Neural Networks in parallel
- [[Stochastic Gradient Descent (SGD)]] is a popular optimization algorithm to train neural networks (Bottou, 2012; Dean et al., 2012; Kingma & Ba, 2014). 
- As for the parallelization of SGD algorithms (suppose we use $M$ machines for the parallelization), one can choose to do it in either a synchronous or asynchronous way.
- In **synchronous** SGD (SSGD), local workers compute the gradients over their own mini-batches of data, and then add the gradients to the global model. By using a barrier, these workers wait for each other, and will not continue their local training until the gradients from all the M workers have been added to the global model. It is clear that the training speed will be dragged by the slowest worker. 
- To improve the training efficiency, **asynchronous** SGD (ASGD) has been adopted, with which no barrier is imposed, and each local worker continues its training process right after its gradient is added to the global model. Although ASGD can achieve faster speed due to no waiting overhead, it suffers from another problem which we call delayed gradient. That is, before a worker wants to add its gradient $g(w_t)$ (calculated based on the model snapshot $w_t$) to the global model, several other workers may have already added their gradients and the global model has been updated to $w_{t + \tau}$ (here $\tau$ is called the delay factor). Adding gradient of model $w_t$ to another model $w_{t+\tau}$ does not make a mathematical sense, and the training trajectory may suffer from unexpected turbulence.


# Large Scale Distributed Deep Networks
[[NIPS-2012-large-scale-distributed-deep-networks-Paper.pdf|Initial paper on ASGD]]

### Asynchronous Stochastic Gradient Descent Summary
- This approach is used for training huge networks where the model is so big it needs to be ==split up among different GPUs==.
- In order to speed up training, the different workers don't wait for each other to send gradient updates and instead just use whatever the last available set of parameters were.
- The paper was able to train a deep network 30x larger than previously reported (back in 2012).
- MapReduce wasn't well suited, so they designed their own framework.
<!--SR:!2027-01-27,1383,350-->

## Model Parallelism
- The first way the paper parallelized was by splitting up a model into different sections to run on individual CPU cores (this was back in 2012).
- This approach splits up a **single instance of the model**.
![[large-scale-distributed-deep-networks-fig1.png]]
- As you can see in the above figure, a five layer deep network is partitioned across four machines (blue rectangles).
- Only those nodes with edges that cross partition boundaries (thick lines) will need to have their state transmitted between machines.
- Models with local connectivity structures are easier to distribute than fully-connected structures.
- The typical cause of less-than-ideal speedups is variance in processing times across the different machines, leading to many machines waiting for the single slowest machine to finish a given phase of computation.

## Distributed Optimization Algorithms
![[large-scale-distributed-deep-networks-fig2.png]]
- To train large models in a reasonable amount of time, they needed to parallelize computation across **multiple model instances** (vs. just splitting up a single model).
- The paper uses two approaches for distribution:
    - Downpour SGD:
    - Sandblaster L-BFGS:

### Downpour SGD
- [[Stochastic Gradient Descent (SGD)]] is often used for training deep neural networks.
- Unfortunately, the traditional formulation of SGD (update weights for one mini-batch at a time) is inherently sequential, making it impractical to apply to very large data sets where the time required to move through the data in an entirely serial fashion is prohibitive.
- They divide dataset into subsets and run a copy of the model on each of these subsets.
- The parameters of the model are sharded across a server (ex. each shard keeps track of 1/10th of the model parameters).

**Implementation**:
- Before processing each mini-batch, a model replica asks the parameter server service for an updated copy of its model parameters.
- After receiving an updated copy of its parameters, the DistBelief model replica processes a mini-batch of data to compute a parameter gradient
- Replica sends the gradient to the parameter server, which then applies the gradient to the current value of the model parameters.
- They used Adagrad to have a seperate adaptive learning rate for each parameter to increase robustness.
- "Warmstarted" model training with a single model replica before starting other replicas to increase stability.

**Analysis**:
- Downpour SGD is **more robust to machines failures** than standard (synchronous) SGD. For synchronous SGD, if one machine fails, the entire training process is delayed; whereas for asynchronous SGD, if one machine in a model replica fails, the other model replicas continue processing their training data and updating the model parameters via the parameter servers.
- Multiple forms of asynchronous processing in Downpour SGD introduce a great deal of additional stochasticity in the optimization procedure.
    - A model replica is almost certainly computing its gradients based on a set of parameters that are slightly out of date, in that some other model replica will likely have updated the parameters on the parameter server in the meantime.
    - Because the parameter server shards (each holds a fraction of the model weights) act independently, there is no guarantee that at any given moment the parameters on each shard of the parameter server have undergone the same number of updates, or that the updates were applied in the same order. This means the **model weights in the different shards could be updated in a different order or a different number of times**.
- Little theoretical grounding for the safety of these operations for nonconvex problems, but in practice we found relaxing consistency requirements to be remarkably effective

### Sandblaster L-BFGS
- A key idea in Sandblaster is distributed parameter storage and manipulation.
- Introduces a **coordinator**. The coordinator issues commands drawn from a small set of operations (e.g., dot product, scaling, coefficient-wise addition, multiplication) that can be performed by each parameter server shard independently.
- This allows running large models (billions of parameters) without incurring the overhead of sending all the parameters and gradients to a single central server.
- Use load-balancing to send small portions of work (smaller than 1/Nth of the total batch size) and assign replicas new portions when they are free. Also use "backup tasks" like in MapReduce.

### ASGD Analysis
- Many practical observations indicate that it usually costs ASGD more iterations to converge than sequential SGD, and sometimes, the converged model of ASGD cannot reach accuracy parity of sequential SGD, especially when the number of workers is large

# Delayed ASGD
[[Asynchronous Stochastic Gradient Descent with Delay Compensation, Shuxin Zheng et al., 2016.pdf|Paper on improving ASGD]]
- DC-ASGD is similar to ASGD in the sense that no worker needs to wait for others. It differs from ASGD in that it does not directly add the local gradient to the global model, but compensates the delay in the local gradient by using the a**pproximate Taylor expansion**. By doing so, it maintains almost the same efficiency as ASGD and **achieves much higher accuracy**