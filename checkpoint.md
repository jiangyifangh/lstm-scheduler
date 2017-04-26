[Proposal](/index.md)          [Checkpoint](/checkpoint.md)

**Yifan Jiang(yjiang1)**		**Xiangguang Zheng(xiangguz)**

### Summary

We are going to build a mini-DSL(a Python library) for defining an LSTM, and then generate code for evaluating the network by assembling basic blocks from cuDNN or cuBLAS.

### Background

Implementing cuDNN operations are very expensive, therefore there are only basic operations defined in cuDNN. If the operation is not defined in cuDNN, the performance would be seriously downgraded, because the alternative implementation in python or C wouldn't take advantages of GPU. So the current situation is that the performance would be either extremely good if cuDNN has implemented the operation or extremely bad if cuDNN does not. We want an intermediate solution that is able to auto generate the code by assembling the basic block from cuDNN in order to get not optimal but acceptable performance, but mostly, much more convenient compared to implementing a new operation directly from cuDNN and provides a more generic interface.

### Challenges

- Though both of us have adequate knowledge in deep learning (mainly convolutional neural network), we have no previous experience in RNN/LSTM. It would take us around a week to fully understand it and implement it in Python.
- Designing the scheduling language is an important component of this project, which requires us a deep understanding of LSTM and a careful thought of what primitives we would include in the language. Neither of us is familiar with code generation. We need to spend some time to learn the code generation procedure.

### Resources

We plan to use P2.xlarge from Amazon aws, equipping with NVIDIA Tesla K80 GPU, or **equivalent** machines equipped with GPUs. 

Since no previous work has been done in this area so we will start from scratch. The reference we will review includes LSTM/RNN, Halide code base(which has a scheduling language for image processing code), cuDNN documentation, Numba and code generation tutorial etc.

### Goals and Deliverables

The deliverable is a DSL (a Python program) that is able to define an LSTM from user's input using basic block implementation from cuDNN. Our goal is to achieve significant speed up compared to the python/C implementation by exploring parallelism to the max extent and the performance should be close to the native cuDNN implementation. The extra goal is to make our DSL to support more generic deep learning model.

Extra Goal: make our DSL to support more generic deep learning model. Specifically, our DSL is able to support more deep learning models without need to code extra interfaces for those.

### Platform Choice

The language we decided to use to implement our DSL module is Python, since it is easy to use and learn for non-programmer and has good community support for deep learning packages. In addition, the level of the language is about right, where the user only defines what to do, not how to do, and the developer can optimize the operations a lot under the hood.

### Schedule

**4.11 ~ 4.17** 

- Research and learning (LSTM, cuDNN/cuBLAS, code generation).
- Implement a baseline LSTM model in Python.

**4.18 ~ 4.24**

- Experiment with cuDNN code generation from simple operations.
- Implement LSTM using cuDNN/cuBLAS.
- Design DSL primitives to describe LSTM variants.

**4.25 ~ 5.1**

- Prepare final exam on 5.1
- Generate cuDNN/cuBLAS blocks from DSL.

**5.2 ~ 5.8**

- Optimize scheduling decision/LSTM performance. 

**5.9 ~ 5.11**

- Prepare for final report and presentation (if selected)
