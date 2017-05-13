[Proposal](/index.md)          [Checkpoint](/checkpoint.md)        [Final Report](/final_report.md)

**Yifan Jiang(yjiang1)**		**Xiangguang Zheng(xiangguz)**

Summary
-------
The goal of this project is to implement a LSTM DSL(Domain Specific Language), which provides RNN researchers a set of handy primitives to experiment with different LSTM-like RNN structures, which are then scheduled and run on GPUs in an efficient way. To achieve the goal of scheduling LSTM network dynamically, we started from optimizing the feedforward process of  the classic LSTM network in a static way using Cuda blocks and CuBlas. Referring to the technical blog [link] by NVIDIA's team, we implemented a series of optimizations and achieved a ~776x speedup compared to the native python implementation (not our main focus) and a ~10x speedup compared to the naive LSTM implementation in Cuda (both run on GHC machines).  Then we designed a generalized and schedulable LSTM engine, targeting to get closed to the performance achieved through static optimization as much as possible. Due to time constraint, we shifted our focus to building an IR (intermediate representation) in C++ and cuda, which acts as the backend engine of the DSL frontend in our original plan. This LSTM scheduler implementation applied several optimizationsis expected to achieve a speedup around ~4x compared to the naive CUDA implementation and ~0.8x compared to the static LSTM optimization.

Background
----------
Long short-term memory(LSTM) is a recurrent neural network architecture that is capable of learning long-term dependencies. It has been proven to be very powerful in classifying, processing and predicting inputs with time series (composing articles, translating etc.). The image below shows a classic LSTM cell and the operations involved in it:

<img src="images/classic_lstm.png" width="600" align="center">
<p align="center"><b>Figure #</b></p>





Challenges in LSTM scheduler
----------------------------
1. There exist dependencies between gates for some LSTM variants(shown in Figure #). Some matrix multiplications can not be combined since they depend on each other. The scheduler needs to explore independent matrix multiplications and allocate the weights in contiguous memory so that they can be combined.
2. LSTM variants have different element-wise operations that can not be scheduled statically. The scheduler needs to explore potential element-wise blocks and generate a fused Cuda kernel for each of them.
3. LSTM variants could have recurrent loops on different data. Traditional LSTM has both recurrent state and output, while GRU has only recurrent output. The scheduler should allow users to determine which part of the being recurrent.

<img src="images/gru_labeled.png" width="400" align="center">
<p align="center"><b>Figure #</b></p>

LSTM Invariants
---------------
1. Hyperparameters that all LSTM variants care about. Some example are number of layers, number of time steps for each layer, number of hidden units etc.
2. All LSTM variants (well-known) can be expressed with the same set of primitives, as shown in Figure #.

<img src="images/invariants.png" width="600" align="center">
<p align="center"><b>Figure #</b></p>

Approach
--------
Our approach is inspired by Tensorflow's architecture - using a DAG to represent a computation flow. Each node in the graph either represents a data point(weights/vectors) or operator.
1. Some primitives are provided for users to define an arbitrary LSTM model. Here shown a piece of code that define a forget gate in classic LSTM.
<img src="images/code.png" width="600" align="center">
<p align="center"><b>Figure #</b></p>

2. The user-defined model is then pre-processed by the intermediate layer so that:
..1. All element-wise operation blocks are explored and fused into a special fused node.
..2. All the potential matrix multiplication comibination are explored.
..3. Temporary data requriement is analyzed. In LSTM scheduler, temporary data is used to store the output of matrix multiplication and fused element-wise nodes.

<img src="images/pre_processed.png" width="600" align="center">
<p align="center"><b>Figure #</b></p>

3. Cuda/Cublas file is then generated based on the pre-processed LSTM model. To generate the file, the computation graph is traversed using BFS with taking into considration the computation dependencies between nodes.

<img src="images/workflow.png" width="600" align="center">
<p align="center"><b>Figure #</b></p>


