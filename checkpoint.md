[Proposal](/proposal.md)&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;
[Checkpoint](/checkpoint.md)&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;
[Final Report](/index.md)

Project Checkpoint
==================

**Yifan Jiang(yjiang1)**		**Xiangguang Zheng(xiangguz)**

What we have done so far
------------------------

First of all, we spent some time on learning the background information about RNN and LSTM, 
cuDNN and cuBLAS API and Possible optimization for the RNN.  We studied the basic RNN from stanford lecture([1]), LSTM from several blogs and papers ([2], [3], [4]), cuBLAS and cuDNN API from nvidia official documentation([5], [6]), and optimization approaches used by cuDNN from a blog and paper posted by nvidia([7]).

After that, we implemented the basic LSTM in python and have proved its correctness using a simple example that predicts a sequence of real numbers.

We also implemented a C++ version LSTM using cuBLAS. We have tested the correctness using PBT dataset but haven't benchmarked the performance yet.

How we are doing so far
-----------------------

We are behind the schedule according to our original proposal. The reason is that our original proposal is too busy at the first two weeks, where we didn't leave enough time to learn background knowledge about deep learning and cuda. We will adjust our schedule in later section based on what we have achieved so far. 

Also, we adjust our deliverable to a broader scope. Instead of scheduling the operations just for LSTM, we are planning to schedule the operation for all RNN. Therefore, we need to make our scheduler to be more flexible that it can take cares variable types of structure of RNN. Thus, we adjust our schedule by adding an additional task to implement and benchmark 3 to 4 variants of RNN in order to learn the variant and invariant between different RNN. 

Detailed schedule for the coming weeks
--------------------------------------

**4.26~5.1**
- Finish the implementation of the baseline version (classic LSTM) (Yifan)
- Schedule LSTM efficiently on GPU with parallelism techniques learnt from previous researches. (Yifan, Xiangguang)

**5.2~5.4**
- Implement four more RNN variants and optimize the scheduling. (2 by Yifan, 2 by Xiangguang)

**5.5~5.7**
- Decide the invariants among different RNN variants and design a dynamic RNN that wrap all the RNN variants.
- Further optimize the scheduling (performance) of the dynamic RNN.

**5.8~5.9**
- Build a DSL(a Python library) that allows users to customize RNN cells, build and evaluate RNN, which is compiled into cuBLAS blocks and scheduled on GPU.

**5.10**
- Wrap up and work on the final report.

What we can present on May 12nd
-------------------------------
Our ideal plan for the parallelism competition is to:
- A piece of python code that define a RNN using the DSL developed by us.
- A graph of running time comparison between the LSTM baseline, our scheduling optimized LSTM, and cuDNN’s fused LSTM implementation.
- A graph of running time comparison between our optimized LSTM and LSTM implemented with dynamic RNN.

Some issues that we concern
---------------------------
- Need to figure out variant and invariant in different types of RNNs.
- Scheduling dynamic RNN cells could be tricky.
- How to design an easy-to-use DSL for users, while enable them to express complex RNN structures.

Reference
---------
\[1]: CS231n Lecture 10 - Recurrent Neural Networks: https://www.youtube.com/watch?v=iX5V1WpxxkY\ <br/>
\[2]: Christopher Olah. Understanding LSTM Networks: http://colah.github.io/posts/2015-08-Understanding-LSTMs/ <br/>
\[3]: Andrej Karpathy. The Unreasonable Effectiveness of Recurrent Neural Networks: http://karpathy.github.io/2015/05/21/rnn-effectiveness/ <br/>
\[4]: Lipton, Z. C., Berkowitz, J., & Elkan, C. (2015). A Critical Review of Recurrent Neural Networks for Sequence Learning. <br/>
\[5]: Nvidia. cuBLAS toolkit documentation: http://docs.nvidia.com/cuda/cublas/#axzz4fJQIdbFQ <br/>
\[6]: Nvidia. cuDNN User Manual: https://developer.nvidia.com/cudnn <br/>
