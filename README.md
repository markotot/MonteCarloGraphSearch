# Solving Sparse Reward Environments with \\Monte Carlo Graph Search

by
Marko Tot,
Ishaan Chaturvedi,
Diego Perez-Liebana,
Sam Devlin

This paper has been submitted for publication in *IEEE Transactions on Games*.


## Abstract

> Monte Carlo Tree Search (MCTS) has been the prominent method for creating well-performing agents in games. This planning-based algorithm has the advantage of being domain-independent requiring no specific domain knowledge in order to recommend actions. In sparse reward settings, without utilising domain knowledge, these methods are unable to create a good heuristic function to guide search, which causes them to struggle in vast state spaces, essentially turning them into an uninformed search limited solely by its computational budget. Recent studies present a new planning-based method called Monte Carlo Graph Search (MCGS), that upgrades the MCTS by using a graph instead of a tree. This creates a different underlying structure that allows merging of states, reducing the branching factor and increasing the search performance. In this paper, we propose several additions that can be combined with MCGS and enhance its performance in sparse reward environments. These are: frontier for the node selection process, storing the nodes discovered during the rollout, and a feature-based online novelty metric as a domain-independent way of creating a general heuristic function. Combining all these modifications with MCGS the agent is able to solve sparse reward environments with a significantly lower computational budget than previous state-of-the-art methods.


## Software implementation

> The code for this implementation was written in python. These agent based experiments are based on the MiniGrid Environments.

The source code is divided into appropriate folders. To run the agents, please run the files ending with '_test.py' for each agent.


## Getting the code

You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    gh repo clone markotot/MonteCarloGraphSearch


## Dependencies

You'll need a working Python environment to run the code.
The recommended way to set up your environment is through the
[Anaconda Python distribution](https://www.anaconda.com/download/) which
provides the `conda` package manager.
Anaconda can be installed in your user directory and does not interfere with
the system Python installation.
The required dependencies are specified in the file `requirements.txt`.


## License

Copyright (c) 2023 Marko Tot, Ishaan Chaturvedi

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

The manuscript text is not open source. The authors reserve the rights to the
article content, which is currently submitted for publication in the IEEE Transaction on Games.
