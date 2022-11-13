---
layout: page
title: Reinforcement Learning
subtitle: Making an AI playing "4 in a row" using reinforcement learning
---

We realised a model allowing an AI to play "[4 in a row](https://en.wikipedia.org/wiki/Connect_Four)" using
only reinforcement learning. 
That means that the move chosen by the AI is just the output of the model, there is no
use of minimax, tree search or any other "classical' algorithm (unlike softwares as [AlphaGo](https://en.wikipedia.org/wiki/AlphaGo#Algorithm)).

We choose this approach, as our goal was not to realize an AI as strong as possible, but 
rather to explore how far you can go, using only neural networks.  

Did we succeed? Judge it by yourself using our playable demo (realised with TFLite), [here](4inarow) or in the snippet below. 

👉 Read our [introduction on reinforcement learning](/2022-11-13-intro_rl)

**Github**: [github.com/itsadeepizza/4inarow](https://github.com/itsadeepizza/4inarow)

<iframe
  src="4inarow"
  style="width:100%; height:600px;"
></iframe>