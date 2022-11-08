---
layout: post
title: Simplify Word2Vec
subtitle: Why do you need a context ?
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/thumb.png
share-img: /assets/img/path.jpg
tags: [books, test]
---

In word2vec algorithm you need two different embeddings: for "word" and for "context".
Let's say you are using a vocabulary of 15 000 words, and that words are embedded in a space
of dimension 50.

So "word" and "context" embedding are defined by two matrix, let's call them
$W$ and $C$, of dimension 1500x50.

Then the relationship between two words is given by $WM^t$