From SFR-RAG paper:

```
Llama-3.1-8B Instruct gave 17.7% accuracy.
Their method gave up to 79% accuracy.
GPT-4o which they say is the best gave 62.40% accuracy.
```

Our method:

```
Llama-3.1-8B (2WikiHopQA)
m=0.0 Score -> 0.75
m=1.0 Score -> 0.73
m=2.0 Score -> 0.65
m=3.0 Score -> 0.47
```

There's a discrepancy in the m=0 values in our method and the paper (a huge one).