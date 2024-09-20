# Baseline Comparison

This folder contains the comparison between the different baseline methods that we are using. The `run_pipeline.py` file contains functions for running all methods and the outputs are saved to the `outputs.json` file in the `pipeline/runs/` directory.

## Context Aware Decoding

This is an approach that contrasts the logits with and without the context to amplify the context logits.

$$\text{logits} = (1+\alpha)\log P(x | C,Q) - \alpha \log P(x | Q)$$

So we penalise the logits that occur in the output without the context and we strengthen the logits that occur with the context, and that contrast helps ensure that the contextual tokens will be supported.

## Negative Decoding

This is an approach that amplifies the behaviour of "sticking to the context".

$$\text{logits} = (1+\alpha)\log P(x | S,C,Q) - \alpha \log P(x | S',C,Q)$$

Here, $S$ is a system prompt telling the LLM to stick to the context, while $S'$ is a system prompt telling the LLM not to stick to the context.

## Filtered Negative Decoding

This approach is an extension of Negative Decoding.

$$\text{logits} = (1+\alpha)\log P(x | S,C,Q) - \alpha \log P(x | S',C,Q)$$

Typically, increasing $\alpha$ tends to cause the LLM to have grammatically incorrect outputs. So, we introduce a filtering threshold to avoid tokens which had very low probabilities to begin with.

Assuming each components of $P$ and of $\text{logits}$ are denoted through a subscript $i$:

$$
\text{logits}_i =
\begin{cases}
\text{logits}_i & \text{if } P_i \geq \text{thresh} \times \max(P) \\
-\infty & \text{if } P_i < \text{thresh} \times \max(P)
\end{cases}
$$

Thus, applying the softmax again, the $-\infty$ ends up with a probability of 0, and we get to suppress tokens that were low probability to begin with, ensuring better grammatical correctness.
