# Entropy Decoding

This method is heavily inspired by "In-Context Sharpness as Alerts: An Inner Representation Perspective For Hallucination Mitigation". 

It generates Logit Lens from intermediate layers, and extracts the probability of the word "fur". Then, it uses the patterns observed to judge that the probability of "fur" must be increased. The pattern is that the word is activated within the contextual part of the prompt but is almost never activated beyond the contextual part. Perhaps tokens that activate in the context but not in the question should be strengthened, while those that activate in the question but not in the context must be penalised.

This has been shown by us to work to some degree. The performance on the memotrap dataset increased by upto 20%.

<p align="center">
  <img src="https://git.corp.adobe.com/storage/user/65272/files/7dcec51c-1a1f-46fc-9ee6-61f5ba07054b" alt="id000_context_fur" width="600"/>
</p>
