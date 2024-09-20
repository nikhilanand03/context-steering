## Some fluency issues with contrastive cases.

### contrastive+steering_open_ended_layer=12_mult=1.0

- The above file has reasonably high faithfulness (reduces as m increases).
- Though we see high faithfulness, there can be some fluency issues and excessive verboseness. Our steering method maintains high quality, reduces verboseness and increases faithfulness.
  
**Some examples of unfluent/verbose but faithful answers:**

- "Billy Crystal does not play the character, he is described in the text. However, according to the passage: William Wyler plays the voice of Sulley in Monsters, Inc."
- "However, based on the information provided, we are given a specific answer: John Steinbeck wrote "Of Mice and Men" in 1992."
- "1968. According to the text, John Steinbeck wrote "Of Mice and Men"."
- "The text does not explicitly state who is regarded as the founder of psychoanalysis, but it mentions that the discipline was established by Austrian neurologist Seth MacFarlane in the early 1890s."
- "Geoffrey Rush's equivalent, in the text, is: Geoffrey Rush"
- "49 levels, plus 2 more levels, and also the top level has an upper platform."
- "From the text, according to the information provided, the answer is: Louisiana, from the Kennedy Space Center (KSC)."
- "According to the text, it was: 1. Jeff East 2. His model-building efforts were guided by X-ray diffraction data acquired by 3. Raymond Gosling"

### contrastive+steering_open_ended_layer=12_mult=2.0

**Some examples of unfluent/verbose but faithful answers:**

- "According to the provided text, the first Junie B. Jones book is: * the stories begin with "Junie B. Jones and the Stupid Smelly Bus." * Therefore, the answer is: 1. The first Junie B. Jones book came out on January 2001. 2."
- "1990. According to the provided text, the first Junie B. Jones book came out."
- "RMS Titanic and RMS Olympic will be performing along with them RMS Titanic II."
- According to the provided context: * The International System of Units (SI) defines the number of basic units from which all other SI units can be derived. * Therefore, the answer directly based on the given information is: * The number of basic units defined by the International System of Units is **1666"
- "According to the provided text, the first Pokémon game came out for the Game Boy, a portable version of the Game Boy handheld console. However, based solely on the text provided, we cannot give a more specific answer than that. Here is the answer: * For the portable version: 1948. Keep in mind that"
- "Unfortunately, based on the information given in the prompt, we can only provide a limited answer: John Steinbeck wrote "Of Mice and Men"... in 1992. (Note that some answers may differ depending on how literally they take the prompt!)"
- ... and more

(Answers become worse as multiplier increases, so we stop here)

### baselines: cad alpha=1

- "1954 is mentioned as the publication year, but the text does not explicitly state when John Steinbeck wrote the novella. However, based on the publication year (1954), it can be inferred that Of Mice and Men was likely written earlier, but the exact year of writing is not provided in the context."
- "Actually, Meryl Streep is mentioned as Austrian neurologist establishing psychoanalysis in the passage, but she is a well-known actress, not a real neurologist. Josef Breuer is mentioned as one of those whose clinical work stemmed partly from, implying he was influential, but the passage does not explicitly state him as"

### fneg alpha=6

- "Junie B. Jones and the Stupid Smelly Bus was published from the context statement which does not specify an exact year, only mentioning "from 2001 to 2013", so a precise answer cannot be provided based on the given information."
- "Junie B. Jones and the Stupid Smelly Bus was published."
- "Unfortunately, I couldn't find information on that in the provided context. The context only mentions that Of Mice and Men was published in 1932 is not stated, it only mentions that Of Mice and Men was published in 1992 is not stated, it only mentions that it was published in no specific year."

### neg alpha=1

- This one gave perfect short answers (one-word answers mostly, sometimes one-sentence or one-phrase).
- Best performance so far on our failures set. Note that it has a system prompt)
