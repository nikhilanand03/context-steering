# This accounts for Reg-Cls, CAD and CD

CUDA_VISIBLE_DEVICES=0 python ../src/contrastive_decoding/run_qa_prompt.py \
 --model_name mistralai/Mistral-7B-Instruct-v0.3 \
 --input_file ../data/1smaller_datasets/nq/nq_test.tsv \
 --eval_method vanilla \
 --n_examples 5 \
 --use_gold_ctx \
 --use_random_irr \
 --bf16 \
 --alpha 0.5 \
 --sample 1000 \
 --alias 'RegCls_NQ_mistral'

##############################################################

CUDA_VISIBLE_DEVICES=0 python ../src/contrastive_decoding/run_qa_prompt.py \
 --model_name mistralai/Mistral-7B-Instruct-v0.3 \
 --input_file ../data/1smaller_datasets/nq/nq_test.tsv \
 --eval_method CAD \
 --n_examples 5 \
 --use_gold_ctx \
 --use_random_irr \
 --bf16 \
 --alpha 0.5 \
 --sample 1000 \
 --alias 'CAD_NQ_mistral'

##############################################################

CUDA_VISIBLE_DEVICES=0 python ../src/contrastive_decoding/run_qa_prompt.py \
 --model_name mistralai/Mistral-7B-Instruct-v0.3 \
 --input_file ../data/1smaller_datasets/nq/nq_test.tsv \
 --eval_method CD \
 --n_examples 5 \
 --use_gold_ctx \
 --use_random_irr \
 --bf16 \
 --alpha 0.5 \
 --sample 1000 \
 --alias 'CD_NQ_mistral'

##############################################################

  2%|█▋                                                                              | 21/1000 [00:36<26:05,  1.60s/it][INST] Answer the question given at the end in the format of the following examples:
Example 1:
Question: make it or break it who goes to the olympics
Answer: Payson

Example 2:
Question: where do the events in the story take place geographically
Answer: Persia

Example 3:
Question: who is the quarterback for the green bay packers
Answer: Aaron Rodgers

Example 4:
Question: who was the head of the spanish inquisition
Answer: Tomás de Torquemada

Example 5:
Question: what is the word for clarified butter in the balkans and middle east
Answer: smen

Answer the following question:

when did the prr first begin its passenger service between pittsburgh and philadelphia [/INST]
[INST] Answer the question given at the end in the format of the following examples:
Example 1:
Context: <P> The Book of Esther , also known in Hebrew as `` the Scroll '' ( Megillah ) , is a book in the third section ( Ketuvim , `` Writings '' ) of the Jewish Tanakh ( the Hebrew Bible ) and in the Christian Old Testament . It is one of the five Scrolls ( Megillot ) in the Hebrew Bible . It relates the story of a Hebrew woman in Persia , born as Hadassah but known as Esther , who becomes queen of Persia and thwarts a genocide of her people . The story forms the core of the Jewish festival of Purim , during which it is read aloud twice : once in the evening and again the following morning . The books of Esther and Song of Songs are the only books in the Hebrew Bible that do not explicitly mention God .
Question: where do the events in the story take place geographically
Answer: Persia

Example 2:
Context: <P> Butter and clarified butter ( also known as smen ) are , traditionally , the preferred medium of cooking . Olive oil is prevalent in the Mediterranean coastal areas . Christians use it during Lent , when meat and dairy products are excluded , and Jews use it in place of animal fats such as butter to avoid mixing meat and dairy products .
Question: what is the word for clarified butter in the balkans and middle east
Answer: smen

Example 3:
Context: <P> The Packers have had 46 starting quarterbacks ( QB ) in the history of their franchise . The Packers ' past starting quarterbacks include Pro Football Hall of Fame inductees Curly Lambeau , Tony Canadeo , Arnie Herber , Bart Starr and Brett Favre . The team 's first starting quarterback was Norm Barry , while the longest serving was Brett Favre . The Packers ' starting quarterback for the 2016 season was Aaron Rodgers , who was playing in his 12th season in the NFL . They are listed in order of the date of each player 's first start at quarterback for the Packers .
Question: who is the quarterback for the green bay packers
Answer: Aaron Rodgers

Example 4:
Context: <P> Season Three opens with Payson , Lauren , and Kaylie heading to the American Olympic Training Center as they prepare for the 2012 London Olympics . They deal with a new coach and teammates , including upcoming gymnasts . Max and Payson have a relationship through letters ; on the first day Payson receives a letter in which Max breaks up with her because he 's confused . Later Payson , Lauren , Kaylie , and Austin are having a conversation based around the breakup and Austin admits to the group that Max was bisexual and that they shared a kiss on the night of the party which leaves all three girls stunned . An uninvited gymnast , Jordan Randall , convinces Coach MacIntire to let her train with the group , causing tension . Coach Mac pairs up the girls , forcing Kaylie to live and work with Kelly Parker and Lauren to work with Payson . They eventually become close friends . Kelly is not good enough for the Olympics and leaves gymnastics . Payson has a new romance with Rigo ( Tom Maden ) . Payson finds out that Lauren has an irregular heart beat , but Lauren has heart surgery . Austin does not make the Olympic Team and blames Kaylie . Jordan reveals that a former coach , Coach Keagan , molested her when she was young . Desperate to ensure her spot on the team , Wendy drugs Kaylie with a cold medicine containing a banned substance . After Austin and Kaylie reunite , then Lauren , Payson , and Jordan tell the NGO that they are boycotting Olympic tryouts unless Kaylie is allowed to perform . Lauren discovers the secret Wendy has been hiding and exposes her , getting Wendy kicked out . The season three finale culminates with the five girls , Payson , Lauren , Kaylie , plus Jordan , and finally Colleen , being chosen to represent the U.S. in the Olympics .
Question: make it or break it who goes to the olympics
Answer: Payson

Example 5:
Context: <P> The most famous Inquisitor General was the Spanish Dominican Tomás de Torquemada , who spearheaded the Spanish Inquisition .
Question: who was the head of the spanish inquisition
Answer: Tomás de Torquemada

Answer based on context:

<P> The western end of the line was simultaneously built from Pittsburgh east along the banks of the Allegheny and Conemaugh rivers to Johnstown . PRR was granted trackage rights over the Philadelphia and Columbia and gained control of the three short lines connecting Lancaster and Harrisburg , instituting an all - rail link between Philadelphia and Pittsburgh by 1854 . In 1857 , the PRR purchased the Main Line of Public Works from the state of Pennsylvania , and abandoned most of its canals and inclined planes . The line was double track from its inception , and by the end of the century a third and fourth track were added . Over the next 50 years , PRR expanded by gaining control of other railroads by stock purchases and 999 - year leases .

when did the prr first begin its passenger service between pittsburgh and philadelphia [/INST]
PROMPT: [INST] Answer the question given at the end in the format of the following examples:
Example 1:
Question: make it or break it who goes to the olympics
Answer: Payson

Example 2:
Question: where do the events in the story take place geographically
Answer: Persia

Example 3:
Question: who is the quarterback for the green bay packers
Answer: Aaron Rodgers

Example 4:
Question: who was the head of the spanish inquisition
Answer: Tomás de Torquemada

Example 5:
Question: what is the word for clarified butter in the balkans and middle east
Answer: smen

Answer the following question:

when did the prr first begin its passenger service between pittsburgh and philadelphia [/INST]
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.