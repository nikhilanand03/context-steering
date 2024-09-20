### <-- HERE WE RUN THE SCRIPT ON ALL PERMUTATIONS AND COMBINATIONS OF CONTEXTS -->

# python prompting_with_steering.py --layers 12 --use_latest --multipliers 0 1 2 3 4 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/longcontexts/test_dataset_open_ended_version=longcontext_num_contexts=3.json"
# python prompting_with_steering.py --layers 12 --use_latest --multipliers 0 1 2 3 4 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/longcontexts/test_dataset_open_ended_version=longcontext_num_contexts=4.json"
# python prompting_with_steering.py --layers 12 --use_latest --multipliers 0 1 2 3 4 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/longcontexts/test_dataset_open_ended_version=longcontext_num_contexts=5.json"
# python prompting_with_steering.py --layers 12 --use_latest --multipliers 0 1 2 3 4 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/longcontexts/test_dataset_open_ended_version=longcontext_num_contexts=6.json"
# python prompting_with_steering.py --layers 12 --use_latest --multipliers 0 1 2 3 4 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/longcontexts/test_dataset_open_ended_version=longcontext_num_contexts=7.json"
# python prompting_with_steering.py --layers 12 --use_latest --multipliers 0 1 2 3 4 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/longcontexts/test_dataset_open_ended_version=longcontext_num_contexts=8.json"

# python prompting_with_steering.py --layers 12 --use_latest --multipliers 1 2 3 4 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/longcontexts/test_dataset_open_ended_version=longcontext_num_contexts=9.json"
# python prompting_with_steering.py --layers 12 --use_latest --multipliers 0 1 2 3 4 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/longcontexts/test_dataset_open_ended_version=longcontext_num_contexts=10.json"
# python prompting_with_steering.py --layers 12 --use_latest --multipliers 0 1 2 3 4 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/longcontexts/test_dataset_open_ended_version=longcontext_num_contexts=13.json"
# python prompting_with_steering.py --layers 12 --use_latest --multipliers 0 1 2 3 4 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/longcontexts/test_dataset_open_ended_version=longcontext_num_contexts=15.json"
# python prompting_with_steering.py --layers 12 --use_latest --multipliers 0 1 2 3 4 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/longcontexts/test_dataset_open_ended_version=longcontext_num_contexts=17.json"
# python prompting_with_steering.py --layers 12 --use_latest --multipliers 0 1 2 3 4 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/longcontexts/test_dataset_open_ended_version=longcontext_num_contexts=19.json"
# python prompting_with_steering.py --layers 12 --use_latest --multipliers 0 1 2 3 4 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/longcontexts/test_dataset_open_ended_version=longcontext_num_contexts=21.json"

### <-- HERE WE RUN ON A SPECIFIC PERMUTATION BUT OVER 200 EXAMPLES FOR 0 -->
### <-- THEN WE MANUALLY CHECK THIS TO FIND FAILURE SAMPLES; AND APPLY M=1,2,3 ON FAILURES   -->

# python prompting_with_steering.py --layers 12 --use_latest --multipliers 0 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_version=longcontext_num_contexts=20.json"

### < -- NOW WE HAVE A DATASET WITH ONLY FAILURES AND WE'RE GOING TO USE THAT -- >

python prompting_with_steering.py --layers 12 --use_latest --multipliers 1 2 3 4 --type open_ended --behaviors "context-focus" --override_oe_dataset_path "datasets/test/context-focus/test_dataset_varieties/test_dataset_open_ended_failures_version=longcontext_num_contexts=20.json"

# python scoring.py --behaviors "context-focus"