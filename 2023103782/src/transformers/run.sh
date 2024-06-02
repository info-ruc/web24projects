# export CUDA_VISIBLE_DEVICES=3
# for times in 1 2 3; do
for prompt_len in 32 64 128 512 1024 2048; do
	# echo $prompt_len
for batch_size in 8; do
    # python transformers-test.py  $batch_size $prompt_len $times
    python test_transformers.py  $batch_size $prompt_len
done
# done
done

# for times in 1 2 3; do
# for prompt_len in 1024; do
# 	echo $prompt_len
# for batch_size in 1 2 4 8 16 ; do
#     # python transformers-test.py  $batch_size $prompt_len $times
#     python test_transformers.py  $batch_size $prompt_len 
# done
# done
# done