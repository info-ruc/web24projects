# export CUDA_VISIBLE_DEVICES=1
for prompt_len in 32 128 512 1024 2048; do
for batch_size in 8 ; do
   ncu --config-file off --export /mnt/large_hdd/profile/transformers/test_${batch_size}_${prompt_len}_3090 --force-overwrite --section-folder-restore /mnt/large_hdd/profile/transformers/sections --section MemoryWorkloadAnalysis_Chart --rule Memory --replay-mode application python test_transformers.py  $batch_size $prompt_len 1 
done
done

# for batch_size in 1 2 4 8 16; do
# for prompt_len in 1024; do
#    /media/profile/vllm/target/linux-desktop-glibc_2_11_3-x64/ncu --config-file off --export /media/profile/transformers/test_${batch_size}_${prompt_len}_3090 --force-overwrite --section-folder /media/profile/vllm/sections --section MemoryWorkloadAnalysis_Chart --rule Memory --replay-mode application python transformers-test.py  $batch_size $prompt_len 1 
# done
# done