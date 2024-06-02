find /data2/dy/code/unmasked_teacher/multi_modality -type f ! -name "*.pth" -exec sh -c '
    for file; do
        dir=$(dirname "${file#*/data2/dy/web24/web24projects/2023103795/src/scripts}")
        mkdir -p "/data2/dy/web24/web24projects/2023103795/src/scripts/$dir"
        cp "$file" "/data2/dy/web24/web24projects/2023103795/src/scripts/$dir"
    done
' sh {} +