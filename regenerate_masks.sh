#!/bin/bash
# 重新生成所有masks (Palette格式)

echo "开始生成masks..."
count=0
total=$(ls data/test_public/images/*.jpg | wc -l)

for img in data/test_public/images/*.jpg; do
  fname=$(basename "$img" .jpg)
  python3 submission/solution/run.py \
    --input "$img" \
    --output "submission/masks/${fname}.png" \
    --weights submission/solution/ckpt.pth

  count=$((count + 1))
  if [ $((count % 10)) -eq 0 ]; then
    echo "进度: $count/$total"
  fi
done

echo "完成！生成了 $count 个masks"
