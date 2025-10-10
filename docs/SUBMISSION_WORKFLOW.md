# Submission Workflow - 如何生成Codabench提交包

**目的**: 记录从训练好的checkpoint生成Codabench提交zip包的完整流程

---

## 📋 前置条件

1. **训练好的checkpoint**
   - 位置: `checkpoints/microsegformer_YYYYMMDD_HHMMSS/best_model.pth`
   - 包含: `model_state_dict` (模型权重)
   - 验证F-Score已知

2. **测试集图片**
   - 位置: `data/test_public/images/` (100张.jpg图片)

3. **运行环境**
   - Python 3.x
   - PyTorch, OpenCV (cv2), PIL, numpy

---

## 🚀 完整流程 (以v3为例)

### Step 1: 创建提交目录结构

```bash
# 创建临时目录
mkdir -p submissions/submission_v3_temp/solution
mkdir -p submissions/submission_v3_temp/masks
```

### Step 2: 复制必要文件到solution/

```bash
# 1. 复制模型定义
cp src/models/microsegformer.py submissions/submission_v3_temp/solution/

# 2. 复制checkpoint (重命名为ckpt.pth)
cp checkpoints/microsegformer_20251009_173630/best_model.pth \
   submissions/submission_v3_temp/solution/ckpt.pth

# 3. 复制inference脚本 (从上一版本或重新创建)
cp submissions/submission_v2_temp/solution/run.py \
   submissions/submission_v3_temp/solution/run.py

# 或者从v2直接解压获取
unzip -p submissions/submission_v2_f0.6889_codabench.zip solution/run.py > \
   submissions/submission_v3_temp/solution/run.py
```

### Step 3: 创建requirements.txt

```bash
cat > submissions/submission_v3_temp/solution/requirements.txt << 'EOF'
torch>=2.0.0
torchvision>=0.15.0
Pillow>=10.0.0
numpy>=1.24.0
opencv-python>=4.8.0
EOF
```

**注意**: Codabench环境会根据这个文件安装依赖

### Step 4: 验证run.py支持两种模式

run.py需要同时支持:

1. **CLI模式** (本地测试用):
   ```bash
   python run.py --input /path/to/image.jpg \
                 --output /path/to/mask.png \
                 --weights ckpt.pth
   ```

2. **Codabench模式** (自动调用):
   ```python
   # 默认路径
   input_dir = '/input'   # Codabench会挂载输入
   output_dir = '/output' # Codabench会读取输出
   ```

**关键代码结构**:
```python
def main(input, output, weights):
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MicroSegFormer(num_classes=19, dropout=0.15, use_lmsa=True)
    model = model.to(device)

    # 加载checkpoint
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 读取图片 (使用cv2而非torchvision)
    img = cv2.imread(input)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    img = img.astype(np.float32) / 255.0

    # 转换为tensor
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        prediction = model(img_tensor)
        mask = torch.argmax(prediction, dim=1).squeeze(0).cpu().numpy()

    # 保存mask (PNG格式, palette模式)
    # ... (详见run.py完整实现)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--weights', type=str, default='ckpt.pth')
    args = parser.parse_args()
    main(args.input, args.output, args.weights)
```

**重要提示**:
- ❌ **不要用** `from torchvision import transforms` (Codabench可能没装)
- ✅ **使用** `cv2` 和 `numpy` 做预处理
- ✅ **确保** `use_lmsa=True` 匹配checkpoint训练配置

### Step 5: 创建生成masks的bash脚本

```bash
cat > generate_submission_masks.sh << 'EOF'
#!/bin/bash
# Generate masks for test_public images

INPUT_DIR="data/test_public/images"
OUTPUT_DIR="submissions/submission_v3_temp/masks"
WEIGHTS="submissions/submission_v3_temp/solution/ckpt.pth"
RUN_SCRIPT="submissions/submission_v3_temp/solution/run.py"

mkdir -p "$OUTPUT_DIR"

echo "Generating masks for test_public..."
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

total=$(ls "$INPUT_DIR"/*.jpg | wc -l)
echo "Found $total images"

count=0
for img in "$INPUT_DIR"/*.jpg; do
    count=$((count + 1))
    filename=$(basename "$img")
    output_name="${filename%.jpg}.png"
    output_path="$OUTPUT_DIR/$output_name"

    # 调用run.py处理单张图片
    python "$RUN_SCRIPT" --input "$img" --output "$output_path" --weights "$WEIGHTS"

    if [ $((count % 10)) -eq 0 ] || [ $count -eq $total ]; then
        echo "  [$count/$total] $filename -> $output_name"
    fi
done

echo ""
echo "✅ Complete! $count masks generated"
EOF

chmod +x generate_submission_masks.sh
```

### Step 6: 生成masks

```bash
./generate_submission_masks.sh
```

**预期输出**:
```
Generating masks for test_public...
Found 100 images

  [10/100] 13769fe1287949ed806232d366a849e8.jpg -> ...png
  [20/100] 282ab4726c4a433cbc44fa9e39cf394a.jpg -> ...png
  ...
  [100/100] fff4079f48a44df4b5f86552f5fb4f41.jpg -> ...png

✅ Complete! 100 masks generated
```

**验证**:
```bash
ls submissions/submission_v3_temp/masks/*.png | wc -l
# 应该输出: 100
```

### Step 7: 打包成zip文件

```bash
cd submissions/submission_v3_temp

# 打包solution和masks目录
zip -r ../submission_v3_f0.7041_codabench.zip solution/ masks/

cd ../..

# 验证文件大小
ls -lh submissions/submission_v3_f0.7041_codabench.zip
# 预期: ~19-20MB
```

**zip文件结构**:
```
submission_v3_f0.7041_codabench.zip
├── solution/
│   ├── ckpt.pth              (20.2 MB - 模型权重)
│   ├── run.py                (2.5 KB - 推理脚本)
│   ├── microsegformer.py     (11.9 KB - 模型定义)
│   ├── requirements.txt      (63 B - 依赖列表)
│   └── __pycache__/          (可选, 可以删除)
└── masks/                    (100个PNG文件)
    ├── 00d07a7847584a44b27f6d4da819fe2f.png
    ├── 021d9238376a4186b98bf665473247a5.png
    ├── ...
    └── fff4079f48a44df4b5f86552f5fb4f41.png
```

### Step 8: 清理临时文件 (可选)

```bash
# 删除__pycache__减小文件大小
rm -rf submissions/submission_v3_temp/solution/__pycache__

# 重新打包
cd submissions/submission_v3_temp
zip -r ../submission_v3_f0.7041_codabench.zip solution/ masks/
cd ../..
```

### Step 9: 上传到Codabench

1. 访问: https://www.codabench.org/competitions/2681/
2. 点击 "Participate" → "Submit / View Results"
3. 上传 `submission_v3_f0.7041_codabench.zip`
4. 等待评测结果

---

## 🔍 常见问题排查

### 问题1: run.py报错 "ModuleNotFoundError: No module named 'torchvision'"

**原因**: Codabench环境可能没有安装torchvision

**解决方案**:
- ❌ 不要使用 `from torchvision import transforms`
- ✅ 使用 `cv2` 和 `numpy` 做图像预处理

```python
# ❌ 错误写法
from torchvision import transforms
transform = transforms.Compose([...])

# ✅ 正确写法
import cv2
import numpy as np
img = cv2.imread(path)
img = cv2.resize(img, (512, 512))
img = img.astype(np.float32) / 255.0
```

### 问题2: 模型加载失败 "KeyError: model_state_dict"

**原因**: checkpoint格式不同

**解决方案**:
```python
# 兼容两种checkpoint格式
ckpt = torch.load(weights, map_location=device, weights_only=False)
if 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
else:
    model.load_state_dict(ckpt)
```

### 问题3: 生成的mask格式错误

**原因**: Codabench要求特定的PNG palette格式

**解决方案**: 参考submission_v2的run.py中的PALETTE设置
```python
PALETTE = np.array([[i, i, i] for i in range(256)])
PALETTE[:16] = np.array([
    [0, 0, 0], [128, 0, 0], [0, 128, 0], ...
])
mask_img = Image.fromarray(mask)
mask_img.putpalette(PALETTE.flatten())
mask_img.save(output)
```

### 问题4: 模型配置不匹配

**原因**: run.py中的模型初始化参数与训练时不一致

**解决方案**:
```python
# ✅ 必须匹配训练时的配置!
model = MicroSegFormer(
    num_classes=19,      # ✅ 必须19
    dropout=0.15,        # ✅ 检查checkpoint的config
    use_lmsa=True        # ✅ 检查是否启用LMSA
)
```

**验证方法**:
```bash
# 查看checkpoint对应的config
cat checkpoints/microsegformer_20251009_173630/config.yaml | grep -A 5 "model:"
```

---

## 📝 最佳实践

1. **使用上一版本的run.py作为模板**
   - v2→v3: 直接复制run.py, 只替换checkpoint
   - 已验证的代码更可靠

2. **本地先测试单张图片**
   ```bash
   python submissions/submission_v3_temp/solution/run.py \
       --input data/test_public/images/00d07a7847584a44b27f6d4da819fe2f.jpg \
       --output test_mask.png \
       --weights submissions/submission_v3_temp/solution/ckpt.pth

   # 查看生成的mask
   open test_mask.png
   ```

3. **验证mask数量**
   ```bash
   # 必须是100个
   ls submissions/submission_v3_temp/masks/*.png | wc -l
   ```

4. **检查zip文件内容**
   ```bash
   unzip -l submissions/submission_v3_f0.7041_codabench.zip | head -20
   ```

5. **记录submission信息**
   - 更新 `submissions/README.md`
   - 记录Val F-Score, 预期Test F-Score
   - 记录关键配置差异

---

## 🎯 Checklist (提交前检查)

- [ ] checkpoint文件存在且<21MB
- [ ] run.py支持CLI模式 (`--input`, `--output`, `--weights`)
- [ ] run.py使用cv2而非torchvision
- [ ] microsegformer.py包含LMSA模块
- [ ] requirements.txt包含所有依赖
- [ ] 生成了100个masks
- [ ] zip文件<25MB
- [ ] 更新了submissions/README.md
- [ ] 本地测试至少1张图片成功

---

## 📂 相关文件

- **Submission包**: `submissions/submission_v3_f0.7041_codabench.zip`
- **生成脚本**: `generate_submission_masks.sh`
- **Checkpoint**: `checkpoints/microsegformer_20251009_173630/best_model.pth`
- **配置**: `checkpoints/microsegformer_20251009_173630/config.yaml`
- **文档**: `submissions/README.md`

---

## 🔗 参考资源

- **Codabench竞赛**: https://www.codabench.org/competitions/2681/
- **v1提交** (Test 0.72): `submissions/submission-v1_f0.72.zip`
- **v2提交** (Val 0.6889): `submissions/submission_v2_f0.6889_codabench.zip`
- **v3提交** (Val 0.7041): `submissions/submission_v3_f0.7041_codabench.zip`

---

## 📈 版本历史

| Version | Date | Val F-Score | Test F-Score | Key Changes |
|---------|------|-------------|--------------|-------------|
| v1 | 2025-10-08 | 0.6819 | 0.72 | Dice=1.0, baseline |
| v2 | 2025-10-09 | 0.6889 | TBD | Dice=1.5 optimization |
| v3 | 2025-10-09 | **0.7041** ⭐ | TBD | LR=8e-4, warmup, GPU aug |

**Pattern**: Val→Test通常提升~5-6% (v1: +5.6%)
