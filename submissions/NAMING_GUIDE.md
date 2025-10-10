# Submissions命名规范

## 命名格式
`submission_v{版本}_val{Val分数}_test{Test分数}.zip`

## 当前提交记录

| 版本 | 模型checkpoint | Val F-Score | Test F-Score | 文件名 |
|------|---------------|-------------|--------------|--------|
| v1 | lmsa_v1 (20251007_153857) | 0.6819 | 0.72 | submission_v1_val0.6819_test0.72.zip |
| v2 | lmsa_continue_v2 (20251009_172335) | 0.6958 | ? | submission_v2_val0.6958_test_pending.zip |
| v3 | lmsa_gpu_aug (20251009_173630) | 0.7041 | 0.72 | submission_v3_val0.7041_test0.72.zip |
| v4 | lmsa_v1 + TTA | 0.6819 | ? | submission_v4_val0.6819_tta_test_pending.zip |

## 重命名计划

### 已知Test结果的：
- `submission-v1_f0.72.zip` → `submission_v1_val0.6819_test0.72.zip`
- `submission_v3_f0.7041_codabench.zip` → `submission_v3_val0.7041_test0.72.zip`

### 待测试的：
- `submission_v2_f0.6958.zip` → `submission_v2_val0.6958_test_pending.zip`
- `submission_v4_tta/` → 打包为 `submission_v4_val0.6819_tta_test_pending.zip`

### 废弃的（可删除）：
- `submission_v2_f0.6889_codabench.zip` (v2早期版本，已被0.6958替代)
- `submission_v3_temp/` (临时文件夹)

## 备注
- Val分数来自验证集最佳F-Score
- Test分数来自Codabench公开测试集
- `_pending` 表示尚未在Codabench测试
- TTA表示使用了Test-Time Augmentation
