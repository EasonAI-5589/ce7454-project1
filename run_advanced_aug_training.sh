#!/bin/bash
# 在服务器上启动高级数据增强训练
# 使用方法: bash run_advanced_aug_training.sh

echo "====================================================================="
echo "启动LMSA + MixUp/CutMix高级增强训练"
echo "====================================================================="

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU信息:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    echo ""
else
    echo "警告: 未检测到GPU，将使用CPU训练（非常慢）"
    echo ""
fi

# 创建日志目录
mkdir -p logs

# 时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/lmsa_advanced_aug_${TIMESTAMP}.log"

echo "配置信息:"
echo "  实验名称: lmsa_advanced_aug"
echo "  配置文件: configs/lmsa_advanced_aug.yaml"
echo "  日志文件: ${LOG_FILE}"
echo ""

echo "增强策略:"
echo "  ✓ MixUp (prob=0.3, alpha=0.2)"
echo "  ✓ CutMix (prob=0.3, alpha=1.0)"
echo "  目标: 提高泛化能力，缩小Val-Test gap"
echo ""

echo "预期效果:"
echo "  当前: Val 0.7041 → Test 0.72 (gap 1.6%)"
echo "  目标: Val 0.71-0.72 → Test 0.73-0.74 (gap <1%)"
echo ""

echo "====================================================================="
echo "开始训练..."
echo "====================================================================="
echo ""

# 启动训练（后台运行）
nohup python main.py --config configs/lmsa_advanced_aug.yaml > "${LOG_FILE}" 2>&1 &

# 获取进程ID
PID=$!

echo "训练已在后台启动!"
echo "  进程ID: ${PID}"
echo "  日志文件: ${LOG_FILE}"
echo ""

echo "====================================================================="
echo "监控命令:"
echo "====================================================================="
echo ""
echo "# 实时查看训练日志:"
echo "tail -f ${LOG_FILE}"
echo ""
echo "# 查看最近的训练进度:"
echo "tail -100 ${LOG_FILE}"
echo ""
echo "# 检查进程是否还在运行:"
echo "ps aux | grep ${PID}"
echo ""
echo "# 停止训练:"
echo "kill ${PID}"
echo ""
echo "# 查看GPU使用情况:"
echo "watch -n 1 nvidia-smi"
echo ""

echo "====================================================================="
echo "等待5秒，检查训练是否正常启动..."
echo "====================================================================="
sleep 5

if ps -p ${PID} > /dev/null; then
    echo "✓ 训练进程运行正常!"
    echo ""
    echo "最近的日志输出:"
    echo "---------------------------------------------------------------------"
    tail -30 "${LOG_FILE}"
    echo "---------------------------------------------------------------------"
else
    echo "✗ 训练进程启动失败，请检查日志:"
    echo "cat ${LOG_FILE}"
fi

echo ""
echo "训练将持续约200个epoch，预计需要数小时"
echo "请定期检查日志文件监控训练进度"
