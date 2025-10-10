#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gpu_occupy_stable.py
稳定版GPU占用脚本：增强容错性，防止跑着跑着断掉
改进：
1. 完善的异常处理和自动恢复
2. 内存泄漏防护
3. 健康检查机制
4. 渐进式资源分配
5. 更保守的默认参数
"""
import argparse
import os
import sys
import time
import signal
import threading
import traceback
from typing import List, Tuple, Optional
from collections import deque

# --------- 依赖检查 ---------
try:
    import torch
except Exception as e:
    print("需要安装 PyTorch：pip install torch", file=sys.stderr)
    raise

_have_pynvml = True
try:
    import pynvml
except Exception:
    _have_pynvml = False
    print("警告：未安装 pynvml，建议安装以获得更好的监控能力：pip install pynvml", file=sys.stderr)

_have_psutil = True
try:
    import psutil
except Exception:
    _have_psutil = False
    print("提示：未安装 psutil，无法识别进程用户：pip install psutil", file=sys.stderr)

# --------- 工具函数 ---------
def parse_gpu_list(gpu_str: str, max_count: int) -> List[int]:
    if gpu_str.strip().lower() in ("all", "auto", ""):
        return list(range(max_count))
    out = []
    for s in gpu_str.split(","):
        s = s.strip()
        if not s:
            continue
        try:
            out.append(int(s))
        except ValueError:
            print(f"警告：无效的GPU编号 '{s}'", file=sys.stderr)
    return out

def sizeof_fmt(num, suffix="B"):
    for unit in ["", "K", "M", "G", "T"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}P{suffix}"

# --------- NVML 封装（带完整异常处理）---------
_nvml_inited = False
_nvml_handles = {}
_nvml_lock = threading.Lock()

def nvml_init_once():
    global _nvml_inited
    if _nvml_inited or not _have_pynvml:
        return
    with _nvml_lock:
        if _nvml_inited:
            return
        try:
            pynvml.nvmlInit()
            _nvml_inited = True
        except Exception as e:
            print(f"NVML初始化失败: {e}", file=sys.stderr)

def get_handle(i) -> Optional[object]:
    if not _nvml_inited:
        return None
    if i in _nvml_handles:
        return _nvml_handles[i]
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        _nvml_handles[i] = h
        return h
    except Exception as e:
        print(f"获取GPU{i} handle失败: {e}", file=sys.stderr)
        return None

def get_gpu_util(i) -> int:
    """返回GPU利用率，-1表示无法获取"""
    h = get_handle(i)
    if h is None:
        return -1
    try:
        u = pynvml.nvmlDeviceGetUtilizationRates(h)
        return int(u.gpu)
    except Exception:
        return -1

def get_mem_info_bytes(i) -> Tuple[int, int]:
    """返回 (free, total) in bytes，(-1, -1) 表示无法获取"""
    h = get_handle(i)
    if h is not None:
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            return int(mem.free), int(mem.total)
        except Exception:
            pass

    # Fallback to torch
    try:
        torch.cuda.set_device(i)
        free, total = torch.cuda.mem_get_info(i)
        return int(free), int(total)
    except Exception:
        return (-1, -1)

def running_gpu_pids(i) -> List[int]:
    """返回当前GPU上运行的进程PID列表"""
    h = get_handle(i)
    if h is None:
        return []
    try:
        procs = []
        try:
            infos = pynvml.nvmlDeviceGetComputeRunningProcesses_v3(h)
        except Exception:
            infos = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
        for p in infos:
            if hasattr(p, 'pid') and p.pid > 0:
                procs.append(int(p.pid))
        return procs
    except Exception:
        return []

def pids_same_user(pids: List[int], uid: int) -> List[int]:
    if not _have_psutil:
        return []
    out = []
    for pid in pids:
        try:
            puid = psutil.Process(pid).uids().real
            if puid == uid:
                out.append(pid)
        except Exception:
            pass
    return out

# --------- 显存管理（带容错）---------
class MemoryHolder:
    """渐进式显存分配，带自动回退"""
    def __init__(self, device: int, block_mb: int = 128, dtype=torch.uint8):
        self.device = device
        self.block_bytes = block_mb * 1024 * 1024
        self.dtype = dtype
        self.blocks = []
        self.failed_attempts = 0
        self.max_failed_attempts = 3

    def total_bytes(self) -> int:
        return len(self.blocks) * self.block_bytes

    def grow_to_bytes(self, target_bytes: int, reserve_bytes: int = 0):
        """渐进式分配，失败后自动回退"""
        torch.cuda.set_device(self.device)

        while self.total_bytes() + self.block_bytes <= max(target_bytes - reserve_bytes, 0):
            try:
                t = torch.empty(self.block_bytes, dtype=self.dtype, device=self.device)
                self.blocks.append(t)
                self.failed_attempts = 0  # 成功后重置失败计数
            except RuntimeError as e:
                self.failed_attempts += 1
                if self.failed_attempts >= self.max_failed_attempts:
                    # 连续失败，回退一些显存
                    self.shrink_blocks(max(1, len(self.blocks) // 4))
                    self.failed_attempts = 0
                torch.cuda.synchronize(self.device)
                break
            except Exception as e:
                print(f"[GPU{self.device}] 显存分配异常: {e}", file=sys.stderr)
                break

    def shrink_to_bytes(self, target_bytes: int):
        """释放显存到目标大小"""
        while self.total_bytes() > max(target_bytes, 0) and self.blocks:
            self.blocks.pop()
        self._cleanup()

    def shrink_blocks(self, num_blocks: int):
        """释放指定数量的块"""
        for _ in range(min(num_blocks, len(self.blocks))):
            if self.blocks:
                self.blocks.pop()
        self._cleanup()

    def _cleanup(self):
        """强制清理"""
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    def clear(self):
        """完全清空"""
        self.blocks.clear()
        self._cleanup()

# --------- SM压力测试（带稳定性改进）---------
class SMStressor:
    """SM压力测试，防止内存泄漏"""
    def __init__(self, device: int, start_dim: int = 4096, max_dim: int = 16384,
                 start_conc: int = 2, max_conc: int = 8):
        self.device = device
        self.dim = start_dim
        self.max_dim = max_dim
        self.min_dim = 2048
        self.conc = start_conc
        self.max_conc = max_conc
        self.min_conc = 1
        self.streams = []
        self.A = []
        self.B = []
        self.rebuild_count = 0
        self.max_rebuild = 100  # 限制重建次数，防止频繁分配
        self._build_kernels()

    def _build_kernels(self):
        """构建计算核心，带异常处理"""
        try:
            torch.cuda.set_device(self.device)

            # 清理旧资源
            self._cleanup()

            self.streams = [torch.cuda.Stream(device=self.device) for _ in range(self.conc)]
            self.A = []
            self.B = []

            # 根据GPU能力选择dtype
            try:
                capability = torch.cuda.get_device_capability(self.device)
                dtype = torch.float16 if capability[0] >= 7 else torch.float32
            except Exception:
                dtype = torch.float32

            for _ in range(self.conc):
                try:
                    a = torch.randn((self.dim, self.dim), device=self.device, dtype=dtype)
                    b = torch.randn((self.dim, self.dim), device=self.device, dtype=dtype)
                    self.A.append(a)
                    self.B.append(b)
                except RuntimeError:
                    # OOM，停止分配
                    break

            torch.cuda.synchronize(self.device)
            self.rebuild_count += 1
        except Exception as e:
            print(f"[GPU{self.device}] 构建kernels失败: {e}", file=sys.stderr)

    def _cleanup(self):
        """清理资源"""
        self.A.clear()
        self.B.clear()
        self.streams.clear()

    def burn_for(self, seconds: float = 1.0):
        """运行计算负载"""
        if not self.A or not self.B:
            return

        try:
            t0 = time.time()
            iters = 0
            max_iters = 1000  # 防止无限循环

            while time.time() - t0 < seconds and iters < max_iters:
                for i in range(min(len(self.streams), len(self.A))):
                    try:
                        with torch.cuda.stream(self.streams[i]):
                            a = self.A[i]
                            b = self.B[i]
                            c = torch.matmul(a, b)
                            c = torch.tanh(c)
                            self.A[i] = c
                    except Exception:
                        pass  # 单次计算失败不影响其他stream

                torch.cuda.synchronize(self.device)
                iters += 1
        except Exception as e:
            print(f"[GPU{self.device}] 计算异常: {e}", file=sys.stderr)

    def adjust_towards(self, current_util: int, target_util: int):
        """自适应调整，减少重建频率"""
        if current_util < 0 or self.rebuild_count >= self.max_rebuild:
            return

        # 更大的容忍范围，减少频繁调整
        if current_util < target_util - 10:
            if self.conc < self.max_conc:
                self.conc += 1
                self._build_kernels()
            elif self.dim < self.max_dim:
                self.dim = min(int(self.dim * 1.2), self.max_dim)
                self._build_kernels()
        elif current_util > target_util + 10:
            if self.dim > self.min_dim:
                self.dim = max(int(self.dim / 1.2), self.min_dim)
                self._build_kernels()
            elif self.conc > self.min_conc:
                self.conc -= 1
                self._build_kernels()

# --------- Worker线程（带健康检查）---------
class Worker(threading.Thread):
    def __init__(self, device: int, args):
        super().__init__(daemon=True)
        self.device = device
        self.args = args
        self._stop = False
        self.holder = None
        self.stressor = None
        self.last_heartbeat = time.time()
        self.error_count = 0
        self.max_errors = 10

        # 性能统计
        self.util_history = deque(maxlen=10)

    def stop(self):
        self._stop = True

    def heartbeat(self):
        """更新心跳"""
        self.last_heartbeat = time.time()

    def someone_else_using(self) -> bool:
        """检测是否有他人使用GPU"""
        try:
            pids = running_gpu_pids(self.device)
            my_pid = os.getpid()
            pids = [p for p in pids if p != my_pid]

            if not pids:
                return False

            if self.args.respect == "none":
                return False

            if self.args.respect == "mine":
                my_uid = os.getuid()
                mine = pids_same_user(pids, my_uid)
                return len(mine) > 0

            return True
        except Exception:
            return False

    def cleanup_resources(self):
        """清理所有资源"""
        try:
            if self.holder:
                self.holder.clear()
            if self.stressor:
                self.stressor._cleanup()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"[GPU{self.device}] 清理资源失败: {e}", file=sys.stderr)

    def run(self):
        device = self.device

        try:
            torch.cuda.set_device(device)
            torch.backends.cudnn.benchmark = True

            # 初始化资源
            self.holder = MemoryHolder(device, block_mb=self.args.mem_block_mb)
            self.stressor = SMStressor(
                device,
                start_dim=self.args.matmul_dim,
                max_dim=self.args.matmul_dim_max,
                start_conc=self.args.streams,
                max_conc=self.args.streams_max
            )

            print(f"[GPU{device}] Worker启动成功")

            while not self._stop:
                try:
                    self.heartbeat()

                    # 检查停止文件
                    if os.path.exists(self.args.stop_file):
                        print(f"[GPU{device}] 检测到停止文件，退出")
                        break

                    # 检测他人使用
                    if self.someone_else_using():
                        if self.holder and self.holder.total_bytes() > 0:
                            print(f"[GPU{device}] 检测到他人使用，释放资源")
                        self.cleanup_resources()
                        time.sleep(self.args.poll_interval)
                        continue

                    # 显存分配
                    free_b, total_b = get_mem_info_bytes(device)
                    if free_b > 0 and total_b > 0:
                        target_bytes = int(total_b * self.args.target_mem_frac)
                        reserve_bytes = int(self.args.reserve_mem_gb * 1024**3)
                        self.holder.grow_to_bytes(target_bytes, reserve_bytes=reserve_bytes)

                    # SM负载
                    self.stressor.burn_for(seconds=self.args.work_window)

                    # 监控与调整
                    util = get_gpu_util(device)
                    if util >= 0:
                        self.util_history.append(util)
                        avg_util = sum(self.util_history) / len(self.util_history)

                        if self.args.verbose:
                            print(f"[GPU{device}] util={util:3d}% (avg={avg_util:.1f}%) | "
                                  f"mem={sizeof_fmt(self.holder.total_bytes())}")

                        # 每5个周期调整一次，避免频繁重建
                        if len(self.util_history) >= 5:
                            self.stressor.adjust_towards(int(avg_util), self.args.target_sm)
                            self.util_history.clear()

                    time.sleep(self.args.idle_gap)
                    self.error_count = 0  # 成功后重置错误计数

                except Exception as e:
                    self.error_count += 1
                    print(f"[GPU{device}] 错误 ({self.error_count}/{self.max_errors}): {e}",
                          file=sys.stderr)

                    if self.error_count >= self.max_errors:
                        print(f"[GPU{device}] 错误次数过多，退出", file=sys.stderr)
                        break

                    # 出错后清理资源，尝试恢复
                    self.cleanup_resources()
                    time.sleep(5)

                    # 重新初始化
                    try:
                        self.holder = MemoryHolder(device, block_mb=self.args.mem_block_mb)
                        self.stressor = SMStressor(
                            device,
                            start_dim=self.args.matmul_dim,
                            max_dim=self.args.matmul_dim_max,
                            start_conc=self.args.streams,
                            max_conc=self.args.streams_max
                        )
                    except Exception as e2:
                        print(f"[GPU{device}] 重新初始化失败: {e2}", file=sys.stderr)

        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"[GPU{device}] 致命错误: {e}", file=sys.stderr)
            traceback.print_exc()
        finally:
            self.cleanup_resources()
            print(f"[GPU{device}] Worker已退出")

# --------- 主程序 ---------
def main():
    parser = argparse.ArgumentParser(description="稳定版GPU占用脚本")
    parser.add_argument("--gpus", type=str, default="all", help="GPU编号，如 0,1,2 或 all")
    parser.add_argument("--target-sm", type=int, default=90, help="目标SM利用率 (0-100)")
    parser.add_argument("--target-mem-frac", type=float, default=0.85, help="目标显存占用比例 (0-1)")
    parser.add_argument("--reserve-mem-gb", type=float, default=2.0, help="保留显存 (GB)")
    parser.add_argument("--mem-block-mb", type=int, default=128, help="显存分配块大小 (MB)")
    parser.add_argument("--streams", type=int, default=2, help="初始并发streams数")
    parser.add_argument("--streams-max", type=int, default=8, help="最大并发streams数")
    parser.add_argument("--matmul-dim", type=int, default=4096, help="初始矩阵维度")
    parser.add_argument("--matmul-dim-max", type=int, default=16384, help="最大矩阵维度")
    parser.add_argument("--work-window", type=float, default=1.0, help="计算窗口时长 (秒)")
    parser.add_argument("--idle-gap", type=float, default=0.5, help="间隔时长 (秒)")
    parser.add_argument("--poll-interval", type=float, default=30.0, help="检测他人使用的间隔 (秒)")
    parser.add_argument("--respect", type=str, choices=["all", "mine", "none"], default="none",
                        help="是否让出GPU: all=任何人, mine=同用户, none=不让出")
    parser.add_argument("--stop-file", type=str, default="/tmp/occupy.stop",
                        help="停止文件路径")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("未检测到CUDA/GPU", file=sys.stderr)
        sys.exit(1)

    world = torch.cuda.device_count()
    gpu_ids = parse_gpu_list(args.gpus, world)

    if not gpu_ids:
        print("没有可用的GPU", file=sys.stderr)
        sys.exit(1)

    print(f"总GPU数: {world}, 将占用: {gpu_ids}")
    print(f"目标: SM={args.target_sm}%, 显存={args.target_mem_frac*100:.1f}%")

    # 环境优化
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # 初始化NVML
    if _have_pynvml:
        nvml_init_once()

    # 信号处理
    workers = []
    _stop_flag = {"v": False}

    def handle_sig(signum, frame):
        print("\n收到停止信号，正在清理...")
        _stop_flag["v"] = True
        for w in workers:
            w.stop()

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    # 启动workers
    for gid in gpu_ids:
        try:
            w = Worker(gid, args)
            w.start()
            workers.append(w)
            time.sleep(0.5)  # 错开启动时间
        except Exception as e:
            print(f"启动GPU{gid} worker失败: {e}", file=sys.stderr)

    if not workers:
        print("没有成功启动的worker", file=sys.stderr)
        sys.exit(1)

    # 主循环：监控worker健康状态
    try:
        while not _stop_flag["v"]:
            time.sleep(5.0)

            # 健康检查
            now = time.time()
            for w in workers:
                if not w.is_alive():
                    print(f"[GPU{w.device}] Worker已停止", file=sys.stderr)
                elif now - w.last_heartbeat > 60:
                    print(f"[GPU{w.device}] Worker无响应 (心跳超时)", file=sys.stderr)

    except KeyboardInterrupt:
        handle_sig(None, None)
    finally:
        print("等待所有worker退出...")
        for w in workers:
            w.join(timeout=10)
        print("所有worker已退出")

if __name__ == "__main__":
    main()
