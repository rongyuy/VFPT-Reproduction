import torch
import os
import numpy as np
import re

# ================= 配置区域 =================
# ⚠️ 这里已经替换为 EuroSAT 的路径，请确认前缀 (/disks/sata2/...) 是否和你的机器一致
target_dir = "/disks/sata2/kaiqian/workspace/VFPT/output_experiment/eurosat_test_finalfinal/vtab-eurosat/sup_vitb16_224/lr12.5_wd0.001"
# ===========================================

def get_epoch_num(key_str):
    """从 'epoch_99' 这样的字符串中提取数字"""
    match = re.search(r'epoch_(\d+)', key_str)
    return int(match.group(1)) if match else -1

print(f"🌍 正在分析 EuroSAT 实验结果: {target_dir}")
print("=" * 65)
print(f"{'Run ID':<10} | {'最终轮 (Final)':<18} | {'历史最高 (Peak)':<18} | {'回落 (Drop)':<10}")
print("-" * 65)

final_accs = []
peak_accs = []

for i in range(1, 6):
    run_folder = os.path.join(target_dir, f"run{i}")
    result_file = os.path.join(run_folder, "eval_results.pth")
    
    if os.path.exists(result_file):
        try:
            data = torch.load(result_file, map_location="cpu")
            
            epoch_data = {} # 格式: {epoch_num: accuracy}
            
            for key, value in data.items():
                if "epoch" in key and "classification" in value:
                    cls_res = value["classification"]
                    # 寻找包含 "test" 的 key (EuroSAT 也是一样的逻辑)
                    test_key = next((k for k in cls_res.keys() if "test" in k), None)
                    
                    if test_key:
                        # 转换 0.95 -> 95.0
                        acc = float(cls_res[test_key]["top1"]) * 100
                        epoch_num = get_epoch_num(key)
                        epoch_data[epoch_num] = acc
            
            if not epoch_data:
                print(f"{f'Run {i}':<10} | {'N/A':<18} | {'N/A':<18} | {'N/A':<10}")
                continue

            # 1. 找到最后一轮 (Final)
            last_epoch = max(epoch_data.keys())
            final_acc = epoch_data[last_epoch]
            
            # 2. 找到历史最高分 (Peak)
            peak_acc = max(epoch_data.values())
            
            # 3. 计算回落
            drop = peak_acc - final_acc
            
            final_accs.append(final_acc)
            peak_accs.append(peak_acc)
            
            print(f"{f'Run {i}':<10} | {final_acc:.2f}% (ep{last_epoch})     | {peak_acc:.2f}%             | -{drop:.2f}%")

        except Exception as e:
            print(f"Run {i}: 读取出错 - {e}")
    else:
        print(f"Run {i}: 文件不存在 (可能路径不对)")

print("=" * 65)

if final_accs:
    final_avg = np.mean(final_accs)
    final_std = np.std(final_accs)
    peak_avg = np.mean(peak_accs)
    peak_std = np.std(peak_accs)
    diff = peak_avg - final_avg

    print("\n📊 EuroSAT 统计报告:")
    print("-" * 30)
    print(f"🏁 最终结果 (Final):  {final_avg:.2f}% ± {final_std:.2f}%")
    print(f"🏔️ 峰值结果 (Peak):   {peak_avg:.2f}% ± {peak_std:.2f}%")
    print(f"📉 平均性能回落:      -{diff:.2f}%")
    
    print("\n💡 简要分析:")
    if diff < 0.5:
        print(f"  ✅ 训练非常稳定！Peak 和 Final 几乎没有差别 (仅差 {diff:.2f}%)。")
        print("  说明在这个数据集上，模型没有严重的过拟合问题。")
    else:
        print(f"  ⚠️ 存在一定的过拟合 (回落 {diff:.2f}%)。")
else:
    print("❌ 未找到有效数据，请检查 'target_dir' 路径是否正确。")