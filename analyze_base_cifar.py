import torch
import os
import numpy as np
import glob
import re

# ================= 配置区域 =================
# ⚠️ 请将此处替换为你的 _finalfinal 文件夹的实际完整路径
target_dir = "/disks/sata2/kaiqian/workspace/VFPT/output_experiment/cifar100_gpu_test_finalfinal/vtab-cifar(num_classes=100)/sup_vitb16_224/lr12.5_wd0.001"
# ===========================================

def get_epoch_num(key_str):
    """从 'epoch_99' 这样的字符串中提取数字 99"""
    match = re.search(r'epoch_(\d+)', key_str)
    return int(match.group(1)) if match else -1

print(f"📂 正在深度分析目录: {target_dir}")
print("=" * 60)
print(f"{'Run ID':<10} | {'最终轮 (Final)':<15} | {'历史最高 (Peak)':<15} | {'回落幅度 (Drop)':<15}")
print("-" * 60)

final_accs = []
peak_accs = []

for i in range(1, 6):
    run_folder = os.path.join(target_dir, f"run{i}")
    result_file = os.path.join(run_folder, "eval_results.pth")
    
    if os.path.exists(result_file):
        try:
            data = torch.load(result_file, map_location="cpu")
            
            # 1. 提取所有包含 test 结果的 epoch
            epoch_data = {} # 格式: {epoch_num: accuracy}
            
            for key, value in data.items():
                if "epoch" in key and "classification" in value:
                    cls_res = value["classification"]
                    # 寻找包含 "test" 的 key (例如 test_vtab-cifar...)
                    test_key = next((k for k in cls_res.keys() if "test" in k), None)
                    
                    if test_key:
                        acc = float(cls_res[test_key]["top1"]) * 100
                        epoch_num = get_epoch_num(key)
                        epoch_data[epoch_num] = acc
            
            if not epoch_data:
                print(f"{f'Run {i}':<10} | {'N/A':<15} | {'N/A':<15} | {'N/A':<15}")
                continue

            # 2. 找到最后一轮 (Final) 和最高分 (Peak)
            last_epoch = max(epoch_data.keys())
            final_acc = epoch_data[last_epoch]
            peak_acc = max(epoch_data.values())
            drop = peak_acc - final_acc
            
            # 3. 存储并打印
            final_accs.append(final_acc)
            peak_accs.append(peak_acc)
            
            print(f"{f'Run {i}':<10} | {final_acc:.2f}% (ep{last_epoch})  | {peak_acc:.2f}%          | -{drop:.2f}%")

        except Exception as e:
            print(f"Run {i}: 读取出错 - {e}")
    else:
        print(f"Run {i}: 文件不存在")

print("=" * 60)

if final_accs:
    print("\n📊 统计总结 (Mean ± Std):")
    print("-" * 30)
    print(f"🏁 最终结果 (Final):  {np.mean(final_accs):.2f}% ± {np.std(final_accs):.2f}%")
    print(f"🏔️ 峰值结果 (Peak):   {np.mean(peak_accs):.2f}% ± {np.std(peak_accs):.2f}%")
    print(f"📉 平均性能回落:      -{np.mean(np.array(peak_accs) - np.array(final_accs)):.2f}%")
    
    print("\n💡 分析建议:")
    diff = np.mean(peak_accs) - np.mean(final_accs)
    if diff > 1.0:
        print(f"  检测到显著的过拟合/不稳定现象 (平均回落 {diff:.2f}%)。")
        print("  建议在报告中汇报 'Peak' 结果，并讨论 Early Stopping 的必要性。")
    else:
        print("  训练结果比较稳定，Final 和 Peak 差异不大。")
else:
    print("未找到有效数据。")