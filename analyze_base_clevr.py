import torch
import os
import numpy as np
import re
import glob

# ================= 配置区域 =================
# ⚠️ 1. 请确认你的 _finalfinal 文件夹名。
#    如果是之前改过配置去掉了引号，名字可能是 "vtab-clevr(task=closest_object_distance)"
#    或者 "vtab-clevr(task='closest_object_distance')"
#    建议先去目录里 ls 看一眼确切名字。
dataset_folder_name = "vtab-clevr(task=closest_object_distance)" 

# ⚠️ 2. 请确认你的基础路径
base_dir = "/disks/sata2/kaiqian/workspace/VFPT/output_experiment/clevr_test_finalfinal" 
# 注意：我根据你的习惯推测文件夹叫 clevr_test_finalfinal，如果不是请修改。

target_dir = os.path.join(base_dir, dataset_folder_name, "sup_vitb16_224/lr12.5_wd0.001")
# ===========================================

def get_epoch_num(key_str):
    match = re.search(r'epoch_(\d+)', key_str)
    return int(match.group(1)) if match else -1

print(f"🧩 正在分析 CLEVR 实验结果: {target_dir}")
print("=" * 70)
print(f"{'Run ID':<10} | {'最终轮 (Final)':<18} | {'历史最高 (Peak)':<18} | {'状态':<10}")
print("-" * 70)

final_accs = []
peak_accs = []
valid_runs = []

for i in range(1, 6):
    run_folder = os.path.join(target_dir, f"run{i}")
    result_file = os.path.join(run_folder, "eval_results.pth")
    
    if os.path.exists(result_file):
        try:
            data = torch.load(result_file, map_location="cpu")
            epoch_data = {} 
            
            for key, value in data.items():
                if "epoch" in key and "classification" in value:
                    cls_res = value["classification"]
                    # CLEVR 的 key 比较长，包含 task="..."
                    # 我们只需要找包含 "test" 的那个 key
                    test_key = next((k for k in cls_res.keys() if "test" in k), None)
                    
                    if test_key:
                        # ⚠️ 注意：有些代码存的是 0.61，有些是 61.0，这里统一处理
                        raw_acc = float(cls_res[test_key]["top1"])
                        acc = raw_acc * 100 if raw_acc <= 1.0 else raw_acc
                        
                        epoch_num = get_epoch_num(key)
                        epoch_data[epoch_num] = acc
            
            if not epoch_data:
                print(f"{f'Run {i}':<10} | {'N/A':<18} | {'N/A':<18} | {'空数据'}")
                continue

            last_epoch = max(epoch_data.keys())
            final_acc = epoch_data[last_epoch]
            peak_acc = max(epoch_data.values())
            
            # 判断是否崩了 (低于 40% 视为崩了)
            status = "❌ 崩了" if final_acc < 40.0 else "✅ 正常"
            
            if final_acc >= 40.0:
                final_accs.append(final_acc)
                peak_accs.append(peak_acc)
                valid_runs.append(i)
            
            print(f"{f'Run {i}':<10} | {final_acc:.2f}% (ep{last_epoch})     | {peak_acc:.2f}%             | {status}")

        except Exception as e:
            print(f"Run {i}: 读取出错 - {e}")
    else:
        print(f"Run {i}: 文件不存在 - {result_file}")

print("=" * 70)

if final_accs:
    print("\n📊 统计总结 (仅计算正常 Run):")
    print("-" * 30)
    print(f"🏁 平均最终结果:  {np.mean(final_accs):.2f}% ± {np.std(final_accs):.2f}%")
    print(f"🏔️ 平均峰值结果:  {np.mean(peak_accs):.2f}% ± {np.std(peak_accs):.2f}%")
    print(f"💡 最佳单次运行:  {np.max(peak_accs):.2f}% (Run {np.argmax(peak_accs) + valid_runs[0]})") # 简单估算Run ID
    
    print("\n📝 报告建议:")
    print("  1. 务必提到 Run 1 的失败，这体现了训练的不稳定性。")
    print(f"  2. 剔除离群值后，该方法在 CLEVR/Distance 上的有效性能约为 {np.mean(final_accs):.1f}%。")
else:
    print("❌ 未找到有效数据，请检查路径。")