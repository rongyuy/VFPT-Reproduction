import torch
import os
import numpy as np
import re
import glob

# ================= ⚙️ 配置区域 (请仔细核对路径) =================

# ⚠️ 注意：请确认 VFPT 的文件夹名称是否正确
# 如果你之前跑 VFPT 复现时用的 OUTPUT_DIR 是 "./output_experiment/cifar100_vfpt"
# 并且跑完了5次最终测试，那么系统自动生成的文件夹名通常是 "./output_experiment/cifar100_vfpt_finalfinal"

experiments = {
    # ------------------- CIFAR-100 -------------------
    "CIFAR-100 [Baseline]":   "./output_experiment/cifar100_vpt_baseline_finalfinal",
    "CIFAR-100 [VFPT]":       "./output_experiment/cifar100_gpu_test_finalfinal", 
    "CIFAR-100 [DCT+Gating]": "./output_experiment/cifar100_vfpt_dct_gating_finalfinal",
    
    # ------------------- EuroSAT -------------------
    "EuroSAT [Baseline]":     "./output_experiment/eurosat_vpt_baseline_finalfinal",
    "EuroSAT [VFPT]":         "./output_experiment/eurosat_test_finalfinal",  
    "EuroSAT [DCT+Gating]":   "./output_experiment/eurosat_vfpt_dct_gating_finalfinal",

    # ------------------- CLEVR -------------------
    "CLEVR [Baseline]":       "./output_experiment/clevr_vpt_baseline_finalfinal",
    "CLEVR [VFPT]":           "./output_experiment/clevr_test_finalfinal",   
    "CLEVR [DCT+Gating]":     "./output_experiment/clevr_vfpt_dct_gating_finalfinal",
}

# ==========================================================

def find_result_folder(base_dir):
    """
    自动寻找包含 eval_results.pth 的 run1 文件夹的上级目录。
    """
    # 搜索模式: base_dir/**/run1/eval_results.pth
    search_pattern = os.path.join(base_dir, "**", "run1", "eval_results.pth")
    matches = glob.glob(search_pattern, recursive=True)
    
    if not matches:
        return None
    
    # 回退两层得到 lr_wd 那一层的路径
    run1_path = os.path.dirname(matches[0]) 
    lr_wd_path = os.path.dirname(run1_path) 
    return lr_wd_path

def get_epoch_num(key_str):
    match = re.search(r'epoch_(\d+)', key_str)
    return int(match.group(1)) if match else -1

def analyze_single_experiment(name, base_path):
    print(f"\n🌍 正在分析: {name}")
    
    if not os.path.exists(base_path):
        print(f"❌ 路径不存在 (跳过): {base_path}")
        return None

    # 1. 自动定位
    target_dir = find_result_folder(base_path)
    if not target_dir:
        print(f"❌ 未找到结果文件 (eval_results.pth)，请检查是否跑完: {base_path}")
        return None
        
    print(f"📍 路径确认: {target_dir}")
    print("-" * 65)
    print(f"{'Run ID':<10} | {'最终精度 (Final)':<18} | {'最高精度 (Peak)':<18} | {'回落 (Drop)':<10}")
    print("-" * 65)

    final_accs = []
    peak_accs = []

    # 遍历 run1 到 run5
    for i in range(1, 6):
        run_folder = os.path.join(target_dir, f"run{i}")
        result_file = os.path.join(run_folder, "eval_results.pth")
        
        if os.path.exists(result_file):
            try:
                data = torch.load(result_file, map_location="cpu")
                epoch_data = {} 
                
                # 解析数据
                for key, value in data.items():
                    if "epoch" in key and "classification" in value:
                        cls_res = value["classification"]
                        test_key = next((k for k in cls_res.keys() if "test" in k), None)
                        if not test_key:
                            test_key = next((k for k in cls_res.keys() if "val" in k), None)

                        if test_key:
                            acc = float(cls_res[test_key]["top1"]) * 100
                            epoch_num = get_epoch_num(key)
                            epoch_data[epoch_num] = acc
                
                if not epoch_data:
                    print(f"{f'Run {i}':<10} | {'数据为空':<18} | {'-':<18} | {'-':<10}")
                    continue

                last_epoch = max(epoch_data.keys())
                final_acc = epoch_data[last_epoch]
                peak_acc = max(epoch_data.values())
                drop = peak_acc - final_acc
                
                final_accs.append(final_acc)
                peak_accs.append(peak_acc)
                
                print(f"{f'Run {i}':<10} | {final_acc:.2f}% (ep{last_epoch})     | {peak_acc:.2f}%             | -{drop:.2f}%")

            except Exception as e:
                print(f"Run {i}: 读取错误 - {e}")
        else:
            print(f"{f'Run {i}':<10} | {'未找到文件':<18} | {'-':<18} | {'-':<10}")

    print("-" * 65)
    
    if final_accs:
        return {
            "name": name,
            "final_avg": np.mean(final_accs),
            "final_std": np.std(final_accs),
            "peak_avg": np.mean(peak_accs),
            "peak_std": np.std(peak_accs)
        }
    return None

# ================= 主程序 =================

results_summary = []

for exp_name, exp_path in experiments.items():
    res = analyze_single_experiment(exp_name, exp_path)
    if res:
        results_summary.append(res)

print("\n\n" + "="*85)
print(f"{'📊 三种方法对比实验结果汇总':^75}")
print("="*85)
print(f"{'实验名称':<25} | {'最终精度 (Final Accuracy)':<30} | {'峰值精度 (Peak Accuracy)':<30}")
print("-" * 85)

# 按数据集分组打印，方便对比
current_dataset = ""
for res in results_summary:
    # 提取数据集名称 (如 CIFAR-100)
    dataset_name = res['name'].split(" [")[0]
    if dataset_name != current_dataset:
        if current_dataset != "": print("-" * 85)
        current_dataset = dataset_name
    
    final_str = f"{res['final_avg']:.2f}% ± {res['final_std']:.2f}%"
    peak_str = f"{res['peak_avg']:.2f}% ± {res['peak_std']:.2f}%"
    print(f"{res['name']:<25} | {final_str:<30} | {peak_str:<30}")

print("="*85)