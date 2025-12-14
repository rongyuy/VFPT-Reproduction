import tensorflow_datasets as tfds
import os

# ================= 配置区域 =================
# 请将此处修改为你服务器上实际想存放数据的路径
# 建议使用绝对路径，例如 /data/your_name/vtab-1k/
DATA_DIR = "/disks/sata2/kaiqian/workspace/VFPT/data/vtab-1k/" 
# ===========================================

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

print(f"开始下载数据至: {DATA_DIR} ...")

# 1. Natural 组 (7个)
natural_tasks = [
    "caltech101:3.*.*",
    "cifar100:3.*.*",
    "dtd:3.*.*",
    "oxford_flowers102:2.*.*",
    "oxford_iiit_pet:3.*.*",
    "sun397/tfds:4.*.*",
    "svhn_cropped:3.*.*"
]

# 2. Structured 组 (5个)
structured_tasks = [
    "clevr:3.*.*",
    "dmlab:2.0.1",
    "dsprites:2.*.*",
    # 注意：kitti 有时版本会有问题，如果报错尝试 3.2.0
    "kitti:3.2.0", 
    "smallnorb:2.*.*"
]

# 3. Specialized 组 (4个)
# 注意：resisc45 和 diabetic_retinopathy 通常需要手动下载，见下方说明
specialized_tasks = [
    "patch_camelyon:2.*.*",
    "eurosat/rgb:2.*.*",
    # "resisc45:3.*.*",  # 建议手动下载，见注释
    # "diabetic_retinopathy_detection/btgraham-300:3.*.*" # 建议手动下载，见注释
]

# 合并所有任务 (暂时排除需要手动下载的)
all_tasks = natural_tasks + structured_tasks + [t for t in specialized_tasks if "resisc" not in t and "diabetic" not in t]

for task_name in all_tasks:
    print(f"正在处理: {task_name}")
    try:
        builder = tfds.builder(task_name, data_dir=DATA_DIR)
        builder.download_and_prepare()
        print(f"--> {task_name} 完成!")
    except Exception as e:
        print(f"--> {task_name} 失败! 错误信息: {e}")

print("\n=== 特殊数据集说明 ===")
print("1. resisc45: 由于版权原因，TFDS 通常无法自动下载。请手动下载 rar 文件并解压到 manual_dir。")
print("2. diabetic_retinopathy: 需要从 Kaggle 下载。")
print("具体操作请参考项目中的 VTAB_SETUP.md 文件。")