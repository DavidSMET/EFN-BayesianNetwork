import subprocess
from pathlib import Path

datasets = ['alarm_interventional_bic','barley_interventional_bic', 
    'child_interventional_bic', 'insurance_interventional_bic', 
    'mildew_interventional_bic', 'water_interventional_bic']

Path("output").mkdir(parents=True, exist_ok=True)

for dataset in datasets:
    output_dir = f"output/{dataset}"
    
    cmd = [
        "python",
        "train.py",
        "--batch_size", "32",
        "--output_folder", output_dir,   # ← 必须在 dataset 之前！
        dataset                          # ← 必须是最后一个参数！
    ]
    
    print(f"\n🚀 Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
  
    if result.returncode != 0:
        print(f"❌ Failed on {dataset}")
        break
    else:
        print(f"✅ Success on {dataset}\n")

print("🎉 All done!")