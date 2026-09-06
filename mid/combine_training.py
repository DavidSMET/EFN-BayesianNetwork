import shutil
import subprocess
import os

def training_part():
    datasets = ['alarm_interventional_bic','barley_interventional_bic', 
    'child_interventional_bic', 'insurance_interventional_bic', 
    'mildew_interventional_bic', 'water_interventional_bic']
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    for dataset in datasets:
        dataset_dir = os.path.join(output_dir, dataset)      
        cmd = [
            "python",
            "train.py",
            "--lr", "1e-4",
            "--lr_scheduler", "reduce_on_plateau",
            "--lr_patience", "100",
            "--lr_factor", "0.3",
            "--lr_min", "1e-7",
            "--batch_size", "32",
            "--output_folder", dataset_dir,
            dataset
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"❌ Failed on {dataset}")
            break
        else:
            print(f"✅ Success on {dataset}\n")
    print("******Training finished!******")
    return 0

def generating_part(): 
    source_file = "continuous_generating.py"
    output_dir = "output"
    destination_file = os.path.join(output_dir, "copied_continuous_generating.py")

    if not os.path.exists(destination_file):   
        shutil.copy(source_file, destination_file)
        print(f"******File copied successfully******")
    
    new_destination_file = destination_file + "models"
    if not os.path.exists(new_destination_file):
        result = subprocess.run(
            ["python", "copied_continuous_generating.py"],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(output_dir)
        )
        if result.returncode != 0:
            print(f"❌ Running error")
        print("******Generating finished!******")
    return 0

def computing_part():
    files = [
        ("inference_asia.py", "asia"),
        ("inference_cancer.py", "cancer"),
        ("inference_earthquake.py", "earthquake"),
        ("inference_sachs.py", "sachs"),
        ("inference_survey.py", "survey")
    ]

    output_dir = "output"
    for source_file, dataset in files:  
        target_dir = os.path.join(output_dir, dataset)
        target_file_name = "copied_" + source_file
        target_file = os.path.join(target_dir, target_file_name)

        if not os.path.exists(target_file):
            shutil.copy(source_file, target_file)
            print(f"******File {source_file} copied successfully******")

    for source_file, dataset in files:  
        target_dir = os.path.join(output_dir, dataset)
        target_file_name = "copied_" + source_file
        target_file = os.path.join(target_dir, target_file_name)
        result = subprocess.run(
            ["python", target_file_name],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(target_dir)
        )
        if result.returncode != 0:
            print(f"❌ Inference file running error")
            break
        else:
            print(f"✅ Inference successfully\n")
    print("******Computing finished!******")
    return 0

g1 = training_part()
g2 = generating_part()
g3 = computing_part()