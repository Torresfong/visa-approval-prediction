import os 
from pathlib import Path # OS = path in window(\) or linux(/) are difference, using path in pathlib to remove this error 

project_name = "us_visa approval prediction"

list_of_files = [
    f"{project_name}/__init__.py",
    f"{project_name}/data/__init__.py",
    f"{project_name}/data/raw/__init__.py",
    f"{project_name}/notebook/__init__.py",
    f"{project_name}/notebook/EDA.ipynb",
    f"{project_name}/notebook/feature_analysis.ipynb",
    f"{project_name}/notebook/model_analysis.ipynb",
    f"{project_name}/notebook/main_utils.py"
    f"{project_name}/src/__init__.py",  
    f"{project_name}/src/data_ingestion.py",
    f"{project_name}/src/data_validation.py",
    f"{project_name}/src/preprocess.py",
    f"{project_name}/src/training.py",
    f"{project_name}/src/model_evaluation.py",
    f"{project_name}/src/prediction.py",
    f"{project_name}/config/__init__.py",
    f"{project_name}/config/data_loader.py",
    f"{project_name}/entity/__init__.py",
    f"{project_name}/entity/config_entity.py",
    f"{project_name}/entity/artifact_entity.py",
    f"{project_name}/constants/__init__.py",
    f"{project_name}/exception/__init__.py",
    f"{project_name}/logger/__init__.py",   
    f"{project_name}/utils/__init__.py",
    "app.py",
    "requirement.txt",
    "Dockerfile",
    ".dockerignore",
    "setup.py",
    "model.yaml",
    "schema.yaml"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir,exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath,"w") as f:
            pass
    else:
        print(f"file is already present at{filepath}")