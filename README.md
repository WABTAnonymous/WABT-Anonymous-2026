# Weight-Adjusting Binary Transformation (WABT)

We provide our WABT model, and necessary utilities in this repository. Containerization will be available in the future. 

Our experiments, shown in our accompanying paper, were conducted on a computer with the processor AMD Ryzen 7 4800H, 2.90 GHz, with 16.0 GB RAM, Windows 11 operating system. 

### Contents

 - wabt.py: Our entire model
 - main.ipynb: Contains a method for running a model on a dataset, which the user can modify accordingly. Also contains some example runs of WABT, and conducting the nemenyi test.
 - nemenyi.py: Method used for nemenyi analysis. It also contains a copy of the graph related methods from the [Orange](https://pypi.org/project/Orange3/) library, since those methods are now deprecated.
 - utils.py: Methods for loading libraries and evaluation metrics. For fairness in comparisons, the accuracy metric implementations are the same as in the [PBT codebase](https://github.com/o-yildirim/PBT/tree/main).
 - results folder: Contains our results as .csv files
 - datasets folder: Contains the datasets used in the experiments.

### Version info
requirements.txt contains all the libraries we installed at one point in the project. These specific ones should be enough for the project to build and run, refer to requirements.txt if you face any problems. Keep in mind that **different versions can create different test results**:

 - river==0.19.0
 - scikit-learn==1.3.2 
 - scikit-multilearn==0.2.0
 - scipy==1.10.1
 - matplotlib==3.7.5
 - tqdm==4.67.1
 - liac-arff==2.5.0
 - numpy==1.24.4

**Note:** We have observed that River, the library we use for streaming data, has compatibility issues with the latest Python version 3.14. If you also face issues, make make sure to try an earlier python release (this project was done on 3.8 and 3.12).
