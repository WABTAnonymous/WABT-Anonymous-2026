# Weight-Adjusting Binary Transformation (WABT)

We provide our WABT model, and necessary utilities in this repository, also with Docker containerization. 

Our experiments, shown in our accompanying paper, were conducted on a computer with the processor AMD Ryzen 7 4800H, 2.90 GHz, with 16.0 GB RAM, Windows 11 operating system. 

### Contents

 - wabt.py: Our entire model
 - main.ipynb: Contains a method for running a model on a dataset, which the user can modify accordingly. Also contains some example runs of WABT, and conducting the nemenyi test.
 - nemenyi.py: Method used for nemenyi analysis. It also contains a copy of the graph related methods from the [Orange](https://pypi.org/project/Orange3/) library, since those methods are now deprecated.
 - utils.py: Methods for loading libraries and evaluation metrics. For fairness in comparisons, the accuracy metric implementations are the same as in the [PBT codebase](https://github.com/o-yildirim/PBT/tree/main).
 - results folder: Contains our results as .csv files
 - datasets folder: Contains the datasets used in the experiments.

### Version info
requirements.txt contains the necessary libraries with their versions that we used to run the project. Keep in mind that **different versions can create different test results**:

**Note:** We have observed that River, the library we use for streaming data, has compatibility issues with the latest Python version 3.14. If you also face issues, make make sure to try an earlier python release (this project was done on 3.8 and 3.12).

### Running via Docker
We included a Dockerfile in the repository, which you can use to easily build and run the project.

#### 1. Make sure Docker is installed on your computer.

#### 2. Build the Docker image
Open your terminal in the project root directory and run:
```bash
  docker build -t wabt-repro .
```

#### 3. Run the container
In the same directory, run:
```bash
    docker run -p 8888:8888 wabt-repro
```

#### 4. Access the notebook
Once the terminal shows that the server is running, open your browser and navigate to:
```bash
    http://localhost:8888
```
From there, open main.ipynb and run the cells.




