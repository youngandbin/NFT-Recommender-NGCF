# NFT-Recommender-NGCF

### config
- The user-defined parameters needed for the experimental setup are stored.
  - e.g., data path, epoch, batch size, metrics, 
- File name: fixed\_config\_{model_name}.config

### dataset
- **collections**
  - For each collection, there are .inter and .itememb files included.
  - The .inter file represents user-item interactions, and the .itememb file represents item embeddings.

### hyper
- Used for hyperparameter optimization.
- hyperparameter search ranges are included.
- File name: {model}.hyper

### hyper_result
- Contains the results of hyperparameter optimization.
- The .best_params file contains the optimal hyperparameters that are saved after the hyperparameter optimization process.
- The .result file contains the performance of all hyperparameter combinations.
- File name: {model}\_{dataset}.hyper

### result
- The results of the performance evaluation on the test set are saved in the result folder.
- File name: {model}-{dataset}.csv

### run.sh
- Contains shell scripts that can be used to run the main file.

### saved
- The best model that shows the lowest valid metric during the model training process is saved.

### Create_dataset.ipynb
- Code to create the input data file in the format required by Recbole.

### Create_config.ipynb
- Code to create the configurations in the format required by Recbole.

### **main.py**
- The code to run experiments using our model.

### Model.py
- The code contains our model architecture.
