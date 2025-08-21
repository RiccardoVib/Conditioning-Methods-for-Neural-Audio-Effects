# Conditioning Methods for Neural Audio Effects

This code repository for the article _Conditioning Methods for Neural Audio Effects_, Proceedings of the SMC Conferences. SMC Network, 2024.

This repository contains all the necessary utilities to use our architectures. Find the code located inside the "./Code" folder

Visit our [companion page with audio examples](https://riccardovib.github.io/Conditioning-Methods-for-Neural-Audio-Effects_pages/)


### Contents

1. [Datasets](#datasets)
2. [How to Train and Run Inference](#how-to-train-and-run-inference)

<br/>

# Datasets

Datsets are available at the following links:
- [Synthetic Compressor and Overdrive](https://www.kaggle.com/datasets/riccardosimionato/compressor-and-overdrive-audio-effect-datasets)

# How To Train and Run Inference 

First, install Python dependencies:
```
cd ./Code
pip install -r requirements.txt
```

To train models, use the starter.py script.
Ensure you have loaded the dataset into the chosen datasets folder

Available options: 
* --model_save_dir - Folder directory in which to store the trained models [str] (default ="./models")
* --data_dir - Folder directory in which the datasets are stored [str] (default="./datasets")
* --datasets - The names of the datasets to use. [str] (default=" ")
* --epochs - Number of training epochs. [int] (defaut=60)
* --batch_size - The size of each batch [int] (default=1)
* --mini_batch_size - The mini batch size [int] (default=2048)
* --units = The hidden layer size (amount of units) of the network. [ [int] ] (default=8)
* --learning_rate - the initial learning rate [float] (default=3e-4)
* --order - Order of transformation (valid only if FiLM)[int] (default=1)
* --technique - Conditioning technique [ExtraInp, GAF, FILM-GLU, FILM-GCU]. [str] (default=" ")
* --only_inference - When True, skips training and runs only inference on the pre-model. When False, runs training and inference on the trained model. [bool] (default=False)

Example training case: 
```
cd ./Code/

python starter.py --datasets Compressor --technique FILM-GLU --order 3 --epochs 500 
```

To only run inference on an existing pre-trained model, use the "only_inference". In this case, ensure you have the existing model and dataset (to use for inference) both in their respective directories with corresponding names.

Example inference case:
```
cd ./Code/
python starter.py --datasets Compressor --technique FILM-GLU --order 3 --only_inference True
```

# Bibtex

If you use the code included in this repository or any part of it, please acknowledge 
its authors by adding a reference to these publications:

@inproceedings{simionato2024conditioning,
  title={Conditioning Methods for Neural Audio Effects},
  author={Simionato, R and Fasciani, S},
  booktitle={Proceedings of the International Conference on Sound and Music Computing (SMC24)},
  year={2024}
}