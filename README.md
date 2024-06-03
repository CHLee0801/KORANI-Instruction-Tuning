# Deep Exploration of Cross-Lingual Zero-Shot Generalization in Instruction Tuning

This is the official github repository for 'Deep Exploration of Cross-Lingual Zero-Shot Generalization in Instruction Tuning' [ACL 2024 Findings].

### 0. Install Dependencies
```
# install torch with the correct cuda version, check nvcc --version
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
# install Hugging Face Libraries
pip install "transformers==4.37.0" "datasets==2.19.1" "accelerate==0.25.0" "evaluate==0.4.0" --upgrade
# install deepspeed and ninja for jit compilations of kernels
pip install "deepspeed==0.9.3" ninja --upgrade
# install additional dependencies needed for training
pip install rouge-score nltk py7zr tensorboard scikit-learn
pip install sentencepiece
pip install wandb
pip install gdown
pip install konlpy
pip install absl-py
```

If you want to get Korean rouge score for generation tasks, you have to download the Mecab.
```
cd KORANI-Instruction-Tuning
wget https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz
tar xvfz mecab-0.996-ko-0.9.2.tar.gz
cd mecab-0.996-ko-0.9.2
./configure
make
make check
sudo make install
```
```
wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz
tar xvfz mecab-ko-dic-2.1.1-20180720.tar.gz
cd mecab-ko-dic-2.1.1-20180720
./configure
make
sudo make install
```
If errors occurred in the command above, please follow the below.
```
cd .. 
rm -r mecab-ko-dic-2.1.1-20180720

tar xvfz mecab-ko-dic-2.1.1-20180720.tar.gz
cd mecab-ko-dic-2.1.1-20180720
autoreconf
./configure
make
sudo make install
```
And finally,
```
pip install mecab-python
```

### 1. KORANI Datasets
![korani_task_taxonomy](https://github.com/CHLee0801/KORANI-Instruction-Tuning/assets/87512263/606c0f59-3de5-4993-bd91-b37d8f227678)
KORANI datasets and task taxonomy. Green datasets are NLG datasets. Yellow datasets are NLU datasets.


Please download the KORANI datasets and unzip it to utilize it.
```
gdown https://drive.google.com/uc?id=1W8tXUZFK-J09kYV1-ZxDOE6QDOrYh_6x
jar xvf KORANI.zip
```

All of the train/eval data are saved in csv file format. For the datasets without license for releasing dataset as an open source, please visit the directory and follow the README.md in the directory. Please visit the following directory in KORANI folder.
```
coreference_resolution/nikl_coref
extractive_qa/korquad1
hatespeech/unsmile
paraphrasing/similar_corpus
```

### 2. Train any LMs in huggingface
Before training, you have to gather instruction tuning datasets in accordance with your held-out setting. If you have decided which dataset to be included in the training dataset, please gather and save it as one csv file. Please refer to following directory as an example.
```
cd data/sample_ko
```

Before actual training, we will tokenize the train dataset before the actaul training. After filling out the configurations in 'run_configs' directory, run the following code for preprocessing the train data
```
bash preprocess_data.sh
```

Finally, run the run.sh file to train! ('localhost:{num1, num2}' to designate the CUDA_VISIBLE_DEVICES=num1, num2. Code will use all GPUs as default)
```
bash run.sh
```

### 2. Evaluate any LMs in huggingface

Run the inference.sh file to evaluate! You can either choose task cluster(s) to evaluate or specific task(s) to evaluate. 
```
bash inference.sh
```