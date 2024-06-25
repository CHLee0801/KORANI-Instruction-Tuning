# Deep Exploration of Cross-Lingual Zero-Shot Generalization in Instruction Tuning

This is the official github repository for 'Deep Exploration of Cross-Lingual Zero-Shot Generalization in Instruction Tuning' [[ACL 2024 Findings](https://arxiv.org/abs/2406.08796)].

Citation:
```
@misc{han2024deep,
    title={Deep Exploration of Cross-Lingual Zero-Shot Generalization in Instruction Tuning},
    author={Janghoon Han and Changho Lee and Joongbo Shin and Stanley Jungkyu Choi and Honglak Lee and Kynghoon Bae},
    year={2024},
    eprint={2406.08796},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

KORANI datasets and task taxonomy. Green datasets are NLG datasets. Yellow datasets are NLU datasets.

![korani_task_taxonomy](https://github.com/CHLee0801/KORANI-Instruction-Tuning/assets/87512263/606c0f59-3de5-4993-bd91-b37d8f227678)

## 0. Install Dependencies
```
conda create -n korani python=3.10
conda activate korani
```

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
./autogen.sh
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

## 1. KORANI Datasets

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

### Source and License of KORANI Datasets
* Note that AIHub License allows distribution of derivative work, but not the original datasets.

| Name               | Task                       | License            | Link                                                                                                     |
| ------------------ | -------------------------- | ------------------ | -------------------------------------------------------------------------------------------------------- |
| NSMC               | Sentiment                  | CC0                | https://github.com/e9t/nsmc                                                                              |
| Naver Shopping     | Sentiment                  | Public Domain      | https://github.com/bab2min/corpus/tree/master/sentiment                                                  |
| AIHub Emo          | Sentiment                  | AIHub License      | https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=86    |
| Sosang Sentiment   | Sentiment                  | AIHub License      | https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=102   |
| Kobest Sentineg    | Sentiment                  | CC-BY-SA           | https://huggingface.co/datasets/skt/kobest_v1                                                            |
| BEEP!              | HateSpeech                 | CC-BY-SA           | https://github.com/kocohub/korean-hate-speech                                                            |
| Curse Detection    | HateSpeech                 | Apache License 2.0 | https://github.com/2runo/Curse-detection-v2                                                              |
| Unsmile            | HateSpeech                 | CC-BY-NC-ND        | https://github.com/smilegate-ai/korean_unsmile_dataset                                                   |
| Apeach             | HateSpeech                 | CC-BY-SA           | https://github.com/jason9693/APEACH                                                                      |
| KLUE TC            | Topic Classification       | CC-BY-SA           | https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000066/data/ynat-v1.1.tar.gz       |
| Ko Conversation    | Topic Classification       | AIHub License      | https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=116   |
| Callcenter         | Topic Classification       | AIHub License      | https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=98    |
| Daily Chat         | Intent                     | AIHub License      |https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=extrldata&dataSetSn=288|
| Sae4k              | Intent                     | CC-BY-SA           | https://github.com/warnikchow/sae4k                                                                      |
| StyleKQC           | Intent                     | CC-BY-SA           | https://github.com/cynthia/stylekqc                                                                      |
| ParaKQC            | Paraphrase Identification  | CC-BY-SA           | https://github.com/warnikchow/paraKQC                                                                    |
| KLUE STS           | Paraphrase Identification  | CC-BY-SA           | https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000067/data/klue-sts-v1.1.tar.gz   |
| Question Pair      | Paraphrase Identification  | MIT                | https://github.com/songys/Question_pair                                                                  |
| KorSTS             | Paraphrase Identification  | CC-BY-SA           | https://github.com/kakaobrain/KorNLUDatasets                                                             |
| KorSS              | Paraphrase Identification  | MIT                | https://github.com/yoongi0428/Kor-Sentence-Similarity                                                    |
| KLUE NLI           | Natural Language Inference | CC-BY-SA           | https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000068/data/klue-nli-v1.1.tar.gz   |
| KorNLI             | Natural Language Inference | CC-BY-SA           | https://github.com/kakaobrain/KorNLUDatasets                                                             |
| Similar Corpus     | Paraphrasing               | NOT FREE           | https://corpus.korean.go.kr/main.do                                                                      |
| Kobest WiC         | Word Sense Disambiguation  | CC-BY-SA           | https://huggingface.co/datasets/skt/kobest_v1                                                            |
| Kobest Hellaswag   | Sentence Completion        | CC-BY-SA           | https://huggingface.co/datasets/skt/kobest_v1                                                            |
| Kobest COPA        | Sentence Completion        | CC-BY-SA           | https://huggingface.co/datasets/skt/kobest_v1                                                            |
| Com Gen            | Structure-to-Text          | AIHub License      |https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=120&topMenu=100&aihubDataSe=extrldata&dataSetSn=459|
| Dacon News         | Summarization              | AIHub License      |https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=97|
| Book               | Summarization              | AIHub License      | https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=93    |
| Document Editorial | Summarization              | AIHub License      | https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=97    |
| Document News      | Summarization              | AIHub License      | https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=97    |
| Report             | Summarization              | AIHub License      | https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=97    |
| KorQuAD1           | Extractive QA              | CC-BY-ND           | https://korquad.github.io/KorQuad%201.0/                                                                 |
| NIA QA             | Extractive QA              | AIHub License      | https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=106  |
| KLUE MRC           | Extractive QA              | CC-BY-SA           | https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000072/data/klue-mrc-v1.1.tar.gz   |
| AIHub MRC          | Extractive QA              | AIHub License      | https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=89    |
| Book MRC           | Extractive QA              | AIHub License      | https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=92    |
| News QA            | Extractive QA              | AIHub License      | https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=577   |
| Kobest BoolQ       | Extractive QA              | CC-BY-SA           | https://huggingface.co/datasets/skt/kobest_v1                                                            |
| Document QA        | Multiple Choice QA         | AIHub License      |https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=569|
| Twitter            | Dialogs                    | AIHub License      |https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=269|
| AIHub Daily Dial   | Dialogs                    | AIHub License      |https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=543|
| AIHub Emo Dial     | Dialogs                    | AIHub License      | https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=86    |
| AIHub Korean Dial  | Dialogs                    | AIHub License      | https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=116   |
| AIHub Minwon       | Dialogs                    | AIHub License      | https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=98    |
| AIHub TOD          | Dialogs                    | AIHub License      | https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=544   |
| Ko-En Technology   | Translation                | AIHub License      | https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71266 |
| Ko-En Social       | Translation                | AIHub License      | https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=125   |
| Ko-En Parallel     | Translation                | AIHub License      | https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=126   |
| NIKL Coref         | Coreference Resolution     | NOT FREE           | https://corpus.korean.go.kr/main.do                                                                      |
| ETRI QA            | Closed Book QA             | CC-BY-NC           | https://aiopen.etri.re.kr/corpusModel                                                                    |

## 2. Train any LMs in Huggingface

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

## 3. Evaluate any LMs in Huggingface

Run the inference.sh file to evaluate! You can either choose task cluster(s) to evaluate or specific task(s) to evaluate.

```
bash inference.sh
```

### Results on Korean LLMs

We report the ROUGE-L score for the generation task (Summarization and Coreference Resolution), and ACC for the others.

| Model | Sentiment | Summarization | Multiple Choice QA | NLI | Sent. Comp. | WSD | Coref. Resol. | Average |
| ----- | --------- | ------------- | ------------------ | --- | ----------- | --- | ------------- | ------- |
| [PolyGlot 1.3b](https://huggingface.co/EleutherAI/polyglot-ko-1.3b)	| 47.723	| 3.277	| 34.133	| 33.56	| 48.183	| 2.17	| 48.81	| 31.122| 
| [PolyGlot 3.8b](https://huggingface.co/EleutherAI/polyglot-ko-3.8b)	| 47.783	| 3.138	| 34.133	| 33.56	| 48.167	| 0.435	| 48.81	| 30.861| 
| [PolyGlot 5.8b](https://huggingface.co/EleutherAI/polyglot-ko-5.8b)	| 49.076	| 3.995	| 37.333	| 33.75	| 52.817	| 2.497	| 48.837	| 32.615| 
| [PolyGlot 12.8b](https://huggingface.co/EleutherAI/polyglot-ko-12.8b)	| 45.963	| 4.026	| 37.7	| 33.817	| 54.734	| 2.497	| 49.79	| 32.647| 
| [Llama-3-Open-Ko-8B](https://huggingface.co/beomi/Llama-3-Open-Ko-8B)	| 44.303	| 2.256	| 49.267	| 34.355	| 52.983	| 1.28	| 48.81	| 33.322| 
| [T3Q-ko-solar-dpo-v7.0](https://huggingface.co/chihoonlee10/T3Q-ko-solar-dpo-v7.0)	| 76.753	| 10.138	| 67.4	| 61.925	| 74.475	| 6.185	| 78.295	| 53.596| 