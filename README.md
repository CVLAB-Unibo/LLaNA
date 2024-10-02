<br>
<p align="center">
<h1 align="center"><img src="assets/llana_icon-removebg-preview.png" align="center" width="5.0%">LLaNA: Large Language and NeRF Assistant (NeurIPS&nbsp;2024)</h1> 
  <p align="center">
    <a href='https://andreamaduzzi.github.io/' target='_blank'>Andrea Amaduzzi</a>
    <a href='https://www.unibo.it/sitoweb/pierluigi.zama' target='_blank'>Pierluigi Zama Ramirez</a>
    <a href='https://www.unibo.it/sitoweb/giuseppe.lisanti' target='_blank'>Giuseppe Lisanti</a>
    <a href='https://www.unibo.it/sitoweb/samuele.salti' target='_blank'>Samuele Salti</a>
    <a href='https://www.unibo.it/sitoweb/luigi.distefano' target='_blank'>Luigi Di Stefano</a>
    <br>
    Computer Vision Lab, University of Bologna, Italy
  </p>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2406.11840" target='_**blank**'>
    <img src="https://img.shields.io/badge/Paper-PDF-red?">
  </a> 
  <a href="https://andreamaduzzi.github.io/llana/" target='_blank'>
    <img src="https://img.shields.io/badge/Project-Page-red">
  </a>
  <a href="link to hf demo" target='_blank'>
    <img src="https://img.shields.io/badge/Demo-&#x1f917-red">
  </a>
  <a href="" target='_blank'>
    <img src="https://visitor-badge.laobi.icu/badge?page_id=CVLAB-Unibo.LLaNA&left_color=gray&right_color=red">
  </a>
  <a href="https://github.com/CVLAB-Unibo/LLaNA/stargazers" target='_blank'>
    <img src="https://img.shields.io/github/stars/CVLAB-Unibo/LLaNA?style=social">
  </a>
</p>
</p>

## 
<p align="center">
  <img src="assets/teaser_full_video_compressed.gif" alt="Teaser GIF">
</p>


<!-- contents with emoji -->
## 📋 Contents
- [🤖 Online Demo](#-online-demo)
- [🔧 Installation](#-installation)
- [📦 Data Preparation](#-data-preparation)
- [👨‍🎓 Training](#-training)
- [🧑‍🏫 Evaluation](#-evaluation)
- [🗣️ Chatting](#-chatting)
- [🔗 Citation](#-citation)
- [📄 License](#-license)
- [📚 Related Work](#-related-work)
- [👏 Acknowledgements](#-acknowledgements)

## 🤖 Online Demo
LLaNA is online! Try it at [link to hf demo](http://101.230.144.196).

## 🔧 Installation
The code provided in this repository has been tested in the following environment:
- Ubuntu 20.04
- CUDA 11.7
- Python 3.10.0
- PyTorch 2.0.1
- Transformers 4.40.1

To start: 
1. Clone this repository.
```bash
git clone git@github.com:CVLAB-Unibo/LLaNA.git
cd LLaNA
```
2. Install packages
```bash
conda create -n llana python=3.10 -y
conda activate llana
pip install --upgrade pip
pip install -r requirements.txt

# * for training
pip install ninja
pip install flash-attn
```

## 📦 Data Preparation
In this work, we propose the dataset ShapeNeRF-Text, for training and evaluation on the tasks of NeRF captioning, QA and zero-shot classification.
This dataset features paired NeRFs and language annotations for ShapeNet objects, in particular for all the 40K NeRFs available in [nf2vec](https://github.com/CVLAB-Unibo/nf2vec) dataset.
Such data can be downloaded at the following link: TODO ADD LINK.
After the download, the folder structure will be the following:
```plaintext
LLaNA
├── LLaNAdata
│   ├── llana
│   │   ├── train
│   │   │    ├── texts
│   │   │    │    ├── conversations_brief.json
│   │   │    │    ├── conversations_complex.json
│   │   │    ├── vecs     
|   |   |    |    ├── <model_id>.npy
|   |   |    |    ├── ...
|   |   |    |    ├── <model_id>.npy
│   │   ├── val
│   │   │    ├── texts
│   │   │    │    ├── conversations_brief.json
│   │   │    │    ├── conversations_complex.json
│   │   │    ├── vecs     
|   |   |    |    ├── <model_id>.npy
|   |   |    |    ├── ...
|   |   |    |    ├── <model_id>.npy
│   │   ├── test
│   │   │    ├── texts
│   │   │    │    ├── conversations_brief.json
│   │   │    │    ├── conversations_complex.json
│   │   │    ├── vecs     
|   |   |    |    ├── <model_id>.npy
|   |   |    |    ├── ...
|   |   |    |    ├── <model_id>.npy
```

where:
1. texts/ folder contains the language annotations 
2. vecs/ folder contains the embeddings from nf2vec

Optional: the original NeRF weights from which the embeddings have been computed can be downloaded at this link: TODO ADD LINK. Such data are not necessary for training and evaluation of LLaNA.

## 👨‍🎓 Training
### Download the pre-trained weights of LLAMA, to initialize LLaNA
1. In the root folder of this repository, create a directory called `checkpoints`.
2. Download the pre-trained LLAMA weights from TODO ADD LINK and place them inside `checkpoints`.

### Training Stage 1
```bash
cd LLaNA
bash scripts/LLaNA_train_stage1.sh
```
### Training Stage 2
```bash
cd LLaNA
bash scripts/LLaNA_train_stage2.sh
```

### Computational Resources for Training
LLaNA has been trained on 4 NVIDIA A100 with 64GB of VRAM each. Completing both stages requires ∼1 day of training.
The weights of the trained models will be saved inside the `outputs` directory.

## Checkpoints of trained LLaNA
The trained LLaNA checkpoints can be downloaded from TODO ADD LINK. They must be saved inside the `outputs` directory.

## 🧑‍🏫 Evaluation
The evaluation metrics reported in the research paper are computed on the test set of ShapeNeRF-Text, which can be downloaded following the instructions in TODO add link to Data Preparation section.
### NeRF captioning 
NeRF captioning task can be evaluated on three different data sources:
1. Brief textual descriptions, from ShapeNeRF-Text Dataset
2. Detailed textual descriptions, from ShapeNeRF-Text Dataset
3. GPT2Shape HST, from [Looking at words and points with attention](https://github.com/AndreAmaduzzi/CrossCoherence)


```bash
python llana/eval/eval_shapenet_llana.py --model_name PATH_TO_MODEL --text_data brief_description
```

```bash
python llana/eval/eval_shapenet_llana.py --model_name PATH_TO_MODEL --text_data detailed_description
```

```bash
python llana/eval/eval_shapenet_llana.py --model_name PATH_TO_MODEL --hst_dataset
```


```model_name``` provides the path to the model weights, which must be stored inside the `outputs` directory.
These scripts compute the LLaNA textual predictions for the captioning task. Such output captions will be saved in the directory `evaluation_results` as json files.

Once obtained such textual data, the evaluation metrics reported on the research paper (SentenceBERT, SimCSE, BLEU-1, ROUGE-L, METEOR) can be computed with the following code:
```bash
python llana/eval/traditional_evaluator_shapenet.py --results_path PATH_TO_RESULTS
```
where  `results_path` provides the path to the json file with the predictions from LLaNA.

### NeRF QA
NeRF QA task can be evaluated by using the single-round questions and answers, belonging to the test set of ShapeNeRF-Text Dataset.
```bash
python llana/eval/eval_shapenet_llana.py --model_name PATH_TO_MODEL --text_data single_round
```
As for the captioning task described before, the quantitative metrics on NeRF QA can be computed in the following way:
```bash
python llana/eval/traditional_evaluator_shapenet.py --results_path <path to results>
```
where `results_path` provides the path to the json path with the predictions from LLaNA.

### NeRF zero-shot classification
The classification task is evaluated by asking to LLaNA which is the class of the object provided as input, where the object belongs to the test set of the ShapeNeRF-Text Dataset. 

```bash
python llana/eval/eval_shapenet_llana.py --model_name PATH_TO_MODEL --classification
```
This script provides two output json files:
1. the first one contains the predictions from LLaNA
2. the second one contains the classification accuracy.


### Computational Resources for Evaluation
By default, the evaluation is performed using torch float16 data types. Such choice allows to evaluate LLaNA on a single NVIDIA GeForce RTX 3090 with 24GB of VRAM.

## 🗣️ Chatting
You can chat with LLaNA about any NeRF from our dataset by running the following code:
```bash
python llana/eval/LlaNA_chat.py --model_name PATH_TO_MODEL --torch_dtype float16
```

### Computational Resources for Chatting
As for the NeRF Captioning-QA and Classification Tasks, using torch.float16 as data type, the inference of the model can be executed on a single NVIDIA GeForce RTX 3090 with 24GB of VRAM.

## 🔗 Citation

If you find our work helpful, please consider starring this repo 🌟 and cite:

```bibtex
aggiungi nuovo bibtex di NEURIPS 
```

## 📄 License
TODO: correggi licenza
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>
<br />
This work is under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## 📚 Related Work
- [PointLLM](https://arxiv.org/pdf/2308.16911): Our codebase is built upon this work.
- [3D-LLM](https://arxiv.org/pdf/2307.12981)
- [GPT4Point](https://arxiv.org/pdf/2307.12981)
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [LLAMA](https://huggingface.co/docs/transformers/model_doc/llama3)

## 👏 Acknowledgements
[CINECA](https://www.cineca.it/): We acknowledge the CINECA award under the ISCRA initiative, for the availability of high-performance computing resources and support
