# MEVER
This is the pytorch implementation of EACL-2026 paper "[MEVER: Multi-Modal and Explainable Claim Verification with Graph-based Evidence Retrieval](/paper/EACL26-MEVER.pdf)", authored by [Delvin Ce Zhang](http://delvincezhang.com/), Suhan Cui, Zhelin Chu, Xianren Zhang, and [Dongwon Lee](https://pike.psu.edu/dongwon/).

MEVER is a multi-modal language model for evidence retrieval, claim verification, and explanation generation. It aims to retrieve relevant evidence to verify a given claim and generate explanation behind model reasoning. In this paper, we further create a mult-modal dataset in the AI domain, named AIChartClaim, which is released in this repository as well.

![](/paper/model_architecture.jpg)

## Implementation Environment
- python == 3.9
- pytorch == 2.4.0
- transformers == 4.46.0
- numpy == 1.24.1
- sklearn == 1.3.2

## Run

`python ./retrieval/main.py`  # evidence retrieval

`python ./ver_and_exp/main.py`   # claim verification and explanation generation

### Parameter Setting
- -dn: dataset name, default = chart_check (choices = \[chart_check, ai_chart_claim\])
- -m: mode, default = train (choices = \[train, test\])
- -bert: bert version, default = sci (choices = \[sci, base\])  # set it to `sci` for scientific datasets like ai_chart_claim and chart_check, set it to `base` for general datasets
- -vit: vit version, default = base
- -t5: t5 version, default = base
- -ep: evidence provided, default = gold (choices = \[gold, retrieved\])
- -ne: number of training epochs, default = 100  # set it to 100 for small datasets like ai_chart_claim and 30 for large datasets
- -ls: log steps, the model makes evaluation on test set every log_steps, default = 10
- -lr: learning rate, default = 1e-6
- -ms: minibatch size, default = 4
- -ml: maximum length of texts input to the language model, default = 128
- -nt: number of sampled evidence texts for multi-evidence reasoning, default = 5
- -ni: number of sampled images of each evidence text, default = 1
- -mt: number of tokens generated for explanation, default = 128
- -l: lambda for consistency regularizer, default=0.5
- -ddp: whether use distributed training, default = False
- -gpu: gpu
- -rs: random seed

## Data
We release ChartCheck and AIChartClaim datasets [here](https://drive.google.com/file/d/1ph6gkL_vsB8eAizk5fdKocQ_Fuo5Ccww/view?usp=sharing). Please unzip `data.zip` and put the unzipped data into `./data` folder (e.g., `./data/chart_check/***.json`).

Each dataset contains `claims.json`, `evidence.json`, and `./images`.

Below is an example of `claims.json` format. It is a list, and each element in the list is a dictionary containing information of a specific claim. The length of the list is the number of total claims.

```
[
    {
        "claim_id": 0,  # claim id (may not start from 0, may not be an integer)
        "claim_text": "The ratio of the volume of an inscribed ball to the volume of the cube inside which the ball is inscribed decreases as the dimensionality of the space increases.",  # claim text (a string)
        "label": "support",  # label (can be any string, may not strictly be support, refute, or nei)
        "explanation": "The chart shows that as the number of dimensions increases, the ratio of the volume of the inscribed ball to the volume of the cube decreases, the curse of dimensionality.",  # gold explanation (a string)
        "train_dev_test": "train",  # dataset split (optional. If provided with train, dev, or test, the model will follow the split. If not provided, the model will split the data into 72:8:20 for train:dev:test)
        "gold_evid_ids": [  # a list of gold evidence ids (used only when args.evidence_provided == gold, these evidence ids correspond to the ids in evidence.json)
            0
        ],
        "retrieved_evid_ids": [  # a list of retrieved evidence ids (used only when args.evidence_provided == retrieved)
            756,
            1573,
            1447,
            1454,
            0
        ]
    },
    {
        "claim_id": 1,
        "claim_text": "Sosa had the highest Normalized Slugging Percentage in year 8.",
        "label": "support",
        "explanation": "Sosa had a straight upward increase from year one to year to year 5.",
        "train_dev_test": "train",
        "gold_evid_ids": [
            1
        ],
        "retrieved_evid_ids": [
            1382,
            262,
            398,
            633,
            1
        ]
    }
]
```

Below is an example of `evidence.json` format. It is a list, and each element in the list is a dictionary containing information of a specific evidence sentence. The length of the list is the number of total evidence sentences.

```
[
    {
        "evid_id": 0,  # evidence id (this id corresponds to gold_evid_ids and retrieved_evid_ids in claims.json, this id may not start from 0 and may not be an integer)
        "evid_text": "Illustrates the \"curse of dimensionality.\" The graph shows the ratio of the volume of an inscribed ball divided by the volume of the cube inside of which the ball is inscribed.",  # evidence text (a string)
        "evid_images": [  # a list of image ids associated with the current evidence text
            "0.png"
        ]
    },
    {
        "evid_id": 1,
        "evid_text": "Slugging Percentage from 1997-2004 for 6 players, normalized to each player's lowest SLG during that term.",
        "evid_images": [
            "1.jpg"
        ]
    }
]
```

For the `./images` folder, it contains all images associated with claims and evidence. Images are named using their respective image IDs. These image IDs correspond to those in the `evid_images` attribute in the `evidence.json` file.

## Output
- For evidence retrieval, results with AUC, MAP, Precision, and Recall scores on test set will be printed out after each training epoch.
- For claim verification, results with Micro-F1 and Macro-F1 scores on test set will be printed out after every `log_steps (default = 10)` training epochs.
- For explanation generation, results with ROUGE-1, ROUGE-2, ROUGE-L, BLEU-2, BLEU-4, and METEOR on test set will be printed out after every `log_steps (default = 10)` training epochs.
- Model checkpoints are saved to the `./ckpt` folder.

## Reference
If you find our paper useful, including code and data, please cite

```
@inproceedings{mever,
  title={MEVER: Multi-Modal and Explainable Claim Verification with Graph-based Evidence Retrieval},
  author={Zhang, Delvin Ce and Cui, Suhan and Chu, Zhelin and Zhang, Xianren and Lee, Dongwon},
  booktitle={Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={5236--5255},
  year={2026}
}
```
