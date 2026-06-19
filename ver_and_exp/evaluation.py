from sklearn.metrics import f1_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
import sacrebleu
# import bert_score
import numpy as np


def classification(y_pred, y_true):

    print('Micro F1: %.4f' % (f1_score(y_true, y_pred, average='micro')))
    print('Macro F1: %.4f' % (f1_score(y_true, y_pred, average='macro')))


def explanation_generation(exp_pred, exp_true):

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    rouge_scores, meteor_scores, bleu_ref, bleu_hyp = [], [], [], []
    for i in range(len(exp_pred)):
        rs = scorer.score(exp_pred[i], exp_true[i])
        rouge_scores.append(rs)
        mscore = meteor_score([exp_true[i].strip().split()], exp_pred[i].strip().split())
        meteor_scores.append(mscore)
        bleu_ref.append([exp_true[i].strip().split()])
        bleu_hyp.append(exp_pred[i].strip().split())

    rouge_1 = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores])
    rouge_2 = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores])
    rouge_l = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores])

    _meteor_score = np.mean(meteor_scores)

    bleu_2 = corpus_bleu(bleu_ref, bleu_hyp, weights=(0.5, 0.5))
    bleu_4 = corpus_bleu(bleu_ref, bleu_hyp, weights=(0.25, 0.25, 0.25, 0.25))
    bleu_sacrebleu = sacrebleu.corpus_bleu(exp_pred, exp_true).score

    # p, r, f1 = bert_score.score(exp_pred, exp_true, lang='en')
    # bertscore = f1.mean().item()

    print('ROUGE-1: %.4f' % rouge_1)
    print('ROUGE-2: %.4f' % rouge_2)
    print('ROUGE-L: %.4f' % rouge_l)
    print('BLEU-2: %.4f' % bleu_2)
    print('BLEU-4: %.4f' % bleu_4)
    print('METEOR: %.4f' % _meteor_score)
    print('BLEU_sacrebleu: %.4f' % bleu_sacrebleu)
    # print('BERTScore: %.4f' % bertscore)