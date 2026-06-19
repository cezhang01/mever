from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, precision_score, recall_score, ndcg_score
import numpy as np
from copy import deepcopy
import json


def evidence_retrieval(claim_emb, evid_emb, claim_ids, test_claim_evid_pairs, ks=[1, 3, 5, 7, 9]):  # may first remove claims in NEI class

    auc, map, prec, rec, count = 0, 0, [0] * len(ks), [0] * len(ks), 0
    for row_id, row in enumerate(claim_emb):
        claim_id = claim_ids[row_id]
        y_true = np.zeros(len(evid_emb))
        gold_evid_indices = test_claim_evid_pairs[test_claim_evid_pairs[:, 0] == claim_id]
        if len(gold_evid_indices) == 0:
            continue
        y_true[gold_evid_indices[:, 1]] = 1
        if np.sum(y_true) == 0:
            continue
        distance = np.sum(np.square(evid_emb - row), axis=1)
        y_score = - distance
        # y_score = np.matmul(evid_emb, np.expand_dims(row, axis=1))
        auc += roc_auc_score(y_true, y_score)
        map += average_precision_score(y_true, y_score)
        y_score_argsort = np.argsort(y_score)
        for idx, k in enumerate(ks):
            y_pred = np.zeros(len(evid_emb))
            y_pred[y_score_argsort[-k:]] = 1
            prec[idx] += precision_score(y_true, y_pred)
            rec[idx] += recall_score(y_true, y_pred)
        count += 1
    auc /= count
    map /= count
    prec = [p / count for p in prec]
    rec = [r / count for r in rec]
    print('Retrieval AUC: %.4f' % auc)
    print('Retrieval MAP: %.4f' % map)
    for idx, k in enumerate(ks):
        print('Retrieval Prec@%d: %.4f' % (k, prec[idx]))
    for idx, k in enumerate(ks):
        print('Retrieval Rec@%d: %.4f' % (k, rec[idx]))


def save_retrieval_result(data_center, claim_emb, evid_emb, args, k=5):

    retrieved_evid_ids = {}
    for row_id, row in enumerate(claim_emb):
        distance = np.sum(np.square(evid_emb - row), axis=1)
        y_score = - distance
        y_score_argsort = np.argsort(y_score)
        retrieved_evid_ids_one_claim = np.flip(y_score_argsort[-k:])
        retrieved_evid_ids[row_id] = retrieved_evid_ids_one_claim.tolist()

    claims = deepcopy(data_center.claims)
    for claim_id in claims.keys():
        claims[claim_id]['retrieved_evid_ids'] = retrieved_evid_ids[claim_id]
    claims_list = []
    for claim_id in claims.keys():
        claims_list.append(claims[claim_id])

    with open('../../../data/' + args.dataset_name + '/claims_retrieved.json', 'w') as f:
        json.dump(claims_list, f, indent=4)