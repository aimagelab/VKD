import numpy as np
from warnings import warn


def compute_ap_cmc(index, good_index, junk_index,
                   return_is_correct: bool = False):
    ap = 0
    cmc = np.zeros(len(index))

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1.0

    correct_ = np.any(np.in1d(index[:1], good_index))
    predicted_ = index[:1]

    for i in range(ngood):

        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2  # trapezoid approximation

    if return_is_correct:
        return ap, cmc, (correct_, predicted_)
    return ap, cmc


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, return_wrong_matches: bool = False):
    num_q, num_g = distmat.shape
    index = np.argsort(distmat, axis=1)  # from small to large

    num_no_gt = 0  # num of query imgs without groundtruth
    num_r1 = 0
    CMC = np.zeros(len(g_pids))
    AP = 0

    wrong_matches = []

    for i in range(num_q):
        # groundtruth index
        query_index = np.argwhere(g_pids == q_pids[i])
        camera_index = np.argwhere(g_camids == q_camids[i])
        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        if good_index.size == 0:
            num_no_gt += 1
            continue
        # remove gallery samples that have the same pid and camid with query
        junk_index = np.intersect1d(query_index, camera_index)

        ap_tmp, CMC_tmp, (correct_, predicted_) = compute_ap_cmc(index[i], good_index,
                                                                 junk_index, True)

        if not correct_:
            wrong_matches.append((i, predicted_[0]))

        if CMC_tmp[0] == 1:
            num_r1 += 1
        CMC = CMC + CMC_tmp
        AP += ap_tmp

    if num_no_gt > 0:
        warn("{} query imgs do not have groundtruth.".format(num_no_gt))

    CMC = CMC / (num_q - num_no_gt)
    mAP = AP / (num_q - num_no_gt)

    if return_wrong_matches: return CMC, mAP, wrong_matches

    return CMC, mAP
