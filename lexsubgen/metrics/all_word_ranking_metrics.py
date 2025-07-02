from collections import OrderedDict
from typing import List, Dict, Tuple, Set, Union
from collections import Counter

def compute_precision_recall_f1_topk(
    gold_substitutes: List[str],
    pred_substitutes: List[str],
    topk_list: List[int] = (1, 3, 10),
) -> Dict[str, float]:
    """
    Method for computing k-metrics for each k in the input 'topk_list'.

    Args:
        gold_substitutes: Gold substitutes provided by human annotators.
        pred_substitutes: Predicted substitutes.
        topk_list: List of integer numbers for metrics.
        For example, if 'topk_list' equal to [1, 3, 5], then there will calculating the following metrics:
            ['Precion@1', 'Recall@1', 'F1-score@1',
             'Precion@3', 'Recall@3', 'F1-score@3',
             'Precion@5', 'Recall@5', 'F1-score@5']

    Returns:
        Dictionary that maps k-values in input 'topk_list' to computed Precison@k, Recall@k, F1@k metrics.
    """
    k_metrics = OrderedDict()
    golds_set = set(gold_substitutes)
    for topk in topk_list:
        if topk > len(pred_substitutes) or topk <= 0:
            raise ValueError(f"Couldn't take top {topk} from {len(pred_substitutes)} substitues")

        topk_pred_substitutes = pred_substitutes[:topk]         # pred_substitutes的选择

        true_positives = sum(1 for s in topk_pred_substitutes if s in golds_set)
        precision, recall, f1_score = _precision_recall_f1_from_tp_tpfp_tpfn(
            true_positives,
            len(topk_pred_substitutes),
            len(gold_substitutes)
        )
        k_metrics[f"prec@{topk}"] = precision
        k_metrics[f"rec@{topk}"] = recall
        k_metrics[f"f1@{topk}"] = f1_score
    return k_metrics


def compute_precision_recall_f1_vocab(
    gold_substitutes: List[str],
    vocabulary: Union[Set[str], Dict[str, int]],
) -> Tuple[float, float, float]:
    """
    Method for computing basic metrics like Precision, Recall, F1-score on all Substitute Generator vocabulary.
    Args:
        gold_substitutes: Gold substitutes provided by human annotators.
        vocabulary: Vocabulary of the used Substitute Generator.
    Returns:
        Precision, Recall, F1 Score
    """
    true_positives = sum(1 for s in set(gold_substitutes) if s in vocabulary)   # 真正例，出现在词汇表中的gold_substitutes
    precision, recall, f1_score = _precision_recall_f1_from_tp_tpfp_tpfn(
        true_positives,
        len(vocabulary),
        len(gold_substitutes)
    )
    return precision, recall, f1_score


def _precision_recall_f1_from_tp_tpfp_tpfn(
    tp: int, tpfp: int, tpfn: int
) -> Tuple[float, float, float]:
    """
    Computing precision, recall and f1 score
    Args:
        tp: number of true positives
        tpfp: number of true positives + false positives    真正例+假正例
        tpfn: number of true positives + false negatives    真正例+假反例
    Returns:
        Precision, Recall and F1 score
    """
    precision, recall, f1_score = 0.0, 0.0, 0.0
    if tpfp:
        precision = tp / tpfp
    if tpfn:
        recall = tp / tpfn
    if precision and recall:
        f1_score = 2 * precision * recall / (precision + recall)
    return precision, recall, f1_score


# def oot_score(golds: Dict[str, float], substitutes: List[str]):
#     """
#     Method for computing Out-Of-Ten score

#     Args:
#         golds: Dictionary that maps gold word to its annotators number.
#         substitutes: List of generated substitutes.
#     Returns:
#         score: Computed OOT score.
#     """
#     score = 0
#     for subst in substitutes:
#         if subst in golds:
#             score += golds[subst]
#     score = score / sum([value for value in golds.values()])
#     return score



# bestm ootm best oot
from collections import defaultdict
from typing import List


def oot_score(gold_words: List[str], gold_weights: List[float], substitutes: List[str]):
    """
    Method for computing Out-Of-Ten score
    Args:
        gold_words: List of gold words.
        gold_weights: List of weights corresponding to gold words.
        substitutes: List of generated substitutes.
    Returns:
        score: Computed OOT score.
    """
    score = 0
    total_weight = sum(gold_weights)        
    if total_weight == 0:
        return 0
    for i, word in enumerate(gold_words):
        if word in substitutes:
            score += gold_weights[i]
    return score / total_weight

# 最高的候选词的频率等不等，等就无mode
def get_mode(gold_words: List[str], gold_weights: List[float]):
    """
    Get the mode(s) from the gold words and weights.
    If there are multiple modes, return None.
    """
    if not gold_words or not gold_weights:
        return None
    weight_dict = defaultdict(float)
    for word, weight in zip(gold_words, gold_weights):
        weight_dict[word] += weight
    max_weight = max(weight_dict.values())
    modes = [k for k, v in weight_dict.items() if v == max_weight]
    if len(modes) > 1:
        return None
    return modes[0]


def ootm_score(gold_words: List[str], gold_weights: List[float], substitutes: List[str]):
    """
    Method for computing Out-Of-Ten Mode score
    Args:
        gold_words: List of gold words.
        gold_weights: List of weights corresponding to gold words.
        substitutes: List of generated substitutes.
    Returns:
        score: Computed OOTM score.
    """
    mode = get_mode(gold_words, gold_weights)
    if mode is None:
        return 0
    return int(mode in substitutes)     # 并非！候选词的任一个包含mode,最大概率的gold_substitute包含在substitutes中


def best_score(gold_words: List[str], gold_weights: List[float], substitutes: List[str]):
    """
    Method for computing Best score
    Args:
        gold_words: List of gold words.
        gold_weights: List of weights corresponding to gold words.
        substitutes: List of generated substitutes.
    Returns:
        score: Computed Best score.
        （3+2）/2/7     并不是最大权重吧，而是替换词出现在gold里的所有词的权重和
    """
    # if not substitutes or not gold_words or not gold_weights:
    #     return 0
    # total_weight = sum(gold_weights)
    # gold_dict = dict(zip(gold_words, gold_weights))
    # temp_weight = []

    # for sub in substitutes:
    #     if sub in gold_dict:
    #         temp_weight.append(gold_dict[sub])
    # if not temp_weight:
    #     return 0.0
    # return sum(temp_weight) /(len(temp_weight)*total_weight) # calculate wrong？？？ top-1上的best吧，而不是20个上的best
    if not substitutes or not gold_words or not gold_weights:
        return 0
    total_weight = sum(gold_weights)
    gold_dict = dict(zip(gold_words, gold_weights))
    temp_weight = []

    # for sub in substitutes[0]:      # a b c d (X)  只考虑第一个（top-1,当然top-2也可以）候选词！！！！！！！！！！
    if substitutes[0] in gold_dict:
        temp_weight.append(gold_dict[substitutes[0]])
    if not temp_weight:
        return 0.0
    return sum(temp_weight) /(len(temp_weight)*total_weight) # calculate wrong？？？ top-1上的best吧，而不是20个上的best


def bestm_score(gold_words: List[str], gold_weights: List[float], substitutes: List[str]):
    """
    Method for computing Best Mode score
    Args:
        gold_words: List of gold words.
        gold_weights: List of weights corresponding to gold words.
        substitutes: List of generated substitutes.
    Returns:
        score: Computed BestM score.
    """
    mode = get_mode(gold_words, gold_weights)
    if mode is None or not substitutes:
        return 0
    first_substitute = substitutes[0]
    return int(first_substitute == mode)


def compute_oot_best_metrics(gold_words: List[str], gold_weights: List[float], substitutes: List[str]):
    """
    Compute oot, ootm, best, and bestm scores and return them in a dictionary.
    Args:
        gold_words: List of gold words.
        gold_weights: List of weights corresponding to gold words.
        substitutes: List of generated substitutes.
    Returns:
        Dictionary containing oot, ootm, best, and bestm scores.
    """
    oot = oot_score(gold_words, gold_weights, substitutes)
    ootm = ootm_score(gold_words, gold_weights, substitutes)
    best = best_score(gold_words, gold_weights, substitutes)
    bestm = bestm_score(gold_words, gold_weights, substitutes)


    return {
        'oot': oot,
        'ootm': ootm,
        'best': best,
        'bestm': bestm,
    }


    