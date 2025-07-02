'''

'''
import os
from enum import Enum, auto
from functools import lru_cache
from typing import Optional
import nltk
# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 定义自定义下载路径，即在当前目录下创建 nltk_data 文件夹
custom_download_path = os.path.join(current_dir, 'nltk_data')
# 确保目录存在，如果不存在则创建
if not os.path.exists(custom_download_path):
    os.makedirs(custom_download_path)
# 将自定义路径添加到 nltk 数据路径列表中
nltk.data.path.append(custom_download_path)
# 下载 wordnet 数据
# nltk.download('wordnet', download_dir=custom_download_path)

from nltk.corpus import wordnet as wn

# 枚举类定义了所有考虑的 WordNet 关系类型。
class Relation(Enum):
    """
    Class that contains all the considered WordNet relation types.
    """
    synonym = auto()
    co_hyponym = auto()
    co_hyponym_3 = auto()
    transitive_hypernym = auto()
    transitive_hyponym = auto()
    direct_hypernym = auto()
    direct_hyponym = auto()
    similar_to = auto()
    no_path = auto()
    unknown_relation = auto()
    unknown_word = auto()
    mwe = auto()
    same = auto()
    target_form = auto()
    meronym = auto()
    holonym = auto()
    entailment = auto()
    anti_entailment = auto()

# 将不同的词性标记映射到 WordNet 中的词性类型。
# nltk.download('wordnet')
to_wordnet_pos = {
    "n": wn.NOUN,
    "a": wn.ADJ,
    "v": wn.VERB,
    "r": wn.ADV,
    "n.v": wn.VERB,
    "n.a": wn.ADJ,
    "J": wn.ADJ,
    "V": wn.VERB,
    "R": wn.ADV,
    "N": wn.NOUN,
}

# 用于获取给定单词的所有同义词集（synsets），可以选择性地指定词性。
def get_synsets(word: str, pos: Optional[str] = None):
    """
    Acquires synsets for a given word and optionally pos tag.

    Args:
        word: word
        pos: pos tag of a word (optional)

    Returns:
        list of WordNet synsets.
    """
    return wn.synsets(word, pos=pos)

# 查找给定单词的 “similar to” 同义词集，主要针对形容词。使用了 lru_cache 进行缓存，提高性能。
@lru_cache(maxsize=8192)
def get_similar_tos(word: str, pos: Optional[str] = None):
    """
    Find `similar to` synsets for a given word and optionally synset.
    Works with adjectives.

    Args:
        word: word to be analyzed
        pos: pos tag of a word

    Returns:
        set of `simialr to` words
    """
    similar_to_synsets = [
        first_lvl_sn
        for tgt_sns in get_synsets(word, pos=pos)
        for first_lvl_sn in tgt_sns.similar_tos()
    ]
    similar_tos = {
        lemma
        for first_lvl_sn in similar_to_synsets
        for lemma in first_lvl_sn.lemma_names()
    }

    similar_tos = similar_tos.union(
        {
            lemma
            for first_lvl_sn in similar_to_synsets
            for second_lvl_sn in first_lvl_sn.similar_tos()
            for lemma in second_lvl_sn.lemma_names()
        }
    )

    return similar_tos

# 获取给定同义词集的整体关系词（holonyms）
def get_holonyms(synset):
    """
    Acquires holonyms from a given synset.

    Args:
        synset: WordNet synset.

    Returns:
        set of holonyms
    """
    return set(
        synset.member_holonyms() + synset.substance_holonyms() + synset.part_holonyms()
    )

# 获取给定同义词集的部分关系词（meronyms）。
def get_meronyms(synset):
    """
    Acquires meronyms for a given synset.

    Args:
        synset: WordNet synset

    Returns:
        set of meronyms
    """
    return set(
        synset.member_meronyms() + synset.substance_meronyms() + synset.part_meronyms()
    )

# 查找两个同义词集列表之间的最近路径，即找到距离最短的两个同义词集。
def find_nearest_synsets(target_synsets, subst_synsets, pos: Optional[str] = None):
    """
    Finds nearest path between two lists of synsets (target word synsets and substitute word synsets),
    e.g. finds two synsets, one from the
    first list and one from another, distance between which are the shortest.

    Args:
        target_synsets: list of synsets of a target word
        subst_synsets: list of synsets of a substitute word
        pos: pos tag of a target word (optional)

    Returns:
        two closest synsets - one for target word and another for substitute.
    """
    # TODO: Parallelize processing
    dists = [
        (tgt_syn, sbt_syn, dist)
        for tgt_syn in target_synsets
        for sbt_syn in subst_synsets
        for dist in [tgt_syn.shortest_path_distance(sbt_syn)]
        if dist is not None
    ]

    if len(dists) == 0:
        return None, None

    tgt_sense, sbt_sense, _ = min(dists, key=lambda x: x[2])

    return tgt_sense, sbt_sense

# 查找目标单词和替代单词之间的 WordNet 关系。使用了 lru_cache 进行缓存，提高性能。
@lru_cache(maxsize=262144)  # 2**18
def get_wordnet_relation(target: str, subst: str, pos: Optional[str] = None) -> str:
    """
    Finds WordNet relation between a target word and a substitute by analyzing
    their synsets. Optionally one could specify pos tag of the target word for
    more robust analysis.

    Args:
        target: target word
        subst: substitute
        pos: pos tag of the target word

    Returns:
        WordNet relation between the target word and a substitute.
    """
    if pos:
        pos = pos.lower()

    if pos is None:
        pos = wn.NOUN

    if len(subst.split(" ")) > 1:
        return Relation.mwe.name

    if target == subst:
        return Relation.same.name

    if set(wn._morphy(target, pos)).intersection(set(wn._morphy(subst, pos))):
        return Relation.target_form.name

    target_synsets = get_synsets(target, pos=pos)
    subst_synsets = get_synsets(subst, pos=pos)
    if len(subst_synsets) == 0:
        return Relation.unknown_word.name

    target_lemmas = {lemma for ss in target_synsets for lemma in ss.lemma_names()}
    subst_lemmas = {lemma for ss in subst_synsets for lemma in ss.lemma_names()}
    if len(target_lemmas.intersection(subst_lemmas)) > 0:
        return Relation.synonym.name

    if subst in get_similar_tos(target, pos):
        return Relation.similar_to.name

    tgt_sense, sbt_sense = find_nearest_synsets(target_synsets, subst_synsets, pos)

    if tgt_sense is None or sbt_sense is None:
        return Relation.no_path.name

    extract_name = lambda synset: synset.name().split(".")[0]
    tgt_name, sbt_name = extract_name(tgt_sense), extract_name(sbt_sense)

    target_holonyms = get_holonyms(tgt_sense)
    target_meronyms = get_meronyms(tgt_sense)

    if sbt_name in {lemma for ss in target_holonyms for lemma in ss.lemma_names()}:
        return Relation.holonym.name
    if sbt_name in {lemma for ss in target_meronyms for lemma in ss.lemma_names()}:
        return Relation.meronym.name

    target_entailments = {
        lemma for ss in tgt_sense.entailments() for lemma in ss.lemma_names()
    }
    if sbt_name in target_entailments:
        return Relation.entailment.name

    subst_entailments = {
        lemma for ss in sbt_sense.entailments() for lemma in ss.lemma_names()
    }
    if tgt_name in subst_entailments:
        return Relation.anti_entailment.name

    for common_hypernym in tgt_sense.lowest_common_hypernyms(sbt_sense):
        tgt_hyp_path = tgt_sense.shortest_path_distance(common_hypernym)
        sbt_hyp_path = sbt_sense.shortest_path_distance(common_hypernym)

        if tgt_hyp_path == 1 and sbt_hyp_path == 0:
            return Relation.direct_hypernym.name  # substitute is a hypernym of target
        elif tgt_hyp_path == 0 and sbt_hyp_path == 1:
            return Relation.direct_hyponym.name
        elif tgt_hyp_path > 1 and sbt_hyp_path == 0:
            return Relation.transitive_hypernym.name
        elif tgt_hyp_path == 0 and sbt_hyp_path > 1:
            return Relation.transitive_hyponym.name
        elif tgt_hyp_path == 1 and sbt_hyp_path == 1:
            return Relation.co_hyponym.name
        elif max(tgt_hyp_path, sbt_hyp_path) <= 3:
            return Relation.co_hyponym_3.name

    return Relation.unknown_relation.name
