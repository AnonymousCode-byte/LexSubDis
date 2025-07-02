'''
处理 WordNet 语料库的工具类。WordNet 是一个英语词汇数据库，将英语单词按照词义、同义词集合等进行组织。该类的主要功能包括对单词进行词形还原，
根据给定的单词和可选的词性标签从 WordNet 中获取词义、同义词集合以及同义词列表，同时还提供了对单词进行清理的功能，以处理缩写和特殊字符。
'''
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re
import warnings
warnings.filterwarnings("ignore")

class Wordnet:
    def __init__(self):
        # Init the Wordnet Lemmatizer
        self.lemmatizer = WordNetLemmatizer()

        self.final = {}
        self.not_found = {}

    def fix_lemma(self, lemma):
        return lemma.replace('_', ' ')
    
    # 根据给定的单词和可选的词性标签，从 WordNet 中获取该单词的词义、同义词集合以及同义词列表。
    def getSenses(self, word, target_pos=None):

        gloss = []
        lemmas = []
        # 得到基本形式 lemma_word,方便查找
        lemma_word = self.lemmatizer.lemmatize(word.lower())

        if target_pos is not None:
            to_wordnet_pos = {'N': wordnet.NOUN, 'J': wordnet.ADJ, 'V': wordnet.VERB, 'R': wordnet.ADV}
            # from_lst_pos = {'j': 'J', 'a': 'J', 'v': 'V', 'n': 'N', 'r': 'R'}
            from_lst_pos={'N': "N", 'V': "V", 'J':"J", 'R': "R", 'n': "N", 'v':"V", 'a':"J", 'r': "R"}
            try:
                pos_initial = to_wordnet_pos[from_lst_pos[target_pos]]
            except:
                pos_initial = to_wordnet_pos['N']

            synsets = wn.synsets(lemma_word.lower(), pos=pos_initial)
        else:
            synsets = wn.synsets(lemma_word.lower())

        if len(synsets) == 0:
            if target_pos is not None:
                synsets = wn.synsets(word.lower(), pos=pos_initial)
            else:
                synsets = wn.synsets(word.lower())

        if len(synsets) == 0:
            clean_word = self.clean_word(word.lower())
            if target_pos is not None:
                synsets = wn.synsets(clean_word.lower(), pos=pos_initial)
            else:
                synsets = wn.synsets(clean_word.lower())

        for synset in synsets:
            gloss.append(synset.definition())
            all_lemmas = [self.fix_lemma(lemma.name()) for lemma in synset.lemmas()]
            lemmas.append(' , '.join(all_lemmas))

        return gloss, synsets, lemmas

    # 对单词进行清理，处理常见的缩写和特殊字符。
    def clean_word(self, word):
        pat_is = re.compile("(it|he|she|that|this|there|here) \'s", re.I)
        # to find the 's following the letters
        pat_s = re.compile("(?<=[a-zA-Z])\'s")
        # to find the ' following the words ending by s
        pat_s2 = re.compile("(?<=s)\'s?")
        # to find the abbreviation of not
        pat_not = re.compile("(?<=[a-zA-Z]) n\'t")
        # to find the abbreviation of would
        pat_would = re.compile("(?<=[a-zA-Z]) \'d")
        # to find the abbreviation of will
        pat_will = re.compile("(?<=[a-zA-Z]) \'ll")
        # to find the abbreviation of am
        pat_am = re.compile("(?<=[I|i]) \'m")
        # to find the abbreviation of are
        pat_are = re.compile("(?<=[a-zA-Z]) \'re")
        # to find the abbreviation of have
        pat_ve = re.compile("(?<=[a-zA-Z]) \'ve")
        new_text = pat_is.sub(r"\1 is", word)
        new_text = pat_s.sub("", new_text)
        new_text = pat_s2.sub("", new_text)
        new_text = pat_not.sub(" not", new_text)
        new_text = pat_would.sub(" would", new_text)
        new_text = pat_will.sub(" will", new_text)
        new_text = pat_am.sub(" am", new_text)
        new_text = pat_are.sub(" are", new_text)
        text_a_raw = pat_ve.sub(" have", new_text)
        text_a_raw = text_a_raw.replace('\'', ' ')
        text_a_raw = text_a_raw.split(' ')
        while '' in text_a_raw:
            text_a_raw.remove('')
        text_a_raw = " ".join(text_a_raw)
        return text_a_raw