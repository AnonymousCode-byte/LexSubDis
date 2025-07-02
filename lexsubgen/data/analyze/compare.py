import pandas as pd
import ast
from nltk.corpus import wordnet as wn
import nltk
import re
# 论文6怎么选的

# 下载 WordNet 资源
nltk.download('wordnet')

dict_pos={'n': wn.NOUN, 'v': wn.VERB, 'a': wn.ADJ, 'r': wn.ADV}
dict_pos_coinco={'N': wn.NOUN, 'V': wn.VERB, 'J': wn.ADJ, 'R': wn.ADV}
dict_pos_semeval={'n': wn.NOUN, 'v': wn.VERB, 'a': wn.ADJ, 'r': wn.ADV}     # 根据数据集的词汇标注去选词，会有问题。标注的pos在原句里的语义可能不同！


# wn.ADJ_SAT：形容词卫星词（Satellite Adjective）形容词卫星词和一个中心形容词（通常被称为 “头形容词”）存在语义上的关联，它们常常用于描述程度、范围或者对中心形容词的含义进行细化。

def get_related_words(word):
    related_words = set()
    for syn in wn.synsets(word):        # 获取单词在 WordNet 中所有 ** 同义词集（synsets）** 的方法。以下是详细说明：
        # 添加同义词
        for lemma in syn.lemmas():
            related_words.add(lemma.name())
        # 添加上位词
        for hypernym in syn.hypernyms():
            for lemma in hypernym.lemmas():
                related_words.add(lemma.name())
        # 添加下位词
        for hyponym in syn.hyponyms():
            for lemma in hyponym.lemmas():
                related_words.add(lemma.name())
    return related_words


def analyze_csv(file_path):
    data = pd.read_csv(file_path)
    total_count = 0
    match_count = 0
    pos=set()
    target_words=set()
    for idx, row in data.iterrows():
        try:
            gold_subst = ast.literal_eval(row['gold_subst'])
            target_id = row['target_position']
            sentence = ast.literal_eval(row['context'])
            target_word = sentence[target_id]
            target_pos=row["pos_tag"]

            pos.add(target_pos)
            target_words.add(sentence[target_id])

            target_synonyms = get_related_words(target_word)

            for word in gold_subst:
                total_count += 1
                if word in target_synonyms:
                    match_count += 1
        except (KeyError, ValueError):
            continue
    return total_count, match_count,pos,target_words


# 替换为你的两个 CSV 文件路径
file1_path = 'csv/semeval_all.csv' # noun, verb, adjective or adverb      pos:{'r', 'n.a', 'a', 'v', 'n', 'n.v', 'a.n'} 应该是一词多意
file2_path = 'csv/coinco.csv' # pos:{'N', 'J', '.N', 'R', 'V'}

total_count1, match_count1,pos1,target_words1 = analyze_csv(file1_path)
total_count2, match_count2,pos2,target_words2 = analyze_csv(file2_path)

total_count = total_count1 + total_count2
match_count = match_count1 + match_count2

# 计算出现的概率
if total_count > 0:
    probability = match_count / total_count
    probability1=match_count1/total_count1
    probability2=match_count2/total_count2
else:
    probability = 0

print(f'词汇在 WordNet 某一词的同义词、上位词、下位词中出现的概率为: {probability} semeval: {probability1} coinco:{probability2}')
print(f'semeval_all的pos:{pos1},coinco的pos:{pos2}')
# 考虑分词，通过bert和xlnet分词的数量
from transformers import BertTokenizer, XLNetTokenizer

# 加载预训练的分词器
bert_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
xlnet_tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')



# 定义函数进行分词并统计
def tokenize_and_count(tokenizer, model_name, word_set):
    count = 0
    for word in word_set:
        # 进行分词
        tokens = tokenizer.tokenize(word)
        # 统计子词数量
        subword_count = len(tokens)
        if subword_count > 2:
            count += 1
            print(f"模型: {model_name}, 原词: {word}, 分词后的词: {tokens}")
    print(f"模型 {model_name} 分词后子词数量大于 2 的单词数量: {count}")


# 使用 BERT-large 分词器进行处理
tokenize_and_count(bert_tokenizer, "BERT-large", target_words1)
print("-" * 50)
# 使用 XLNet 分词器进行处理
tokenize_and_count(xlnet_tokenizer, "XLNet", target_words1)

# coinco
tokenize_and_count(bert_tokenizer, "BERT-large", target_words2)
print("-" * 50)
tokenize_and_count(xlnet_tokenizer, "XLNet", target_words2)


satellite_adj_synsets = wn.all_synsets(wn.ADJ_SAT)
# 打印前 5 个形容词卫星词的同义词集名称
count = 0
for synset in satellite_adj_synsets:
    if count < 10:
        print(synset.name())
        count += 1
    else:
        break


# wordnet结构
def get_wordnet_structure(word):
    synsets = wn.synsets(word)
    if not synsets:
        print(f"未找到 '{word}' 的同义词集。")
        return

    for synset in synsets:
        print(f"同义词集: {synset.name()} - {synset.definition()}")

        # 获取同义词
        synonyms = [lemma.name() for lemma in synset.lemmas()]
        print(f"同义词: {', '.join(synonyms)}")

        # 获取上位词
        hypernyms = [hypernym.name() for hypernym in synset.hypernyms()]
        if hypernyms:
            print(f"上位词: {', '.join(hypernyms)}")
        else:
            print("未找到上位词。")

        # 获取下位词
        hyponyms = [hyponym.name() for hyponym in synset.hyponyms()]
        if hyponyms:
            print(f"下位词: {', '.join(hyponyms)}")
        else:
            print("未找到下位词。")

        print("-" * 50)

# 测试单词
word = "dog"
get_wordnet_structure(word)


# 论文 6 同义词操作

def created_proposed_list(self, change_word, wordnet_gloss, pos_tag):
    gloss_list, synset, _ = wordnet_gloss.getSenses(change_word, pos_tag)
    synonyms = {}
    synonyms_final = {}

    for syn in synset:
        # adding lemmas
        for l in syn.lemmas():
            synonyms[l.name().lower()] = 0
        # adding hypernyms
        for syn_temp in syn.hypernyms():
            for l in syn_temp.lemmas():
                synonyms[l.name().lower()] = 0
        #adding hyponyms
        for syn_temp in syn.hyponyms():
            for l in syn_temp.lemmas():
                synonyms[l.name().lower()] = 0
    try:
        del synonyms[change_word]
    except:
        pass

    for word in synonyms:
        word_temp = word.replace("_", " ")
        word_temp = word_temp.replace("-", " ")
        word_temp = word_temp.replace("'","")
        synonyms_final[word_temp] = 0
    return synonyms_final                       # 同义词列表


# 获取给定单词的同义词列表。这里仅基于基本的 lemma（即直接的同义词），没有扩展到上位词或下位词。
def adding_noise(self, change_word, wordnet_gloss, pos_tag=None):

    gloss_list, synset, _ = wordnet_gloss.getSenses(change_word, pos_tag)

    synonyms = {}
    synonyms_list = []
    for syn in synset:
        for l in syn.lemmas():
            synonyms[l.name().lower()] = 0
    try:
        del synonyms[change_word]

    except:
        pass
    for word in synonyms:
        synonyms_list.append(word)
    return synonyms_list


# 获取给定单词的所有语义信息，即该单词的各种释义（gloss）、对应的 synset 列表以及每个 synset 中所有词条（lemmas）的列表。
def getSenses(self, word, target_pos=None):
    gloss = []
    lemmas = []
    try:
        lemma_word = self.lemmatizer.lemmatize(word.lower())
        # 函数尝试使用 self.lemmatizer.lemmatize 将输入单词转为其基本形式（lemma），以便更好地匹配 WordNet 中的词条。
    except:
        print(word)
    if target_pos is not None:
        to_wordnet_pos = {'N': wn.NOUN, 'J': wn.ADJ, 'V': wn.VERB, 'R': wn.ADV}
        from_lst_pos = {'j': 'J', 'a': 'J', 'v': 'V', 'n': 'N', 'r': 'R'}
        try:
        # 如果提供了 target_pos 参数，函数会根据内部的映射字典 to_wordnet_pos 和 from_lst_pos 将输入的词性转换为 WordNet 所需的格式。
            pos_initial = to_wordnet_pos[from_lst_pos[target_pos]]
        except:
            pos_initial = to_wordnet_pos[target_pos]

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

    return gloss, synsets, lemmas  # 释义


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