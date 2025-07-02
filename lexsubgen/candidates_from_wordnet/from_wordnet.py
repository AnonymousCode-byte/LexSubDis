
# from nltk.corpus import wordnet as wn

# dict_pos_coinco={'N': wn.NOUN, 'V': wn.VERB, 'J': wn.ADJ, 'R': wn.ADV}
# dict_pos_semeval={'n': wn.NOUN, 'v': wn.VERB, 'a': wn.ADJ, 'r': wn.ADV} 


# 创建一个包含给定单词的同义词、上位词和下位词的列表。
def created_proposed_list(change_word, wordnet_gloss, pos_tag):
    gloss_list, synset, _ = wordnet_gloss.getSenses(change_word, pos_tag)
    synonyms = {}   # 还以为是集合，字典按照加入顺序有序
    synonyms_final = {} # 字典，无重复


    for syn in synset:
        # adding lemmas
        for l in syn.lemmas():
            synonyms[l.name().lower()] = 0
        
        # adding hypernyms
        for syn_temp in syn.hypernyms():
            for l in syn_temp.lemmas():
                synonyms[l.name().lower()] = 0

        #adding hyponyms  earlier than hypernyms——not good
        for syn_temp in syn.hyponyms():
            for l in syn_temp.lemmas():
                synonyms[l.name().lower()] = 0
        
    try:
        del synonyms[change_word]       # 删除原词
    except:
        pass

    for word in synonyms:
        word_temp = word.replace("_", " ")
        word_temp = word_temp.replace("-", " ")
        word_temp = word_temp.replace("'","")
        synonyms_final[word_temp] = 0       # 得分初始化为0，需要proposal score否
    return synonyms_final


# 获取给定单词的同义词列表。
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



