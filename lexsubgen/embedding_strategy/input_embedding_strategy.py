# embedding_preprocessor.py
# subwards是经过模型[分割后]的子词tokenizer
# 处理是在子词的embedding上的处理
# 在单个位置上进行处理，因为模型输出只用取first

import random
from typing import List, Optional
import torch
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer
import torch.nn.functional as F
import numpy as np
from candidates_from_wordnet.from_wordnet import created_proposed_list
from candidates_from_wordnet.wordnet import Wordnet


class EmbeddingPreprocessor:
    def __init__(
        self,
        tokenizer: None,       # 怎么能写死
        word_embeddings: torch.nn.Embedding,  # 新增参数：词向量矩阵
        device: torch.device,
        strategy: str = "keep",
        mixup_alpha: float = 0.25,
        synonym_topn: int = 10,          # 多少个同义词合适？调参 3个可行，5个效果更差,论文6
        gauss_sigma: float = 0.01,       # 标准差（Standard Deviation）为 0.01 表示数据分布的离散程度非常小，
        dropout_rate: float = 0.3,      # 0.5      0.3时，一个位置，和keep一样 0.4也没用----编码错误？dropout和drop-out的错误
    ):
        self.tokenizer = tokenizer
        self.device = device
        self.strategy = strategy
        self.mixup_alpha = mixup_alpha
        self.synonym_topn = synonym_topn
        self.gauss_sigma = gauss_sigma
        self.dropout_rate = dropout_rate
        self.lemmatizer = WordNetLemmatizer()
        
        # 词向量矩阵（不加载完整模型）——从参数传递
        self.word_embeddings = word_embeddings.to(device)
    

    # 属于一个类的方法（从 self 参数可看出）一种约定俗成的私有方法
    def process_word(
        self, 
        original_word: str,  # 新增原始词参数
        subwords: List[str],     # 子词序列
        pos_tag: str
    ) -> List[torch.Tensor]:
        """
        改进后的处理方法
        Args:
            original_word: 直接从上下文获取的原始词
            subwords: 实际分词后的子词序列
        """
        strategy_processor = { 
            "keep": self._process_keep,
            "mask": self._process_mask,
            "gauss": self._process_gauss,
            "dropout": self._process_dropout,
            "mix-up": self._process_mixup
        }.get(self.strategy, self._process_keep)    # 没有就默认方法
        # print(f'选择的输入策略：{self.strategy}')
        return strategy_processor(original_word, subwords,pos_tag)[:len(subwords)]

    
    def _process_keep(self, original_word: str, subwords: List[str],pos_tag: str):
        """保持原始嵌入"""
        embeddings=[self._get_subword_embedding(sw) for sw in subwords] # 多个位置时         
        # 替代为一个位置
        # 堆叠张量
        # stacked_embeddings = torch.stack(embeddings)
        # # 计算平均值,第0个维度消除
        # average_embedding = torch.mean(stacked_embeddings, dim=0)
        # return torch.unsqueeze(average_embedding, dim=0)
        return torch.unsqueeze(embeddings[0],dim=0)   # 只替换第一个位置，针对多个子词的情况

    def _process_mask(self, original_word: str, subwords: List[str],pos_tag: str):
        """使用[MASK]替换"""
        mask_embed = self._get_subword_embedding(self.tokenizer.mask_token)
        # return [mask_embed] * len(subwords)        # 替换目标词就行了，不用子词多占几个空，意味着输出只能取first
        return torch.unsqueeze(mask_embed,dim=0)



    # 随机数种子，即同样需要运行5次……
    def _process_gauss(self, original_word: str, subwords: List[str],pos_tag: str):
        """添加高斯噪声"""
        base_embeds = self._process_keep(original_word, subwords,pos_tag)  # 返回的shape(1,embedding_size)
        temp_embs=torch.stack([e + torch.randn_like(e)*self.gauss_sigma for e in base_embeds])
        combined_embedding=torch.unsqueeze(temp_embs, dim=0)
        return combined_embedding
    

    # 论文1的drop out不是如此吧？partial mask才对，这里大概率返回了原词，embedding基础上mask,0.3执行5次取平均
    def _process_dropout(self, original_word: str, subwords: List[str],pos_tag: str):
        """随机替换策略"""
        emb=self._process_keep(original_word,subwords,pos_tag)  # 返回的维度,0.3没什么效果
        # 生成一个与 embedding 同形状的随机 mask，mask 中大于 dropout_rate 的保留
        mask = (torch.rand_like(emb) > self.dropout_rate).float()      
        dropped_embedding = emb * mask
        
        return dropped_embedding


    # 其同义词分词后的子词数量不一定同原词，这样的考量更加复杂，怎么合并？  ——————不合并试试看，相当于一个创新做法辣（X）
    # 补充合并的操作，论文6做法——怎么合并？embedding是嵌入，不是模型输出的结果，一个词输入得到的embedding维度可能变大
    # 单个词可能被拆分为多个子词，每个子词对应一个嵌入向量。整个词的表示通常是这些子词嵌入的平均值或拼接。
    # 需要考量的只是这一点，平均或者拼接，即，输入和输出都可以考量！！
    # def _process_mixup(self, original_word: str, subwords: List[str]):
    #     """混合原始词嵌入与同义词嵌入"""
    #     synonyms = self.get_synonyms(original_word)
    #     if not synonyms:
    #         return self._process_gauss(original_word, subwords)  # 回退策略

    #     all_syn_embeds = []
    #     for synonym in synonyms:
    #         syn_subwords = self.tokenizer.tokenize(synonym)
    #         syn_embeds = [self._get_subword_embedding(sw) for sw in syn_subwords]
    #         all_syn_embeds.append(syn_embeds)

    #     # 动态长度对齐
    #     max_len = max([len(embeds) for embeds in all_syn_embeds] + [len(subwords)])
    #     orig_embeds = [self._get_subword_embedding(sw) for sw in subwords]
    #     orig_embeds += [orig_embeds[-1]] * (max_len - len(orig_embeds))     # 二者中的最长部分，也即是只有长的

    #     for i in range(len(all_syn_embeds)):
    #         all_syn_embeds[i] += [all_syn_embeds[i][-1]] * (max_len - len(all_syn_embeds[i]))

    #     # 计算所有同义词嵌入的平均
    #     avg_syn_embeds = []
    #     for j in range(max_len):
    #         syn_embeds_at_j = [embeds[j] for embeds in all_syn_embeds]
    #         avg_syn_embeds.append(torch.mean(torch.stack(syn_embeds_at_j), dim=0))

    #     # 线性混合
    #     return [
    #         self.mixup_alpha * orig + (1 - self.mixup_alpha) * syn
    #         for orig, syn in zip(orig_embeds, avg_syn_embeds)
    #     ]

    # 一个位置上的操作
    def _process_mixup(self, original_word: str, subwords: List[str],pos_tag: str):
        """混合原始词嵌入与同义词嵌入"""
        wordNet=Wordnet()
        synonyms = created_proposed_list(original_word,wordNet,pos_tag=pos_tag)    # 同义词、上位词、下位词字典
        synonyms_final = dict(list(synonyms.items())[:self.synonym_topn])

        if not synonyms_final:
            return self._process_keep(original_word, subwords,pos_tag)  # 回退策略

        all_syn_embeds = []
        for synonym in synonyms_final:
            # 对同义词进行分词
            syn_subwords = self.tokenizer.tokenize(synonym)
            if not syn_subwords:
                continue
            # 获取每个子词的嵌入
            syn_embeds = [self._get_subword_embedding(sw) for sw in syn_subwords]
            # 计算该同义词所有子词嵌入的平均值
            avg_syn_embed = torch.mean(torch.stack(syn_embeds), dim=0)
            all_syn_embeds.append(avg_syn_embed)

        # 如果没有同义词的嵌入，则同样使用回退策略,即用原词的表示
        if not all_syn_embeds:
            return self._process_keep(original_word, subwords,pos_tag)
        # 对所有同义词的平均嵌入进行平均，得到最终的同义词嵌入
        sum_synonym = torch.mean(torch.stack(all_syn_embeds), dim=0)

        # 对原始词进行分词，并计算每个子词的嵌入平均值,均值后，第一个维度会消去
        # orig_embeds = [self._get_subword_embedding(sw) for sw in subwords]    就一个位置,取平均
        # sum_orig = torch.mean(torch.stack(orig_embeds), dim=0)
        sum_orig=self._process_keep(original_word,subwords,pos_tag)

        combined_embedding = self.mixup_alpha * sum_orig + (1 - self.mixup_alpha) * sum_synonym
        combined_embedding=torch.unsqueeze(combined_embedding, dim=0)


        # # 只替换第一个位置，且，mix-up的基础上加drop out
        # # 生成一个与 embedding 同形状的随机 mask，mask 中大于 dropout_rate 的保留
        # mask = (torch.rand_like(combined_embedding) > self.dropout_rate).float()       # >0.3为1
        # # 对保留的元素进行缩放，确保期望值保持不变
        # dropped_embedding = combined_embedding * mask 

        return combined_embedding
        

    # subword究竟是分割后的子词还是字符串？
    def _get_subword_embedding(self, subword: str) -> torch.Tensor: 
        """统一子词嵌入获取方法"""
        # 将子词转换为对应的 ID
        subword_id = self.tokenizer.convert_tokens_to_ids(subword)
        # 将子词 ID 转换为 Tensor 并移动到指定设备
        subword_id_tensor = torch.tensor(subword_id, device=self.device)
        # 使用索引操作获取对应子词 ID 的嵌入向量
        return self.word_embeddings[subword_id_tensor]
    
