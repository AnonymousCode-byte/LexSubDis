import logging
import os
import random
from string import punctuation
from pathlib import Path
from typing import List, Tuple, Dict, NoReturn
import numpy as np
import torch
from overrides import overrides
from transformers import XLNetLMHeadModel, XLNetTokenizer, SPIECE_UNDERLINE
from lexsubgen.prob_estimators.embsim_estimator import EmbSimProbEstimator
from lexsubgen.embedding_strategy.input_embedding_strategy import EmbeddingPreprocessor
from lexsubgen.embedding_strategy.output_embedding_strategy import outputlogits_stategy
import json
from lexsubgen.embedding_strategy.output_embedding_strategy import outputlogits_stategy
import torch.nn.functional as F
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

logger = logging.getLogger(Path(__file__).name)


class XLNetProbEstimator(EmbSimProbEstimator):

    _word_embeddings = None
    _tokenizer = None
    def __init__(
        self,
        model_name: str = "xlnet-large-cased",
        masked: bool = True,
        use_input_mask: bool = False,
        embedding_similarity: bool = False,
        temperature: float = 1.0,
        cuda_device: int = 0,
        multi_subword: bool = False,
        top_k_subword: int = 10,
        filter_words: bool = True,
        sim_func: str = "dot-product",
        use_subword_mean: bool = False,
        verbose: bool = False,

        stratagy_input_embedding:str="gauss",          # gauss dropout mask keep mix-up    
        mixup_alpha: float = 0.25,
        synonym_topn: int = 10,
        gauss_sigma: float = 0.01,
        weights:str="exponential",                                        #  ['mean', 'first', 'linear', 'exponential']
        decay_rate:float=0.9
    ):
        """
        Probability estimator based on XLNet model, see
        Z. Yang et al. "XLNet: Generalized Autoregressive Pretraining
        for Language Understanding".

        Args:
            model_name: name of the XLNet model, see https://github.com/huggingface/transformers
            masked: whether to mask target word or not
            use_input_mask: whether to zero out attention weights for pad tokens
            embedding_similarity: whether to compute XLNet embedding similarity instead of the full model
            temperature: temperature by which to divide log-probs
            cuda_device: CUDA device to load model to
                multi_subword: whether to generate multi-subword words
            top_k_subword: branching factor when generating multi-subword words
            filter_words: whether to filter special tokens and word pieces
            sim_func: name of similarity function to use in order to compute embedding similarity
            use_subword_mean: how to handle words that are splitted into multiple subwords when computing
            verbose: whether to print misc information
        """
        super(XLNetProbEstimator, self).__init__(
            model_name=model_name,
            verbose=verbose,
            sim_func=sim_func,
            temperature=temperature,
        )
        self.cuda_device = cuda_device
        self.use_input_mask = use_input_mask
        self.masked = masked
        self.multi_subword = multi_subword
        self.top_k_subword = top_k_subword
        self.filter_words = filter_words
        self.embedding_similarity = embedding_similarity
        self.use_subword_mean = use_subword_mean

        self.stratagy_input_embedding=stratagy_input_embedding
        self.mixup_alpha = mixup_alpha
        self.synonym_topn = synonym_topn
        self.gauss_sigma = gauss_sigma
        self.weights=weights        # 输出的合并方式
        self.decayrate=decay_rate   #指数方式


        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)  指定用一个
        
        if self.cuda_device != -1 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.cuda_device}")
        else:
            self.device = torch.device("cpu")

        self.descriptor = {
            "Prob_generator": {
                "name": "xlnet",
                "model_name": self.model_name,
                "use_input_mask": self.use_input_mask,
                "masked": self.masked,
                "multi_subword": self.multi_subword,
                "use_subword_mean": self.use_subword_mean,
            }
        }

        self.NON_START_SYMBOL = "##"
        self.register_model()   # 默认在cpu     提前加载，放在前面
        self.logger.debug(f"Probability estimator {self.descriptor} is created.")
        self.logger.debug(f"Config:\n{json.dumps(self.descriptor, indent=4)}")

        # 检查类属性是否已经加载了词嵌入和分词器
        if XLNetProbEstimator._word_embeddings is None or XLNetProbEstimator._tokenizer is None:
            XLNetProbEstimator._word_embeddings = torch.tensor(
                self.loaded[self.model_name]["embeddings"], device=self.device
            )
            XLNetProbEstimator._tokenizer = self.loaded[self.model_name]["tokenizer"]
        self.preprocessor = EmbeddingPreprocessor(
            word_embeddings=self._word_embeddings,
            # model_name=model_name,
            tokenizer=self.tokenizer,
            device=self.device,
            strategy=stratagy_input_embedding,

            mixup_alpha=mixup_alpha,
            synonym_topn=synonym_topn,
            gauss_sigma=gauss_sigma,
        )

    def register_model(self) -> NoReturn:
        """
        If the model is not registered this method creates that model and
        places it to the model register. If the model is registered just
        increments model reference count. This method helps to save computational resources
        e.g. when combining model prediction with embedding similarity by not loading into
        memory same model twice.
        """
        if self.model_name not in XLNetProbEstimator.loaded:
            model = XLNetLMHeadModel.from_pretrained(self.model_name)
            model.to(self.device)
            model.eval()
            tokenizer = XLNetTokenizer.from_pretrained(self.model_name)
            word2id = self._get_word2id(tokenizer)
            spiece_ids = [
                idx
                for word, idx in word2id.items()
                if word.startswith(self.NON_START_SYMBOL)
            ]
            all_special_ids = tokenizer.all_special_ids
            word_embeddings = model.transformer.word_embedding.weight.data.cpu().numpy()
            XLNetProbEstimator.loaded[self.model_name] = {
                "model": model,
                "tokenizer": tokenizer,
                "embeddings": word_embeddings,
                "word2id": word2id,
                "spiece_ids": spiece_ids,
                "all_special_ids": all_special_ids,
            }
            XLNetProbEstimator.loaded[self.model_name]["ref_count"] = 1
        else:
            XLNetProbEstimator.loaded[self.model_name]["ref_count"] += 1
    # 去除掉有多个子词的情况
    @property
    def spiece_ids(self):
        """
        Indexes of word pieces, i.e. words that start with special token
        (in original tokenizer that words doesn't start with special underline
        score token so they are non-starting parts of some words). We filter them
        cause they do not represent any word from a target vocabulary.
        即排除掉，以 _ 开头的单词
        Returns:
            list of indexes of word pieces.
        """
        return self.loaded[self.model_name]["spiece_ids"]

    @property
    def all_special_ids(self):
        return self.loaded[self.model_name]["all_special_ids"]

    @property
    def tokenizer(self):
        """
        Tokenizer related to the current model.

        Returns:
            `transformers.XLNetTokenizer`
        """
        return self.loaded[self.model_name]["tokenizer"]

    

    @overrides
    def get_unk_word_vector(self, word) -> np.ndarray:
        """
        This method returns vector to be used as a default if
        word is not present in the vocabulary. If `self.use_subword_mean` is true
        then the word will be splitted into subwords and mean of their embeddings
        will be taken.

        Args:
            word: word for which the vector should be given

        Returns:
            zeros vector
        """
        if self.use_subword_mean:
            sub_token_ids = self.tokenizer.encode(word)[:-2]
            mean_vector = self.embeddings[sub_token_ids, :].mean(axis=0, keepdims=True)
            return mean_vector
        return super(XLNetProbEstimator, self).get_unk_word_vector(word)

    # batch 数据处理，bert的prepare_batch
    # 将句子转换为模型可处理的 token ID 序列 定位目标词的新位置
    def _numericalize_batch(
        self, tokens_lists: List[List[str]], target_ids: List[int]
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Tokenize contexts and numericalize them according to model vocabulary.
        Update target token indexes in new obtained contexts.

        Args:
            tokens_lists: list of contexts
            target_ids: list of target word indexes

        Returns:
            numerical contexts and updated target word positions
        """
        numerical_sentences, target_positions,target_len = [], [],[]
        for tokens, target_id in zip(tokens_lists, target_ids):
            seq, pos,len_sub_word = self.get_new_token_seq_and_pos(tokens, target_id)
            numerical_sentences.append(seq)
            target_positions.append(pos)
            target_len.append(len_sub_word)
        return numerical_sentences, target_positions,target_len

    def _get_word2id(self, tokenizer: XLNetTokenizer, convert: bool = True):
        """
        Get model vocabulary in the form of mapping from words to indexes.

        Args:
            tokenizer: model tokenizer
            convert: whether to convert words with special underline scores characters
            into ordinary words and prepend word pieces with special characters.

        Returns:
            model vocabulary
        """
        word2id = dict()
        for idx in range(tokenizer.vocab_size):
            token: str = tokenizer.convert_ids_to_tokens(idx)
            if convert:
                # Prepare vocab suitable for substitution evaluation
                # Remove sentence piece underline and add special symbol to intra word parts
                if token.startswith(SPIECE_UNDERLINE) and len(token) > 1:
                    token = token[1:]
                else:
                    token = self.NON_START_SYMBOL + token
                word2id[token] = idx
        return word2id

    # 生成带掩码的输入序列
    def get_new_token_seq_and_pos(self, tokens: List[str], target_id: int):
        """
        Transform original context into the form suitable for processing with XLNet model.
        # 单句处理
        Args:
            tokens: context
            target_id: target word id

        Returns:
            transformed context and new target word position index
        """
        target_word = tokens[target_id]
        sentence = " ".join(
            [
                token if idx != target_id else self.tokenizer.mask_token        # 用mask替代了目标词，没有分词情况，此时
                for idx, token in enumerate(tokens)
            ]
        )
        # "Hello , how are you ?" 标点符号“,”和“?”前后出现了多余的空格，这在自然语言中是不符合书写习惯的。
        # tokenizer会先将输入文本拆分为若干token，再将这些token拼接成字符串。如果直接用“join”操作将token以空格连接，可能会引入多余的空格，
        sentence = self.tokenizer.clean_up_tokenization(sentence)
        sent_numerical = self.tokenizer.encode(sentence)[:-2]

        # TODO: test target position search  找到mask的位置  也就是不管是mask还是target都只占一个位置
        def get_target_id(indexes: List[int]) -> int:
            pos = 0
            mask_id = self.tokenizer._convert_token_to_id(self.tokenizer.mask_token)
            # 找一下[mask]标记的位置
            while pos < len(indexes):
                if indexes[pos] == mask_id:
                    break
                pos += 1
            else:
                raise ValueError("Can't find masked token")
            return pos

        target_id = get_target_id(sent_numerical)
        if not self.masked:         # 如果不用mask
            temp=self.tokenizer.encode(target_word)     # 查看末尾特殊标记
            target_codes = self.tokenizer.encode(target_word)[:-2]
            target_word_len=len(target_codes)
            if len(target_codes) > 1:
                if target_codes[0] == SPIECE_UNDERLINE:
                    target_codes = target_codes[1:]
                    target_word_len-=1
            # sent_numerical[target_id] = target_codes[0]       # 论文3 默认采用target第一个编码作为target位置的编码
            sent_numerical=sent_numerical[:target_id]+target_codes+sent_numerical[target_id+1:] # 更改为，都采用
        return sent_numerical, target_id,target_word_len

    # 模型预测扩展，处理多子词情况
    def get_multi_subword_predictions(
        self, predictions: torch.Tensor, sentences: List, target_ids: List
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Predict distribution with multi-subword acquistion.

        Args:
            predictions: model predictions from the last step.
            sentences: list of contexts
            target_ids: list of target word indexes

        Returns:
            predictions extended with multi-subwords and extended vocabulary
        """
        # TODO: refactor multi-subword generation process
        new_word2id = dict(self.word2id)
        extended_sentences = []
        for sentence, target_id in zip(sentences, target_ids):
            # Create new sentence with to contiguous masks
            sentence_ext = (
                sentence[:target_id]
                + [self.tokenizer.mask_token_id] * 2 # 将目标词位置替换为两个连续的 mask 标记，这样做的目的是为多子词预测预留两个位置。
                + sentence[target_id + 1 :]
            )
            extended_sentences.append(sentence_ext)
        inputs = self._prepare_inputs(
            extended_sentences, target_ids, multi_subword=True
        )
        extended_sent_predictions = self.get_predictions(*inputs, exclude_subword=True) # 排除原子词的影响

        new_words = []
        input_ids = inputs[0]
        for idx, target_id in enumerate(target_ids):    # 遍历目标位置
            (
                ext_input_ids,
                perm_mask,
                target_mapping,
                input_mask,
                top_probs,
                top_indexes,
            ) = self._create_extended_top_substitutions(
                # 扩展后的预测、原始输入和目标索引，提取出候选的 top-k 子词预测
                # （返回扩展后的输入、注意力掩码、目标映射、输入掩码、最高概率和对应的索引）。
                extended_sent_predictions,
                input_ids,
                idx,
                target_id,
                top_k=self.top_k_subword,
            )
            ext_predictions = self.get_predictions(
                ext_input_ids,
                perm_mask,
                target_mapping,
                input_mask,
                exclude_subword=False,      # 这里不同了
            )
            # 根据前面得到的概率、索引和扩展预测，生成完整的子词候选（例如，可能将拆分的子词片段组合为完整的词），
            # 同时可以利用 filter_words 进行过滤。
            completed_subwords = self._complete_subwords(
                top_probs, top_indexes, ext_predictions, filter_words=self.filter_words
            )
            new_words.append(completed_subwords)
            # Update word2id
            vocab_size = len(new_word2id)
            for key in completed_subwords:
                if key not in new_word2id:
                    new_word2id[key] = vocab_size
                    vocab_size += 1
        origin_vocab_size = len(self.word2id)
        vocab_size_diff = len(new_word2id) - origin_vocab_size
        if vocab_size_diff <= 0:
            return predictions, self.word2id
        # Extending predictions
        '''
        计算扩展后词汇表与原始词汇表的大小差 vocab_size_diff。如果没有新增子词，则直接返回原始预测和词汇表。
        如果有扩展，则为每个样本构建一个全零张量 subword_predictions，其形状为 [batch_size, vocab_size_diff]。
        遍历每个样本及其对应的候选子词，将预测值填入 subword_predictions 的对应位置（位置由新词在扩展词汇表中的索引减去原始词汇表的大小确定）。
        最后，将原始预测张量与 subword_predictions 在维度1上拼接，得到扩展后的预测张量，并返回该张量及更新后的词汇表。
        '''
        subword_predictions = torch.zeros(
            predictions.size(0), vocab_size_diff, dtype=torch.float32
        )
        for idx, words_dict in enumerate(new_words):
            for key, value in words_dict.items():
                if key not in self.word2id:
                    subword_predictions[
                        idx, new_word2id[key] - origin_vocab_size
                    ] = value
        extended_predictions = torch.cat([predictions, subword_predictions], dim=1)
        return extended_predictions, new_word2id    # 更新词汇表

    
    # 合并两个子词的预测结果，是否可以采纳？？输入分词后，输出的位置有多个。可以合并这种方式吧?
    # 数据集是否有短语的情况
    def _complete_subwords(
        self,
        first_subword_probs: torch.Tensor,
        first_subword_indexes: torch.Tensor,
        second_subword_probs: torch.Tensor,
        filter_words: bool = True,
    ) -> Dict[str, float]:
        """
        Combine two subwords in order to get whole words. The log-probability of combination
        is the mean of their log-probs.

        Args:
            first_subword_probs: tensor containing first subwords distribution.
            first_subword_indexes: tensor containing first subword indexes.
            second_subword_probs: tensor containing second subword distribution.
            filter_words: whether to remove words with punctuation and other special tokens.
 
        Returns:
            mapping from predicted word to their log-probabilities
        """
        indexes_1 = first_subword_indexes.squeeze().data.cpu()
        log_probs_1 = first_subword_probs.squeeze().data.cpu()

        # TODO: which type to use for results (how to aggregate data)
        results = {}
        for i, (idx_1, log_prob_1) in enumerate(zip(indexes_1, log_probs_1)):
            log_prob_2, idx_2 = torch.topk(
                second_subword_probs[i, :].view(1, 1, -1), k=1
            )
            log_prob_2 = log_prob_2.squeeze().item()
            pred_idx_2 = idx_2.squeeze().item()
            tok_1: str = self.tokenizer.convert_ids_to_tokens(idx_1.item())
            tok_2: str = self.tokenizer.convert_ids_to_tokens(pred_idx_2)

            # filter tokens with punctuation
            if tok_2.endswith(punctuation):
                continue

            if filter_words and tok_2.startswith("▁"):
                continue
            # 简单的字符串拼接
            subst = (tok_1 + tok_2).replace(SPIECE_UNDERLINE, " ").strip()
            # 两个子词对数概率的平均值
            mean_log_prob = (log_prob_1 + log_prob_2).item() / 2.0
            results[subst] = mean_log_prob      # 平均对数概率和组合后的词
            # 有点麻烦，占位的词然后各自在词典中的top-k候选词，组合，返回概率。是否可行？
        return results     
    # 这段代码的主要作用是在多子词生成过程中，为目标位置生成多个候选的“扩展”上下文，
    # 并构造相应的掩码和映射张量，以便后续预测下一个子词。
    def _create_extended_top_substitutions(
        self,
        log_probs: torch.Tensor,
        input_ids: torch.Tensor,
        idx: int,
        target_id: int,
        top_k: int = 100,
    ):
        """
        Preprocess inputs for multi-subword generation. Acquire top-k
        first subwords and their indexes. Place them in k new contexts
        and prepare inputs to predict next subword token.

        Args:
            log_probs: log-probs for first subword
            input_ids: input tensor from the previous multi-subword generation step
            idx: index of element from the batch
            target_id: index of the target word
            top_k: branching factor for multi-subword generation

        Returns:
            extended input tensor, permutation mask, target mapping, original input tensor, top k log-probs and indexes
        """
        top_log_probs, top_indexes = torch.topk(
            log_probs[idx, :].view(1, 1, -1), k=top_k
        )
        # 根据选出的top-k候选，然后复制
        ext_input_ids = input_ids[idx].repeat((top_k, 1))
        ext_input_ids[:, target_id] = top_indexes.squeeze()
        ext_input_ids = ext_input_ids.to(self.device)
        # 掩码通常用于控制 XLNet 中的自注意力机制，使得在预测下一个子词时屏蔽特定位置的信息，排列掩码，用于控制注意力；
        perm_mask = torch.zeros(
            (top_k, ext_input_ids.shape[1], ext_input_ids.shape[1]),
            dtype=torch.float,
            device=self.device,
        )
        perm_mask[:, :, target_id + 1] = 1.0
        target_mapping = torch.zeros(
            (top_k, 1, ext_input_ids.shape[1]), dtype=torch.float, device=self.device
        )
        target_mapping[:, 0, target_id + 1] = 1.0
        input_mask = None
        if self.use_input_mask:
            input_mask = (ext_input_ids == self.tokenizer.pad_token_id).type(
                torch.FloatTensor
            )
            input_mask = input_mask.to(perm_mask)
        return (
            ext_input_ids,
            perm_mask,
            target_mapping,
            input_mask,
            top_log_probs,
            top_indexes,
        )

    # 为 XLNet 模型准备输入数据，具体来说，它对输入的 token 序列进行 padding、生成 permutation mask（排列掩码）、
    # 构建 target mapping（目标映射）以及创建 input mask（输入掩码）。
    def _prepare_inputs(
        self, tokens: List[List[int]], target_ids: List[int], multi_subword: bool ,target_len:List[int]
    ):
        """
        Prepare input batch for processing with XLNet model: pad contexts to have same length,
        generate permutation mask according to masking strategy, create target mapping and input mask.

        Args:
            tokens: list of contexts
            target_ids: list of target word indexes
            multi_subword: whether to generate multi-subword words

        Returns:
            input tensor, permutation mask, target mapping and input mask for `transformers.XLNetLMHead` model.
        """
        # 多个 token 序列进行 padding，确保它们的长度一致。接着将处理后的列表转换为 PyTorch 张量，
        tokens_padded = self._pad_batch(tokens)
        input_ids = torch.tensor(tokens_padded)
        input_ids = input_ids.to(self.device)
        # 子词处理方法
        if not multi_subword:
            perm_mask, target_mapping = self._create_perm_mask_and_target_map(
                input_ids.shape[1], target_ids,target_len
            )
        else:
            perm_mask, target_mapping = self._create_perm_mask_and_target_map_sub_word(
                input_ids.shape[1], target_ids
            )
            '''
            将 input_ids 与 padding token 的 id（self.tokenizer.pad_token_id）进行比较，生成一个布尔张量，标识出哪些位置是 padding。
            将布尔张量转换为浮点型（通常 1 表示 padding 的位置），并确保该张量与 permutation mask 在同一设备上，方便后续计算。
            '''
        input_mask = None
        if self.use_input_mask:
            input_mask = (input_ids != self.tokenizer.pad_token_id).type(
                torch.FloatTensor
            )
            # input_mask = input_mask.to(perm_mask)
        return input_ids, perm_mask, target_mapping, input_mask

    def _pad_batch(self, token_ids: List[List[int]]) -> List[List[int]]:
        """
        Pad given batch of contexts.

        Args:
            token_ids: list of contexts

        Returns:
            list of padded contexts all having the same length
        """
        max_len = max([len(ids) for ids in token_ids])
        for ids in token_ids:
            ids.extend(
                [self.tokenizer._convert_token_to_id(self.tokenizer.pad_token)]
                * (max_len - len(ids))
            )
        return token_ids

    def _create_perm_mask_and_target_map(
        self, seq_len: int, target_ids: List[int],subword_lengths: List[int]
    ) -> Tuple[torch.Tensor, ...]:
        """
        Generates permutation mask and target mapping.
        If `self.masked` is true then there is no word that sees target word through attention.
        If it is false then only target word doesn't see itself.

        Args:
            seq_len: length of the sequence (context)
            target_ids: target word indexes
            子词的长度，原始方法只处理了一个位置。
        Returns:
            two `torch.Tensor`s: permutation mask and target mapping
        """
        assert isinstance(target_ids[0], int), "One target per sentence"
        batch_size = len(target_ids)
        # perm_mask = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.float)
        # target_mapping = torch.zeros((batch_size, 1, seq_len))  # 标记目标词的位置

        # for idx, target_id in enumerate(target_ids):
        #     perm_mask[idx, :, target_id] = 1.0  # 设置当前批次中所有位置对目标位置 target_id 的注意力为不可见（值为 1.0）。
        #     target_mapping[idx, 0, target_id] = 1.0
        #     if not self.masked:
        #         perm_mask[idx, :, target_id] = 0.0  # 允许所有位置关注目标位置。（xlnet自回归，双上下文）
        #         perm_mask[idx, target_id, target_id] = 1.0 # 仅禁止目标位置关注自身 
        #  上方原始操作，只关注一个首位置         

        # 初始化排列掩码和目标映射张量,关心子词
        perm_mask = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.float)
        target_mapping = torch.zeros((batch_size, max(subword_lengths), seq_len), dtype=torch.float)

        for idx in range(batch_size):
            start_pos = target_ids[idx]
            length = subword_lengths[idx]

            for i in range(length):
                current_pos = start_pos + i
                # 设置排列掩码：默认情况下，所有位置都不能看到当前子词
                perm_mask[idx, :, current_pos] = 1.0
                # 设置目标映射：标识目标子词的位置
                target_mapping[idx, i, current_pos] = 1.0       # 不影响操作，只关注标记为1的地方

                if not self.masked:
                    # 如果不使用掩码，允许所有位置关注当前子词
                    perm_mask[idx, :, current_pos] = 0.0
                    perm_mask[idx, current_pos, current_pos] = 1.0


        perm_mask = perm_mask.to(self.device)
        target_mapping = target_mapping.to(self.device)
        return perm_mask, target_mapping

    def _create_perm_mask_and_target_map_sub_word(
        self, seq_len: int, target_ids: List[int]
    ) -> Tuple[torch.Tensor, ...]:
        """
        Generates permutation mask and target mapping for multi-subword geenration.
        If `self.masked` is true then there is no word that sees target word through attention.
        If it is false then only target word doesn't see itself.
        ATTENTION. Now we only support generation of words that consists of two subwords.

        Args:
            seq_len: length of the sequence (context)
            target_ids: target word indexes

        Returns:
            two `torch.Tensor`s: permutation mask and target mapping
        """
        batch_size = len(target_ids)
        perm_mask = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.float)
        target_mapping = torch.zeros((batch_size, 2, seq_len))

        for idx, target_id in enumerate(target_ids):
            perm_mask[idx, :, (target_id, target_id + 1)] = 1.0
            perm_mask[idx, target_id + 1, target_id] = 0.0
            target_mapping[idx, 0, target_id] = 1.0
            target_mapping[idx, 1, target_id + 1] = 1.0
        perm_mask = perm_mask.to(self.device)
        target_mapping = target_mapping.to(self.device)
        return perm_mask, target_mapping

    def _exclude_special_symbols(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Zero out probabilities related to special symbols e.g. punctuation.

        Args:
            predictions: original predictions

        Returns:
            filtered predictions
        """
        mask = torch.zeros(predictions.size(-1), dtype=torch.bool)
        mask[self.all_special_ids] = True
        # predictions[:,mask] = -1e9
        predictions[:,:,mask] = -1e9      # 只替换第一个的时候用
        return predictions

    def _exclude_subword_symbols(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Zero out probabilities related to subwords.

        Args:
            predictions: original predictions

        Returns:
            filtered predictions
        """
        mask = torch.zeros(predictions.size(-1), dtype=torch.bool)
        mask[self.spiece_ids] = True
        predictions[:,:,mask] = -1e9       # 多维度
        # predictions[:,mask] = -1e9 
        return predictions
    
    @overrides
    def get_log_probs(
        self, tokens_lists: List[List[str]], target_ids: List[int],target_pos_tag:List[str]
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Compute probabilities for each target word in tokens lists.
        If `self.embedding_similarity` is true will return similarity scores.
        Process all input data with batches.

        Args:
            tokens_lists: list of tokenized sequences,  each list corresponds to one tokenized example.
            target_ids: indices of target words from all tokens lists.
                E.g.:
                token_lists = [["Hello", "world", "!"], ["Go", "to" "stackoverflow"]]
                target_ids_list = [1,2]
                This means that we want to get probability distribution for words "world" and "stackoverflow".
        Returns:
            `numpy.ndarray` of log-probs distribution over vocabulary and the relative vocabulary.
        """
        if self.embedding_similarity:
            # TODO: should we exclude special symbols?
            # Getting scores according to word embeddings similarity
            logits = self.get_emb_similarity(tokens_lists, target_ids)
            return logits, self.word2id
        # Use full model to predict masked word
        # logits, word2id,len_sub_words = self.predict(tokens_lists, target_ids)
        # 这里的logits已经是1个位置的了
        logits, word2id,len_subwords = self.predict(tokens_lists, target_ids,target_pos_tag)
        # processed=[]
        # for idx,logit in enumerate(logits):
        #     # logit 形状为 [num_subwords, vocabsize],且这里的num_subwords为当前批次里的最大子词数，当前批次的logits固定的
        #     sample_tensor = torch.tensor(logit)
        #     if len_subwords[idx]>=2:
        #         aggregated = outputlogits_stategy(sample_tensor.unsqueeze(0), self.weights, self.decayrate)  # 增加一个维度以符合函数输入要求
        #         processed.append(aggregated.squeeze(0).cpu().numpy())  # 去除多余维度
        #     # if len(logit)>=2:     同一批次中，只要有数据分割子词，其它同样就会如此，维度对齐了，所以需要筛选
        #         # print(f'分割子词了，数量为{len(logit)}')
        #     else:
        #         processed.append(sample_tensor[0].cpu().numpy())
        # logits=processed
        logits=np.vstack(logits)     # （batch_size,vocab_size） 最后经过输出都是1条向量
        
        
        return logits, word2id

    def predict(
        self, tokens_lists: List[List[str]], target_ids: List[int],target_pos_tag: List[str]
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Get log probability distribution over vocabulary.

        Args:
            tokens_lists: list of contexts
            target_ids: target word indexes

        Returns:
            `numpy.ndarray`, matrix with rows - log-prob distribution over vocabulary.
        """
        org_words=[]
        for i, target_id in enumerate(target_ids):
            org_words.append(tokens_lists[i][target_id])

        # 这里返回的是经过tokenizer后的值，也就是字典里的id序列，怎么转成embedding且合并目标位置？
        numerical_sentences, target_positions,target_len = self._numericalize_batch(
            tokens_lists=tokens_lists, target_ids=target_ids
        )
        # input_ids长度一样否
        input_ids, perm_mask, target_mapping, input_mask = self._prepare_inputs(
            numerical_sentences, target_positions, multi_subword=False,target_len=target_len
        )
        predictions = self.get_predictions(   # size(50,32000)=(batch_size,vocab_size)
            input_ids, input_mask,target_positions,target_len,target_pos_tag,org_words
        )
        predictions = predictions.cpu()        # why?

        word2id = self.word2id
        if self.multi_subword:
            # TODO: check implementation of multi sub-word generation process
            predictions, word2id = self.get_multi_subword_predictions(
                predictions, numerical_sentences, target_positions
            )
        predictions = predictions.numpy()
        return predictions, word2id,target_len
    
    #  合并为一个的版本，需要依次处理，无法批处理
    # def get_predictions(
    #     self,
    #     input_ids: torch.Tensor,
    #     input_mask: torch.Tensor,
    #     target_positions:List[int],
    #     target_len:List[int],
    #     target_pos_tag:List[str],
    #     target_words:List[str],
    #     exclude_subword: bool = True,
    # ) -> torch.Tensor:
    
    #     """
    #     Get XLNet model predictions for a given input.

    #     Args:
    #         input_ids: input matrix
    #         perm_mask: mask to indicate the attention pattern for each input token with values
    #             selected in ``[0, 1]``:
    #             If ``perm_mask[k, i, j] = 0``, i attend to j in batch k;
    #             if ``perm_mask[k, i, j] = 1``, i does not attend to j in batch k.
    #         target_mapping: mask to indicate the output tokens to use.
    #             If ``target_mapping[k, i, j] = 1``, the i-th predict in batch k is on the j-th token.
    #         input_mask: mask to avoid performing attention on padding token indices.
    #         exclude_subword: whether to remove subwords from final distribution (zero out probabilities)

    #     Returns:
    #         predicted distribution over vocabulary
    #     """
    #     # 预处理，将input ids转为embedding,占一个位置,每一条数据的目标位置的词又不一样，所以怎么能当成批处理呢？？
    #     # 获取嵌入层对象
    #     embedding_layer = self.model.get_input_embeddings()
    #     res=[]
    #     with torch.no_grad():
    #         embeddings = embedding_layer(input_ids)  # 原始嵌入
            
    #         # 遍历处理每个样本
    #         for idx, (target_pos, subword_len,orig_word) in enumerate(zip(target_positions,target_len,target_words)):
    #             left = embeddings[idx, :target_pos]
    #             right = embeddings[idx, target_pos+subword_len:]
    #             subword=self.tokenizer.tokenize(orig_word)

    #             processed_embeds = self.preprocessor.process_word(
    #                 original_word=orig_word,  
    #                 subwords=subword,  
    #                 pos_tag=target_pos_tag[idx][0]
    #             )
    #             if processed_embeds.dim()>2:
    #                 processed_embeds=np.squeeze(processed_embeds,axis=0)         
    #             new_embeddings=torch.cat([left, processed_embeds, right], dim=0).unsqueeze(0)
                
    #             if self.use_input_mask:
    #                 left_mask = input_mask[idx, :target_pos]
    #                 target_mask = torch.ones(len(processed_embeds))
    #                 right_mask = input_mask[idx, target_pos+subword_len:]
    #                 new_mask=torch.cat([left_mask, target_mask, right_mask], dim=0).unsqueeze(0)
                    
            
    #             # 前向传播优化
    #             new_mask=new_mask.to(self.device)

    #             outputs = self.model(inputs_embeds=new_embeddings,attention_mask=new_mask)
    #             logits = outputs.logits 
    #             res.append(logits.squeeze(0))

    #         target_logits = torch.stack([res[i][target_positions[i], :] for i in range(len(res))]) / self.temperature

        
    #         predictions = self._exclude_special_symbols(target_logits)
    #         if exclude_subword:
    #             predictions = self._exclude_subword_symbols(predictions)        
            
            
    #     return predictions



    # 此为替换第一个位置的算法，实验mix-up在第一个位置，然后结果的子词取平均
    def get_predictions(
        self,
        input_ids: torch.Tensor,
        input_mask: torch.Tensor,
        target_positions:List[int],
        target_len:List[int],
        target_pos_tag:List[str],
        target_words:List[str],
        exclude_subword: bool = True,
    ) -> torch.Tensor:
    
        """
        Get XLNet model predictions for a given input.

        Args:
            input_ids: input matrix
            perm_mask: mask to indicate the attention pattern for each input token with values
                selected in ``[0, 1]``:
                If ``perm_mask[k, i, j] = 0``, i attend to j in batch k;
                if ``perm_mask[k, i, j] = 1``, i does not attend to j in batch k.
            target_mapping: mask to indicate the output tokens to use.
                If ``target_mapping[k, i, j] = 1``, the i-th predict in batch k is on the j-th token.
            input_mask: mask to avoid performing attention on padding token indices.
            exclude_subword: whether to remove subwords from final distribution (zero out probabilities)

        Returns:
            predicted distribution over vocabulary
        """
        # 预处理，将input ids转为embedding,占一个位置,每一条数据的目标位置的词又不一样，所以怎么能当成批处理呢？？
        # 获取嵌入层对象
        embedding_layer = self.model.get_input_embeddings()
        input_mask=input_mask.to(self.device)
        res=[]
        with torch.no_grad():
            embeddings = embedding_layer(input_ids).to(self.device)  # 原始嵌入
            
            # 遍历处理每个样本
            for idx, (target_pos, subword_len,orig_word) in enumerate(zip(target_positions,target_len,target_words)):
                subword=self.tokenizer.tokenize(orig_word)

                processed_embeds = self.preprocessor.process_word(
                    original_word=orig_word,  
                    subwords=subword,  
                    pos_tag=target_pos_tag[idx][0]
                )
                
                embeddings[idx][target_pos]=processed_embeds.to(self.device)

            
            outputs = self.model(inputs_embeds=embeddings,attention_mask=input_mask)
            logits = outputs.logits 
            
            target_position = torch.tensor(target_positions).to(self.device)
            target_length=torch.tensor(target_len).to(self.device)
            
        
            predictions = self._exclude_special_symbols(logits)
            if exclude_subword:
                predictions = self._exclude_subword_symbols(predictions)
            
            batch_size = logits.size(0)
            target_logits_list=[]
            # 取出来的时候就做平均，何乐而不为
            for i in range(batch_size):
                subword_logits = predictions[i, target_position[i] : target_position[i] + target_length[i], :]
                avg_logits = subword_logits.mean(dim=0) / self.temperature
                target_logits_list.append(avg_logits)
            target_logits_list=torch.stack(target_logits_list)
        return target_logits_list            
            
        # return predictions

    def get_ordered_synonyms(self, original_word: str, synonyms_from_wordnet: List[str]):
        loaded = XLNetProbEstimator.loaded[self.model_name]
        model = loaded["model"]
        tokenizer = loaded["tokenizer"]
        
        # 准备批处理输入，原始词作为第一个元素
        words = [original_word] + synonyms_from_wordnet
        
        # 批量编码并处理填充
        encodings = tokenizer(words, padding=True, truncation=True, return_tensors='pt').to(self.device)
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        
        # 模型前向传播（批量处理）
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # 提取每个单词的有效subwords嵌入并计算均值
        batch_size = input_ids.size(0)
        lengths = attention_mask.sum(dim=1)  # 各样本的有效token长度
        
        embeddings = []
        for i in range(batch_size):
            length = lengths[i].item()
            if length < 2:
                raise ValueError(f"Invalid token length for word: '{words[i]}'")
            
            # 提取subwords位置（排除CLS和SEP）
            start_idx = 1
            end_idx = length - 1  # 切片不包含end_idx，故取到SEP前一位
            subword_embeds = outputs[0][i, start_idx:end_idx, :]
            mean_embed = subword_embeds.mean(dim=0)
            embeddings.append(mean_embed)
        
        embeddings = torch.stack(embeddings)
        
        # 分离原始词和同义词嵌入
        original_embed = embeddings[0]
        synonym_embeds = embeddings[1:]
        
        # 批量计算余弦相似度
        similarities = F.cosine_similarity(original_embed.unsqueeze(0), synonym_embeds, dim=1)
        
        # 按相似度降序排序
        sorted_indices = torch.argsort(similarities, descending=True)
        sorted_words = [synonyms_from_wordnet[idx] for idx in sorted_indices.cpu().numpy()]
        
        return sorted_words