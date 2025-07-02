'''
输入bert前的处理+返回概率分布
'''
import json
import os
from string import punctuation
from typing import NoReturn, Dict, List, Tuple
import numpy as np
import torch
from overrides import overrides
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM 
# MLM任务专用模型 作用：在BertModel基础上添加MLM预测头，用于预测被遮蔽词。结构：BertModel + 线性分类层（将隐藏状态映射到词汇表）。
from lexsubgen.prob_estimators.embsim_estimator import EmbSimProbEstimator
from lexsubgen.embedding_strategy.input_embedding_strategy import EmbeddingPreprocessor
# from transformers import BertModel  # BERT的核心架构，输出上下文相关的隐藏状态。包含嵌入层和多层Transformer编码器。输出最后一层（或所有层）的隐藏表示，不直接用于任务预测。
from lexsubgen.embedding_strategy.output_embedding_strategy import outputlogits_stategy



class BertProbEstimator(EmbSimProbEstimator):
    # 类属性，用于存储加载的词嵌入和分词器
    _word_embeddings = None
    _tokenizer = None

    def __init__(
        self,
        mask_type: str = "not_masked",
        model_name: str = "bert-base-cased",        # 不同于config中的bert
        embedding_similarity: bool = False, 
        temperature: float = 1.0,
        use_attention_mask: bool = True,
        sim_func: str = "dot-product",
        use_subword_mean: bool = False,         # 未出现的词取平均，就调用父类方法
        verbose: bool = False,
        cuda_device: int = 0,

        stratagy_input_embedding:str="gauss",          # gauss dropout mask keep mix-up    
        mixup_alpha: float = 0.25,                                      # before 0.5
        synonym_topn: int = 10,                         # 这里更改,10足够
        gauss_sigma: float = 0.01,
        weights:str="exponential",                                        #  ['mean', 'first', 'linear', 'exponential']
        decay_rate:float=0.9                        # 前为0.5
    ):
        """
        Probability estimator based on the BERT model.
        See J. Devlin et al. "BERT: Pre-training of Deep
        Bidirectional Transformers for Language Understanding".
        Args:
            mask_type: the target word masking strategy.        输入target处的选择
            model_name: BERT model name, see https://github.com/huggingface/transformers
            embedding_similarity: whether to compute BERT embedding similarity instead of the full model
                true:计算目标词的词嵌入和上下文词嵌入的相似度——评估词汇在特定上下文中的语义相似性。
                false:基于上下文预测目标词的概率分布。
            temperature: temperature by which to divide log-probs
            use_attention_mask: whether to zero out attention on padding tokens
                是否把填充的内容注意力设置为0，为了更好的聚焦于本身的句子上
            cuda_device: CUDA device to load model to
            sim_func: name of similarity function to use in order to compute embedding similarity
            use_subword_mean: how to handle words that are splitted into multiple subwords when computing embedding similarity
                如果一个词被拆分成多个子词（例如 BERT 模型的 WordPiece 分词），此参数决定是否使用这些子词的平均词嵌入作为该词的表示。
            verbose: whether to print misc information
                作用是控制日志输出

            target_output_embedding_type:控制输出target处的embedding选择
        """
        super(BertProbEstimator, self).__init__(
            model_name=model_name,
            temperature=temperature,
            sim_func=sim_func,
            verbose=verbose,
            weights=weights,
            stratagy_input_embedding=stratagy_input_embedding
        )
        self.mask_type = mask_type
        self.embedding_similarity = embedding_similarity
        self.use_attention_mask = use_attention_mask
        self.use_subword_mean = use_subword_mean
        self.cuda_device=cuda_device
        
        self.stratagy_input_embedding=stratagy_input_embedding
        self.mixup_alpha = mixup_alpha
        self.synonym_topn = synonym_topn
        self.gauss_sigma = gauss_sigma
        self.weights=weights        # 输出的合并方式
        self.decayrate=decay_rate   #指数方式


        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)   # 设置！
        if self.cuda_device != -1 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.cuda_device}")
        else:
            self.device = torch.device("cpu")

        self.descriptor = {
            "Prob_estimator": {
                "name": "bert",
                "class": self.__class__.__name__,
                "model_name": self.model_name,
                "mask_type": self.mask_type,
                "embedding_similarity": self.embedding_similarity,
                "temperature": self.temperature,
                "use_attention_mask": self.use_attention_mask,
                "use_subword_mean": self.use_subword_mean,
                "target_output_embedding_type":self.weights
            }
        }
        # 初始化预处理器，gauss keep mask mix-up drop-out
        # 外部加载模型并提取词向量
        # from transformers import BertModel
        # bert_model = BertModel.from_pretrained("bert-base-uncased")
        # word_embeddings = bert_model.embeddings.word_embeddings

        self.register_model()   # 默认在cpu     提前加载，放在前面
        self.logger.debug(f"Probability estimator {self.descriptor} is created.")
        self.logger.debug(f"Config:\n{json.dumps(self.descriptor, indent=4)}")

        # 检查类属性是否已经加载了词嵌入和分词器
        if BertProbEstimator._word_embeddings is None or BertProbEstimator._tokenizer is None:
            BertProbEstimator._word_embeddings = torch.tensor(
                self.loaded[self.model_name]["embeddings"], device=self.device
            )
            BertProbEstimator._tokenizer = self.loaded[self.model_name]["tokenizer"]
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

    # 加载词权重，多个实例化对象只加载一次  嵌入移动到指定设备  register中加载模型后未统一设备
    # 已经在register中加载了，怎么又加载一次呢？
    # @staticmethod
    # def load_token_embeddings(model_name: str, device: torch.device):
    #     # tokenizer = BertTokenizer.from_pretrained(model_name)           # 加载了两次
    #     model = BertForMaskedLM.from_pretrained(model_name)
    #     embeddings = model.bert.embeddings.word_embeddings.weight.data 
    #     return embeddings.to(device)

    



    # 属性：只读属性，返回与当前模型相关的 BERT 分词器。分词器用于将文本转换为模型可以处理的输入格式。——wordpiece
    @property
    def tokenizer(self):
        """
        Model tokenizer.
        Returns:
            `transformers.BertTokenizer` tokenzier related to the model
        """
        return self.loaded[self.model_name]["tokenizer"]
    

    # 注册模型。如果指定的 BERT 模型尚未加载，则加载模型、分词器、词汇表和词嵌入，并将其注册到类的静态变量 loaded 中。如果模型已加载，则增加其引用计数。这有助于节省计算资源，避免重复加载相同的模型。
    def register_model(self) -> NoReturn:
        """
        If the model is not registered this method creates that model and
        places it to the model register. If the model is registered just
        increments model reference count. This method helps to save computational resources
        e.g. when combining model prediction with embedding similarity by not loading into
        memory same model twice.
        """
        if self.model_name not in BertProbEstimator.loaded: # 父类的loaded
            bert_model = BertForMaskedLM.from_pretrained(self.model_name)       # 实现了模型加载，字典
            bert_model=bert_model.to(self.device).eval()                        # 指定设备
            bert_tokenizer = BertTokenizer.from_pretrained(
                self.model_name, do_lower_case=self.model_name.endswith("uncased")
            )
            bert_word2id = BertProbEstimator.load_word2id(bert_tokenizer)
            bert_filter_word_ids = BertProbEstimator.load_filter_word_ids(
                bert_word2id, punctuation
            )
            word_embeddings = (
                bert_model.bert.embeddings.word_embeddings.weight.data.cpu().numpy()
            )
            # 这里将模型的loaded中传入了，因此父类中就可以直接调用了
            BertProbEstimator.loaded[self.model_name] = {
                "model": bert_model,
                "tokenizer": bert_tokenizer,
                "embeddings": word_embeddings,
                "word2id": bert_word2id,
                "filter_word_ids": bert_filter_word_ids,
            }
            BertProbEstimator.loaded[self.model_name]["ref_count"] = 1
        else:
            BertProbEstimator.loaded[self.model_name]["ref_count"] += 1
    # 获取未知词的向量表示。如果 use_subword_mean 为 True，则将未知词拆分为子词，并返回这些子词嵌入的平均值。
    # 否则，调用父类的 get_unk_word_vector 方法获取默认的零向量。
    # bert词汇表可能不包含这个单词
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
            sub_token_ids = self.tokenizer.encode(word)[1:-1]       # CLS esp标记
            # 使用 BERT 分词器（BertTokenizer）将输入的单词 word 编码成模型可以理解的 token ID（整数序列）
            mean_vector = self.embeddings[sub_token_ids, :].mean(axis=0, keepdims=True)
            return mean_vector
        return super(BertProbEstimator, self).get_unk_word_vector(word)
    

        # 加载模型的词汇表，并返回一个从词汇到索引的映射字典。这对于将词汇转换为模型输入的 ID 非常有用
        # pad->0 unk->1    playing->play+##ing代表两个子词
        # 字典的具体内容取决于所使用的 BERT 模型和其训练时使用的词汇表。
        # 不同的模型可能会有不同的词汇表和相应的索引映射。
        # 因此，word2id 字典的内容是模型特定的。
    @staticmethod
    def load_word2id(tokenizer: BertTokenizer) -> Dict[str, int]:
        """
        Loads model vocabulary in the form of mapping from words to their indexes.
        Args:
            tokenizer: `transformers.BertTokenizer` tokenizer
        Returns:
            model vocabulary
        """
        word2id = dict()
        for word_idx in range(tokenizer.vocab_size):
            word = tokenizer.convert_ids_to_tokens([word_idx])[0]
            word2id[word] = word_idx
        return word2id
    

    # 根据给定的过滤字符（如标点符号）生成一个词汇表索引的列表，表示需要从输出分布中过滤掉的词汇。这有助于避免模型生成无意义的标点符号。
    @staticmethod
    def load_filter_word_ids(word2id: Dict[str, int], filter_chars: str) -> List[int]:
        """
        Gathers words that should be filtered from the end distribution, e.g.
        punctuation.

        Args:
            word2id: model vocabulary
            filter_chars: words with this chars should be filtered from end distribution.

        Returns:
            Indexes of words to be filtered from the end distribution.
        """
        filter_word_ids = []
        set_filter_chars = set(filter_chars)
        for word, idx in word2id.items():
            if len(set(word) & set_filter_chars):
                filter_word_ids.append(idx)
        return filter_word_ids


    # 返回需要从输出分布中过滤掉的词汇的索引列表。这些词汇通常是标点符号或其他不需要的词汇。
    @property
    def filter_word_ids(self) -> List[int]:
        """
        Indexes of words to be filtered from the end distribution.
        Returns:
            list of indexes
        """
        return self.loaded[self.model_name]["filter_word_ids"]


    # 将给定的词汇列表转换为 BERT 模型可以处理的子词列表。它使用 BERT 分词器将每个词汇拆分为子词，并返回这些子词的列表。
    # 非首词会添加“##”，无多余空格，没有字符串拼接          直接作用于词元，所以像 ','两边不存在都有空格的情况  .join(' ')
    def bert_tokenize_sentence(
        self, tokens: List[str], tokenizer: BertTokenizer = None
    ) -> List[str]:
        """
        Auxiliary function that tokenize given context into subwords.

        Args:
            tokens: list of unsplitted tokens.
            tokenizer: tokenizer to be used for words tokenization into subwords.

        Returns:
            list of newly acquired tokens
        """
        if tokenizer is None:
            tokenizer = self.tokenizer
        bert_tokens = list()
        for token in tokens:
            bert_tokens.extend(tokenizer.tokenize(token))       # 将输入的文本分割成一个个的词元（tokens）
        return bert_tokens

    # 将一批上下文和目标词索引转换为适合 BERT 模型处理的格式
    # 返回target位置，注意：bert是加CLS和sep，其它模型不一定，在输入模型计算分数的时候，需要添加标记
    def bert_prepare_batch(
        self,
        batch_of_tokens: List[List[str]],
        batch_of_target_ids: List[int],
        tokenizer: BertTokenizer = None,
    ) -> Tuple[List[List[str]], List[int]]:
        """
        Prepares batch of contexts and target indexes into the form
        suitable for processing with BERT, e.g. tokenziation, addition of special tokens
        like [CLS] and [SEP], padding contexts to have the same size etc.
        Args:
            batch_of_tokens: list of contexts
            batch_of_target_ids: list of target word indexes
            tokenizer: tokenizer to use for word tokenization
        Returns:
            transformed contexts and target word indexes in these new contexts
        """
        if tokenizer is None:   # 配置文件有配置，不用默认的
            tokenizer = self.tokenizer

        bert_batch_of_tokens, bert_batch_of_target_ids,bert_len_of_tokens,origin_word = list(), list(),list(),list()
        temp=list()
        max_seq_len = 0     # 最大长度初始为0，并没有规定最大值常量
        # L target R ——sentence句子处理
        for tokens, target_idx in zip(batch_of_tokens, batch_of_target_ids):
            left_context = ["[CLS]"] + self.bert_tokenize_sentence( # 这里是分词后的left_context,target_id变大
                tokens[:target_idx], tokenizer
            )
            right_context = self.bert_tokenize_sentence(
                tokens[target_idx + 1 :], tokenizer
            ) + ["[SEP]"]

            target_tokens = self.bert_tokenize_sentence([tokens[target_idx]], tokenizer)    # 使用bert模型编码成词元， playing = play + ing
            length_target_tokens=1
            # 目标词的mask策略
            if self.mask_type == "masked":
                target_tokens = ["[MASK]"]  # 整体mask
            elif self.mask_type == "not_masked":    # 输出默认提取的【首个位置】的语义——会不会不够啊？
                length_target_tokens=len(target_tokens)
            else:
                raise ValueError(f"Unrecognised masking type {self.mask_type}.")
            
            # xlnet给的灵感，是不是有空格的存在？？？
            context = left_context + target_tokens + right_context
            seq_len = len(context)
            if seq_len > max_seq_len:
                max_seq_len = seq_len

            bert_batch_of_tokens.append(context)
            bert_batch_of_target_ids.append(len(left_context))
            bert_len_of_tokens.append(length_target_tokens)
            origin_word.append(tokens[target_idx])        # 注意句首添加了CLS
        

        bert_batch_of_tokens = [
            tokens + ["[PAD]"] * (max_seq_len - len(tokens))
            for tokens in bert_batch_of_tokens
        ]
        return bert_batch_of_tokens, bert_batch_of_target_ids,bert_len_of_tokens,origin_word
    


    # 获取 目标词 的对数概率分布。它首先调用 bert_prepare_batch 方法准备输入数据，然后将其转换为模型输入的 ID。
    # 接下来，将输入数据传递给 BERT 模型，获取输出的 logits。一个批次一个批次的predict
    # 最后，返回目标词的对数概率分布。
    # def predict(
    #     self, tokens_lists: List[List[str]], target_ids: List[int],
    # ) -> np.ndarray:
    #     """
    #     Get log probability distribution over vocabulary.
    #     Args:
    #         tokens_lists: list of contexts
    #         target_ids: target word indexes
    #     Returns:
    #         `numpy.ndarray`, matrix with rows - log-prob distribution over vocabulary.
    #     """
    #     bert_tokens, bert_target_ids,length_target_tokens,orig_word = self.bert_prepare_batch(tokens_lists, target_ids)
    #     input_ids = np.vstack(  # 垂直方向（行）堆叠数组，一维变二维
    #         [self.tokenizer.convert_tokens_to_ids(tokens) for tokens in bert_tokens]    # 将分词后的词转为数字，bert无法直接处理字符
    #     )
    #     input_ids = torch.tensor(input_ids).to(self.device)

    #     attention_mask = None
    #     if self.use_attention_mask:
    #         attention_mask = (input_ids != self.tokenizer.pad_token_id).type(
    #             torch.FloatTensor
    #         )
    #         attention_mask = attention_mask.to(input_ids)
    #     # 核心：
    #     with torch.no_grad():
    #         outputs = self.model(input_ids=input_ids, attention_mask=attention_mask) # 父类model方法(属性)
    #         logits = outputs[0]
    #         # print(f"logits维度是:{logits.shape}")
    #         # logits:模型最后一层全连接层的原始输出，没有经过 softmax 等激活函数处理，这些值可以用来计算各个类别的概率。没有归一化
    #         # 词汇表的各个词的得分，即作为下一个词出现的可能性，需经过softmax
    #         # 维度是
    #         # output有多个参数
    #         # logits = np.vstack(
    #         #     [
    #         #         logits[idx, target_idx, :].cpu().numpy() / self.temperature     # 同样默认的取第一个
    #         #         # 直接除以temperature 为什么：温度参数可以用来控制生成结果的随机性和多样性。   ————直接取target位置处的预测值并除以温度系数
    #         #         for idx, target_idx in enumerate(bert_target_ids)
    #         #     ]
    #         # ) 批处理，默认取第一个
    #         logits=[
    #                 logits[idx, target_idx:target_idx+len_target_tokens, :].cpu().numpy() / self.temperature     # 同样默认的取第一个
    #                 # 直接除以temperature 为什么：温度参数可以用来控制生成结果的随机性和多样性。
    #                 for idx, (target_idx,len_target_tokens )in enumerate(zip(bert_target_ids,length_target_tokens))
    #                 ]
    #                 #这里返回的，有的维度为（1，vocabsize）有的为（2，vocabsize）
    #         return logits
        
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
            logits = self.get_emb_similarity(tokens_lists, target_ids)
        else:
            logits = self.predict(tokens_lists, target_ids,target_pos_tag)
            # 下为输出的探究,当考虑多个位置的时候需要
            processed=[]
            for idx,logit in enumerate(logits):
                    # logit 形状为 [num_subwords, vocabsize]
                sample_tensor = torch.tensor(logit)
                aggregated = outputlogits_stategy(sample_tensor.unsqueeze(0), self.weights, self.decayrate)  # 增加一个维度以符合函数输入要求
                processed.append(aggregated.squeeze(0).cpu().numpy())  # 去除多余维度
            logits=processed
            logits=np.vstack(logits)     # （batch_size,vocab_size） 最后经过输出都是1条向量
            # logits=np.vstack(logits)
        logits[:, self.filter_word_ids] = -1e9
        return logits, self.word2id

    # 计算每个目标词的对数概率分布，logits经过soft-max后才是概率维度：(batch_size,sequence_length,vocab_size)
    # @overrides
    # def get_log_probs(
    #     self, tokens_lists: List[List[str]], target_ids: List[int]
    # ) -> Tuple[np.ndarray, Dict[str, int]]:
    #     """
    #     Compute probabilities for each target word in tokens lists.
    #     If `self.embedding_similarity` is true will return similarity scores.
    #     Process all input data with batches.
    #     Args:
    #         tokens_lists: list of tokenized sequences,  each list corresponds to one tokenized example.
    #         target_ids: indices of target words from all tokens lists.
    #             E.g.:
    #             token_lists = [["Hello", "world", "!"], ["Go", "to" "stackoverflow"]]
    #             target_ids_list = [1,2]
    #             This means that we want to get probability distribution for words "world" and "stackoverflow".
    #     Returns:
    #         `numpy.ndarray` of log-probs distribution over vocabulary and the relative vocabulary.
    #     """
    #     # 老六，论文的核心，config配置文件的第三个或者第二个模型的embedding_similarity设置为true经过这儿
    #     if self.embedding_similarity:
    #         logits = self.get_emb_similarity(tokens_lists, target_ids)  
    #         # 目标词和bert模型词汇表中的其它所有词的相似度
    #     else:
    #         logits = self.predict(tokens_lists, target_ids)
    #         # 单纯的目标词的对数概率（50，28996）将其它非常见词设置为-1e9
    #         # numpy处理成tensor类型返回，predict函数得到的logits是numpy 
    #         # 输出处理支持批量（batch_size,num_subwords,vocab_size）——批量个der
    #         processed=[]
    #         for sample in logits:
    #             # sample 形状为 [num_subwords, vocabsize]
    #             sample_tensor = torch.tensor(sample)
    #             aggregated = outputlogits_stategy(sample_tensor, self.weights, self.decayrate)  # 增加一个维度以符合函数输入要求
    #             processed.append(aggregated.squeeze(0).cpu().numpy())  # 去除多余维度
    #         logits=processed

    #     logits=np.vstack(logits)     # （batch_size,vocab_size）
    #     logits[:, self.filter_word_ids] = -1e9
    #     return logits, self.word2id
    #     #注意，这里返回的是两个值


    #     ## ____________________________________________________input embedding处理的具体函数，嵌入处理后进行模型预测
    
    # def predict(self,  tokens_lists: List[List[str]], target_ids: List[int]):
    #     # 准备batch数据，bert_tokens有子词标记，填充好了的，pad，一个批次的数据相同
    #     bert_tokens, bert_target_ids, length_target_tokens,original_words = self.bert_prepare_batch(
    #         tokens_lists, target_ids
    #     )
        
    #     # 转换为input_ids
    #     input_ids = torch.tensor([
    #         self.tokenizer.convert_tokens_to_ids(tokens)  for tokens in bert_tokens
    #     ]).to(self.device)
        

    #     attention_mask = None
    #     if self.use_attention_mask:
    #         attention_mask = (input_ids != self.tokenizer.pad_token_id).type(
    #             torch.FloatTensor
    #         )
    #         attention_mask = attention_mask.to(input_ids)
    #     cnt=0 # 处理后的嵌入有多少不同原使嵌入
    #     target_logits=[]
    #     target_pos_left=[]
    #     # 获取原始嵌入
    #     with torch.no_grad():
    #         embeddings = self.model.bert.embeddings(input_ids)  # 原始嵌入

    #         # 遍历处理每个样本
    #         for idx, (target_pos, subword_len, orig_word) in enumerate(zip(bert_target_ids, length_target_tokens, original_words)):
                
    #             # 划分 left, target, right
    #             left = embeddings[idx, :target_pos]
    #             target_subwords = bert_tokens[idx][target_pos:target_pos+subword_len]
    #             right = embeddings[idx, target_pos+subword_len:]
                
    #             # 直接使用原始词进行预处理
    #             processed_embeds = self.preprocessor.process_word(
    #                 original_word=orig_word,  # 直接传入原始词
    #                 subwords=target_subwords  # 子词序列，面临分词的风险
    #             )
    #             # 采用mix-up的时候,对维度进行了“扩充”，根据同义词而来，所以存在不等的情况
    #             # assert len(processed_embeds) == subword_len, "嵌入长度与子词数不匹配"       # 防止手动修改后不匹配 
                
    #             # 拼接更新，对比实验的话可直接在prosessed_embeds上操作，相同维度进行堆叠
    #             processed_embeds_tensor = torch.stack(processed_embeds)
    #             new_embedding = torch.cat([left, processed_embeds_tensor, right], dim=0).unsqueeze(0)

    #             # 更新 embeddings 张量，还想批处理，不行啊，既然更新了subwords，不匹配，无法直接赋值，token数不同
    #             # embeddings[idx] = new_embedding

    #             # 更新 attention_mask——可能面临长度过长的问题
    #             if self.use_attention_mask:
    #                 left_mask = attention_mask[idx, :target_pos]
    #                 target_mask = torch.ones(len(processed_embeds)).long().to(self.device)
    #                 right_mask = attention_mask[idx, target_pos+len(processed_embeds):]
    #                 new_mask = torch.cat([left_mask, target_mask, right_mask], dim=0).unsqueeze(0)
    #                 # attention_mask[idx] = new_mask
                

    #             # target_pos_left.append(len(processed_embeds))
    #             # 处理输入超过最大长度的问题，有个问题，就是扩展了embedding和mask的seq_length长度，会不会出现问题
    #             # 512的维度，拿什么超过

    #             # max_length = self.model.config.max_position_embeddings
    #             # if embeddings.shape[1] > max_length:
    #             #     embeddings = embeddings[:, :max_length]
    #             #     attention_mask = attention_mask[:, :max_length]

    #         # 使得embedding和mask的维度一致，方便批处理
    #         # 找出最大的 num_tokens
    #         # max_num_tokens = max([emb.shape[0] for emb in embeddings])
    #         # # 获取 PAD 标记的嵌入向量——不同模型就不同
    #         # pad_token_id = self.tokenizer.pad_token_id
    #         # pad_embedding = self.model.embeddings.word_embeddings(torch.tensor(pad_token_id)).unsqueeze(0)
    #         # padded_embeddings,padded_attention_masks = [],[]
    #         # for emb,mask in zip(embeddings,attention_mask):
    #         #     num_pad = max_num_tokens - emb.shape[0]
    #         #     if num_pad > 0:
    #         #         pad = pad_embedding.repeat(num_pad, 1)
    #         #         padded_emb = torch.cat([emb, pad], dim=0)
    #         #         padded_embeddings.append(padded_emb)

    #         #         # 填充注意力掩码
    #         #         pad_mask = torch.zeros(num_pad)
    #         #         padded_mask = torch.cat([mask, pad_mask], dim=0)
    #         #         padded_attention_masks.append(padded_mask)
                    
    #         #     else:
    #         #         padded_embeddings.append(emb)
    #         #         padded_attention_masks.append(mask)
    #         # # 创建新的张量
    #         # batch_embeddings = torch.stack(padded_embeddings, dim=0)
    #         # batch_mask = torch.stack(padded_attention_masks, dim=0)

    #             # 前向传播优化
    #             outputs = self.model(
    #                 inputs_embeds=new_embedding,
    #                 attention_mask=new_mask
    #             )
    #             logits = outputs.logits
                
    #             target_logits.append(logits[:,target_pos:target_pos+len(processed_embeds),:].cpu().numpy() / self.temperature)
    #     return target_logits
    
    # shift+tab多行减少缩进
    
    
    
    #________________________________________________________________________________________________________
    # 此版本为，合并子词的输入为一个位置，有子词的情况，合并后加dropout mix-up gauss mask keep
    # def predict(self,  tokens_lists: List[List[str]], target_ids: List[int],target_pos_tag: List[str]):
    #     # 准备batch数据，bert_tokens有子词标记，填充好了的，pad，一个批次的数据相同
    #     bert_tokens, bert_target_ids, length_target_tokens,original_words = self.bert_prepare_batch(
    #         tokens_lists, target_ids
    #     )
    #     # 转换为input_ids
    #     input_ids = torch.tensor([
    #         self.tokenizer.convert_tokens_to_ids(tokens)  for tokens in bert_tokens
    #     ]).to(self.device)
        

    #     attention_mask = None
    #     if self.use_attention_mask:
    #         attention_mask = (input_ids != self.tokenizer.pad_token_id).type(
    #             torch.FloatTensor
    #         )
    #         attention_mask = attention_mask.to(input_ids)
    #     cnt=0 # 处理后的嵌入有多少不同原使嵌入
    #     target_logits=[]
    #     target_pos_left=[]
        
    #     with torch.no_grad():
    #         embeddings = self.model.bert.embeddings(input_ids)  # 原始嵌入
    #         # 遍历处理每个样本
    #         for idx, (target_pos, subword_len, orig_word) in enumerate(zip(bert_target_ids, length_target_tokens, original_words)):
                
    #             # 划分 left, target, right
    #             left = embeddings[idx, :target_pos]
    #             target_subwords = bert_tokens[idx][target_pos:target_pos+subword_len]
    #             right = embeddings[idx, target_pos+subword_len:]
                
    #             # 直接使用原始词进行预处理
    #             processed_embeds = self.preprocessor.process_word(
    #                 original_word=orig_word,  # 直接传入原始词
    #                 subwords=target_subwords,  # 子词序列，面临分词的风险
    #                 pos_tag=target_pos_tag[idx][0]
    #             )
    #             # 采用mix-up的时候,对维度进行了“扩充”，根据同义词而来，所以存在不等的情况
    #             # assert len(processed_embeds) == subword_len, "嵌入长度与子词数不匹配"       # 防止手动修改后不匹配 
                
    #             # 拼接更新，对比实验的话可直接在prosessed_embeds上操作，相同维度进行堆叠
    #             # processed_embeds_tensor = torch.stack(processed_embeds)
    #             if processed_embeds.dim()>2:
    #                 processed_embeds=np.squeeze(processed_embeds,axis=0)
    #             new_embedding = torch.cat([left, processed_embeds, right], dim=0).unsqueeze(0)

    #             # 更新 embeddings 张量，还想批处理，不行啊，既然更新了subwords，不匹配，无法直接赋值，token数不同
    #             # embeddings[idx] = new_embedding

    #             # 更新 attention_mask——可能面临长度过长的问题
    #             if self.use_attention_mask:
    #                 left_mask = attention_mask[idx, :target_pos]
    #                 target_mask = torch.ones(len(processed_embeds)).long().to(self.device)
    #                 right_mask = attention_mask[idx, target_pos+subword_len:]
    #                 new_mask = torch.cat([left_mask, target_mask, right_mask], dim=0).unsqueeze(0)
                    
    #             # 前向传播优化
    #             outputs = self.model(
    #                 inputs_embeds=new_embedding,
    #                 attention_mask=new_mask
    #             )
    #             logits = outputs.logits
                
    #             target_logits.append(logits[:,target_pos,:].cpu().numpy() / self.temperature)
    #     return target_logits
    
    #________________________________________________________________________________________________________
    # 此版本为替换子词的第一个位置
    def predict(self,  tokens_lists: List[List[str]], target_ids: List[int],target_pos_tag: List[str]):
        # 准备batch数据，bert_tokens有子词标记，填充好了的，pad，一个批次的数据相同
        bert_tokens, bert_target_ids, length_target_tokens,original_words = self.bert_prepare_batch(
            tokens_lists, target_ids
        )
        # 转换为input_ids
        input_ids = torch.tensor([
            self.tokenizer.convert_tokens_to_ids(tokens)  for tokens in bert_tokens
        ]).to(self.device)

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).type(
                torch.FloatTensor
            )
            attention_mask = attention_mask.to(input_ids)
        target_logits=[]
        # 获取原始嵌入
        with torch.no_grad():
            embeddings = self.model.bert.embeddings(input_ids)  # 原始嵌入
            # 遍历处理每个样本
            for idx, (target_pos, subword_len, orig_word) in enumerate(zip(bert_target_ids, length_target_tokens, original_words)):
                target_subwords = bert_tokens[idx][target_pos:target_pos+subword_len]        
                # 直接使用原始词进行预处理
                processed_embeds = self.preprocessor.process_word(
                    original_word=orig_word,  # 直接传入原始词
                    subwords=target_subwords,  # 子词序列，面临分词的风险
                    pos_tag=target_pos_tag[idx][0]
                )
                
                # processed_embeds_tensor = torch.stack(processed_embeds)
                embeddings[idx][target_pos]=processed_embeds

            # 前向传播优化
            outputs = self.model(inputs_embeds=embeddings,attention_mask=attention_mask)
            logits = outputs.logits
            
            # 7. 使用 batch 索引同时提取每个样本在其 target 位置的 logits——就一个位置提取，不对啊,补充长度
            # 只返回的目标位置，默认取第一个
            # batch_indices = torch.arange(logits.size(0)).to(self.device)
            target_positions = torch.tensor(bert_target_ids).to(self.device)
            target_length=torch.tensor(length_target_tokens).to(self.device)
            # target_logits = logits[batch_indices, target_positions,:] / self.temperature

            # 访问多个
            batch_size = logits.size(0)
            target_logits_list = [
                logits[i, target_positions[i] : target_positions[i] + target_length[i], :]/self.temperature
                for i in range(batch_size)
            ]
            # .cpu().numpy()
        return target_logits_list
    

    # 待修改
    def get_ordered_synonyms(self, original_word: str, synonyms_from_wordnet: List[str]):
        loaded = BertProbEstimator.loaded[self.model_name]
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