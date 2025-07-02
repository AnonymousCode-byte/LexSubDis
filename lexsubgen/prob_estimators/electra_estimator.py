from transformers import ElectraForPreTraining, ElectraTokenizerFast,ElectraTokenizer, ElectraModel
import torch
from lexsubgen.prob_estimators.embsim_estimator import EmbSimProbEstimator
import numpy as np
from typing import List, Dict,Tuple
import json
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # 空间不够大，gpu超出缓存
from string import punctuation
from overrides import overrides
import torch.nn.functional as F


# ElectraForPreTraining 模型的输出 logits 就是用于做词汇是否替换过的判断的依据。（batch_size,sequen_length）接近0替换，接近1表示未替换
    # electramodel的基础上接了一层，提取cls等，.last_hidden_state
# ElectraModel 是 ELECTRA 模型的基础版本，它主要用于提取输入文本的特征表示。
    # 它不包含用于预训练任务（如判别器任务）的额外层，通常用于下游任务（如文本分类、命名实体识别等）的特征提取。
    # last_hidden_state：形状为 (batch_size, sequence_length, hidden_size)，表示输入序列中每个 token 的最后一层隐藏状态。


class ElectraProbEstimator(EmbSimProbEstimator):
    """
    A probability estimator using ELECTRA embeddings to compute substitution probabilities
    based on embedding similarity.
    """
    def __init__(
        self,
        model_name: str,
        cuda_device:int =7,
        use_subword_mean: bool = False,     # 处理未登录词
        use_attention_mask: bool = True,
        target_input_embedding_type: str = "not_masked",      # (mask_type)targetword mask type=mask、drop out、gauss、mix-up stratage
        verbose: bool = False,
        sim_func: str = "dot-product",      # 配置文件没有的字段
        temperature: float = 1.0,

        embedding_similarity: bool = False,     # 同理，不影响
        target_output_embedding_type:str="None"       # 新字段，输出的target处的词的处理方式，max_pooling、average_pooling
    ):
        super().__init__(model_name, verbose, sim_func, temperature)
        self.embedding_similarity = embedding_similarity
        self.use_attention_mask = use_attention_mask
        self.use_subword_mean = use_subword_mean
        self.cuda_device=cuda_device
        self.target_input_embedding_type=target_input_embedding_type,
        self.target_output_embedding_type=target_output_embedding_type,

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device) 
        if self.cuda_device != -1 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.cuda_device}")
        else:
            self.device = torch.device("cpu")

        self.descriptor = {
            "Prob_estimator": {
                "name": "electra",
                "class": self.__class__.__name__,
                "model_name": self.model_name,
                "target_input_embedding_type": self.target_input_embedding_type,
                "target_output_embedding_type":self.target_output_embedding_type,
                "embedding_similarity": self.embedding_similarity,
                "temperature": self.temperature,
                "use_attention_mask": self.use_attention_mask,
                "use_subword_mean": self.use_subword_mean,
            }
        }
        self.logger.debug(f"Probability estimator {self.descriptor} is created.")
        self.logger.debug(f"Config:\n{json.dumps(self.descriptor, indent=4)}")
        self.register_model()
    

    # hf镜像实现无vpn下载模型
    # 假设本地模型存放在 'path/to/local/bert-base-uncased' 目录下
    # tokenizer = BertTokenizer.from_pretrained('path/to/local/bert-base-uncased')
    # model = BertModel.from_pretrained('path/to/local/bert-base-uncased')
    def register_model(self):
        """加载ELECTRA模型到共享缓存"""
        if self.model_name not in self.loaded:
            # 加载模型和分词器
            tokenizer = ElectraTokenizer.from_pretrained(self.model_name)
            model = ElectraForPreTraining.from_pretrained(self.model_name).to(self.device)
            
            # 获取嵌入矩阵 (shape: [vocab_size, hidden_dim]) word_embeddings
            embeddings = model.get_input_embeddings().weight.data
            
            # 构建词汇表映射    加载过多，负担重，不必要
            # word2id = tokenizer.get_vocab()
            
            # 需要过滤的
            # electra_filter_word_ids = ElectraProbEstimator.load_filter_word_ids(
            #     word2id, punctuation
            # )

            # 将资源存入缓存
            self.loaded[self.model_name] = {
                "model": model,
                "tokenizer": tokenizer,
                "embeddings": embeddings,
                # "word2id": word2id,
                "ref_count": 1,  # 初始化引用计数
                # "electra_filter_word_ids":electra_filter_word_ids,
            }
        else:
            # 增加已有模型的引用计数
            self.loaded[self.model_name]["ref_count"] += 1

    def get_unk_word_vector(self, word: str) -> np.ndarray:
        """处理未登录词：通过子词的平均向量实现"""
        tokenizer = self.loaded[self.model_name]["tokenizer"]
        sub_tokens = tokenizer.tokenize(word)
        
        # 如果没有子词则返回零向量
        if not sub_tokens:
            return super().get_unk_word_vector(word)
        
        # 收集所有有效子词向量
        vectors = []
        for token in sub_tokens:
            if token in self.word2id:
                vectors.append(self.embeddings[self.word2id[token]])
        
        # 返回平均向量或零向量
        if vectors:
            return np.mean(vectors, axis=0, keepdims=True)
        else:
            return super().get_unk_word_vector(word)    # 可以交给父类处理
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
    
    @property
    def tokenizer(self):
        return self.loaded[self.model_name]["tokenizer"]
    

    # 句子填充最大长度且截断
    def tokenize(self, sentence: str):
        tokenizer=self.tokenizer
        return tokenizer(sentence,return_tensors='pt',padding='max_length', max_length=256,truncation=True)


    @staticmethod
    def load_word2id(tokenizer:ElectraTokenizer ) -> Dict[str, int]:
        """
        Loads model vocabulary in the form of mapping from words to their indexes.
        Args:
            tokenizer: `transformers.ElectraTokenizer` tokenizer
        Returns:
            model vocabulary
        """
        word2id = dict()
        for word_idx in range(tokenizer.vocab_size):
            word = tokenizer.convert_ids_to_tokens([word_idx])[0]
            word2id[word] = word_idx
        return word2id

    @property
    def filter_word_ids(self) -> List[int]:
        """
        Indexes of words to be filtered from the end distribution.
        Returns:
            list of indexes
        """
        return self.loaded[self.model_name]["electra_filter_word_ids"]
    
    # 将给定的词汇列表转换为 elctra 模型可以处理的子词列表。它使用 electra 分词器将每个词汇拆分为子词，并返回这些子词的列表
    def electra_tokenize_sentence(
        self, tokens: List[str], tokenizer: ElectraTokenizer = None
    ) -> List[str]:
        """使用ELECTRA分词器处理子词"""
        if tokenizer is None:
            tokenizer = self.tokenizer
        electra_tokens = []
        for token in tokens:
            electra_tokens.extend(tokenizer.tokenize(token))
        return electra_tokens
    
    # 原始句子，处理成模型能处理的
    def electra_prepare_batch(
        self,
        batch_of_tokens: List[List[str]],
        batch_of_target_ids: List[int],
        tokenizer: ElectraTokenizer = None,
    ) -> Tuple[List[List[str]], List[int]]:
        """ELECTRA专用批处理准备"""
        if tokenizer is None:
            tokenizer = self.tokenizer

        processed_tokens, target_positions,electra_len_of_tokens = [], [],[]
        max_seq_len = 0
        
        for tokens, target_idx in zip(batch_of_tokens, batch_of_target_ids):
            # ELECTRA使用[CLS]和[SEP]标记，默认target词就一个，占一个空——不然呢
            left = ["[CLS]"] + self.electra_tokenize_sentence(tokens[:target_idx], tokenizer)
            right = self.electra_tokenize_sentence(tokens[target_idx+1:], tokenizer) + ["[SEP]"]
            
            # 处理目标词——分词，
            target_tokens = self.electra_tokenize_sentence([tokens[target_idx]], tokenizer)
            length_target_tokens=1
            # 应用掩码策略，这里涉及到各种各样的输入掩码策略
            if self.target_input_embedding_type == "masked":
                target_tokens = ["[MASK]"]  # ELECTRA的掩码标记
            elif self.target_input_embedding_type == "not_masked":
                length_target_tokens=len(target_tokens)
            else:
                raise ValueError(f"Unrecognised masking type {self.target_input_embedding_type}.")

            context = left + target_tokens + right
            processed_tokens.append(context)
            electra_len_of_tokens.append(length_target_tokens)
            target_positions.append(len(left))
            
            max_seq_len = max(max_seq_len, len(context))

        # 统一填充长度
        padded_tokens = [
            seq + ["[PAD]"]*(max_seq_len-len(seq)) 
            for seq in processed_tokens
        ]
        return padded_tokens, target_positions,electra_len_of_tokens
    
    def predict(
        self, tokens_lists: List[List[str]], target_ids: List[int],target_pos_tag: List[str]
    ) -> np.ndarray:
        """ELECTRA预测实现"""
        electra_tokens, electra_targets = self.electra_prepare_batch(tokens_lists, target_ids)
        
        # 转换为模型输入
        input_ids = torch.tensor([
            self.tokenizer.convert_tokens_to_ids(tokens)
            for tokens in electra_tokens
        ]).to(self.device)

        # 注意力掩码
        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.last_hidden_state  # ELECTRA的输出位置
            
        # 提取目标位置logits    ——这里涉及到输出位置的target的聚合方式
        # if self.target_output_embedding_type="":
        
        batch_logits = []
        for idx, pos in enumerate(electra_targets):
            target_logits = logits[idx, pos, :].cpu().numpy()
            batch_logits.append(target_logits / self.temperature)
            
        return np.vstack(batch_logits)

    # 计算每个目标词的对数概率分布
    @overrides
    def get_log_probs(
        self, tokens_lists: List[List[str]], target_ids: List[int],target_pos_tag:List[str]
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """概率计算（保持与父类兼容）"""
        if self.embedding_similarity:
            logits = self.get_emb_similarity(tokens_lists, target_ids)
        else:
            logits = self.predict(tokens_lists, target_ids,target_pos_tag)
        # 过滤非常用词
        logits[:, self.filter_word_ids] = -1e9
        return logits, self.word2id
    
    
    
    # 别盲目替代，这是replaced token detection score，最后都是要取平均,包括了CLS、sep
    # 需要取CLS吗？？辅助性的作用，所以不取CLS   句子的similarity用专门的sentenceBert
    # 此版本需要将目标位置词及其候选词进行处理，占一个空，取embedding平均就行了！ 还将目标位置合并成一个，感觉不对
    def get_rtd_score(self,tokens_lists: List[List[str]],pred_substitutes: List[List[str]],
                      target_ids: List[int]) -> torch.Tensor:
        overall_differences=[]
        for i, (tokens,idx )in enumerate(zip(tokens_lists,target_ids)):
            candidate_logits = []
            # 将原句转换为input_ids和attention_mask
            original_inputs = self.tokenize(" ".join(tokens)).to(self.device)   # 默认加CLS和SEP
            input_ids = original_inputs["input_ids"]            # 和mask一样，size=(1,256)
            attention_mask = original_inputs["attention_mask"]      # 为original_inputs的属性
            
            # 获取原句的嵌入表示
            original_embeddings = self.model.get_input_embeddings()(input_ids)
              
            left_text = " ".join(tokens[:idx])
            right_text = " ".join(tokens[idx+1:])

            # 目的求长度，所以不需要加CLS和SEP,已经是id序列了
            left_encoded = self.tokenizer(left_text, add_special_tokens=False, return_tensors="pt")
            target_encoded = self.tokenizer(tokens[idx], add_special_tokens=False, return_tensors="pt")
            right_encoded = self.tokenizer(right_text, add_special_tokens=False, return_tensors="pt")
            
            
            num_left = left_encoded["input_ids"].shape[1]
            target_subword_count = target_encoded["input_ids"].shape[1]
            num_right = right_encoded["input_ids"].shape[1]   
            
            # Depending on your tokenizer settings, original_inputs may have added a [CLS] token at the beginning.
            # Adjust the index accordingly. For example, if [CLS] is present as the first token:
            start_idx = num_left+1  # 1 for [CLS] and num_left tokens before the target 
            
            # Extract the embeddings corresponding to the target word’s subwords.
            target_embeddings = original_embeddings[0, start_idx: start_idx + target_subword_count]
            
            # If the target word has been split into multiple subwords, average their embeddings.
            if target_subword_count > 1:
                target_embedding = target_embeddings.mean(dim=0, keepdim=True)  # keepdim=true
            else:
                target_embedding = target_embeddings
            
            left_emb = original_embeddings[0,:start_idx]  
            right_emb = original_embeddings[0,start_idx+target_subword_count:] 

            left_mask=attention_mask[0,:start_idx]
            right_mask=attention_mask[0,start_idx+target_subword_count:]

            updated_embeddings = torch.cat([left_emb, target_embedding, right_emb], dim=0)
            updated_mask=torch.cat([left_mask,torch.tensor([1]).to(self.device),right_mask],dim=0)

            # 释放未使用的缓存，空间不够
            torch.cuda.empty_cache()
            
            candidate_batch_embeddings = []
            candidate_batch_masks = []
            # Add the original sentence (with target word) as the first element.
            candidate_batch_embeddings.append(updated_embeddings)
            candidate_batch_masks.append(updated_mask.to(self.device))

            for candidate in pred_substitutes[i]:
                # Tokenize the candidate word (without adding special tokens)
                candidate_encoded = self.tokenizer(candidate, add_special_tokens=False, return_tensors="pt")
                candidate_input_ids = candidate_encoded["input_ids"].to(self.device)
                
                # Get the candidate word's embeddings.
                candidate_emb_all = self.model.get_input_embeddings()(candidate_input_ids)
                candidate_subword_count = candidate_input_ids.shape[1]
                # If the candidate is split into multiple subwords, average their embeddings.
                if candidate_subword_count > 1:
                    candidate_embedding = candidate_emb_all.mean(dim=1)  # shape: (1, embedding_dim)
                else:
                    candidate_embedding = candidate_emb_all.squeeze(1)    # shape: (1, embedding_dim)
                

                # Build the candidate sentence embedding by replacing the target's embedding.
                candidate_updated_embeddings = torch.cat([left_emb, candidate_embedding, right_emb], dim=0)
                candidate_updated_mask = torch.cat([left_mask, torch.tensor([1]).to(self.device), right_mask], dim=0)
                
                candidate_batch_embeddings.append(candidate_updated_embeddings)
                candidate_batch_masks.append(candidate_updated_mask)

            # Stack the embeddings and masks into a batch.
            # The first element is the original sentence; subsequent elements are candidate replacements.
            batch_embeddings = torch.stack(candidate_batch_embeddings, dim=0)  # shape: (batch_size, seq_len, embedding_dim)
            batch_masks = torch.stack(candidate_batch_masks, dim=0)            # shape: (batch_size, seq_len)
            
            # Forward pass the batch through the model to get logits.
            with torch.no_grad():   # 原始没加这句话，保留了梯度！！！！！！！！！！！！！！！！！！
                outputs = self.model(inputs_embeds=batch_embeddings, attention_mask=batch_masks)
                logits = outputs.logits  # Assuming outputs contain a 'logits' attribute
            
            # The first logit corresponds to the original sentence and the remaining ones correspond to each candidate.
            # candidate_logits.append(logits[1:])

            original_logits = logits[0]  # shape: (原句1+替代词num_sub, vocab_size)
            sentence_candidate_diffs = []  # 用来存储当前句子所有候选替换的差值
            
            for cand_logits in logits[1:]:
                # 计算目标位置的 logits 差值（对 vocab 维度求绝对值后求和）
                # diff = torch.abs(original_logits[start_idx] - cand_logits[start_idx])

                diff = torch.abs(torch.sigmoid(original_logits) - torch.sigmoid(cand_logits))
                token_diff = diff[1:num_left+num_right+2]        # 包含CLS否，即从0还是从1开始算，包含有所下降
                score = torch.mean(token_diff)                  # 差值是越大越不好，所以排序的时候注意！
                

                # sigmoid函数才对,目标位置处的
                # score=torch.sigmoid(cand_logits[start_idx])
                sentence_candidate_diffs.append(score.item())
                
                # sentence_candidate_diffs.append(diff.item())      # 只取目标位置

                
            # 将当前句子候选词的差值列表存入总体结果二维 list 中
            overall_differences.append(sentence_candidate_diffs)

           
        return overall_differences


                
    
    

    #______版本为不区分CLS和SEP标记且存在输入的长度不一致的情况--淘汰--kl散度淘汰
    # 实验electra模型做validataion score 捕捉词的细粒度关系
    # def get_validation_score(self,tokens_lists: List[List[str]],pred_substitutes: List[List[str]],
    #                   target_ids: List[int]) -> torch.Tensor:
    #     overall_differences=[]
    #     for i, (tokens,idx )in enumerate(zip(tokens_lists,target_ids)):
    #         candidate_logits = []
    #         # 将原句转换为input_ids和attention_mask
    #         original_inputs = self.tokenize(" ".join(tokens)).to(self.device)   # 默认加CLS和SEP
    #         input_ids = original_inputs["input_ids"]            # 和mask一样，size=(1,256)
    #         attention_mask = original_inputs["attention_mask"]      # 为original_inputs的属性
            
    #         # 获取原句的嵌入表示
    #         original_embeddings = self.model.get_input_embeddings()(input_ids)
              
    #         left_text = " ".join(tokens[:idx])
    #         right_text = " ".join(tokens[idx+1:])

    #         # 目的求长度，所以不需要加CLS和SEP,已经是id序列了
    #         left_encoded = self.tokenizer(left_text, add_special_tokens=False, return_tensors="pt")
    #         target_encoded = self.tokenizer(tokens[idx], add_special_tokens=False, return_tensors="pt")
    #         right_encoded = self.tokenizer(right_text, add_special_tokens=False, return_tensors="pt")
            
            
    #         num_left = left_encoded["input_ids"].shape[1]
    #         target_subword_count = target_encoded["input_ids"].shape[1]
    #         num_right = right_encoded["input_ids"].shape[1]   
            
    #         start_idx = num_left+1  # 1 for [CLS] and num_left tokens before the target 
            
    #         # Extract the embeddings corresponding to the target word’s subwords.
    #         target_embeddings = original_embeddings[0, start_idx: start_idx + target_subword_count]
            
    #         # If the target word has been split into multiple subwords, average their embeddings.
    #         if target_subword_count > 1:
    #             target_embedding = target_embeddings.mean(dim=0, keepdim=True)  # keepdim=true
    #         else:
    #             target_embedding = target_embeddings
            
    #         left_emb = original_embeddings[0,:start_idx]  
    #         right_emb = original_embeddings[0,start_idx+target_subword_count:] 

    #         left_mask=attention_mask[0,:start_idx]
    #         right_mask=attention_mask[0,start_idx+target_subword_count:]

    #         updated_embeddings = torch.cat([left_emb, target_embedding, right_emb], dim=0)
    #         updated_mask=torch.cat([left_mask,torch.tensor([1]).to(self.device),right_mask],dim=0)

    #         # 释放未使用的缓存，空间不够
    #         torch.cuda.empty_cache()
            
    #         candidate_batch_embeddings = []
    #         candidate_batch_masks = []
    #         # Add the original sentence (with target word) as the first element.
    #         candidate_batch_embeddings.append(updated_embeddings)
    #         candidate_batch_masks.append(updated_mask.to(self.device))

    #         for candidate in pred_substitutes[i]:
    #             # Tokenize the candidate word (without adding special tokens)
    #             candidate_encoded = self.tokenizer(candidate, add_special_tokens=False, return_tensors="pt")
    #             candidate_input_ids = candidate_encoded["input_ids"].to(self.device)
                
    #             # Get the candidate word's embeddings.
    #             candidate_emb_all = self.model.get_input_embeddings()(candidate_input_ids)
    #             candidate_subword_count = candidate_input_ids.shape[1]
    #             # If the candidate is split into multiple subwords, average their embeddings.
    #             if candidate_subword_count > 1:
    #                 candidate_embedding = candidate_emb_all.mean(dim=1)  # shape: (1, embedding_dim)
    #             else:
    #                 candidate_embedding = candidate_emb_all.squeeze(1)    # shape: (1, embedding_dim)
                

    #             # Build the candidate sentence embedding by replacing the target's embedding.
    #             candidate_updated_embeddings = torch.cat([left_emb, candidate_embedding, right_emb], dim=0)
    #             candidate_updated_mask = torch.cat([left_mask, torch.tensor([1]).to(self.device), right_mask], dim=0)
                
    #             candidate_batch_embeddings.append(candidate_updated_embeddings)
    #             candidate_batch_masks.append(candidate_updated_mask)

        
    #         batch_embeddings = torch.stack(candidate_batch_embeddings, dim=0)  # shape: (batch_size, seq_len, embedding_dim)
    #         batch_masks = torch.stack(candidate_batch_masks, dim=0)            # shape: (batch_size, seq_len)
            
            
    #         with torch.no_grad():   # 原始没加这句话，保留了梯度！！！！！！！！！！！！！！！！！！
    #             outputs = self.model(inputs_embeds=batch_embeddings, attention_mask=batch_masks)
    #             logits = outputs.logits  # Assuming outputs contain a 'logits' attribute

    #         original_logits = logits[0]  # shape: (seq_len, vocab_size)
    #         sentence_candidate_diffs = []  # 用来存储当前句子所有候选替换的差值
            
            
            
    #         for cand_logits in logits[1:]:
    #             diff = torch.abs(original_logits - cand_logits)
    #             token_diff = diff[1:num_left+num_right+2]
    #             score = torch.mean(token_diff)
    #             sentence_candidate_diffs.append(score.item())
                
    #             # sentence_candidate_diffs.append(diff.item())      # 只取目标位置

                
    #         # 将当前句子候选词的差值列表存入总体结果二维 list 中
    #         overall_differences.append(sentence_candidate_diffs)

           
    #     return overall_differences

    
    
    
