import sys
import os
import argparse

# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 定位到项目根目录（假设项目根目录在上级的上级目录）
project_root = os.path.dirname(os.path.dirname(current_dir))
# 将根目录添加到sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# print(f'当前目录是：{current_dir}，项目根目录是：{project_root}')

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, NoReturn, Optional

import numpy as np
import pandas as pd
from fire import Fire
from overrides import overrides
from collections import OrderedDict
from tqdm import tqdm

from lexsubgen.data.lexsub import DatasetReader
from lexsubgen.evaluations.task import Task
from lexsubgen.metrics.all_word_ranking_metrics import (
    compute_precision_recall_f1_topk,
    compute_precision_recall_f1_vocab,
    compute_oot_best_metrics,
    get_mode
)
from lexsubgen.metrics.candidate_ranking_metrics import gap_score
from lexsubgen.subst_generator import SubstituteGenerator
from lexsubgen.utils.batch_reader import BatchReader
from lexsubgen.utils.file import dump_json
from lexsubgen.utils.params import build_from_config_path,read_config
from lexsubgen.utils.wordnet_relation import to_wordnet_pos, get_wordnet_relation
from lexsubgen.prob_estimators.electra_estimator import ElectraProbEstimator    # replaced token detectino score
from sentence_transformers import SentenceTransformer, util
from candidates_from_wordnet.from_wordnet import created_proposed_list
from candidates_from_wordnet.wordnet import Wordnet
logger = logging.getLogger(Path(__file__).name)

DEFAULT_RUN_DIR = Path(__file__).resolve().parent.parent / "run"/"debug" / Path(__file__).stem

# print("lexsubgen/evaluations/lexsub.py DEFAULT_RUN_DIR=")
# print(DEFAULT_RUN_DIR)

def sort_substitutes_rtd_score(pred_substitutes, pred_score):
    sorted_result = []
    for substitutes, scores in zip(pred_substitutes, pred_score):
        # 将替换词和分数组合成元组列表
        combined = list(zip(substitutes, scores))
        # 根据分数从高到低排序——差值不应该是从低到高吗？？？
        # combined.sort(key=lambda x: x[1], reverse=True)
        combined.sort(key=lambda x: x[1], reverse=False)
        # 提取排序后的替换词
        sorted_substitutes = [substitute for substitute, _ in combined]
        sorted_result.append(sorted_substitutes)
    return sorted_result

def sort_substitutes_sentence_sim(pred_substitutes, pred_score):
    sorted_result = []
    for substitutes, scores in zip(pred_substitutes, pred_score):
        # 将替换词和分数组合成元组列表
        combined = list(zip(substitutes, scores))
        combined.sort(key=lambda x: x[1], reverse=True)
        # 提取排序后的替换词
        sorted_substitutes = [substitute for substitute, _ in combined]
        sorted_result.append(sorted_substitutes)
    return sorted_result



class LexSubEvaluation(Task):
    def __init__(
        self,
        substitute_generator: SubstituteGenerator = None,   # 传入的对象实例，模型做词汇替换 {post_processing、pre_processing、prob_estimator}
        dataset_reader: DatasetReader = None,
        verbose: bool = True,
        k_list: List[int] = (1, 3, 10),
        batch_size: int = 50,
        # 三个参数干啥用的
        save_instance_results: bool = True,
        save_wordnet_relations: bool = False,
        save_target_rank: bool = False,
    ):
        """
        Main class for performing Lexical Substitution task evaluation.
        This evaluation computes metrics for two subtasks in Lexical Substitution task:
        - Candidate-ranking task (GAP, GAP_normalized, GAP_vocab_normalized).
        - All-word-ranking task (Precision@k, Recall@k, F1@k for k-best substitutes).
        Args:
            substitute_generator: Object that generate possible substitutes.
            dataset_reader: Object that can read datasets for Lexical Substitution task.
            verbose: Bool flag for verbosity.
            k_list: List of integer numbers for metrics. For example, if 'k_list' equal to [1, 3, 5],
                then there will calculating the following metrics:
                    - Precion@1, Recall@1, F1-score@1
                    - Precion@3, Recall@3, F1-score@3
                    - Precion@5, Recall@5, F1-score@5
            batch_size: Number of samples in batch for substitute generator.
        """
        super(LexSubEvaluation, self).__init__(
            substitute_generator=substitute_generator,
            dataset_reader=dataset_reader,
            verbose=verbose,
        )
        self.batch_size = batch_size
        self.k_list = k_list
        self.save_wordnet_relations = save_wordnet_relations
        self.save_target_rank = save_target_rank
        self.save_instance_results = save_instance_results

        self.gap_metrics = ["gap", "gap_normalized", "gap_vocab_normalized"]
        self.base_metrics = ["precision", "recall", "f1_score"]
        self.oot_best_metrics=['oot','ootm','best','bestm'] # 补充
        k_metrics = []
        for k in self.k_list:
            k_metrics.extend([f"prec@{k}", f"rec@{k}", f"f1@{k}"])
        self.metrics = self.gap_metrics + self.base_metrics + k_metrics+self.oot_best_metrics



        # replaced token detection模型计算分数的调用处，参考cilex1
        self.electra_model=ElectraProbEstimator(model_name='google/electra-large-discriminator')
        # self.sentence_sim_model=SentenceTransformer('sentence-transformers/all-roberta-large-v1').to("cuda:5")
        # 1024维度向量,效果比不加的更好，比rtd好
        

    @overrides
    def get_metrics(self, dataset: pd.DataFrame,**kwargs) -> Dict[str, Any]:
        """
        子类重写父类方法，参数数量需要保持一致，keyword aruguments，表示可接受多个参数
        Method for calculating metrics for Lexical Substitution task.
        Args:
            dataset: pandas DataFrame with the whole datasets.
        Returns:
            metrics_data: Dictionary with two keys:
                - all_metrics: pandas DataFrame, extended 'datasets' with computed metrics
                - mean_metrics: Dictionary with mean values of computed metrics
        """
        logger.info(f"Lexical Substitution for {len(dataset)} instances.")

        progress_bar = BatchReader(
            dataset["context"].tolist(),
            dataset["target_position"].tolist(),
            dataset["pos_tag"].tolist(),
            dataset["gold_subst"].tolist(),
            dataset["gold_subst_weights"].tolist(),
            dataset["candidates"].tolist(),
            dataset["target_lemma"].tolist(),   # “target_lemma” 指的是目标词的词元形式。 cats——cat
            batch_size=self.batch_size, # 批量
        )
        # 添加进度条，批量读取数据
        if self.verbose:
            progress_bar = tqdm(
                progress_bar,
                desc=f"Lexical Substitution for {len(dataset)} instances"
            )

        all_metrics_data, columns = [], None
        alpha=0.7       # alpha*sentence_similarity+(1-alph)*token_similarity
        beta=0.3
        # top-k=20 (sentence , token) = 0.7 0.3 > 0.3 0.7 | 0.8 0.2 | 0.75 0.25
        #  0.5 0.5 > 0.65 0.35  | 0.6 0.4 |0.7 0.3  ——except presion@1
        #  0.5 0.5 > 0.4 0.6 except presion@3
        #  0.45 0.55 in some little item better than 0.5 0.5

        wordNet=Wordnet()
        num_mode=0.0      # 数据集中，有多少数据有mode,方便计算bestm和ootm
        for (
            tokens_lists,
            target_ids,
            pos_tags,
            gold_substitutes,
            gold_weights,
            candidates, 
            target_lemmas,
        ) in progress_bar:
            
            
            # 实例化后的对象，里面存有多个对象，看config文件【post_processing\pre_processing\prob_estimator----elmo_estimator】
            # Computing probability distribution over possible substitutes
            # probs就是target位置处的概率分布，代表者可能的选择。——组合子词的地方
            probs, word2id = self.substitute_generator.get_probs(
                tokens_lists, target_ids, pos_tags
            )   # 一个progress_bar就是一batch条数据，target_id[batch],token_list[batch][batch],gold_weight,candidates同，二维list
                # probs:shape(batch,25779)

            
            # Selecting most probable substitutes from the obtained distribution
            # 根据top-k和top p选出来的词（选择范围为模型词汇表） pred_substitutes不会出现##吧？
            pred_substitutes,pred_substitutes_and_probs = self.substitute_generator.substitutes_from_probs(
                probs, word2id, tokens_lists, target_ids
            )

           

            # pred_substitutes_and_probs类型：list[dict{word:porbs},dict{word:probs}]  size:batch_size,top-k
            # 外部资源补充候选词，将目标位置的最大概率值当作候选词的替代,先实验组合技能
            # 字典按照【插入顺序】有序，判断是否在字典中出现，防止被洗刷

            # 遍历每个 tokens_list（及其对应 target_id 与 pos_tag）
            ordered_pred_and_synonyms=[]
            probs_score=[]
            for i, tokens in enumerate(tokens_lists):
                # substitution_batches = []  # 存放所有生成的新 tokens 列表,一条数据多个同义词，组成一个批次
                target_index = target_ids[i]
                pos_tag = pos_tags[i][0]
                
                original_word = tokens[target_index]
                word_temp = original_word.replace("_", " ")
                word_temp = word_temp.replace("-", " ")
                word_temp = word_temp.replace("'","")
                
                candidate_synonyms = created_proposed_list(original_word, wordNet, pos_tag)
                
                # 清除掉模型产生的词出现原始词的情况
                # pred_substitutes[i]=[word for word in pred_substitutes[i] if original_word not in word] # 子字符串判断了！
                pred_substitutes[i] = [word for word in pred_substitutes[i] if word != original_word]
                
                # 有个bug但是影响应该不大，即，取的pred_substitutes[i]是20个范围内的，后面会截取掉的风险——排名【后10】个，截取掉问题不大
                filtered_synonyms = [syn for syn in candidate_synonyms if syn not in pred_substitutes[i] and syn!=word_temp]

                # 针对每个筛选后的同义词，构造新的 tokens 列表，将目标词替换，若同义词为空则不遍历   
                num_synonyms=10         # 补充,10个有点不全,semeval of the first gold_candidates don't exist-同义优先
                
                if filtered_synonyms:       # can't find the synonyms   dont restrict the synonyms
                    synonym_ordered=self.substitute_generator.get_ordered_synonyms(word_temp,filtered_synonyms)
                    if len(synonym_ordered)>=10:
                        ordered_pred_and_synonyms.append(pred_substitutes[i][:10]+synonym_ordered[:10])
                    else:
                        ordered_pred_and_synonyms.append(pred_substitutes[i][:20-len(synonym_ordered)]+synonym_ordered)
                else:
                    ordered_pred_and_synonyms.append(pred_substitutes[i])
                    
            

            pred_substitutes=ordered_pred_and_synonyms
            # model predict probs  if in the wordnet, give sixth probs of the pred
            for pred_list,pred_list_probs in zip(pred_substitutes,pred_substitutes_and_probs):
                word_probs = []
                for d in pred_list:
                    if d in pred_list_probs:
                        word_probs.append(pred_list_probs[d])
                    else:
                        prob_values = list(pred_list_probs.values())[5]  # 之前设置的5，提权wordnet的词 设置3，4效果不行,2试试？
                        word_probs.append(prob_values)
                probs_score.append(word_probs)

            # no synoms version
            # for pred_list_probs in pred_substitutes_and_probs:
            #     word_probs = []
            #     for d in pred_list_probs:
            #         word_probs.append(pred_list_probs[d])
            #     probs_score.append(word_probs)

            #——————————————————————是否需要经过后处理将词汇还原？有做工作,handler默认为None，多个词元才能有还原的可能，
            # 所以唯一能做的就是将选出来的特殊标记，【去除】：##possible——>possible
            # pred_substitutes是【二维】的！ 这两个socore已经是【批量】的了，后面再for的是单条处理，这里需要list取平均
            # pred_substitutes的rank——validation sentence_similarity_score attention_score token_similarity_score
            # list[list[float]]每一个元素大小为len(pred_substitutes)分别代表和原始句子的相似度
            # token_similarity_score,sentence_similarity_score,attention_score,validation_score=self.substitute_generator.get_model_score(
            #     tokens_lists,pred_substitutes,target_ids
            # )

            # sentence similarity score by sentenceBERT
            # scores_list=[]
            # for idx,(sentence,target_id) in enumerate(zip(tokens_lists,target_ids)):
            #     pred_sub=pred_substitutes[idx]
            #     new_sentences = []
            #     original_sentence=' '.join(sentence)

            #     new_sentences.append(original_sentence)
            #     for c in pred_sub:
            #         new_sent = sentence.copy()
            #         new_sent[target_id] = c
            #         new_sentences.append(' '.join(new_sent))
                
            #     embeddings = self.sentence_sim_model.encode(new_sentences, convert_to_tensor=True)
                
            #     cosine_scores = util.cos_sim(embeddings[0],embeddings[1:])
            #     scores_list.append(cosine_scores.squeeze(0).tolist())
            

            # pred_substitutes的replaced_token_detection_score
            # 不同模型得到的子词，传入另一个模型，如果有特殊标记，需要先进行预处理。比如bert中的“##”！
            # 不同模型的值不同，##和G，设想，目标词完整，概率选出的词带有特殊标记，即不完整，怎么办？舍去
            # filtered_predsubstitute=self.substitute_generator.remove_special_tokens(pred_substitutes)
            # 返回二维tensor,一维度代表着10个候选词的rtd
            rtd_score=self.electra_model.get_rtd_score(tokens_lists,pred_substitutes,target_ids)
            
            
            # 组合rtd得分和句子相似度得分  alph*sentence_similarity+(1-alph)*token_similarity  alph=[0.5,0.7]
            # 注：rtd_score越大越不好，因为计算的是差值绝对值！而sentence_similarity越大越好  两种值都在（0，1）之间
            # 注：还有一个原始的概率，即从词典中选出来的概率值，排好序的
            # 注：wordnet中没有的词，完全可以从模型词典中【多选几个出来】以谁为基准很关键
            
            # 按行遍历两个二维列表，三个分值的版本
            # final_scores=[]
            # for row_sentence, row_word ,prob_score in zip(scores_list, rtd_score,probs_score):
            #     combined_row = []
            #     for s, w ,p in zip(row_sentence, row_word,prob_score):
            #         combined = ( (1 - w)+p)    # rp and s=0
            #         # 对 word score 进行转换：1 - w，使得值越大越好,  token_similarity and probs_score = word score
            #         # combined = alpha * s + beta *( (1 - w)+p)   # better than seperate
            #         # combined = alpha * s + beta * (1 - w)
                    
            #         combined_row.append(combined)
            #     final_scores.append(combined_row)

            # 两个分值的版本 r and p
            final_scores=[]
            for  row_word ,prob_score in zip( rtd_score,probs_score):
                combined_row = []
                for  w ,p in zip(row_word,prob_score):
                    combined = ( (1 - w)+p)    # rp and s=0
                    combined_row.append(combined)
                final_scores.append(combined_row)
            
            # 两个分支版本，s and p
            # final_scores=[]
            # for row_sentence, prob_score in zip(scores_list,probs_score):
            #     combined_row = []
            #     for s, p in zip(row_sentence,prob_score):
            #         # 对 word score 进行转换：1 - w，使得值越大越好,  token_similarity and probs_score = word score
            #         combined = alpha * s + beta *p    
            #         combined_row.append(combined)
            #     final_scores.append(combined_row)

            
            # pred_substitutes=sort_substitutes_sentence_sim(pred_substitutes,final_scores)  

            # rank pred_substitutes，选出top_k个候选词后根据得分排序
            # pred_substitutes=sort_substitutes_sentence_sim(pred_substitutes=pred_substitutes,pred_score=scores_list)
            # pred_substitutes=sort_substitutes_rtd_score(pred_substitutes,rtd_score)
            # pred_substitutes=sort_substitutes_sentence_sim(pred_substitutes,probs_score)

            pred_substitutes=sort_substitutes_sentence_sim(pred_substitutes,final_scores)  # rtd + predict


            # Ranking candidates using the obtained distribution    数据集有golden candidates 考虑进来
            # candidates在模型词汇表中的词，和不在模型词汇表中的词的排序（根据target处的probs进行排序）
            ranked = self.substitute_generator.candidates_from_probs(
                probs, word2id, candidates
            )
            ranked_candidates_in_vocab, ranked_candidates = ranked
            
            
            # 一次处理一行数据，循环bath次，分数取平均,这里的pred_substitutes维度（batch_size,num_substitutes）
            for i in range(len(pred_substitutes)):
                instance_results = OrderedDict([    # 记住元素插入的顺序
                    ("target_word", tokens_lists[i][target_ids[i]]),
                    ("target_lemma", target_lemmas[i]),
                    ("target_pos_tag", pos_tags[i]),
                    ("target_position", target_ids[i]),
                    ("context", json.dumps(tokens_lists[i])),
                ])
                # 数据集总共有多少条数据有mode
                mode =get_mode(gold_substitutes[i],gold_weights[i])
                if mode is not None:
                    num_mode+=1


                # Metrics computation
                # Compute GAP, GAP_normalized, GAP_vocab_normalized and ranked candidates
                gap_scores = gap_score(
                    gold_substitutes[i], gold_weights[i],
                    ranked_candidates_in_vocab[i], word2id,
                )
                for metric, gap in zip(self.gap_metrics, gap_scores):
                    instance_results[metric] = gap

                # Computing basic Precision, Recall, F-score metrics，这和产生的词也不相关啊？
                # 为了和下面预测的做比较
                base_metrics_values = compute_precision_recall_f1_vocab(
                    gold_substitutes[i], word2id
                )
                for metric, value in zip(self.base_metrics, base_metrics_values):
                    instance_results[metric] = value

                # Computing Top K metrics for each K in the k_list
                k_metrics = compute_precision_recall_f1_topk(
                    gold_substitutes[i], pred_substitutes[i], self.k_list
                )
                for metric, value in k_metrics.items():
                    instance_results[metric] = value

                # computing oot ootm best bestm，注意是考察的前10个，单条数据还没取平均呢,之前的计算ootm和bestm有失误
                oot_and_best=compute_oot_best_metrics(gold_substitutes[i],gold_weights[i],pred_substitutes[i][:10])
                for metric,value in oot_and_best.items():
                    instance_results[metric]=value

                if self.save_instance_results:
                    additional_results = self.create_instance_results(
                        tokens_lists[i], target_ids[i], pos_tags[i],
                        probs[i], word2id, gold_weights[i],
                        gold_substitutes[i], pred_substitutes[i],
                        candidates[i], ranked_candidates[i]
                    )
                    instance_results.update(
                        (k, v) for k, v in additional_results.items()
                    )

                all_metrics_data.append(list(instance_results.values()))

                if columns is None:
                    columns = list(instance_results.keys())
        # print(f'数据集的postag是：{dict_data_pos_tags}')
        all_metrics = pd.DataFrame(all_metrics_data, columns=columns)

        # 此前计算出错，除以的是总的数据条数，改成有mode的数据条数
        mean_metrics = {}
        for metric in self.metrics:
            if metric in ['ootm', 'bestm']:
                value = round(all_metrics[metric].sum(skipna=True) / num_mode * 100, 2)
            else:
                value = round(all_metrics[metric].mean(skipna=True) * 100, 2)
            mean_metrics[metric] = value

        

        weights=self.substitute_generator.prob_estimator.weights
        stratagy_input_embedding=self.substitute_generator.prob_estimator.stratagy_input_embedding

        return {"mean_metrics": mean_metrics, "instance_metrics": all_metrics,"weight":weights,"stratagy_input_embedding":stratagy_input_embedding}

    def create_instance_results(
        self,
        tokens: List[str], target_id: int, pos_tag: str, probs: np.ndarray,
        word2id: Dict[str, int], gold_weights: Dict[str, int],
        gold_substitutes: List[str], pred_substitutes: List[str],
        candidates: List[str], ranked_candidates: List[str],
    ) -> Dict[str, Any]:
        instance_results = OrderedDict()
        pos_tag = to_wordnet_pos.get(pos_tag, None)
        target = tokens[target_id]
        instance_results["gold_substitutes"] = json.dumps(gold_substitutes)
        instance_results["gold_weights"] = json.dumps(gold_weights)
        instance_results["pred_substitutes"] = json.dumps(pred_substitutes)
        instance_results["candidates"] = json.dumps(candidates)
        instance_results["ranked_candidates"] = json.dumps(ranked_candidates)

        if hasattr(self.substitute_generator, "prob_estimator"):
            prob_estimator = self.substitute_generator.prob_estimator
            if target in word2id:
                instance_results["target_subtokens"] = 1
            elif hasattr(prob_estimator, "tokenizer"):
                target_subtokens = prob_estimator.tokenizer.tokenize(target)
                instance_results["target_subtokens"] = len(target_subtokens)
            else:
                instance_results["target_subtokens"] = -1

        if self.save_target_rank:
            target_rank = -1
            if target in word2id:
                target_vocab_idx = word2id[target]
                target_rank = np.where(np.argsort(-probs) == target_vocab_idx)[0][0]
            instance_results["target_rank"] = target_rank

        if self.save_wordnet_relations:
            relations = [
                get_wordnet_relation(target, s, pos_tag)
                for s in pred_substitutes
            ]
            instance_results["relations"] = json.dumps(relations)

        return instance_results

    # 写入结果数据
    @overrides
    def dump_metrics(
        self, metrics: Dict[str, Any], run_dir: Optional[Path] = None, log: bool = False
    ):
        """
        Method for dumping input 'metrics' to 'run_dir' directory.

        Args:
            metrics: Dictionary with two keys:
                - all_metrics: pandas DataFrame, extended 'datasets' with computed metrics
                - mean_metrics: Dictionary with mean values of computed metrics
            run_dir: Directory path for dumping Lexical Substitution task metrics.
            log: Bool flag for logger.
        """
        if run_dir is not None:
            with (run_dir / "metrics.json").open("w") as fp:
                json.dump(metrics["mean_metrics"], fp, indent=4)
            with (run_dir / "metrics.json").open("a") as fp:    # 输入
                json.dump(metrics["stratagy_input_embedding"], fp, indent=4)
            with (run_dir / "metrics.json").open("a") as fp:    # 输出
                json.dump(metrics["weight"], fp, indent=4)

            if self.save_instance_results:
                metrics_df: pd.DataFrame = metrics["instance_metrics"]
                metrics_df.to_csv(run_dir / "results.csv", sep=",", index=False)
                metrics_df.to_html(run_dir / "results.html", index=False)
            if log:
                logger.info(f"Evaluation results were saved to '{run_dir.resolve()}'")
        if log:
            logger.info(json.dumps(metrics["mean_metrics"], indent=4))


    # 程序入口——开始处
    # 指明 模型配置+数据集配置+其它参数
    def solve(
        self,
        substgen_config_path: str,
        dataset_config_path: str,
        run_dir: str = DEFAULT_RUN_DIR,     # 是否需要根据模型+数据名动态变化run_dir，涉及到保存配置文件和运行目录
        mode: str = "evaluate",
        force: bool = False,
        auto_create_subdir: bool = True,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
    ) -> NoReturn:
        """
        Evaluates task defined by configuration files.
        Builds datasets reader from datasets dataset_config_path and substitute generator from substgen_config_path.
        Args:
            substgen_config_path: path to a configuration file.
            dataset_config_path: path to a datasets configuration file.
            run_dir: path to the directory where to [store experiment data].
            mode: evaluation mode - 'evaluate' or 'hyperparam_search'
            force: whether to rewrite data in the existing directory.
            auto_create_subdir: if true a subdirectory will be created automatically
                and its name will be the current date and time
            MLFlow
            experiment_name: results of the run will be added to 'experiment_name' experiment in MLflow.
            run_name: this run will be marked as 'run_name' in MLflow.
        """
        config = {
            "class_name": "evaluations.lexsub.LexSubEvaluation",
            "substitute_generator": substgen_config_path,       # 配置不再构建对象，直接保存路径
            "dataset_reader": dataset_config_path,
            "verbose": self.verbose,
            # 开始调用前，实例化了对象，这里规定batch大小
            "k_list": self.k_list,
            "batch_size": self.batch_size,
            "save_instance_results": self.save_instance_results,
            "save_wordnet_relations": self.save_wordnet_relations,
            "save_target_rank": self.save_target_rank,
        }
        from lexsubgen.runner import Runner
        runner = Runner(config,run_dir, force, auto_create_subdir)
        
        if mode == "evaluate":
            runner.evaluate(
                config=config,
                experiment_name=experiment_name,
                run_name=run_name
            )
        elif mode == "hyperparam_search":   # 超参数搜索模式
            runner.hyperparam_search(
                config_path=Path(run_dir) / "config.json",
                experiment_name=experiment_name
            )


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(name)-16s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # 相对路径，运行时候有问题，别的地方的文件函数运行时候，相对出错——改为绝对
    import os
    # 获取当前脚本的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # print(f"绝对路径:{current_dir}")
    # 获取上一层目录
    parent_dir = os.path.dirname(current_dir)
    # 获取上上层目录
    grandparent_dir = os.path.dirname(parent_dir)


    # 构建配置文件的绝对路径，脚本方式运行
    
    # substgen_config_path = os.path.join(current_dir, "../../configs/subst_generators/xlnet_embs.jsonnet")
    # # dataset_config_path = os.path.join(current_dir, "../../configs/dataset/dataset_readers/coinco.jsonnet")
    # dataset_config_path = os.path.join(current_dir, "../../configs/data/dataset_readers/semeval_all.jsonnet")

    # lexsub_evaluation = LexSubEvaluation()
    # lexsub_evaluation.solve(
    #     # 跳到base_estimator是否是因为这里的配置是bert??
    #     substgen_config_path=substgen_config_path,
    #     dataset_config_path=dataset_config_path,
    #     run_dir=DEFAULT_RUN_DIR,  # 结果保存目录
    #     mode='evaluate',  # 评估模式
    #     force=False,  # 是否覆盖已有数据
    #     auto_create_subdir=True,  # 是否自动创建子目录
    #     experiment_name="bert-large-semeval",  # 实验名称
    #     run_name="test"  # 运行名称
    # )


    parser = argparse.ArgumentParser(description='Run lexical substitution evaluation.')
    parser.add_argument('--substgen-config-path', type=str, required=True, help='Path to substitute generator config file')
    parser.add_argument('--dataset-config-path', type=str, required=True, help='Path to dataset config file')
    parser.add_argument('--run-dir', type=str, default=DEFAULT_RUN_DIR, help='Directory to save results')
    parser.add_argument('--mode', type=str, default="evaluate", help='Evaluation mode')
    parser.add_argument('--force', action='store_true', help='Overwrite existing data')
    parser.add_argument('--auto-create-subdir', action='store_true', help='Automatically create subdirectory')
    parser.add_argument('--experiment-name', type=str, default="lexsub-all-models", help='Experiment name')
    parser.add_argument('--run-name', type=str, default="semeval_all_bert", help='Run name')

    args = parser.parse_args()
    
    lexsub_evaluation = LexSubEvaluation()
    # print(f'配置文件路径：{args.substgen_config_path}')
    substgen_config_path = os.path.join(grandparent_dir, args.substgen_config_path)
    dataset_config_path = os.path.join(grandparent_dir, args.dataset_config_path)
    
    # 调用solve方法进行任务评估
    lexsub_evaluation.solve(
        # 跳到base_estimator是否是因为这里的配置是bert??
        substgen_config_path=substgen_config_path,
        dataset_config_path=dataset_config_path,
        run_dir=DEFAULT_RUN_DIR,  # 结果保存目录
        mode=args.mode,  # 评估模式
        force=False,  # 是否覆盖已有数据
        auto_create_subdir=True,  # 是否自动创建子目录
        experiment_name=args.experiment_name,  # 实验名称
        run_name=args.run_name  # 运行名称
    )
