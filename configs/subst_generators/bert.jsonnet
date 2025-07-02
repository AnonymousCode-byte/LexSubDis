local pre_processing = import '../data/dataset_preprocess/bert_preprocessor.jsonnet';
local prob_estimator = import '../prob_estimators/bert.jsonnet';
local post_processing = import '../data/dataset_postprocess/lower_nltk_spacy.jsonnet';
{
    class_name: "SubstituteGenerator",
    pre_processing: pre_processing,
    prob_estimator: prob_estimator,
    post_processing: post_processing,
    top_k: 20                               # top p，不好，无法量化p的值,10个足够了，gold 最多才差不多5个
    # 但是为了顾及哪些没有在wordnet中有同义词的词，需要选20个，
    # wordnet中同样挑20个
}
