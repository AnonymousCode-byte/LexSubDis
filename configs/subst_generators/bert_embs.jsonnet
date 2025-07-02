local pre_processing = import '../data/dataset_preprocess/bert_preprocessor.jsonnet';
local prob_estimator = import '../prob_estimators/bert_embs.jsonnet';
local post_processing = import '../data/dataset_postprocess/lower_nltk_spacy.jsonnet';

{
    class_name: "SubstituteGenerator",
    pre_processing: pre_processing,
    prob_estimator: prob_estimator,
    post_processing: post_processing,
    top_k: 20
}
