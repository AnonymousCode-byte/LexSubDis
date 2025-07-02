{
    class_name: "lexsubgen.prob_estimators.combiner.BcombCombiner",
    prob_estimators: [
        {
            class_name: "lexsubgen.prob_estimators.bert_estimator.BertProbEstimator",
            mask_type: "not_masked",
            model_name: "bert-large-cased",
            embedding_similarity: false,
            temperature: 1.0,
            use_attention_mask: true,
            cuda_device: 0,
            verbose: false,

            weights:"mean",   # [ 'first','mean', 'linear', 'exponential'][max][min]  模型初始化受到影响
            stratagy_input_embedding:"mix-up" # mask\gauss\mix-up\dropout\keep
        },
        {
            class_name: "lexsubgen.prob_estimators.bert_estimator.BertProbEstimator",
            mask_type: "not_masked",
            model_name: "bert-large-cased",
            embedding_similarity: true,
            temperature: 0.1,
            use_attention_mask: true,
            use_subword_mean: true,
            cuda_device: 0,
            verbose: false,

            weights:"mean",   # ['first'，'mean', , 'linear', 'exponential'][max][min]  模型初始化受到影响
            stratagy_input_embedding:"mix-up" # mask\gauss\mix-up\dropout\keep
        }
    ],
    verbose: false,
    weights:"mean",   # ['first', 'mean', 'linear', 'exponential'][max][min]  模型初始化受到影响
    stratagy_input_embedding:"mix-up" # mask\gauss\mix-up\dropout\keep
}
