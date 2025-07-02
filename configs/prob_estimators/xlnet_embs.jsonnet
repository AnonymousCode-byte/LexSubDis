{
    class_name: "lexsubgen.prob_estimators.combiner.BcombCombiner",
    prob_estimators: [
        {
            class_name: "lexsubgen.prob_estimators.xlnet_estimator.XLNetProbEstimator",
            masked: false,
            model_name: "xlnet-large-cased",
            embedding_similarity: false,
            temperature: 1.0,
            use_input_mask: true,
            multi_subword: false,
            cuda_device: 4,
            verbose: false,

            weights:"first",   # ['mean', 'first', 'linear', 'exponential'] [max][min] 模型初始化受到影响  linear效果更好
            stratagy_input_embedding:"mix-up" # mask\gauss\mix-up\dropout\keep
        },
        {
            class_name: "lexsubgen.prob_estimators.xlnet_estimator.XLNetProbEstimator",
            masked: false,
            model_name: "xlnet-large-cased",
            embedding_similarity: true,
            temperature: 0.1,
            use_input_mask: true,
            multi_subword: false,
            cuda_device: 5,
            use_subword_mean: true,
            verbose: false,

            weights:"first",   # ['mean', 'first', 'linear', 'exponential'] [max][min] 模型初始化受到影响  linear效果更好
            stratagy_input_embedding:"mix-up" # mask\gauss\mix-up\dropout\keep
        }
    ],
    verbose: false,
    weights:"first",   # ['mean', 'first', 'linear', 'exponential'] [max][min] 模型初始化受到影响  linear效果更好
    stratagy_input_embedding:"mix-up" # mask\gauss\mix-up\dropout\keep
}
