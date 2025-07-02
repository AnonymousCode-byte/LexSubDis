{
    class_name: "lexsubgen.prob_estimators.xlnet_estimator.XLNetProbEstimator",
    masked: false,              # 不用mask标记，且单词的话就取第一位而不是多个子词
    model_name: "xlnet-large-cased",
    embedding_similarity: false,
    temperature: 1.0,
    use_input_mask: true,
    multi_subword: false,
    cuda_device: 4,
    verbose: false,
    
    weights:"first",   # ['mean', 'first', 'linear', 'exponential'] [max][min] 模型初始化受到影响  linear效果更好
    stratagy_input_embedding:"mix-up" # mask\gauss\mix-up\dropout\keep  补一个linear
}
