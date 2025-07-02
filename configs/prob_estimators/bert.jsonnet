{
    class_name: "lexsubgen.prob_estimators.bert_estimator.BertProbEstimator",
    mask_type: "not_masked",    # 保留所有分词，mask的话就是合并后mask
    model_name: "bert-large-cased",
    embedding_similarity: false,
    temperature: 1.0,
    use_attention_mask: true,
    cuda_device: 4,
    verbose: false,

    weights:"mean",   # [ 'first', 'mean','linear', 'exponential'][max][min]  模型初始化受到影响  linear效果更好
    stratagy_input_embedding:"mix-up" # mask\gauss\mix-up\dropout\keep
}
