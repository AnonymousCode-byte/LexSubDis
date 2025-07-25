# python ../lexsubgen/evaluations/lexsub.py  --substgen-config-path configs/subst_generators/bert.jsonnet --dataset-config-path configs/data/dataset_readers/semeval_all.jsonnet --run-dir='debug/lexsub-all-models/semeval_all_bert' --force --experiment-name='lexsub-all-models' --run-name='semeval_all_bert'
# python ../lexsubgen/evaluations/lexsub.py  --substgen-config-path configs/subst_generators/bert.jsonnet --dataset-config-path configs/data/dataset_readers/coinco.jsonnet --run-dir='debug/lexsub-all-models/coinco_bert' --force --experiment-name='lexsub-all-models' --run-name='coinco_bert'


# python ../lexsubgen/evaluations/lexsub.py  --substgen-config-path configs/subst_generators/bert_embs.jsonnet --dataset-config-path configs/data/dataset_readers/semeval_all.jsonnet --run-dir='debug/lexsub-all-models/semeval_all_bert_embs' --force --experiment-name='lexsub-all-models' --run-name='semeval_all_bert_embs'
# python ../lexsubgen/evaluations/lexsub.py  --substgen-config-path configs/subst_generators/bert_embs.jsonnet --dataset-config-path configs/data/dataset_readers/coinco.jsonnet --run-dir='debug/lexsub-all-models/coinco_bert_embs' --force --experiment-name='lexsub-all-models' --run-name='coinco_bert_embs'


# python ../lexsubgen/evaluations/lexsub.py  --substgen-config-path configs/subst_generators/xlnet.jsonnet --dataset-config-path configs/data/dataset_readers/semeval_all.jsonnet --run-dir='debug/lexsub-all-models/semeval_all_xlnet' --force --experiment-name='lexsub-all-models' --run-name='semeval_all_xlnet'
# python ../lexsubgen/evaluations/lexsub.py  --substgen-config-path configs/subst_generators/xlnet.jsonnet --dataset-config-path configs/data/dataset_readers/coinco.jsonnet --run-dir='debug/lexsub-all-models/coinco_xlnet' --force --experiment-name='lexsub-all-models' --run-name='coinco_xlnet'

# python ../lexsubgen/evaluations/lexsub.py  --substgen-config-path configs/subst_generators/xlnet_embs.jsonnet --dataset-config-path configs/data/dataset_readers/semeval_all.jsonnet --run-dir='debug/lexsub-all-models/semeval_all_xlnet_embs' --force --experiment-name='lexsub-all-models' --run-name='semeval_all_xlnet_embs'
python ../lexsubgen/evaluations/lexsub.py  --substgen-config-path configs/subst_generators/xlnet_embs.jsonnet --dataset-config-path configs/data/dataset_readers/coinco.jsonnet --run-dir='debug/lexsub-all-models/coinco_xlnet_embs' --force --experiment-name='lexsub-all-models' --run-name='coinco_xlnet_embs'
