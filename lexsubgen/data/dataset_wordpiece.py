import pandas as pd
from transformers import BertTokenizer


def analyze_target_words_in_csv(file_path):
    # 加载预训练的BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 假设目标词在 'target' 列
    total_count = len(df)
    splittable_count = 0

    for index, row in df.iterrows():
        # 使用WordPiece分词
        target_id=row["target_position"]
        sentence=eval(row["context"])
        temp=sentence[target_id]
        tokens = tokenizer.tokenize(sentence[target_id])
        # 判断是否能分词（分词结果数量大于1表示可分词）
        if len(tokens) > 1:
            splittable_count += 1
            print(tokens)
    
    tp=tokenizer.tokenize("dislike")
    print(tp)
    print(f'{tokenizer.tokenize("unacceptable")}')
    print(f'{tokenizer.tokenize("unhappy")}')


    return splittable_count, total_count

# file_path = '66_lexsubFormyself/lexsubgen/datasets/csv/coinco.csv'  # 967能分词 bert
file_path='66_lexsubFormyself/lexsubgen/datasets/csv/semeval_all.csv' # 28个能分词 bert
splittable_count, total_count = analyze_target_words_in_csv(file_path)
print(f"能分词的目标词个数: {splittable_count}")
print(f"数据总条数: {total_count}")
