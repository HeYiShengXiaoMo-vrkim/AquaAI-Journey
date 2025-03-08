"""
1. 将一段文本使用 张量 进行展示,其中一般将词汇表示成一个向量,称作词向量,再有各个词向量按照顺序组成矩阵形成文本表示
2. 将文本表示成张量形式,能够使语言文本可以作为计算机处理程序的输入，进行接下来的一系列的解析工作
3. 常用的文本 张量 表示方法: -one-hot编码 -Word2vec -Word Embedding
4. one-hot编码:独热编码,将每个词表示成具有n个元素的向量,这个词向量中只有一个元素是1,其余元素都是0,不同元素为0的位置不同,不同词汇元素为0的位置不同,其中n的大小是整个语料中不同词汇的数量
    这样的向量称为one-hot向量,将所有的词向量按照顺序组成矩阵,称为one-hot矩阵,这种表示方法的优点是简单易懂,
    缺点是向量维度高,稀疏度高,计算量大,无法表示词与词之间的关系
"""
# 导入用于对象保存与加载的joblib
import joblib
# 导入keras中的词汇映射器Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer  # 恢复原始导入方式

# 假定vocab为语料集所有不同词汇集合
vocab = ['我', '爱', '你', '中国', '北京', '上海', '杭州', '南京', '广州', '深圳']
# 初始化一个词汇映射器
tokenizer = Tokenizer(num_words=5000, char_level=False) # 设定词汇表的大小为5000, 设定为字符级别的映射器
# 使用映射器拟合现有文本数据
tokenizer.fit_on_texts(vocab)

for tokens in vocab:
    zero_list = [0] * len(vocab)
    # 使用映射器将转化现有文本数据,每个词汇对应从1开始的自然数
    # 返回样式如: [[2]], 取出其中的数字需要使用[0][0] - 1
    token_index = tokenizer.texts_to_sequences([tokens])[0][0] - 1 # 修正索引值,从0开始
    zero_list[token_index] = 1
    print(f"{tokens}的one-hot编码为:{zero_list}")

# 保存映射器,以便以后使用
tokenizer_path = './Tokenizer'
joblib.dump(tokenizer, tokenizer_path)