import jieba

# 1. 分词
# 全模式分词,将句子中所有可以成词的词语都扫描出来,速度非常快,但是不能消除歧义
content = '吃葡萄不吐葡萄皮，不吃葡萄倒吐葡萄皮'
jieba.cut(content, cut_all=True)
jieba.lcut(content, cut_all=True)
# 精确分词,适合文本分析
jieba.cut(content, cut_all=False)
jieba.lcut(content,cut_all=False)
# 搜索引擎模式分类,在精确模式的基础上,对长词再次切分,提高召回率,适合用于搜索引擎
jieba.cut_for_search(content)
jieba.lcut_for_search(content)
# 添加自定义词典（在文件开头导入后添加）
jieba.load_userdict('user_dict.txt')  # 文件路径为当前目录或绝对路径

# 2. 词性标注
# 标注句子分词后每个词的词性,采用和ictclas兼容的标记法
import jieba.posseg as pseg
print(pseg.lcut('我爱北京天安门'))
import hanlp
tagger = hanlp.load(hanlp.pretrained.pos.PTB_POS_RNN_FASTTEXT_EN)
tagger(['我', '爱', '北京', '天安门'])