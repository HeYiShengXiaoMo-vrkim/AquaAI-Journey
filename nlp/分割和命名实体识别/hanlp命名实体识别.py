"""
import hanlp

try:
    # 更换为支持当前环境的通用中文模型
    tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_PKU_NAME_MERGED_SIX_MONTHS)
    tokens = tokenizer('2022年HanLPv2.1为生产环境带来次世代最先进的多语种NLP技术。阿婆主来到北京立方庭参观自然语义科技公司。')
    print(tokens)
except Exception as e:
    print(f"处理出错：{e}")
"""

import hanlp
# 加载中文命名实体识别的预训练模型MSRA_NER_BERT_ZH
recognizer_Chinese = hanlp.load(hanlp.pretrained.ner.MSRA_NER_BERT_BASE_ZH)
# 输入是对句子进行字符分割的列表,因此在句子前加上了List()
list('上海华安工业(集团)公司董事长谭旭光和秘书长张晚霞来到美国纽约现代化艺术博物馆参观')
recognizer_Chinese(list('上海华安工业(集团)公司董事长谭旭光和秘书长张晚霞来到美国纽约现代化艺术博物馆参观'))
# 加载对英文进行处理的包
recognizer_English = hanlp.load(hanlp.pretrained.ner.CONLL03_NER_BERT_BASE_CASED_EN)
