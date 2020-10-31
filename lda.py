from nltk.stem.wordnet import WordNetLemmatizer
import string
import pandas as pd
import gensim
from gensim import corpora

doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father."
doc2 = "My father spends a lot of time driving my sister around to dance practice."
doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."
doc4 = "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better."
doc5 = "Health experts say that Sugar is not good for your lifestyle."


df = pd.read_excel("/Users/pineleaf/Desktop/15年数据.xlsx",usecols=['ABST'],names=None)
df = pd.DataFrame(df.astype(str))
# print(df)
list_1 = df.values.tolist()
list_2 = []
# print(list_1)
for i in list_1:
    list_2.append(i[0])
print(list_2)

# 整合文档数据
doc_complete = [doc1, doc2, doc3, doc4, doc5]
# print(doc_complete)
# 加载停用词
stopwords = pd.read_csv('/Users/pineleaf/PycharmProjects/English_word_segmentation/English_stop_words.txt',
                        index_col=False, quoting=3, sep="\t", names=['stopword'], encoding='utf-8')
stopwords = stopwords['stopword'].values

exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stopwords])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


doc_clean = [clean(doc).split() for doc in list_2]

# 创建语料的词语词典，每个单独的词语都会被赋予一个索引
dictionary = corpora.Dictionary(doc_clean)

# 使用上面的词典，将转换文档列表（语料）变成 DT 矩阵
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# 使用 gensim 来创建 LDA 模型对象
Lda = gensim.models.ldamodel.LdaModel

# 在 DT 矩阵上运行和训练 LDA 模型
ldamodel = Lda(doc_term_matrix, num_topics=20, id2word=dictionary, passes=50)

# 输出结果
print(ldamodel.print_topics(num_topics=20, num_words=10))