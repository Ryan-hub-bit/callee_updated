import pandas as pd
import gensim
import os
from gensim.parsing.preprocessing import preprocess_documents
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import codecs
import chardet
import numpy
from gensim.test.utils import get_tmpfile
import shutil


# with codecs.open('/home/isec/Documents/tokenized_slices/1b230cafde18feaca285062334af84ab0b97e8a384dae61accac56fd1bf4786a_6.slice', 'r', encoding='ISO-8859-1') as f:
#     df = pd.read_csv(f)

# df = pd.read_csv('/home/isec/Documents/tokenized_slices/1b230cafde18feaca285062334af84ab0b97e8a384dae61accac56fd1bf4786a_6.slice', delimiter='\t', header=None,encoding='latin-1')
text_corpus = []
# text_corpus.append(df.values)
# print(numpy.array2string(df.values))
i = 1
for root, dirs, files in os.walk("/home/isec/Documents/t_slice"):
    for file in files:
        if file.endswith('.uniq'):
            abpath = os.path.join(root,file)
            df=pd.read_csv(abpath, delimiter='\t',header = None,encoding='latin-1')
            print(i)
            text_corpus.append(df.values)
            i += 1
            if i >= 100000:
                break

# print(text_corpus)
# doc = ''.join(str(t) for t in text_corpus)
doc = []
for t in text_corpus:
    # print(t)
    doc.append(''.join(str(t)))
# print(doc)
print("gj")
processed_corpus =  preprocess_documents(doc)
print("process_tagged_corpus_ing")
tagged_corpus = [TaggedDocument(d, [i]) for i, d in enumerate(processed_corpus)]
# print(tagged_corpus)
print("tagged_corpus generated")
model = Doc2Vec(tagged_corpus, dm=0, vector_size=100, min_count=1, epochs=10, hs=1)
model_save_name = get_tmpfile('doc2vec_model3')
print(model_save_name)
model.save(model_save_name)
shutil.copy(model_save_name, "/home/isec/Documents/Callee")

# new_doc = gensim.parsing.preprocessing.preprocess_string(new_doc)
# test_doc_vector = model.infer_vector(new_doc)
# sims = model.docvecs.most_similar(positive = [test_doc_vector])
# for s in sims:
#     print(f"{(s[1])} | {df['Title'].iloc[s[0]]}")