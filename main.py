from sentence_retriever import SentenceRetriever
import pandas as pd
import time

start = time.time()

corpus_path = '../masterthesis/data/scidtb.csv' # Just an exemplary corpus as a tab-separated file
corpus = pd.read_csv(corpus_path, sep='\t', header=None) # the file has no header
sentences = corpus[2] # the sentences are in the third column

end = time.time()

print('Corpus loaded, {:.2f}s'.format(end - start))


start = time.time()

retriever = SentenceRetriever(sentences) # initialize the SentenceRetriever

end = time.time()

print('SentenceRetriever initialized, {:.2f}s'.format(end - start))


start = time.time()

# An exemplary query
query = "Efficient software packages are rarely employed in practice for tuning NLP models because their interoperability with deep learning frameworks such as PyTorch is not optimized."
print(retriever.retrieve(query)) # Retrieve the 5 most similar sentences

end = time.time()

print()
print('Retrieval, {:.2f}s'.format(end - start))

