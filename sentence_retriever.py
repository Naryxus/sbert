import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class SentenceRetriever:

	# corpus_sentences has to be a list of strings
	def __init__(self, corpus_sentences):
		self.corpus_sentences = corpus_sentences

		self.model = SentenceTransformer('bert-base-nli-mean-tokens') # Default config of SBERT

		# Compute embeddings of corpus just once to a #samples x #features matrix
		self.embeddings = np.stack(self.model.encode(corpus_sentences))

	# query is a string and k determines the number of retrieved most similar sentences
	# Result are k pairs of sentences and corresponding cosine-similarity scores
	def retrieve(self, query, k=5):
		query_embedding = self.model.encode([query])[0].reshape(1, -1)
		score_map = zip(self.corpus_sentences, cosine_similarity(self.embeddings, query_embedding).flatten().tolist())
		return sorted(score_map, key=lambda v: v[1], reverse=True)[:k]
