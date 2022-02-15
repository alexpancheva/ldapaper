import sys
import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation

dataset = pd.read_csv(sys.argv[1],header=0, index_col=0)
print ("Done reading file")
numTopics = np.arange(int(sys.argv[2]),int(sys.argv[3]))
valsPerplexity = []
for k in numTopics:
    lda = LatentDirichletAllocation(n_components=k, doc_topic_prior=1, topic_word_prior=0.005)
    output = lda.fit_transform(dataset.T)
    perp = lda.perplexity(dataset.T)
    valsPerplexity.append(int(perp))


print ("Writing results")    
result = pd.DataFrame({'topics': numTopics,'perplexity':valsPerplexity})
result.to_csv("perplexities.csv")

