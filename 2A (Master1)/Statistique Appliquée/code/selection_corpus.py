from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#import nltk
import nltk
from nltk.corpus import stopwords

#stop-words
nltk.download('stopwords')


def spectral_entropy(matrix, tol = 10**(-2)):
    eigenvalues, _ = np.linalg.eig(matrix)
    eigenvalues = np.real(eigenvalues)
    
    for i,val in enumerate(eigenvalues) :
        if val < tol :
            eigenvalues[i] = 0
    
    eigenvalues = eigenvalues/np.sum(eigenvalues)

    entr = 0
    for val in eigenvalues:
        #print(val)

        if val > 0 :
            #print(val*np.log(val))
            entr += val*np.log(val)
    #print("entr :",entr)

    eigenvalues = np.power(eigenvalues,-eigenvalues)

    return np.exp(-entr),eigenvalues

def select_n_highest(array, n):
    sorted_indices = np.argsort(-array)
    selected_indices = sorted_indices[:n]
    selected_indices.sort()
    selected_values = array[selected_indices]
    return selected_values, selected_indices


def select_least_similar3(corpus, size_selected, num_subcorpus = 10, max_features = 5000, seed_value = 345, tol = 10**(-2), n_components = None, affiche_entrop = False):
    
    n = len(corpus)

    if n_components is None:
        n_components = round((size_selected) ** 0.5)
    max_vendi = 0

    stop_words = nltk.corpus.stopwords.words('english')
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=max_features)
    selected_indices = []

    for i in tqdm(range(num_subcorpus)) :
        
        np.random.seed(seed_value + i)
        random_indices = np.random.choice(n, size_selected, replace=False)
        sub_corpus = [corpus[idx] for idx in random_indices]

        tfidf_matrix = vectorizer.fit_transform(sub_corpus)
        lsa_model = TruncatedSVD(n_components=n_components, n_iter=10, random_state=seed_value)
        lsa_matrix = lsa_model.fit_transform(tfidf_matrix)
        similarity_matrix = cosine_similarity(lsa_matrix, lsa_matrix)

        vendi, aux = spectral_entropy(similarity_matrix, tol)
        #print("vendi :",vendi)

        if vendi > max_vendi :
            selected_indices = random_indices
            max_vendi = vendi
    
    if affiche_entrop:
        print(max_vendi)

    return data.select(selected_indices)