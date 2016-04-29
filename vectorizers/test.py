from vectorizer2 import Vectorizer2

v = Vectorizer2(stop_words='english')
print(v)

with open('data/tweets.txt') as f:
    documentos = f.readlines()

#documentos = ["This is text 1", "This is other hell of a bad text", "This is the best awesome text of life"]
aux = v.fit_transform(documentos)
print(aux)