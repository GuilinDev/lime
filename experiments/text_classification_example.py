import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np
from lime import lime_text
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt

# Load 20newsgroups dataset
categories = ['alt.atheism', 'soc.religion.christian']
newsgroups_train = sklearn.datasets.fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = sklearn.datasets.fetch_20newsgroups(subset='test', categories=categories)

# Create TF-IDF vectorizer and classifier
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train_vectors, newsgroups_train.target)

# Create pipeline
c = make_pipeline(vectorizer, rf)

# Use LIME to explain prediction
explainer = LimeTextExplainer(class_names=categories)
idx = 1
exp = explainer.explain_instance(newsgroups_test.data[idx], c.predict_proba, num_features=6)
print('Document id: %d' % idx)
print('Predicted class =', categories[c.predict([newsgroups_test.data[idx]])[0]])
print('True class: %s' % categories[newsgroups_test.target[idx]])

# Display explanation - use matplotlib instead of notebook
plt.figure(figsize=(10,6))
exp.as_pyplot_figure()
plt.tight_layout()
plt.show()

# Print text explanation
print('\nExplanation as list:')
for feature, weight in exp.as_list():
    print(f'{feature}: {weight}') 