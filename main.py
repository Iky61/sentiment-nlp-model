import pandas as pd
import numpy as np 
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS
import string

# get the stop words using Sastrawi library
factory = StopWordRemoverFactory()

# stop words from Sastrawi
stopwords_1 = factory.get_stop_words()

# indonesian stop words
stopwords_2 = [
    'yang', 'di', 'dan', 'untuk', 'pada', 'dari', 'dengan', 'karena', 'oleh', 'atau',
    'tetapi', 'jika', 'ketika', 'sehingga', 'agar', 'namun', 'juga', 'ini', 'itu',
    'adalah', 'maka', 'saat', 'belum', ' sudah', 'hanya', 'saja', 'belum', 'telah',
    'akan', 'kami', 'kamu', 'mereka', 'saya', 'dia', 'anda', 'kita', 'kalian',
    'dimana', 'bagaimana', 'kenapa', 'siapa', 'apa', 'mana', 'berapa', 'cara', 'apa',
    'setiap', 'beberapa', 'semua', 'tidak', 'ya', 'tidak', 'ya', 'mungkin', 'pasti',
    'harus', 'boleh', 'tidak', 'boleh', 'oleh', 'kali', 'jadi', 'misalnya', 'dapat',
    'bisa', 'sudah', 'masih', 'lagi', 'setelah', 'sebelum', 'pada', 'kepada', 'dalam',
    'luar', 'dari', 'sampai', 'hingga', 'melalui', 'dengan', 'tanpa', 'bagi', 'tentang',
    'terhadap', 'menurut', 'sesuai', 'antara', 'dari', 'pada', 'ke', 'di', 'itu', 'ini',
    'atau', 'dan', 'lainnya', 'dll', 'nya', 'tapi', 'disini', 'nya', 'yg', 'sangat', 'palu', 
    'satu','kota','skli', 'buat', 'selalu', 'menggunakan', 'banyak', 'mudah','membantu',
    'digunakan','bagus','baik','terus', 'lebih','kasih','terima','hebat','berguna','memperbarui',
]

# english stop words
stopwords_3 = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
    'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off',
    'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
    'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn',
    'weren', 'won', 'wouldn','comment'
]

# append all stopwords
stop_words = stopwords_1 + stopwords_2 + stopwords_3

# create the function to remove stop words from sentence
def stop_words_processing(sentence):    
    # remove the sentence from any simbols and make it lower sentence
    sentence = sentence.translate(str.maketrans('','',string.punctuation)).lower()
    sentence = ' '.join(sentence.split())
    
    # get the words is not stop words
    non_stopwords = [word for word in sentence.split() if word not in stop_words]
    non_stopwords = ' '.join(non_stopwords)

    return non_stopwords


# create function to count all hot topics
def get_target_count(sentence, target):
    msg=pd.DataFrame({'text':sentence.split()})
    msg=msg['text'].value_counts().reset_index().rename(columns={'text':'count', 'index':'text'})
    msg=msg[msg.text.isin(target)==True]
    msg['sentences']=sentence
    return msg


class Predict(Resource):
    def predict():
        try:
            # Get input data from the request
            data = request.json['sentences']
            n_topics = request.json['ntopics']

            # Sample text data
            data = pd.DataFrame({'sentences':data})
            data['sentences2']=data.sentences.apply(stop_words_processing)
            documents=data.sentences.tolist()

            # Vectorize the text data
            vectorizer = CountVectorizer(stop_words=stop_words)
            dtm = vectorizer.fit_transform(documents)
            
            # Apply LDA
            num_topics = 1  # Adjust based on your needs
            lda = LatentDirichletAllocation(n_components=num_topics)
            lda.fit(dtm)
            
            # Display the top words for each topic
            topics = []
            feature_names = vectorizer.get_feature_names_out()
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[:-(n_topics) - 1:-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics.append(top_words)

            data['sentences3']=data.sentences2.apply(lambda x: get_target_count(x, topics[0]))
            report=pd.concat(data.sentences3.tolist())
            report=report.groupby(['text'])[['count']].sum().sort_values('count', ascending=False).reset_index()
            report['p_count']=report['count']/len(feature_names)

            return {'topics':report.text.tolist(), 'count':report['count'].tolist(), 'normalize':report.p_count.tolist()}
        
        except Exception as e:
            return {'error': str(e)}
    

# inisiasi objeck flask
app = Flask(__name__)

# inisasi object flask_restful
api = Api(app)

# inisiasi object flask_cors
CORS(app)

# setup resource
api.add_resource(Predict, "/api/predict", methods=["GET", "POST"])

if __name__ == "__main__":
    app.run(debug=True)