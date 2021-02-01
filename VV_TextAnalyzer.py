


#basics
import numpy as np 
import pandas as pd
import collections
import re
import functools
import operator
from datetime import datetime, timedelta
from accessify import protected
from copy import deepcopy

#NLP
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from string import punctuation
nltk.download("stopwords")
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
nltk.download("wordnet")
from pymystem3 import Mystem
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import gensim
from sklearn.manifold import TSNE
from nltk import ngrams

#visualise
import matplotlib.pyplot as plt
import chart_studio.plotly as py
#import plotly.graph_obs as go
from plotly.offline import iplot
import cufflinks as cf
import matplotlib.cm as cm
import plotly.express as px


#fastai
from sklearn.model_selection import train_test_split
import fastai
from fastai.text.transform import Tokenizer
from fastai.text.learner import text_classifier_learner
from fastai.text.models import AWD_LSTM
from fastai.text.models import awd_lstm_lm_config


#extra



'''

это модель тональности
ее нужно вызывать вот так
mod = SentimentalModel()
mod.fit()
далее она будет писать тебе что сформировала батчи и обучилась
после этого можно предсказывать 
mod.predict('твое предложение')
и он отдаст тебе тензор, если тот показыват 0, то негативное, если 1 позитивное сообщение
с этим нужно быть аккуратным, там также сбоку циферки, которые в сумме дают 1, это вероятности позитивного
или негативного
посматривай на них обязательно
а еще чем больше слов ты передашь, тем будет точнее

ВАЖНО!
в директории, в которой находится твой питоновский скрипт обязательно должны быть четыре файла
негатив_сенти.цсв
позитив_сенти.цсв
тв_лстм
фт_енк
'''
class SentimentalModel:
    
    def __init__(self, pos='positive_senti.csv', neg='negative_senti.csv'):
        self.pos = pos
        self.neg = neg
        self.path  = ''
        noise = stopwords.words('russian') + list(punctuation)
        upnoise = [letter.upper() for letter in noise]
        self.sum_noise = noise+upnoise+['.','»','«', 'Коллега', "коллега", "это",'спасибо', 
                           'такой',"уважаемый", "квартира", "который", "свой", "пожалуйста"]
        
    def token_text(self, text):
        return [word for word in word_tokenize(text.lower()) if word not in self.sum_noise]
    
    def preprocess_text(self, text):
        text = text.lower().replace("ё", "е")
        text = text.lower().replace("USER", "")
        text = text.lower().replace("rt", "")
        text = text.lower().replace("URL", "")
        text = text.lower().replace("", "")
        text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text)
        text = re.sub('@[^\s]+', 'USER', text)
        text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
        text = re.sub(' +', ' ', text)
        return text.strip()
    
    def create_data(self):
        n = ['id', 'date', 'name', 'text', 'typr', 'rep', 'rtw', 'faw', 'stcount', 'foll', 'frien', 'listcount']
        data_positive = pd.read_csv(self.pos, sep=';', error_bad_lines=False, names=n, usecols=['text'])
        data_negative = pd.read_csv(self.neg, sep=';', error_bad_lines=False, names=n, usecols=['text'])

        # Формируем сбалансированный датасет
        sample_size = min(data_positive.shape[0], data_negative.shape[0])
        raw_data = np.concatenate((data_positive['text'].values[:sample_size],
                                   data_negative['text'].values[:sample_size]), axis=0)
        self.labels = [1] * sample_size + [0] * sample_size
        
        
        self.data = [self.preprocess_text(t) for t in raw_data]
        self.data_tok = pd.Series(self.data).apply(self.token_text)

        self.WORDS = set()
        for sent in self.data_tok:
            for w in sent:
                self.WORDS.add(w)
                
                
        df_train=pd.DataFrame(columns=['Text', 'Label'])
        df_test=pd.DataFrame(columns=['Text', 'Label'])

        df_train['Text'], df_test['Text'], df_train['Label'], df_test['Label'] = train_test_split(self.data,
                                                                                                  self.labels,
                                                                                                  test_size=0.2,
                                                                                                  random_state=1)
        
        df_val=pd.DataFrame(columns=['Text', 'Label'])
        self.df_train, self.df_val = train_test_split(df_train, test_size=0.2, random_state=1)
        
        print('data created')
        
    def fit(self):
        
        self.create_data()
        
        
        
        tokenizer=Tokenizer(lang='xx')
        data_lm = fastai.text.data.TextLMDataBunch.from_df(self.path, tokenizer=tokenizer,
                                                   bs=16, train_df=self.df_train, valid_df=self.df_val,
                                                           text_cols=0)
        print('batches formed')
        
        data_test_clas = fastai.text.data.TextClasDataBunch.from_df(self.path, vocab=data_lm.train_ds.vocab,
                                                            bs=32, train_df=self.df_train, valid_df=self.df_val,
                                                            text_cols=0, label_cols=1, tokenizer=tokenizer)
        
        
        config = fastai.text.models.awd_lstm_clas_config.copy()
        config['n_hid'] = 1150
        self.learn_test =text_classifier_learner(data_test_clas, AWD_LSTM, config=config, drop_mult=0.5)
        
        self.learn_test.load_encoder('/home/victor/fb/ft_enc')
        self.learn_test.load('/home/victor/fb/tw_lstm')
        
        print('model learned')
        
    def predict(self, obj:'str'):
        
        return self.learn_test.predict(obj)
'''
Это анализатор текстов Вити Виталича
первое и наверно самое главное - формат данных
это должен быть датафрейм в пандасе
там должны обязательно быть 3 колонки, с которыми происходит вся работы 
и называться они должны только так и никак иначе "post", "comment" и "date"
соответственно это посты комменты и даты
важно в каком формате все должно быть
колонка post должна содержать str строку 
колонка comment это жуткий костыль:
пустые комменты: вот так '[]', то есть это строка с двумя скобками как будто это лист
непустые комменты вот так "['2 млн лучше брать в кредит', 'У меня была аналогичная ситуация']"
то есть это строка, внутри которой как будто бы лист из строк через запятую)
date - это дата и она должна быть в формате таймдейт

Ну поехали
вызывать это вот так
test = TextAnalyzer(df), где df - это данные, в формате как я указал выше
Если пользоваться моим парсером, то они уже такие 

Функции:

1.frequency(numb, text_type, start_date = None, end_date = None, ngram = 1, draw=True)
это частота слов, то есть сколько таких слов было за период
1. в нее нужно передать обязательно число топ-слов которые ты хочешь вывести
2. обязательно тип данных - коммент или пост
Необязательные параметры:
1. дата начала в формате листа [день,месяц,год]
2. дата конца в таком же формате
если их не передавать, то берется статистика по всем имеющимся датам
3. словосочетания. По умолчанию смотрит одно слово, можно указать искать встречаемость словосочетаний 
из 2,3 и тд
4. отрисовка, если мешают графики можно вырубить

Пример:
test.frequency(20, 'comment', start_date=[28,9,2020], ngram=2,draw=False)
он выведет 20 самых частых словосочетаний из 2 слов из комментов начиная с 28 сентября 20 года без графика

2.date_top(text_type, nmb, start_date = None, end_date = None, draw = False)
это топ слово рядом с датой
обязательные параметры:
1. тип данных. см пред ф-ию
2. количество топ слов по этой дате 
необязательные
дата начала и  конца, см пред. ф-ию
Пример:
test.date_top('comment', 1) - 1 самое популярное слово в комментах на каждый день

3. word_in_time(word)
интерактивный график использования слова во времени
Параметр 1 - слово которое хочешь смотретт
Пример:
test.word_in_time('право') - увидишь график слова "право" во времени

4.date_unique(nmb, start_date = None, end_date = None)
Это отдельная функция, которая рассчитана на то, что ты указываешь даты
то есть ее бесполезно применять на всем промежутке
смысл ее в том, что она выделяет слова, которые значимы именно для этих дат,
относительно всего корпуса текстов

обязательные параметры:
1. количество таких топ-слов
Необязательные
1.дата начала
2.дата конца

Пример:
test.date_unique(15, start_date=[3,10,2020]) - 15 наиболее важных слов для периода 3.10.20-нынешняя дата

5.context(word, start_date = None, end_date = None, k = 15, ngram = 1, draw=True)
Эта штука смотрит на контекст слова, но смотрит довольно втупую, то есть какое слово часто идет рядом с
тем, которое ты указала
Обязательные параметры:
1. Слово)
Необязательные
1.дата начала
2. дата конца
3. количество слов контекста, по умолчанию 15. А также этот параметр регулирует окно, то есть сколько слов
смотри влево и вправо от твоего указанного в каждом предложении
4. словосочетания. По умолчанию смотрит одно слово, можно указать искать встречаемость словосочетаний 
из 2,3 и тд
5. отрисовка - если не хочешь график выключи

Пример:
test.context('право', k=25) - 25 самых частых слов контекста у слова "право"

6.visualise_context(keys, n, start_date=None, end_date=None, min_cnt = 5, wind = 20, 
                         draw=True, title='Context Visualisation', a=0.7)
Эта штука находит контексты умнее и рисует их на графике. 
чем ближе слова на графике, тем ближе они по контексту
обязательные параметры:
1. слова для которых ищется контекст, они должны быть в формате листа из строк ['слово', "второе слово"]
2. количество слов контекста требуемых
необязательные параметры:
1. дата начала и дата конца
2.рисовать или нет
3. название графика title
4. остальные тяжело обьяснить, просто не трогай их
Пример:
test.visualise_context(['залог'], 15) - нарисован график на котором слово "залог" и 15 контекстных слов
'''  
class TextAnalyzer:
    def __init__(self, df):
        noise = stopwords.words('russian') + list(punctuation)
        upnoise = [letter.upper() for letter in noise]
        self.sum_noise = noise+upnoise+['.','»','«', 'Коллега', "коллега", "это",'спасибо', 
                           'такой',"уважаемый", "квартира", "который", "свой", "пожалуйста"]
        ##########TIME#######################
        for i in range(len(df)):
            try:
                df['time'][i] = datetime.strptime(df["time"][i], '%Y-%m-%d')
            except TypeError:
                continue
            
        self.df = deepcopy(self.tokeniz(df))
        self.df2 = deepcopy(df)
        
    @protected
    def tokeniz(self, df):

        #########COMMENTS#####################
        for i in range(len(df)):
            df["comment"][i] = list(df["comment"][i][2:-2].replace("'", '').split(','))
        tw = TweetTokenizer()
        det = TreebankWordDetokenizer()
        for i in (range(len(df))):
            for j in range(len(df["comment"][i])):
                tokenized_example = (tw.tokenize(df["comment"][i][j]))
                filtered_example = [word for word in tokenized_example if not word in self.sum_noise]
                df["comment"][i][j] = det.detokenize(filtered_example)
        mystem_analyzer = Mystem(entire_input=False)
        for i in (range(len(df))):
            df["comment"][i] = [mystem_analyzer.lemmatize(w) for w in df["comment"][i]]
            df["comment"][i] = list(filter(None, df["comment"][i]))
        for i in range(len(df)):
            for j in range(len(df['comment'][i])):
                df['comment'][i][j] = [word for word in df['comment'][i][j] if not word in self.sum_noise]


        ##########POSTS##############
        for i in (range(len(df))):
                tokenized_example = (tw.tokenize(df["post"][i]))
                filtered_example = [word for word in tokenized_example if not word in self.sum_noise]
                df["post"][i] = det.detokenize(filtered_example)
        for i in (range(len(df))):
            a = []
            a.append(df['post'][i])
            df["post"][i] = a
        for i in (range(len(df))):
            df["post"][i] = [mystem_analyzer.lemmatize(w) for w in df["post"][i]][0]
        for i in range(len(df)):
            df['post'][i] = [word for word in df['post'][i] if not word in self.sum_noise]
        
        return df
    
    @protected
    def date_slice(self, start_date = None, end_date = None):
        if start_date == None and end_date == None:
                self.df1 = deepcopy(self.df)
        
        if (type(start_date) == list or type(end_date) == list):
            if (not start_date == None ) and (not end_date==None):
                start_date = datetime(start_date[2], start_date[1], start_date[0], 0, 0)
                end_date = datetime(end_date[2], end_date[1], end_date[0], 0, 0)
                self.df1 = self.df[self.df['time']<end_date]
                self.df1 = self.df1[self.df1['time']>start_date]
            elif (not start_date == None ) and (end_date==None):
                start_date = datetime(start_date[2], start_date[1], start_date[0], 0, 0)
                self.df1 = self.df[self.df['time']>start_date]
            elif (start_date == None ) and (not end_date==None):
                end_date = datetime(end_date[2], end_date[1], end_date[0], 0, 0)
                self.df1 = self.df[self.df['time']<end_date]
        elif (type(start_date) == datetime  and type(end_date) == datetime):
            self.df1 = self.df[self.df['time']<end_date]
            self.df1 = self.df1[self.df1['time']>=start_date]
            
    @protected
    def date_slice1(self, df, start_date = None, end_date = None):
        if start_date == None and end_date == None:
                self.df1 = deepcopy(df)
        
        if (type(start_date) == list or type(end_date) == list):
            if (not start_date == None ) and (not end_date==None):
                start_date = datetime(start_date[2], start_date[1], start_date[0], 0, 0)
                end_date = datetime(end_date[2], end_date[1], end_date[0], 0, 0)
                self.df1 = df[df['time']<end_date]
                self.df1 = self.df1[self.df1['time']>start_date]
            elif (not start_date == None ) and (end_date==None):
                start_date = datetime(start_date[2], start_date[1], start_date[0], 0, 0)
                self.df1 = df[df['time']>start_date]
            elif (start_date == None ) and (not end_date==None):
                end_date = datetime(end_date[2], end_date[1], end_date[0], 0, 0)
                self.df1 = df[df['time']<end_date]
        elif (type(start_date) == datetime  and type(end_date) == datetime):
            self.df1 = df[df['time']<end_date]
            self.df1 = self.df1[self.df1['time']>=start_date]
            
    def frequency(self,  numb, text_type, start_date = None, end_date = None, ngram = 1, draw=True, output=True):
        #start&end date = [d, m ,y]
        if ngram>1:
            draw = False
        self.numb = numb
        self.wordcount = {}
        self.text_type = text_type
        self.draw = draw
        #self.start_date = start_date
        #self.end_date = end_date
       
        self.date_slice1(self.df, start_date, end_date)
        
        if self.text_type == 'comment':
            
            self.df1.index = np.arange(len(self.df1))
            if ngram>1:
                for i in range(len(self.df1)):
                    for j in range(len(self.df1['comment'][i])):
                        self.df1['comment'][i][j] = list(ngrams(self.df1['comment'][i][j],ngram))
            for i in range(len(self.df1)):
                for j in range(len(self.df1['comment'][i])):
                    for word in (self.df1['comment'][i][j]):
                       # if ngram==1:
                        #    word = word.replace(".","")
                         #   word = word.replace(",","")
                          #  word = word.replace(":","")
                           # word = word.replace("\"","")
                           # word = word.replace("!","")
                           # word = word.replace("â€œ","")
                           # word = word.replace("â€˜","")
                           # word = word.replace("*","")
                           # word = word.replace(" ","")
                        if word not in self.sum_noise:
                            if word not in self.wordcount.keys():
                                self.wordcount[word] = 1
                            else:
                                self.wordcount[word] += 1
        if self.text_type == 'post':
            self.df1.index = np.arange(len(self.df1))
            if ngram>1:
                for i in range(len(self.df1)):
                    self.df1['post'][i] = list(ngrams(self.df1['post'][i],ngram))
            for i in range(len(self.df1)):   
                for word in self.df1['post'][i]:
                  #  if ngram==1:
                   #     word = word.replace(".","")
                    ##    word = word.replace(",","")
                     #   word = word.replace(":","")
                     #   word = word.replace("\"","")
                     #   word = word.replace("!","")
                     #   word = word.replace("â€œ","")
                     #   word = word.replace("â€˜","")
                     #   word = word.replace("*","")
                     #   word = word.replace(" ","")
                    if word not in self.sum_noise:
                        if word not in self.wordcount.keys():
                            self.wordcount[word] = 1
                        else:
                            self.wordcount[word] += 1
        self.word_counter = collections.Counter(self.wordcount)
        if output:
            print("{} наиболее часто встречающихся слов\n".format(self.numb))
            for word, count in self.word_counter.most_common(self.numb):
                print(word, ": ", count)
        if self.draw:
            lst = self.word_counter.most_common(self.numb)
            df_cnt = pd.DataFrame(lst, columns = ['Word', 'Count'])
            plt.figure(figsize=(20, 12))
            plt.bar(df_cnt['Word'], df_cnt['Count'])
        if ngram>1:
            global df 
            df = pd.read_csv("fb.csv", index_col=0)
            tokeniz(df)
            self.df = df
    def date_top(self, text_type, nmb, start_date = None, end_date = None, draw = False):
        self.date_slice1(self.df, start_date, end_date)
            
        self.dates = np.sort(self.df1['time'].unique())
        self.top_words = []
        print("Самое частое слово за день\n")
        for i in range(len(self.dates)):
            self.frequency(nmb, text_type, start_date = self.dates[i], end_date = self.dates[i]+timedelta(days=1), draw = draw, output=False)
            #self.top_words.append(self.word_counter.most_common(1)[0][0])
            if not len(self.word_counter.most_common(nmb)) == 0:
                print(self.word_counter.most_common(nmb), self.dates[i].date(), '\n')
            else:
                print('no comments', self.dates[i].date(), '\n')
    def word_in_time(self, word):
        self.word = word
        self.dates = np.sort(self.df['time'].unique())
        self.wrd_cnt = []
        for i in range(len(self.dates)):
            self.count = 0
            self.dfwrd = self.df[self.df['time'] == self.dates[i]]
            self.dfwrd.index = np.arange(len(self.dfwrd))
            for j in range(len(self.dfwrd)):
                for word1 in self.dfwrd['post'][j]:
                    if word1 == self.word:
                        self.count +=1
                for comm in self.dfwrd['comment'][j]:
                    for word2 in comm:
                        if word2 == self.word:
                            self.count +=1
            self.wrd_cnt.append(self.count)
        self.plots = pd.DataFrame(self.wrd_cnt, columns=['Количество упоминаний'])
        self.plots['Дата'] = self.dates
        f = plt.figure(figsize=(19, 15))
        fig = px.line(self.plots, x='Дата', y = 'Количество упоминаний', title = 'Встречаемость слова "{}" во времени'.format(word))
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        fig.show()
        
    def date_unique(self, nmb, start_date = None, end_date = None):
        self.date_slice1(self.df, start_date, end_date)
        self.df1_nd = deepcopy(self.df.iloc[np.delete(np.arange(len(self.df)), self.df1.index),:])
        self.df1.index = np.arange(len(self.df1))
        self.nmb = nmb
        self.d = ''
        det = TreebankWordDetokenizer()
        for i in range(len(self.df1)):
            self.d += ' ' + det.detokenize(self.df1['post'][i])
        for i in range(len(self.df1)):
            for j in range(len(self.df1['comment'][i])):
                self.d += ' ' + det.detokenize(self.df1['comment'][i][j])
        self.df1_nd.index = np.arange(len(self.df1_nd))
        self.nd = ''
        for i in range(len(self.df1_nd)):
            self.nd += ' ' + det.detokenize(self.df1_nd['post'][i])
        for i in range(len(self.df1_nd)):
            for j in range(len(self.df1_nd['comment'][i])):
                self.nd += ' ' + det.detokenize(self.df1_nd['comment'][i][j])
        self.ls_dt = [self.d, self.nd]
        vectorizer = TfidfVectorizer()
        self.X = vectorizer.fit_transform(self.ls_dt)
        self.tt = pd.DataFrame(self.X.toarray(), columns=vectorizer.get_feature_names())
        self.dick = dict(zip(self.tt.columns, self.tt.loc[0]))
        self.utp = sorted(self.dick, key=self.dick.get, reverse=True)
        print ('{} самых значимых слов для данного промежутка\n'.format(self.nmb))
        for i in range(nmb):
            print('{}:'.format(i+1)+' '+self.utp[i])
    def context(self, word, start_date = None, end_date = None, k = 15, ngram = 1, draw=True):
        self.date_slice1(self.df, start_date, end_date)
        self.closest = []
        if ngram ==1:     
            for i in range(len(self.df1)):
                j = 0
                try:
                    index = self.df1['post'][i][j:].index(word)
                    if index>k:
                        if len(self.df1['post'][i][j:][index:])>k:
                            context_ls = self.df1['post'][i][j:][index-k:index+k]
                        elif len(self.df1['post'][i][j:][index:])<=k:
                            context_ls = self.df1['post'][i][j:][index-k:]
                    elif index<=k:
                        if len(self.df1['post'][i][j:][index:])>k:
                            context_ls = self.df1['post'][i][j:][:index+k]
                        elif len(self.df1['post'][i][j:][index:])<=k:
                            context_ls = self.df1['post'][i][j:][:]
                    self.closest.append(context_ls)
                    j = index
                except ValueError:
                    continue
            for i in range(len(self.df1)):
                for f in range(len(self.df1['comment'][i])):
                    j = 0
                    try:
                        index = self.df1['comment'][i][f][j:].index(word)
                        if index>k:
                            if len(self.df1['comment'][i][f][j:][index:])>k:
                                context_ls = self.df1['comment'][i][f][j:][index-k:index+k]
                            elif len(self.df1['comment'][i][f][j:][index:])<=k:
                                context_ls = self.df1['comment'][i][f][j:][index-k:]
                        elif index<=k:
                            if len(self.df1['comment'][i][f][j:][index:])>k:
                                context_ls = self.df1['comment'][i][f][j:][:index+k]
                            elif len(self.df1['comment'][i][f][j:][index:])<=k:
                                context_ls = self.df1['comment'][i][f][j:][:]
                        self.closest.append(context_ls)
                        j = index
                    except ValueError:
                        continue
            self.wordcount_contxt = {}
            for i in range(len(self.closest)):   
                        for word in self.closest[i]:
                            word = word.replace(".","")
                            word = word.replace(",","")
                            word = word.replace(":","")
                            word = word.replace("\"","")
                            word = word.replace("!","")
                            word = word.replace("â€œ","")
                            word = word.replace("â€˜","")
                            word = word.replace("*","")
                            if word not in self.sum_noise:
                                if word not in self.wordcount_contxt:
                                    self.wordcount_contxt[word] = 1
                                else:
                                    self.wordcount_contxt[word] += 1
            print("{} наиболее часто встречающихся слов контекста\n".format(k))
            self.wordcount_contxt.pop(word)
            word_counter = collections.Counter(self.wordcount_contxt)
            for word, count in word_counter.most_common(k):
                   print(word, ": ", count)
                
        if ngram>1:
            print('не работает, извините(')
            draw=False
            '''
            
            for i in range(len(self.df1)):
                    for j in range(len(self.df1['comment'][i])):
                        self.df1['comment'][i][j] = list(ngrams(self.df1['comment'][i][j],ngram))
            for i in range(len(self.df1)):
                    self.df1['post'][i] = list(ngrams(self.df1['post'][i],ngram))
                    
            self.contxt_ls = []
            for i in range(len(self.df1)):
                for j in range(len(self.df1['post'][i])):
                    if word in self.df1['post'][i][j]:
                        self.contxt_ls.append(self.df1['post'][i][j])
            for i in range(len(self.df1)):
                for j in range(len(self.df1['comment'][i])):
                    for f in range(len(self.df1['comment'][i][j])):
                        if word in self.df1['comment'][i][j][f]:
                            self.contxt_ls.append(self.df1['comment'][i][j][f])
            self.wordcount_contxt = {}
            for word in (self.contxt_ls):   
                if word not in self.sum_noise:
                    if word not in self.wordcount_contxt:
                        self.wordcount_contxt[word] = 1
                    else:
                        self.wordcount_contxt[word] += 1
            self.wordcount_contxt.pop(word)
            print("{} наиболее часто встречающихся слов контекста\n".format(k))
            word_counter = collections.Counter(self.wordcount_contxt)
            for word, count in word_counter.most_common(k):
                   print(word, ": ", count) 
            
            global df
            df = pd.read_csv("fb.csv", index_col=0)
            tokeniz(df)
            self.df = deepcopy(df)
            '''
        if draw:
            lst = word_counter.most_common(k)
            df_cnt = pd.DataFrame(lst, columns = ['Word', 'Count'])
            plt.figure(figsize=(20, 12))
            plt.bar(df_cnt['Word'], df_cnt['Count'])

    def visualise_context(self, keys:"list", n, start_date=None, end_date=None, min_cnt = 5, wind = 20, 
                         draw=True, title='Context Visualisation', a=0.7):
        
        self.date_slice1(self.df2, start_date, end_date)
        
        corpus = []
        for ls in self.df1['post']:
            corpus.append(ls)
        for i in range(len(self.df1)):
            for j in range(len(self.df1['comment'][i])):
                corpus.append(self.df1['comment'][i][j])
                
        
        self.keys = keys
        model = gensim.models.word2vec.Word2Vec(sentences = corpus, min_count = min_cnt, window=wind,
                                                workers=4)

        embedding_clusters = []
        word_clusters = []
        for word in keys:
            embeddings = []
            words = []
            for similar_word, _ in model.most_similar(word, topn=n):
                words.append(similar_word)
                embeddings.append(model[similar_word])
            words.append(word)
            embeddings.append(model[word])
            embedding_clusters.append(embeddings)
            word_clusters.append(words)
        embedding_clusters = np.array(embedding_clusters)
        n, m, k = embedding_clusters.shape
        try:
            tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
            embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
        except:
            tsne_model_en_2d = TSNE(perplexity=15, n_components=2, n_iter=3500, random_state=32)
            embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
        
        if draw:
            
            plt.figure(figsize=(20, 12))
            colors = cm.rainbow(np.linspace(0, 1, len(self.keys)))
            for label, embeddings, words, color in zip(self.keys, embeddings_en_2d, word_clusters, colors):
                x = embeddings[:, 0]
                y = embeddings[:, 1]
                plt.scatter(x, y, c=color.reshape(1,-1), alpha=a, label=label)
                for i, word in enumerate(words):
                    plt.annotate(word, alpha=3, xy=(x[i], y[i]), xytext=(5, 2),
                                 textcoords='offset points', ha='right', va='bottom', size=15)
            plt.legend(loc=4)
            plt.title(title)
            plt.grid(True)
            plt.show()
