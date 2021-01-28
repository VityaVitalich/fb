
import requests      # Библиотека для отправки запросов
import numpy as np   # Библиотека для матриц, векторов и линала
import pandas as pd  # Библиотека для табличек
import time          # Библиотека для тайм-менеджмента
from bs4 import BeautifulSoup
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from fake_useragent import UserAgent
from tqdm import tqdm
from webdriver_manager.firefox import GeckoDriverManager
from datetime import datetime
import hashlib
from selenium.webdriver.common.by import By
from sys import platform
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from selenium.webdriver import FirefoxOptions
from string import punctuation
nltk.download("stopwords")
import datetime
import time


MonthDict={ "января" : 1,
  "февраля": 2,
   "марта": 3,
   "апреля": 4,
   "мая": 5,
   "июня": 6,
   "июля": 7,
   "августа": 8,
   "сентября": 9,
   "октября": 10,
   "ноября": 11,
   "декабря": 12
}

def init():

    options = FirefoxOptions()
    #opts.add_argument("--headless")

    #options.add_argument("--headless")
    options.add_argument("--window-size=1920,1080")
    #options.add_argument('--MOZ_LOG_FILE=/root/firefox.log')
    #options.add_argument("start-maximized")
    #options.add_argument("disable-infobars")
    #options.add_argument("--disable-extensions")
    #options.add_argument("--no-sandbox")
    #options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Firefox( firefox_options=options, executable_path=GeckoDriverManager().install())

    return driver

def search(request, driver):
    #вызов нужного сайта
    import time
    driver.get(request)
    time.sleep(6)
    ok_button = driver.find_elements_by_xpath("//span[contains(text(), 'OK')]")
    try:
        ok_button[0].click()
    except:
        pass

def scrolling(times_scroll, driver):
    #Цикл для повторения прокрутки
    for i in tqdm(range(times_scroll)):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        if i%5==0: #Здесь каждую пятую прокрутку цикл засыпает на 1 секунду, чтобы страница прогрузилась
                   #Если убрать эту строчку, может не дорабатывать до конца и прогружать меньше
            time.sleep(1)
    print('scrolled')

def opening_comms(driver, num_iter=3):
    #Костыли для открытия комментов
    #Костыли, созданные, чтобы открыть все длинные посты. К сожалению к 3 циклам пришлось прибегнуть, так как
    #часть кнопок не нажимаются с первого раза из-за того, что сайт недогрузился или не доскролился до нужной кнопки            
    for j in range(num_iter):       
        should_restart = True
        while should_restart:
            should_restart = False
            buttons = driver.find_elements_by_xpath("//span[contains(text(), 'Показать ещё')]")
            for i in tqdm(range(len(buttons))):
                try: 
                    buttons[i].click()
                except:
                    continue

        buttons = driver.find_elements_by_xpath("//span[contains(text(), 'ответ')]")
        for i in tqdm(range(len(buttons))):
            try: 
                buttons[i].click()
            except:
                continue
        buttons = driver.find_elements_by_xpath("//span[contains(text(), 'ответов')]")
        for i in tqdm(range(len(buttons))):
            try: 
                buttons[i].click()
            except:
                continue

        buttons = driver.find_elements_by_xpath("//span[contains(text(), 'Показать предыдущие комментарии')]")
        for i in tqdm(range(len(buttons))):
            try: 
                buttons[i].click()
            except:
                continue
        buttons = driver.find_elements_by_xpath("//div[contains(text(), 'Ещё')]")
        for i in tqdm(range(len(buttons))):
            try: 
                buttons[i].click()
            except:
                continue
        print('comms opened', j+1, ' iteration')

def get_soup(driver):
    selen_page = driver.page_source
    soup = BeautifulSoup(selen_page,'html.parser')

    posts = (soup.findAll('div', attrs = {'class':"kvgmc6g5 cxmmr5t8 oygrvhab hcukyx3x c1et5uql ii04i59q"}))
    b = soup.findAll('span', attrs = {'class':"j1lvzwm4 stjgntxs ni8dbmo4 q9uorilb gpro0wi8"})
    #текст постов и время размещения *не спрашивайте почему b*
    print('got soup')
    return soup, b, posts


def date_parse(b, posts):
    #Загрузка сегодняшней даты в нужном формате
    day_td = str(datetime.datetime.now().day)
    day_ytd = str((datetime.datetime.now() - datetime.timedelta (days = 1)).day)
    month_td = list(MonthDict.keys())[list(MonthDict.values()).index(datetime.datetime.now().month)]
    year_td = str(datetime.datetime.now().year)
    date_td = day_td+month_td+year_td
    date_ytd = day_ytd+month_td+year_td
    #Куча ИФ циклов для чистки и преобразования данных о времени
    parsed_ls = []
    #Первый цикл, для создания листа, где не будет лишних знаков
    for i in tqdm(range(len(b))):
        parsed = (b[i].text.replace('=','').replace('·','').replace('\xa0', ''))
        if b[i].text != '':
            parsed_ls.append(parsed)
    #Выкидывание ненужной информации и замена часов, минут и дней на дату
    for i in tqdm(range(len(parsed_ls))):
        parsed_ls[i] = parsed_ls[i].replace(' ', '')
        parsed_ls[i] = parsed_ls[i].replace('г.Москва', '')
        parsed_ls[i] = parsed_ls[i].replace('г.', '')
        parsed_ls[i] = parsed_ls[i].replace('Толькочто', 'ч.')
        if 'ч.' in parsed_ls[i]:
            parsed_ls[i] = date_td
        if 'мин.' in parsed_ls[i]:
            parsed_ls[i] = date_td
        if 'Вчера' in parsed_ls[i]:
            parsed_ls[i] = date_ytd
    #Перевод дат в формат д/м/г
    for i in tqdm(range(len(parsed_ls))):
        if (parsed_ls[i][-3]==':')==True:
            if '2020' in parsed_ls[i]:
                parsed_ls[i] = parsed_ls[i][:-6]
            else:
                parsed_ls[i]= parsed_ls[i][:-6]+year_td
        if parsed_ls[i][-4:-2]!='20':
            parsed_ls[i] = parsed_ls[i]+parsed_ls[i-1][-4:]
        if parsed_ls[i][:2].isnumeric():
            month = str(MonthDict[parsed_ls[i][2:-4]])
            date = (parsed_ls[i][:2])
            year = (parsed_ls[i][-4:])
            parsed_ls[i] = date+'/'+month+'/'+year
        if parsed_ls[i][:2].isnumeric()==False:
            month = str(MonthDict[parsed_ls[i][1:-4]])
            date = (parsed_ls[i][:1])
            year = (parsed_ls[i][-4:])
            parsed_ls[i] = date+'/'+month+'/'+year
    #перевод дат в формат datetime, перевод листа с хтмл кодом в текстовый лист
    dt_ls = []
    for date in (parsed_ls):
        dt_ls.append(datetime.datetime.strptime(date, '%d/%m/%Y'))
    for i in range(len(posts)):
        posts[i] = posts[i].text
    
    print('date parsed')    
    return dt_ls,posts


def df_creation(dt_ls, posts):
    Dict = {'time': dt_ls[:len(posts)], 'post': posts}
    df = pd.DataFrame.from_dict(Dict)
    return df


def read_comments(soup):
    new_cl = 'd2edcug0 hpfvmrgz qv66sw1b c1et5uql gk29lw5a a8c37x1j keod5gw0 nxhoafnm aigsh9s9 d9wwppkn fe6kdd0r mau55g9w c8b282yb hrzyx87i jq4qci2q a3bd9o3v knj5qynh m9osqain'
    #считывание комментариев и количества комментариев под каждым постом
    comms = (soup.findAll('div', attrs = {'class':"tw6a2znq sj5x9vvc d1544ag0 cxgpxx05"}))
    #comm_num_cl = 'd2edcug0 hpfvmrgz qv66sw1b c1et5uql gk29lw5a a8c37x1j keod5gw0 nxhoafnm aigsh9s9 d9wwppkn fe6kdd0r mau55g9w c8b282yb hrzyx87i jq4qci2q a3bd9o3v knj5qynh m9osqain'
    comm_num = (soup.findAll('span', attrs = {'class':new_cl}))
    return comms, comm_num

def number_of_comments(df, comm_num):
    #выкидываем кол-во репостов
    comm_num_ls = []
    for i in tqdm(range(len(comm_num))):
        if 'Комментарии' in comm_num[i].text:
            comm_num_ls.append(int(comm_num[i].text.replace('Комментарии: ', '')))
    #почти всегда получается так, что посты без комментов технически сьедают комменты,
    #а нижние посты остаются без них. Добьем лист нулями. Общая картина текстов не теряется
    if len(comm_num_ls)<len(df):
        for i in range(len(df)-len(comm_num_ls)):
            comm_num_ls.append(0)
    #Добавляем колонку с кол-вом комментов
    df['number_comments'] = np.array(comm_num_ls[:len(df)])

def clear_comms(comms, df):
    #токенизируем комменты, чтобы выкинуть мусор и убрать обращение по имени
    #шумовые слова выкидываю сразу, чтобы не засорять датасет
    tw = TweetTokenizer()
    det = TreebankWordDetokenizer()
    noise = stopwords.words('russian') + list(punctuation)
    upnoise = [letter.upper() for letter in noise]
    sum_noise = noise+upnoise+['.','»','«']
    for i in tqdm(range(len(comms))):
        comms[i] = comms[i].text
        tokenized_example = (tw.tokenize(comms[i])[2:])
        #filtered_example = [word for word in tokenized_example if not word in noise]
        comms[i] = det.detokenize(tokenized_example)
    #разбиваем и клеим так, чтобы они были одним предложением. А также чтобы у каждого поста нужное кол-во
    comms_ls = []
    j = 0 
    for i in tqdm(range(len(df))):
        comms_ls.append(comms[j:j+df.iloc[i,2]])
        j+=df.iloc[i,2]
    print('comments parsed')
    df['comment'] = comms_ls
    
def md5(df):
    md5 = []
    for post in df['post']:
        hash_object = hashlib.md5(post.encode())
        md5.append(hash_object.hexdigest())
    df['hash_summ'] = md5

def check_and_save(df, name, format = 'csv'):
    df.to_csv(name+'.'+format)
    print('saved')
