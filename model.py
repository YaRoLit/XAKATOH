# Загрузка библиотек
import spacy
from spacy.matcher import Matcher
import random
import json
from pathlib import Path
import pandas as pd
import re



# Загрузка собственными руками обученной модели из директории
model_dir = Path('./model/')
nlp = spacy.load(model_dir)


# Подгружаем базу с известными компаниями. В базе содержится уникальный идентификатор - номер налогоплательщика GST REG
# И привязанные к нему адреса и названия компании.
try:
    df = pd.read_csv('./data/company.csv', delimiter = ',', dtype='object')
except:
    print("Ошибка загрузки базы данных компаний!")
    df = pd.DataFrame(columns=['company', 'adress', 'GST_no'])

# Задаем паттерны поиска сущностей по шаблону
# Паттерн для поиска индивидуального налогового номера GST REG №__
matcher = Matcher(nlp.vocab)
pattern = [
    {"LOWER": 'gst'},
    {"OP": "?"},
    {"OP": "?"},
    {"OP": "?"},
    {"OP": "?"},
    {"OP": "?"},
    {"IS_DIGIT": True, "LENGTH": 12},
]
matcher.add("GST_REG_pattern", [pattern])

# Паттерн для поиска итоговой цены (Total) в чеке
matcher_total = Matcher(nlp.vocab)
pattern = [
    {"LOWER": 'total'},
    {"OP": "?"},
    {"LIKE_NUM": True}
]
matcher_total.add("total_pattern", [pattern])


# Определяем необходимые функции
def find_gst(doc):
    '''
    Функция осуществляет поиск индивидуального малазийского номера налогоплательщика в чеке
    на вход принимает строку с текстом чека, ищет по определенному выше паттерну 
    возвращает 12 значный номер gst налогоплательщика (если находит его), или None
    '''  
    gst=''
    matches = matcher(doc)
  
    for match_id, start, end in matches:
        matched_span = doc[start:end]
        gst = str(matched_span[-1])
    
        return gst


def find_total(doc):
    '''
    Функция осуществляет поиск суммы чека по шаблону
    на вход принимает строку с текстом чека
    возвращает итоговую сумму (если находит её), или None
    '''  
    gst=''
    matches = matcher_total(doc)
    old_total = 0
    
    for match_id, start, end in matches:
        matched_span = doc[start:end]
        gst = str(matched_span)
        gst = gst.split()
        total = gst[-1]
        
        if '.' not in total:
            continue
        try:
            total = float(total)
        except:
            continue

        if total > old_total:
            old_total = total       
    
    return old_total


def find_date(chk):
    '''
    Функция осуществляет поиск даты в чеке по шаблону (если сущность не распознана моделью)
    на вход принимает строку с текстом чека
    возвращает дату (если находит её), или None
    '''

    # TD!!! Дописать алгоритма перебор всех найденных по регулярке значений и проверку на соответствие правилам даты

    pattern = re.compile('(?:\d{1,2}[\/](?:\d{1,2}|[A-Za-z]{3})[\/]\d{2,4})|(?:\d{1,2}[ ](?:\d{1,2}|[A-Za-z]{3})[ ]\d{2,4})|(?:\d{1,2}[-](?:\d{1,2}|[A-Za-z]{3})[-]\d{2,4})|(?:\d{1,2}[.](?:\d{1,2}|[A-Za-z]{3})[.]\d{2,4})')

    result = pattern.findall(chk)

    if len(result):
        return result[0]


def find_entities(chk):
    '''
    Функция осуществляет поиск сущностей в чеке с помощью предобученной модели
    возвращает словарь, содержащий ключи {'company': '', 'date': '', 'address': '', 'total': ''}
    в который записаны найденные соответствующие сущности в тексте чека (если они найдены).
    '''  
    if not chk:
        raise('Передана пустая строка')
    
    doc = nlp(chk)

    gst = find_gst(doc)

    check = {'company': None, 'date': None, 'address': None, 'total': None}

    # Ищем в тексте чека именованные сущности с помощью модели
    for entity in doc.ents:
        if entity.label_ == 'COMPANY':
            check['company'] = entity.text

        if entity.label_ == 'DATE':
            check['date'] = entity.text

        if entity.label_ == 'ADRESS':
            check['address'] = entity.text

    # В случае неудачи нахождения какой-то из сущностей моделью пробуем найти её другими методами
    if (check['company']==None) and (len(df[df.GST_no==gst])!=0):
        check['company'] = df.company[df.GST_no==gst].to_list()[0]
  
    if (check['address']==None) and (len(df[df.GST_no==gst])!=0):
        check['address'] = df.adress[df.GST_no==gst].to_list()[0]

    if (check['date']==None):
        try:
            check['date'] = find_date(chk)
        except:
            check['date'] = ''
            
    # В случае успешного распознавания моделью новой компании, которой нет в базе, добавляем её туда
    if gst and len(df[df.GST_no==gst])==0:
        if ((check['company']!='') or (check['address']!='')):
            df.loc[len(df.index)]=[check['company'], check['address'], gst]

    check['total'] = find_total(doc)

    return check



#--------------------------------------------------------------------------------------
if __name__ == "__main__":

    chk = '31803053 GOLDEN KEY MAKER (000760274-K) Selangor TEL03-58919941 FAX03-58919941 E-Mail:goldenkeymaker@gmail.com Website:www.goldenkeymaker.com S/PHOE Inv No : CS-0014134 24-Mar-2018 03:44:22 PM INVOICE No.Desc/Code Qty Price Disc% Amount 1 NORMAL KEY 33100 4 4.00 0.00 16.00 2 NORMAL KEY LONG 33101 1 5.00 0.00 5.00 Total Qty : 5 Sub-Total : 21.00 Discount : 0.00 Net Total : 21.00 Cash Received : 50.00 Change : 29.00 GOODS SOLD ARE NOT RETURNABLE/REFUNDABLE '
    print(find_entities(chk))
