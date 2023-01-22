# Загрузка библиотек
import spacy
from spacy.matcher import Matcher
import random
import json
from pathlib import Path
import pandas as pd
import re
import oraqul



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


def find_entities(chk, boxes, words):
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

    try:
        check['total'] = oraqul.oraqul_answer(boxes, words, question = "What is the total price?")
    except:
        print('fail total oraqul answer')

    if (check['total']==None):
        check['total'] = find_total(doc)

    # Последняя попытка добиться результата - обращение к оракулу
    try:
        if (check['company']==None):
            check['company'] = oraqul.oraqul_answer(boxes, words, question = "What is the company name?")
    except:
        print('fail company oraqul answer')

    try:
        if (check['address']==None):
            check['address'] = oraqul.oraqul_answer(boxes, words, question = "What is the adress?")
    except:
        print('fail adress oraqul answer')

    try:
        if (check['date']==None):
            check['date'] = oraqul.oraqul_answer(boxes, words, question = "What is the date?")
    except:
        print('fail date oraqul answer')
    
    return check



#--------------------------------------------------------------------------------------
if __name__ == "__main__":

    chk = '31803053 GOLDEN KEY MAKER (000760274-K) Selangor TEL03-58919941 FAX03-58919941 E-Mail:goldenkeymaker@gmail.com Website:www.goldenkeymaker.com S/PHOE Inv No : CS-0014134 24-Mar-2018 03:44:22 PM INVOICE No.Desc/Code Qty Price Disc% Amount 1 NORMAL KEY 33100 4 4.00 0.00 16.00 2 NORMAL KEY LONG 33101 1 5.00 0.00 5.00 Total Qty : 5 Sub-Total : 21.00 Discount : 0.00 Net Total : 21.00 Cash Received : 50.00 Change : 29.00 GOODS SOLD ARE NOT RETURNABLE/REFUNDABLE '

    words = ['TAN WOON YANN', 'MR D.T.Y. (JOHOR) SDN BHD', '(CO.REG : 933109-X)',
            'LOT 1851-A & 1851-B JALAN KPB 6', 'KAWASAN PERINDUSTRIAN BALAKONG', '43300 SERI KEMBANGAN SELANGOR',
            '(MR DIY TESCO TERBAU)', '-INVOICE-', 'CHOPPING BOARD 35.5X25.5CM 803M#', 'EZ10HD05 - 24',
            '8970669', '1', 'X', '19.00', '19.00', 'AIR PRESSURE SPRAYER SX-575-1 1.5L',
            'HC03-7 - 15', '9066468', '1', 'X', '8.02', '8.02', 'WAXCO WINDSHILED CLEANER 120ML',
            'WA14-3A - 48', '9557031100236', '1', 'X', '3.02', '3.02', 'BOPP TAPE 48MM*100M CLEAR',
            'FZ-04 - 36', '6935818350846', '1', 'X', '3.88', '3.88', 'ITEM(S) : 4', 'QTY(S) : 4',
            'TOTAL', 'RM 33.92', 'ROUNDING ADJUSTMENT', '-RM 0.02', 'TOTAL ROUNDED', 'RM 33.90',
            'CASH', 'RM 50.00', 'CHANGE', 'RM 16.10', '12-01-19 21:13 SH01 ZK09', 'T4 R000027830',
            'OPERATOR TRAINEE CASHIER', 'EXCHANGE ARE ALLOWED WITHIN', '7 DAYS WITH RECEIPT.',
            'STRICTLY NO CASH REFUND.']

    boxes = [[119, 47, 367, 80], [93, 161, 352, 183], [137, 185, 320, 204], [49, 201, 391, 227],
            [60, 228, 382, 246], [70, 249, 384, 267], [116, 268, 331, 289], [176, 290, 268, 308],
            [14, 328, 358, 351], [16, 349, 158, 372], [20, 371, 95, 393], [249, 373, 259, 393],
            [266, 376, 282, 393], [290, 374, 343, 395], [358, 375, 412, 396], [15, 394, 371, 417],
            [17, 413, 136, 436], [14, 436, 93, 456], [245, 438, 258, 458], [265, 437, 282, 460],
            [296, 439, 343, 459], [364, 436, 413, 460], [17, 457, 332, 479], [15, 477, 146, 497],
            [18, 501, 159, 521], [245, 498, 259, 522], [267, 502, 283, 518], [294, 502, 344, 523],
            [368, 502, 412, 523], [16, 521, 278, 539], [21, 542, 124, 559], [19, 563, 155, 583],
            [245, 561, 256, 583], [265, 562, 282, 586], [296, 561, 340, 589], [367, 563, 424, 586],
            [21, 604, 132, 624], [329, 607, 431, 628], [17, 643, 79, 665], [347, 650, 432, 668],
            [16, 667, 216, 688], [345, 667, 436, 689], [16, 688, 155, 711], [347, 688, 431, 712],
            [16, 710, 62, 728], [343, 711, 435, 729], [15, 730, 85, 754], [349, 733, 432, 753],
            [22, 773, 269, 791], [297, 772, 431, 794], [18, 794, 269, 812], [82, 835, 361, 855],
            [122, 858, 325, 876], [101, 877, 347, 899]]

    print(find_entities(chk, boxes, words))
