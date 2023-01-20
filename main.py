import os
import sys
from functools import reduce
import cv2
from paddleocr import PaddleOCR, draw_ocr
import json

import model # подключаемый наш модуль, в котором происходит распознавание именованных сущностей с помощью модели



def read_args():
    '''
    Читаем аргументы, с которыми запущен скрипт, для определения пути к исходным
    файлам и пути для вывода результатов
    '''
    try:
        output_path = sys.argv[2]
        
    except IndexError:
        output_path = os.getcwd()
        
    return sys.argv[1], output_path


def get_files(path):
    '''
    Функция определяет список файлов для обработки
    !!!!! потом нужно сюда дописать обход вложенных директорий
    '''

    for files in os.walk(path):
        return files[2]


def preprocessing_img(file):
    '''
    Функция осуществляет загрузку и предобработку изображения из файла
    возвращает уже обработанное изображение. (пока только цветовая перекодировка,
    но можно попробовать что-нибудь ещё потом)
    '''
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img
    

def read_checks(input_path, output_path, files):
    '''
    Функция распознает текстовку чеков с помощью библиотеки paddleocr,
    передает результаты распознавания в виде box функции записи в файл
    '''
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
    for file in files:
        filename = input_path + '/' + file
        
        try:
            img = preprocessing_img(filename)
        except:
            print('Проблема с загрузкой файла изображения')
            continue

        result = ocr.ocr(img, cls=True)
        
        box_name = output_path + '/box/' + file.split('.')[0] + '.txt'
        chk = save_box(box_name, result)

        checks_file = output_path + '/result.txt'
        check_str = str(file.split('.')[0]) + '\t\t\t' + chk + '\n'
        save_checks(checks_file, check_str)

        json_name = output_path + '/entities/' + file.split('.')[0] + '.txt'
        save_json(json_name, model.find_entities(chk))
                

def save_checks(filename, chk):
    '''
    Функция сохраняет на диск файл с текстовкой чеков
    принимает на вход имя файла для записи результатов, текстовые строки
    '''
    with open(filename, 'a') as f:
        f.write(chk)


def save_json(filename, ent):
    '''
    Функция сохраняет на диск файл с выявленными именованными сущностями в виде json
    принимает на вход имя файла для записи результатов, словарь с результатами
    '''
    with open(filename, 'w', encoding='utf8') as f:
        json.dump(ent, f, indent=4, ensure_ascii=False)

    
def save_box(filename, result):
    '''
    Функция сохраняет на диск файл с координатами выявленных текстовых блоков
    принимает на вход имя файла для записи результатов, массив строк с результатами
    пишет это в файл, а также перегоняет всю текстовку из чека в одну строку и возвращает её
    '''
    check = ''

    with open(filename, 'w') as f:
        for idx in range(len(result)):
            res = result[idx] # Это распознанные текстовые блоки с их координатами
            for line in res:
                f.write('%s\n' % line)
                check += str(line[-1][0]) + ' '

    return check


        
if __name__ == '__main__':
    
    try:
        input_path, output_path = read_args()
        
    except IndexError:
        input_path = os.getcwd() + '/input_images'
        output_path = os.getcwd() + '/results'

    file_list = get_files(input_path)
    read_checks(input_path, output_path, file_list)
