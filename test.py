# Проверка работоспособности модуля обученной модели
# В функцию model.find_entities(check) подается текстовое содержание чека
# Функция возвращает распознанные именованные сущности

import model


with open('./results/result.txt', "r", encoding="utf8") as read_file:
    test_txt = read_file.readlines()

    for check in test_txt:
        if check != '\n':
            try:
                check = check.split('\t\t\t')
                print(str(check[0] + '.txt'))
                print(model.find_entities(str(check[1])))
            except:
                print('FAIL!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                print(check)
