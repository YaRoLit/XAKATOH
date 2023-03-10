# XAKATOH
Хакатон в УрФУ 2023

**Команда Cobra Cai. Состав:**
1. Бакулин Семен
2. Прасолова Евгения
3. Макушев Данил
4. Тряпицын Денис
5. Литаврин Ярослав
6. Охотников Павел
7. * и еще какой-то парень

## Внимание! Текущая версия скрипта 2.0. 22.01.2023 были внесены изменения: исправлена выгрузка в файл box'ов с координатами текстовых блоков (теперь записывается без скобок, в точном соответствии с заданием). Добавлена ещё одна модель для распознавания именованных сущностей.

### Для запуска скрипта необходимо поместить файлы со сканами чеков в формате .jpg в папку /input_images и запустить скрипт main.py. Результаты обработки этих сканов в виде текстовых файлов будут помещены в папку /results в подпапки /box (с распознанными текстовыми блоками и их координатами на изображении чека) и /entities (с распознанными именованными сущностями - 'company', 'adress', 'date', 'total'). Работоспособность скрипта проверена на нескольких машинах.


Скрипт состоит из двух модулей. Main.py является основным, осуществляет загрузку файлов с изображениями чеков из папки /input_images, их обработку и распознание текста с помощью библиотеки PaddleOCR, а также запись файлов с результатами на диск. Model.py является вспомогательным, в нём содержатся функции работы с предобученной нами моделью для распознавания именованных сущностей в текстовом блоке (модель находится в папке /model), а также функции поиска текста по шаблону и другие. Модуль model.py целиком посвящен поиску именованных сущностей в текстовом массиве. Модуль test.py создан для проверки работоспособности model.py.

Модель обучена нами "с нуля" с использованием размеченных данных. Подготовленные тренировочные датасеты и скрипты для их создания, а также для создания и обучения модели в этом гитхабе не размещены, но при необходимости могут быть добавлены.  
