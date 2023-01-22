# Импорт библиотеки, отвечающей на вопросы
from transformers import AutoTokenizer, TFLayoutLMForQuestionAnswering
import tensorflow as tf


# Загрузка модели, которая даёт ответы на вопросы по документу (в прямом смысле)
tokenizer = AutoTokenizer.from_pretrained("impira/layoutlm-document-qa", add_prefix_space=True)
model = TFLayoutLMForQuestionAnswering.from_pretrained("impira/layoutlm-document-qa", revision="1e3ebac")


def oraqul_answer(boxes, words, question):
    '''
    Функция получения ответов на вопросы по документу. Входные параметры - боксы с координатами распознанных в документе
    слов (списком, только левый верхний и правый нижний углы), перечень слов документа (списком), а также вопрос текстом.
    Возвращает ответ на вопрос
    '''
    encoding = tokenizer(
        question.split(), words, is_split_into_words=True, return_token_type_ids=True, return_tensors="tf"
    )
    bbox = []
    for i, s, w in zip(encoding.input_ids[0], encoding.sequence_ids(0), encoding.word_ids(0)):
        if s == 1:
            bbox.append(boxes[w])
        elif i == tokenizer.sep_token_id:
            bbox.append([1000] * 4)
        else:
            bbox.append([0] * 4)
    encoding["bbox"] = tf.convert_to_tensor([bbox])

    word_ids = encoding.word_ids(0)
    outputs = model(**encoding)
    loss = outputs.loss
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    start, end = word_ids[tf.math.argmax(start_scores, -1)[0]], word_ids[tf.math.argmax(end_scores, -1)[0]]
    answer = words[start : end + 1][0:2] #" ".join(words[start : end + 1])
    answer = " ".join([x for x in answer if not x.isalpha()])
  
    return answer


#--------------------------------------------------------------------------------------
if __name__ == "__main__":

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


    print(oraqul_answer(boxes, words, question = "What is the total price?"))
    print(oraqul_answer(boxes, words, question = "What is the date?"))
    print(oraqul_answer(boxes, words, question = "What is the company?"))
    print(oraqul_answer(boxes, words, question = "What is the adress?"))
