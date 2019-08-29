"""
对SequenceToSequence模型进行基本的参数组合测试
"""

import sys
import random
import json
import pickle
import os
import numpy as np
import tensorflow as tf
# import jieba
# from nltk.tokenize import word_tokenize
from langconv import *


def nlp(question):
    """测试不同参数在生成的假数据上的运行结果"""
    params = json.load(open(os.path.dirname(__file__) + '/params.json'))
    from sequence_to_sequence import SequenceToSequence
    from data_utils import batch_flow

    x_data, _ = pickle.load(open(os.path.dirname(__file__)+'/data/chatbot.pkl', 'rb'))
    ws = pickle.load(open(os.path.dirname(__file__)+'/data/ws.pkl', 'rb'))

# for x in x_data[:5]:
# print(' '.join(x))

    config = tf.ConfigProto(
        device_count={'CPU': 1, 'GPU': 0},
        allow_soft_placement=True,
        log_device_placement=False
    )

    # save_path = '/tmp/s2ss_chatbot.ckpt'
    save_path = os.path.dirname(__file__)+'/data/s2ss_chatbot.ckpt'

    # 测试部分
    tf.reset_default_graph()
    model_pred = SequenceToSequence(
        input_vocab_size=len(ws),
        target_vocab_size=len(ws),
        batch_size=1,
        mode='decode',
        beam_width=0,
        **params
    )
    init = tf.global_variables_initializer()

    with tf.Session(config=config) as sess:
        sess.run(init)
        model_pred.load(sess, save_path)

        while True:
            # user_text = input('Input Chat Sentence:')
            question = Converter('zh-hans').convert(question)  # 繁體轉簡體

            if question in ('exit', 'quit'):
                exit(0)
            x_test = [list(question.lower())]
            # x_test = [word_tokenize(user_text)]
            bar = batch_flow([x_test], ws, 1)
            x, xl = next(bar)
            x = np.flip(x, axis=1)

            pred = model_pred.predict(
                sess,
                np.array(x),
                np.array(xl)
            )
            # print(ws.inverse_transform(x[0]))   -->['</s>', '饭', '吃', '去', '要', '我']

            for p in pred:
                ans = ws.inverse_transform(p)
                str1 = ''.join(ans)  # list轉str
                line = str1.strip("</s>")
                line = Converter('zh-hant').convert(line)  # 簡體轉繁體
            return line


#print(nlp("下雨冷網速慢無趣"))   # [TEST RESULT]感覺很恐怖啊,那就不起來了


"""def main():
    # 入口程序
    import json
    import os
    nlp(json.load(open(os.path.dirname(__file__)+'/params.json')))


if __name__ == '__main__':
    main()"""
