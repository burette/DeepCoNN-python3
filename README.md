# DeepCoNN

This is our implementation for the paper:


*Lei Zheng, Vahid Noroozi, and Philip S Yu. 2017. Joint deep modeling of users and items using reviews for recommendation.
In WSDM. ACM, 425-434.*


Two models:

1、DeepCoNN: This is the state-of-the-art method that uti-lizes deep learning technology to jointly model user and itemfrom textual reviews.

2、DeepCoNN++: We extend DeepCoNN by changing its share layer from FM to our neural prediction layer.


The two methods are used as the baselines of our method **NARRE** in the paper:


*Chong Chen, Min Zhang, Yiqun Liu, and Shaoping Ma. 2018. [Neural Attentional Rating Regression with Review-level Explanations.](http://www.thuir.cn/group/~YQLiu/publications/WWW2018_CC.pdf) 
In WWW'18.*

**Please cite our WWW'18 paper if you use our codes. Thanks!**

```
@inproceedings{chen2018neural,
  title={Neural Attentional Rating Regression with Review-level Explanations},
  author={Chen, Chong and Zhang, Min and Liu, Yiqun and Ma, Shaoping},
  booktitle={Proceedings of the 2018 World Wide Web Conference on World Wide Web},
  pages={1583--1592},
  year={2018},
}
```

Author: Chong Chen (cstchenc@163.com)

## Environments

- python 2.7
- Tensorflow (version: 0.12.1)
- numpy
- pandas


## Dataset

In our experiments, we use the datasets from  Amazon 5-core(http://jmcauley.ucsd.edu/data/amazon) and Yelp Challenge 2017(https://www.yelp.com/dataset_challenge).

## Example to run the codes		

Data preprocessing:

```
python loaddata.py	
python data_pro.py
```

Train and evaluate the model:

```
python train.py
```



Last Update Date: Jan 3, 2018

-------------分割线-------------------

感谢原repo，之前跑过，最近又跑了一遍，python3下问题记录

-------------分割线-------------------
## Environments
- python 3.5
- Tensorflow 1.13.1

## 更改说明
1.print()加括号问题

2.itervalue()改为value()

3.在train.py中，xrange()改为range()

4.load_data和data_pro中has_key()问题，改为in

5.load_data中报错""AttributeError: Can't pickle local object 'numerize.<locals>.<lambda>’"，在58行numberize()函数中的map改为list(map())

6.data_pro.py和train.py中"AttributeError: _parse_flags：""，注释掉""# FLAGS._parse_flags()"即可

7.data_pro中报错""UnicodeDecodeError: 'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte："，在data_pro中改为
f1 = open(user_review,'rb')
f2 = open(item_review,'rb')

8.DeepCoNN中，有三处concat()函数，将tf.concat(3,pooled_outputs_u)改为tf.concat(pooled_outputs_u, 3),其余的也是一样

9.train.py中加载预训练模型读取词向量：
原来的

                with open(FLAGS.word2vec, "rb") as f:
                    header = f.readline()
                    vocab_size, layer1_size = map(int, header.split())
                    binary_len = np.dtype('float32').itemsize * layer1_size
                    for line in xrange(vocab_size):
                        word = []
                        while True:
                            ch = f.read(1)
                            if ch == ' ':
                                word = ''.join(word)
                                break
                            if ch != '\n':
                                word.append(ch)
                        idx = 0

                        if word in vocabulary_user:
                            u = u + 1
                            idx = vocabulary_user[word]
                            initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                        else:
                            f.read(binary_len)

改为

                with open(FLAGS.word2vec, "rb") as f:
                    header = f.readline()
                    vocab_size, layer1_size = map(int, header.split())
                    binary_len = np.dtype('float32').itemsize * layer1_size
                    for line in tqdm(range(vocab_size)):
                        word = b''
                        while True:
                            ch = f.read(1)
                            if ch == b' ':
                                break
                            if ch != '\n':
                                word += (ch)
                        idx = 0

                        if word in vocabulary_user:
                            u = u + 1
                            idx = vocabulary_user[word]
                            initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                        else:
                            f.read(binary_len)