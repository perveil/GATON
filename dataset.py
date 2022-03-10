#导入相关模块
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import pickle  as pkl
import dgl
from collections import defaultdict



class DocData(object):
    def __init__(self, datapath,device="cuda:0"): 

        ins_tr, _, vocab,tgt_keys, pretrained_emb = pkl.load(open(datapath, 'rb'))
        # ins_tr：训练集，list，每个元素是一个(doc, label)的tuple，doc是单词id的列表，label是一个数
        # ins_ts：测试集，格式同上
        # vocab：词汇表，列表，每个元素是一个单词
        # tgt_keys：类别关键词，列表，每个元素是一个类别的关键词列表
        

        self.ins_tr= ins_tr
        self.idx2word={}
        self.word2id={}
        self.vocab=vocab
        self.tgt_keys=tgt_keys
        
        for idx,word in enumerate(vocab):
            self.idx2word[idx]=word
            self.word2id[word]=idx


        #建立一个二部图
        doc_idx = []
        word_idx = []
        for idx,(doc,label) in enumerate(ins_tr):
            for word in doc:
                doc_idx.append(idx)
                word_idx.append(word)

        bipartite_graph_data = {
            ('doc', 'include', 'word'): (torch.from_numpy(np.array(doc_idx)), torch.from_numpy(np.array(word_idx))),
            ('word', 'included', 'doc'): (torch.from_numpy(np.array(word_idx)),(torch.from_numpy(np.array(doc_idx))))
        }
        graph= dgl.heterograph(bipartite_graph_data)
        graph=graph.to(torch.device('cuda:0'))
        self.graph=graph

        self.word_embedding_size=300
        self.vocabulary_size=len(vocab)
        #读取预定义的word-embedding
        self.word_embeddings=[]

        for word in vocab:
            if word not in pretrained_emb.keys() or  type(pretrained_emb[word])!=np.ndarray:
                word_embedding=list(np.random.rand(1,self.word_embedding_size))[0]
            else:
                word_embedding=list(pretrained_emb[word])
            self.word_embeddings.append(word_embedding)
        self.word_embeddings=torch.from_numpy(np.array(self.word_embeddings,dtype=np.float32)).to('cuda:0')

        #计算doc 的表示
        self.doc_word_frequency=[]
        self.class_dict = defaultdict(int)
        self.labels=[]
        for doc,label in ins_tr:
            self.doc_word_frequency.append(self.doc_preprocessing(self.idx2vec(doc)))
            # self.doc_word_frequency.append(self.idx2vec(doc))
            self.labels.append(label)
            self.class_dict[label]+=1

        self.doc_word_frequency=torch.from_numpy(np.array(self.doc_word_frequency,dtype=np.float32)).to('cuda:0')

        

    def idx2vec(self, token_idxs):
        vec = np.zeros(self.vocabulary_size).astype('float32')
        for idx in token_idxs:
            vec[idx] += 1.0
        return vec
    def doc_preprocessing(self, bow): #对词袋表征进行预处理
        max_row = np.log(1 + np.max(bow))
        return   np.log(1 + bow) / max_row


