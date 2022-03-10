import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
#DGL
import dgl
from dgl import DGLGraph

from dataset import DocData
from model import GATON,MyLoss
from evaluate import computeCoherence,computeClassificationMetric
#超参

from apex import amp

# 设置seed

seed=2022
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

topic_num=20
graph_input_dim=64
graph_hidden_dim=128
device="cuda:0"
head_num=2
epochs=100

data=DocData("/home/v-ruiruiwang/notebooks/code/GATON/data/20NG_mindf_97_vocab_2004_pretrain.pkl")


model = GATON(
    data.graph,
    topic_num,
    data.vocabulary_size,
    data.word_embedding_size,
    graph_input_dim,
    graph_hidden_dim,
    head_num
    ).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=2e-3,weight_decay=0.0005)

loss_fn=MyLoss()

model.train()

for epoch in range(epochs):

    doc_input = data.doc_word_frequency.to(device) # 初始的 doc 表示 
    word_input = data.word_embeddings.to(device)    #初始的 word 表示

    doc_topic_prob,word_topic_prob = model(doc_input,word_input)

    #计算loss
    doc_word_occ = torch.matmul(doc_topic_prob,word_topic_prob.permute(1,0))

    recon_loss = loss_fn(doc_word_occ,doc_input)

    optimizer.zero_grad()
    recon_loss.backward()
    optimizer.step()

    print("==========================================================================")
    print('cur reconstruct loss:',recon_loss.item())

    model.eval()
    with torch.no_grad():
        word_topic_dis = word_topic_prob.cpu().numpy().transpose()
        doc_topic_dis = doc_topic_prob.cpu().argmax(dim=1).numpy()
    #验证分类
    computeClassificationMetric(doc_topic_dis,np.array(data.labels),True)
    #验证Topic Coherence
    computeCoherence(word_topic_dis,data.ins_tr,data.tgt_keys,data.idx2word,10,True)




