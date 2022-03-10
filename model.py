from re import S
from time import sleep
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl




class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim,dropout_rate=0.3,device='cuda:0'):
        super(GATLayer, self).__init__()
        self.g = g
        self.device = device
        self.dropout_rate = dropout_rate

        # self.w2d = nn.Linear(2 * in_dim,in_dim)
        # self.d2w = nn.Linear(2 * in_dim,in_dim)

        self.attn_fc_d2w = nn.Linear(2 * in_dim, 1)
        self.attn_fc_w2d = nn.Linear(2 * in_dim, 1)

        self.fc=nn.Linear(in_dim,out_dim)
        self.dropout=nn.Dropout(self.dropout_rate)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e' : F.leaky_relu(a)}
    def message_func(self, edges):
        return {'z' : edges.src['z'], 'e' : edges.data['e']}
    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h' : h}

    def edge_attention_type_1(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc_d2w(z2)
        return {'dew' : F.leaky_relu(a)}
    

    def edge_attention_type_2(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc_w2d(z2)
        return {'wed' : F.leaky_relu(a)}
 
    
    def message_func_d2w(self, edges):
        return {'z' : edges.src['z'], 'dew' : edges.data['dew']}
    
    def reduce_func_d2w(self, nodes):
        alpha = F.softmax(nodes.mailbox['dew'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h' : h}

    def message_func_w2d(self, edges):
        return {'z' : edges.src['z'], 'wed' : edges.data['wed']}
    
    def reduce_func_w2d(self, nodes):
        alpha = F.softmax(nodes.mailbox['wed'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h' : h}
 

    def forward(self,doc_hidden,word_hidden):
    
        self.g.nodes['doc'].data['z'] = doc_hidden
        self.g.nodes['word'].data['z'] = word_hidden
        
        self.g.apply_edges(self.edge_attention_type_1,etype='include')  # doc  -> word
        self.g.apply_edges(self.edge_attention_type_2,etype='included') # word -> doc
        self.g.update_all(self.message_func_d2w, self.reduce_func_d2w,etype='include')  # doc  -> word
        self.g.update_all(self.message_func_w2d, self.reduce_func_w2d,etype='included') # word -> doc


        self.g.nodes['doc'].data['h'] = F.softmax(self.dropout(self.fc(self.g.nodes['doc'].data['h'])),1)
        self.g.nodes['word'].data['h'] = F.softmax(self.dropout(self.fc(self.g.nodes['word'].data['h'])),1)

        #Torch Topk
        # self.g.nodes['doc'].data['h']=self.g.nodes['doc'].data['h'].masked_fill_(
        #         torch.argsort(
        #             self.g.nodes['doc'].data['h'],dim=1).gt(10),0)
        # self.g.nodes['word'].data['h']=self.g.nodes['word'].data['h'].masked_fill_(
        #         torch.argsort(
        #             self.g.nodes['word'].data['h'],dim=1).gt(10),0)

        return self.g.nodes['doc'].data['h'],self.g.nodes['word'].data['h']


class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='mean'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge
 
    def forward(self, doc_hidden,word_hidden):

        doc_head_outs=[]
        word_head_outs=[]
        for attn_head in self.heads :
            doc_head_out,word_head_out=attn_head(doc_hidden,word_hidden)
            
            doc_head_outs.append(doc_head_out)
            word_head_outs.append(word_head_out)
        
        #对多头的attention 结果进行fusion
        if self.merge == 'cat':
            # 对输出特征维度（第1维）做拼接
            return torch.cat(doc_head_outs, dim=1),torch.cat(word_head_outs, dim=1)
        else:
            # 用求平均整合多头结果
            return torch.mean(torch.stack(doc_head_outs,1),1),torch.mean(torch.stack(word_head_outs,1),1)

class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        # 注意输入的维度是 hidden_dim * num_heads 因为多头的结果都被拼接在了
        # 一起。 此外输出层只有一个头。
        # head cat
        # self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1) 
        #head mean
        self.layer2 = MultiHeadGATLayer(g, hidden_dim, out_dim, 1)
 
    def forward(self, doc_hidden,word_hidden):
        doc_hidden,word_hidden = self.layer1(doc_hidden,word_hidden)
        doc_hidden = F.elu(doc_hidden)
        word_hidden = F.elu(word_hidden)
        doc_out,word_out = self.layer2(doc_hidden,word_hidden)

        return doc_out,word_out


class GATON(nn.Module):
    
    def __init__(self, g,topic_num,vocabulary_size,init_wordEmbedding_size,graph_input_dim,graph_hidden_dim,nums_head,device='cuda:0'):
        super(GATON, self).__init__()
        
        self.g = g
        self.device=device
        self.topic_num = topic_num
        self.vocabulary_size = vocabulary_size
        self.graph_input_dim = graph_input_dim
        self.graph_hidden_dim = graph_hidden_dim
        self.nums_head = nums_head

        #先将word 和doc的表示投影到同一空间
        self.word_linear=nn.Linear(init_wordEmbedding_size,graph_input_dim)
        self.doc_linear=nn.Linear(vocabulary_size,graph_input_dim)

        self.gat =GAT(self.g,self.graph_input_dim,self.graph_hidden_dim,self.topic_num,self.nums_head)
    
    #用一个整图在训练
    def forward(self,doc_input,word_input):

        # 先将 word  & doc 的表示转换到同一空间下
        word_hidden=self.word_linear(word_input)
        doc_hidden=self.doc_linear(doc_input)
        

        doc_topic_dist,word_topic_dist=self.gat(doc_hidden,word_hidden)

        # doc_topic_dist=F.softmax(doc_topic_dist,1)
        # word_topic_dist=F.softmax(word_topic_dist,1)

        return doc_topic_dist,word_topic_dist  #返回卷积之后的feature


class MyLoss(nn.Module):

    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, output, target):
        mask = target.eq(0)
        output.masked_fill_(mask,0)
        loss=torch.sum((output.subtract(target))**2)
        # loss=torch.mean((output.subtract(target))**2)
        return loss