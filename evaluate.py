import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score



def coherence(words_list1, words_list2, doc_size, word_occurrence, metric='npmi'):
    topic_num = len(words_list1)
    assert len(words_list1)==len(words_list2)

    mean_coherence_list = []
    for i in range(topic_num):
        words1 = words_list1[i]
        words2 = words_list2[i]
        sum_score = 0.0
        sum_count = 0
        for n in range(len(words1)):
            set_n = word_occurrence[words1[n]]
            p_n = len(set_n) / doc_size
            for l in range(len(words2)):
                if words1[n] == words2[l]:
                    continue
                set_l =  word_occurrence[words2[l]]
                p_l = len(set_l)
                p_nl = len(set_n & set_l)
                if p_n * p_l * p_nl > 0:
                    p_l = p_l / doc_size
                    p_nl = p_nl / doc_size
                    pmi = np.log(p_nl / (p_l * p_n))
                    if metric == 'npmi':
                        sum_score += pmi / -np.log(p_nl)
                    else:
                        sum_score += pmi
                else:
                    sum_score += -1.0
                sum_count += 1
        mean_coherence_list.append(sum_score/sum_count)

    return mean_coherence_list

def coherence_gaton(words_list1, words_list2, word_occurrence):
    topic_num = len(words_list1)
    assert len(words_list1)==len(words_list2)

    mean_coherence_list = []
    for i in range(topic_num):
        words1 = words_list1[i]
        words2 = words_list2[i]
        sum_score = 0.0
        for n in range(len(words1)):
            set_n = word_occurrence[words1[n]]
            for l in range(len(words2)):
                if words1[n] == words2[l]:
                    continue
                set_l =  word_occurrence[words2[l]]

                #拿到两个topic owrd 共同出现doc 数
                to_set=set_n.intersection(set_l)
                sum_score +=np.log((len(to_set)+1) / len(set_n))

               
        mean_coherence_list.append(sum_score)

    return mean_coherence_list

def computeCoherence(topic_word_dis, corpus,tgt_keys, idx2word, topk, verbose=False):
    def getWordOccurrence(corpus, idx2word):
        word_occurrence = defaultdict(set) #该 word 在 corpus 的出现次数
        for i in range(len(corpus)):
            doc = corpus[i][0]
            for token in doc:
                word_occurrence[idx2word[token]].add(i)
        return word_occurrence
    word_occurrence = getWordOccurrence(corpus, idx2word)

    
    topic_bow =topic_word_dis

    topic_freq_idxs = np.argsort(topic_bow, 1)[:, ::-1][:, :topk]
    topics = [[idx2word[idx] for idx in topic_freq_idxs[i][:topk]] for i in range(topic_freq_idxs.shape[0])]

    #计算GATON 所给出的topic coherence


    tc_gaton=coherence_gaton(topics, topics, word_occurrence)

    npmi_list = coherence(topics, topics, len(corpus), word_occurrence)
    if verbose:
        print("==========================================================================")
        print("Average Topic Coherence = %.3f" % np.mean(npmi_list))
        print("Average  gaton Topic Coherence = %.3f" % np.mean(tc_gaton))
        
    for i in range(len(topics)):
        print("==========================================================================")
        print(npmi_list[i])
        print("主题先验===================================================================")
        print(tgt_keys[i])
        print(topics[i])
        print("模型结果====================================================================")
    
    return np.mean(npmi_list)




def computeClassificationMetric(doc_topic_dis,label,verbose=False):
    f1_macro_dev = f1_score(label, doc_topic_dis, average='macro')
    f1_micro_dev = f1_score(label, doc_topic_dis, average='micro')
    precision = precision_score(label,doc_topic_dis,average='micro')
    accuracy = accuracy_score(label,doc_topic_dis)
    recall = recall_score(label,doc_topic_dis,average='micro')
    if verbose:
      print("==========================================================================")
      print('f1 macro:', f1_macro_dev, 'f1 micro:', f1_micro_dev ,'precision:',precision,'accuracy:',accuracy,'recall',recall)
