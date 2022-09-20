# coding: UTF-8
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time

class Config(object):

    """配置参数"""
    def __init__(self):
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 15                         # 类别数
        self.class_list = range(15)                         # 类别数
        self.num_epochs = 100                                             # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.learning_rate = 1e-2                                       # 学习率
        self.hidden_size = 300
        self.save_path = './data/torch_model/hash_sup'

class Model(nn.Module):

     def __init__(self, config):
         super(Model, self).__init__()
         #self.fc = nn.Linear(config.hidden_size, config.num_classes)
         self.fc1 = nn.Linear(config.hidden_size, 384)
         self.fc2 = nn.Linear(384, config.num_classes)

     def forward(self, x):
         #print (pooled.size()) #torch.Size([128, 768])
         out1 = F.relu(self.fc1(x))
         out = F.relu(self.fc2(out1))
         return out

#def train1(B,label):
def train1(B,label,B_test,label_test,B_dev,label_dev):
    #label = torch.eye(20)[label,:]
    config = Config()
    model = Model(config)
    model.train()
    label = torch.LongTensor([_ for _ in label])
    B = B.T
    B = torch.FloatTensor(B)
    B.requires_grad = True

    label_test = torch.LongTensor([_ for _ in label_test])
    B_test = B_test.T
    B_test = torch.FloatTensor(B_test)

    label_dev = torch.LongTensor([_ for _ in label_dev])
    B_dev = B_dev.T
    B_dev = torch.FloatTensor(B_dev)
    #print(label.shape)
    #print(label)
    #print(B.shape)
    dev_best_loss = float('inf')
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    for epoch in range(config.num_epochs):
        #fc1 = nn.Linear(B.shape[1],hidden_size)
        #fc2 = nn.Linear(hidden_size,class_num)
        #out1 = F.relu(fc1(B))
        #out = F.relu(fc2(out1))
        output = model(B)
        model.zero_grad()
        #print(str("output") + "\t" + str(output))
        #print( str("label") + "\t" + str(label))
        loss = F.cross_entropy(output, label)
        loss.backward()
        #print (B.grad)
        #B = B - B.grad
        optimizer.step()
        true = label.data.cpu()
        predic = torch.max(output.data, 1)[1].cpu()
        #print( str("predic") + "\t" + str(predic))
        train_acc = metrics.accuracy_score(true, predic)
        dev_acc, dev_loss = evaluate(config, model, B_dev, label_dev)
        if dev_loss < dev_best_loss:
            dev_best_loss = dev_loss
            torch.save(model.state_dict(),config.save_path)
            improve = '*'
        else:
            improve = ''
        #model.train()
        msg = 'Train Loss: {0:>5.2},  Train Acc: {1:>6.2%}, Val Loss: {2:>5.2},  Val Acc: {3:>6.2%} {4}'
        print(msg.format(loss.item(), train_acc, dev_loss, dev_acc, improve))
        #print(msg.format(loss.item(), train_acc, dev_loss, dev_acc, improve), file=sys.stderr)
    #test(config, model, B_test, label_test)
    return B.grad, dev_best_loss 

def test(config, model, B_test, label_test):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, B_test, label_test, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print(msg.format(test_loss, test_acc), file=sys.stderr)
    print("Precision, Recall and F1-Score...")
    print("Precision, Recall and F1-Score...", file=sys.stderr)
    print(test_report)
    print(test_report, file=sys.stderr)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, texts, labels, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        outputs = model(texts)
        loss = F.cross_entropy(outputs, labels)
        loss_total += loss
        labels = labels.data.cpu().numpy()
        predic = torch.max(outputs.data, 1)[1].cpu().numpy()
        labels_all = np.append(labels_all, labels)
        predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(labels), report, confusion
    return acc, loss_total / len(labels)
