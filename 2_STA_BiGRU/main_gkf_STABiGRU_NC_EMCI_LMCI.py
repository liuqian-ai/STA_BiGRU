import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.io import loadmat,savemat
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import torch.optim as optim
from attn import Self_Attention,Time_Attention
from regulation import l2_regularization, spatial_regularization, temporal_regularization
import time
import os
import warnings
from torch.utils.data import DataLoader

use_gpu = torch.cuda.is_available()
warnings.filterwarnings("ignore")

#####################模型###############
class classifier(nn.Module):
    def __init__(self, c_out, seq_len, hidden_size):
        super(classifier, self).__init__()
        self.fcn_1 = nn.Sequential(
            nn.Linear(seq_len * hidden_size * 2, hidden_size),
            nn.Dropout(0.5, inplace=True),
            nn.ReLU(inplace=True)
        )
        self.fcn_2 = nn.Linear(hidden_size, c_out)

    def forward(self, x):
        output = x.view(x.size(0), x.size(1) * x.size(2))
        output = self.fcn_1(output)
        output = self.fcn_2(output)
        output = F.softmax(output, dim=-1)
        return output

class Model(nn.Module):
    def __init__(self, n_layer, f_in, c_out, seq_len, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_norm = nn.LayerNorm(f_in, eps=1e-6)
        self.BiGRU = nn.GRU(f_in, hidden_size, n_layer, bidirectional=True)
        # attention
        self.spatial_attn = Self_Attention(f_in)
        self.time_attn = Time_Attention(f_in, seq_len)
        # classifier
        self.con_classifier = classifier(c_out, seq_len, hidden_size)

    def forward(self, x):
        enc_out = self.layer_norm(x)
        spatial_softmax, spatial_score = self.spatial_attn(enc_out)
        spatial_attention = torch.mul(spatial_score, enc_out)
        bigru_out, h_n = self.BiGRU(spatial_attention)
        time_score, time_attention = self.time_attn(enc_out,bigru_out)
        output = self.con_classifier(time_attention)
        return output, spatial_softmax, spatial_score, time_score,bigru_out

# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class GetLoader(torch.utils.data.Dataset):
	# 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    n_class = 3
    lr = 10e-5
    num_epochs = 100
    batch_size = 600
    hidden_size = 64
    lambda_loss_amount = 0.0015
    l1_spatial_lambda = 0.01
    l1_temporal_lambda = 1
    n_layer = 1  # 只使用一层BiGRU

    for kalman in range(9, 10):
        time_start = time.time()
        kalman = kalman/10
        kalman = str(kalman)
        if not os.path.exists('../results/NC_EMCI_LMCI/gkf_STA_BiGRU/' + kalman):
            os.mkdir('../results/NC_EMCI_LMCI/gkf_STA_BiGRU/' + kalman)
        out_file = open('../results/NC_EMCI_LMCI/gkf_STA_BiGRU/'+kalman+'/result.txt', 'a')

        # 载入数据文件
        Featurefile = '../0_A_data/NC_EMCI_LMCI/GLfeatures/kalmancorr_0.01_' + kalman + '_1121.mat'
        print(Featurefile)
        datas = loadmat(Featurefile)
        corr = datas['datas']
        sample_nums = len(corr)

        # 取数据矩阵
        X = np.array([np.array(corr[i][0], dtype=np.float32) for i in range(sample_nums)])
        _, n_step, featureNum = X.shape
        labels = np.array([0 if corr[i][3] == 'NC  ' else 1 if corr[i][3] == 'EMCI' else 2 for i in range(sample_nums)],dtype=np.int32)
        X = torch.Tensor(X)
        labels = torch.Tensor(labels)
        labels = labels.long()

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        test_kflodCount = 1
        cfms = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        aucs = []

        # 针对类别不平衡的样本，设置不同的误差惩罚权重，比例按样本数量反比设置
        weight_CE = torch.FloatTensor([1.1953, 1, 1.5149])
        if use_gpu:
            weight_CE = weight_CE.cuda()
        loss_func = torch.nn.CrossEntropyLoss(weight_CE)

        print('begin:', file=out_file)
        for train_idx, test_idx in kf.split(X, labels):
            # 划分80%训练集，20% 测试集
            X_train = X[train_idx]
            X_test = X[test_idx]
            Y_train = labels[train_idx]
            Y_test = labels[test_idx]
            print('--------------------------------')
            print('trainX.shape: ', X_train.shape)
            print('trainY.shape: ', Y_train.shape)
            print('testX.shape: ', X_test.shape)
            print('testY.shape: ', Y_test.shape)
            print('--------------------------------')

            train_dataset = GetLoader(X_train, Y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)

            model = Model(n_layer=n_layer, f_in=featureNum, c_out=n_class, seq_len=n_step, hidden_size=hidden_size)
            if use_gpu:
                model.cuda()
            optimizer = optim.Adam(model.parameters(), lr)

            print('test_kflodCount', test_kflodCount)

            trainloss, testloss, trainacc, testacc = [], [], [], []
            # 训练开始
            for epoch in range(num_epochs):
                print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                print('-' * 10)
                train_total_loss = 0
                train_total_acc = 0
                train_batch_num = len(train_loader)
                for batch_idx, (data, target) in enumerate(train_loader):
                    # 梯度初始化为0
                    optimizer.zero_grad()
                    data = data.float()
                    if use_gpu:
                        data = data.cuda()
                        target = target.cuda()
                    output,_,spatial_score, time_score,_= model(data)
                    train_loss = loss_func(output, target)
                    # 时空正则化
                    spatial_reg = spatial_regularization(spatial_score, l1_spatial_lambda,n_step)
                    temporal_reg = temporal_regularization(time_score, l1_temporal_lambda,n_step)
                    # l2-正则化,GRU和全连接层的权重均加入l2正则化
                    l2_loss = l2_regularization(model, lambda_loss_amount)

                    train_final_loss = train_loss + l2_loss + spatial_reg + temporal_reg
                    train_final_loss = train_final_loss.mean()
                    train_final_loss.backward()
                    optimizer.step()

                    train_acc = metrics.accuracy_score(target.cpu(), output.argmax(axis=1).cpu())
                    train_total_loss += train_final_loss.item()
                    train_total_acc += train_acc
                train_avg_loss = train_total_loss / train_batch_num
                train_avg_acc = train_total_acc / train_batch_num
                trainloss.append(train_avg_loss)
                trainacc.append(train_avg_acc)

                # 训练完成后，测试
                with torch.no_grad():
                    X_test = X_test.float()
                    if use_gpu:
                        X_test = X_test.cuda()
                        Y_test = Y_test.cuda()
                    test_output, test_spatial_softmax, test_spatial_score, test_time_score,test_bigru_out = model(X_test)
                    test_loss = loss_func(test_output, Y_test)  #也求一下测试损失，但是不参与训练
                    # 时空正则化损失
                    test_spatial_reg = spatial_regularization(test_spatial_score, l1_spatial_lambda,n_step)
                    test_temporal_reg = temporal_regularization(test_spatial_softmax, l1_temporal_lambda,n_step)
                    # l2-正则化损失
                    test_l2_loss = l2_regularization(model, lambda_loss_amount)
                    test_final_loss = test_loss + test_l2_loss + test_spatial_reg + test_temporal_reg
                    test_final_loss = test_final_loss.mean()

                    # 指标计算
                    cfm = confusion_matrix(Y_test.cpu(), test_output.argmax(axis=1).cpu()).ravel()
                    test_acc = metrics.accuracy_score(Y_test.cpu(), test_output.argmax(axis=1).cpu())
                    auc = metrics.roc_auc_score(torch.nn.functional.one_hot(Y_test, 3).cpu(), test_output.cpu(),
                                                    average="weighted")
                    testloss.append(test_final_loss.item())
                    testacc.append(test_acc)
                print("Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
                          % (train_avg_loss,
                             train_avg_acc, test_final_loss.item(), test_acc))
            # 保存注意力模型
            np.savez(
                "../results/NC_EMCI_LMCI/gkf_STA_BiGRU/"+kalman+"/spatial_softmax_out_" + str(
                    test_kflodCount),test_spatial_softmax.cpu())
            np.savez("../results/NC_EMCI_LMCI/gkf_STA_BiGRU/"+kalman+"/spatial_score_out_" + str(
                test_kflodCount), test_spatial_score.cpu())
            np.savez("../results/NC_EMCI_LMCI/gkf_STA_BiGRU/"+kalman+"/time_score_out_" + str(
                test_kflodCount),test_time_score.cpu())
            np.savez("../results/NC_EMCI_LMCI/gkf_STA_BiGRU/"+kalman+"/bi_gru_out_" + str(
                test_kflodCount),test_bigru_out.cpu())
            print("numpy save success!")
            # 输出结果
            Y_test, test_output = np.array(Y_test.cpu()), np.array(test_output.argmax(axis=1).cpu())
            savemat('../results/NC_EMCI_LMCI/gkf_STA_BiGRU/'+kalman+'/result'+'_' +str(test_kflodCount)+'.mat',{'l1_temporal_lambda': l1_temporal_lambda, 'l1_spatial_lambda': l1_spatial_lambda,
                                       'test_kflodCount': test_kflodCount, 'spatial_score_out': test_spatial_score.cpu(),
                                       'time_score_out': test_time_score.cpu(), 'bi_gru_out': test_bigru_out.cpu(),'labeltest': Y_test, 'predicttest': test_output})

            cfms = np.sum([cfms, cfm], axis=0)
            aucs.append(auc)

            # 画每一折的loss和acc曲线
            plt.clf()
            plt.plot(range(0, num_epochs), np.array(trainloss), 'g-', label=u'train_loss')
            # ‘’g‘’代表“green”,表示画出的曲线是绿色，“-”代表画的曲线是实线，可自行选择，label代表的是图例的名称，一般要在名称前面加一个u，如果名称是中文，会显示不出来，目前还不知道怎么解决。
            plt.plot(range(0, num_epochs), np.array(testloss), 'r-', label=u'val_loss')
            plt.legend()
            plt.xlabel(u'epochs')
            plt.ylabel(u'loss')
            plt.savefig('../results/NC_EMCI_LMCI/gkf_STA_BiGRU/'+kalman+'/loss_' + str(test_kflodCount) + '.jpg')
          #  plt.show()

            plt.clf()
            plt.plot(range(0, num_epochs), np.array(trainacc), 'g-', label=u'train_acc')
            # ‘’g‘’代表“green”,表示画出的曲线是绿色，“-”代表画的曲线是实线，可自行选择，label代表的是图例的名称，一般要在名称前面加一个u，如果名称是中文，会显示不出来，目前还不知道怎么解决。
            plt.plot(range(0, num_epochs), np.array(testacc), 'r-', label=u'val_acc')
            plt.legend()
            plt.xlabel(u'epochs')
            plt.ylabel(u'acc')
            plt.savefig('../results/NC_EMCI_LMCI/gkf_STA_BiGRU/'+kalman+'/acc_' + str(test_kflodCount) + '.jpg')
          #  plt.show()
            test_kflodCount += 1

        # 计算各类别的准确率
        acc_nc = cfms[0] / (cfms[0] + cfms[1] + cfms[2])
        acc_emci = cfms[4] / (cfms[3] + cfms[4] + cfms[5])
        acc_lmci = cfms[8] / (cfms[6] + cfms[7] + cfms[8])
        total_acc = (cfms[0] + cfms[4] + cfms[8]) / np.sum(cfms)

        # 结果输出
        print(
            '5-fold mean of Test accuracy, acc_nc, acc_emci, acc_lmci, auc\n{}% {}% {}% {}% {}'.format(
                total_acc * 100, acc_nc * 100, acc_emci * 100, acc_lmci * 100, np.mean(aucs)), file=out_file)
        print(
            '5-fold mean of Test accuracy, acc_nc, acc_emci, acc_lmci, auc\n{}% {}% {}% {}% {}'.format(
                total_acc * 100, acc_nc * 100, acc_emci * 100, acc_lmci * 100, np.mean(aucs)))

        print('cfms', cfms, file=out_file)
        print('cfms', cfms)

        # 统计程序运行时间
        time_end = time.time()
        time_c = time_end - time_start
        print('time cost', time_c, 's')
        print('time cost', time_c, 's', file=out_file)
        out_file.close()

