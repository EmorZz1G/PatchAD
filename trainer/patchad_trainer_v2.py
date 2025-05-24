import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from patchad_model.models import PatchMLPAD
from data.data_loader2 import get_loader_segment
from einops import rearrange,repeat
from metrics.combine_all_scores import combine_all_evaluation_scores
import warnings
warnings.filterwarnings('ignore')
from tkinter import _flatten

from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
from sklearn.metrics import accuracy_score

def my_best_f1(score, label):
    best_f1 = (0,0,0)
    best_thre = 0
    best_pred = None
    # score = minmax_scale(score)
    for q in np.arange(0.01, 0.901, 0.01):
    # for q in np.arange(0.01, 0.901, 0.01):
        thre = np.quantile(score, 1-q)
        pred = score > thre
        pred = pred.astype(int)
        label = label.astype(int)
        p,r,f1,_ = precision_recall_fscore_support(label, pred, average='binary')
        print(f'q: {q}, p: {p}, r: {r}, f1: {f1}')
        if f1 > best_f1[2]:
            best_f1 = (p, r, f1)
            best_thre = thre
            best_pred = pred

    return (best_f1, best_thre, best_pred)

def my_kl_loss(p, q):
    # B N D
    res = p * (torch.log(p + 0.0000001) - torch.log(q + 0.0000001))
    # B N
    return torch.sum(res, dim=-1)


def inter_intra_dist(p,q,w_de=True,train=1,temp=1):
    # B N D
    if train:
        if w_de:
            p_loss = torch.mean(my_kl_loss(p,q.detach()*temp)) + torch.mean(my_kl_loss(q.detach(),p*temp))
            q_loss = torch.mean(my_kl_loss(p.detach(),q*temp)) + torch.mean(my_kl_loss(q,p.detach()*temp))
        else:
            p_loss = -torch.mean(my_kl_loss(p,q.detach())) 
            q_loss = -torch.mean(my_kl_loss(q,p.detach())) 
    else:
        if w_de:
            p_loss = my_kl_loss(p,q.detach()) + my_kl_loss(q.detach(),p)
            q_loss = my_kl_loss(p.detach(),q) + my_kl_loss(q,p.detach())

        else:
            p_loss = -(my_kl_loss(p,q.detach())) 
            q_loss = -(my_kl_loss(q,p.detach())) 

    return p_loss,q_loss


def normalize_tensor(tensor):
    # tensor: B N D
    sum_tensor = torch.sum(tensor,dim=-1,keepdim=True)
    normalized_tensor = tensor / sum_tensor
    return normalized_tensor

def anomaly_score(patch_num_dist_list,patch_size_dist_list, win_size, train=1, temp=1, w_de=True):
    for i in range(len(patch_num_dist_list)):
        patch_num_dist = patch_num_dist_list[i]
        patch_size_dist = patch_size_dist_list[i]



        # patch_num_dist = repeat(patch_num_dist,'b n d -> b (n rp) d',rp=win_size//patch_num_dist.shape[1])
        # patch_size_dist = repeat(patch_size_dist,'b p d -> b (rp p) d',rp=win_size//patch_size_dist.shape[1])

        patch_num_dist = repeat(patch_num_dist,'b n d -> b (rp n) d',rp=win_size//patch_num_dist.shape[1])
        patch_size_dist = repeat(patch_size_dist,'b p d -> b (p rp) d',rp=win_size//patch_size_dist.shape[1])

        patch_num_dist = normalize_tensor(patch_num_dist)
        patch_size_dist = normalize_tensor(patch_size_dist)

        patch_num_loss,patch_size_loss = inter_intra_dist(patch_num_dist,patch_size_dist,w_de,train=train,temp=temp)

        if i==0:
            patch_num_loss_all = patch_num_loss
            patch_size_loss_all = patch_size_loss
        else:
            patch_num_loss_all += patch_num_loss
            patch_size_loss_all += patch_size_loss

    return patch_num_loss_all,patch_size_loss_all



def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = val_loss
        score2 = val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        print('Save model')
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2

        
class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        datapth = ''+self.data_path
        self.dataset = self.data_name
        self.patch_size = self.patch_sizes
        self.num_epochs = self.epochs
        self.model_save_path = os.path.join(self.model_save_path, self.data_name)
        os.makedirs(self.model_save_path, exist_ok=True)
        self.res_pth = os.path.join(self.res_pth, self.data_name)
        os.makedirs(self.res_pth, exist_ok=True)
        self.lr = self.learning_rate
        datapth = os.path.join(datapth, self.dataset)

        self.train_loader = get_loader_segment(self.index, datapth, batch_size=self.batch_size, win_size=self.win_size, mode='train', dataset=self.dataset, step=self.stride)
        self.vali_loader = get_loader_segment(self.index, datapth, batch_size=self.batch_size, win_size=self.win_size, mode='val', dataset=self.dataset, step=self.stride)
        self.test_loader = get_loader_segment(self.index, datapth, batch_size=self.batch_size, win_size=self.win_size, mode='test', dataset=self.dataset, step=self.stride)
        self.thre_loader = get_loader_segment(self.index, datapth, batch_size=self.batch_size, win_size=self.win_size, mode='thre', dataset=self.dataset, step=self.stride)

        self.build_model()
        
        

    def build_model(self):
        self.model = PatchMLPAD(win_size=self.win_size, e_layer=self.e_layer, patch_sizes=self.patch_size, dropout=0.1, activation="relu", output_attention=True,
                                    channel=self.input_c,d_model=self.d_model,cont_model=self.win_size,norm='n')
        
        if torch.cuda.is_available():
            self.model = self.model.to(self.device)
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        
    @torch.no_grad()
    def vali(self, vali_loader):
        self.model.eval()
        loss_1 = []
        loss_2 = []
        win_size=self.win_size
        loss_mse = nn.MSELoss()
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            patch_num_dist_list,patch_size_dist_list,patch_num_mx_list,patch_size_mx_list,recx = self.model(input)

            patch_num_loss, patch_size_loss = anomaly_score(patch_num_dist_list,patch_size_dist_list,win_size=win_size,train=1)
            patch_num_loss = patch_num_loss / len(patch_num_dist_list)
            patch_size_loss = patch_size_loss / len(patch_num_dist_list)

            # loss3 = patch_size_loss + patch_num_loss
                
            p_loss = patch_size_loss 
            q_loss = patch_num_loss

            loss_1.append((p_loss).item())
            loss_2.append((q_loss).item())

        return np.average(loss_1), np.average(loss_2)


    def train(self):

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        win_size = self.win_size
        early_stopping = EarlyStopping(patience=5, verbose=True, dataset_name=self.data_name)
        train_steps = len(self.train_loader)

        mse_loss = nn.MSELoss()

        for epoch in range(self.num_epochs):
            iter_count = 0

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)

                # input = input + torch.rand_like(input)*0.2


                patch_num_dist_list,patch_size_dist_list,patch_num_mx_list,patch_size_mx_list,recx = self.model(input)

                loss = 0.

                cont_loss1,cont_loss2 = anomaly_score(patch_num_dist_list,patch_size_mx_list,win_size=win_size,train=1,temp=1)
                cont_loss_1 = cont_loss1 - cont_loss2
                loss -= self.patch_mx *cont_loss_1

                cont_loss12,cont_loss22 = anomaly_score(patch_num_mx_list,patch_size_dist_list,win_size=win_size,train=1,temp=1)
                cont_loss_2 = cont_loss12 - cont_loss22
                loss -= self.patch_mx *cont_loss_2

                patch_num_loss, patch_size_loss = anomaly_score(patch_num_dist_list,patch_size_dist_list,win_size=win_size,train=1,temp=1)
                patch_num_loss = patch_num_loss / len(patch_num_dist_list)
                patch_size_loss = patch_size_loss / len(patch_num_dist_list)

                loss3 = patch_num_loss - patch_size_loss

                
                loss -= loss3 * (1-self.patch_mx)

                loss_mse = mse_loss(recx, input)
                # print(loss_mse)
                # loss += loss_mse

                loss.backward()
                self.optimizer.step()


                # self.analysis(from_file=0)
                # self.model.train()

                if (i + 1) % 20 == 0:
                    print(f'MSE {loss_mse.item()} Loss {loss.item()}')
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                    epo_left = speed * (len(self.train_loader))
                    print('Epoch time left: {:.4f}s'.format(epo_left))
                    # self.test(from_file=0)
                    # self.model.train()

            vali_loss1, vali_loss2 = self.vali(self.test_loader)
            print('Vali',vali_loss1, vali_loss2)

            print(
                "Epoch: {0}, Cost time: {1:.3f}s ".format(
                    epoch + 1, time.time() - epoch_time))
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)
            self.test(from_file=0)
            

    
    @torch.no_grad()
    def test(self,from_file=1):
        if from_file:
            print('load model from file')
            self.model.load_state_dict(
                torch.load(
                    os.path.join(str(self.model_save_path), str(self.data_name) + '_checkpoint.pth')))
        self.model.eval()
        temperature = 1 #+ (self.patch_mx*10)
        win_size = self.win_size
        use_project_score = 0
        # (1) stastic on the train set
        attens_energy = []

        cont_beta = self.cont_beta

        mse_loss = nn.MSELoss(reduction='none')
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            patch_num_dist_list,patch_size_dist_list,patch_num_mx_list,patch_size_mx_list,recx = self.model(input)

            if use_project_score:
                patch_num_loss, patch_size_loss = anomaly_score(patch_num_mx_list,patch_size_mx_list,win_size=win_size,train=0,temp=temperature)
            else:
                patch_num_loss, patch_size_loss = anomaly_score(patch_num_dist_list,patch_size_dist_list,win_size=win_size,train=0)
            patch_num_loss = patch_num_loss / len(patch_num_dist_list)
            patch_size_loss = patch_size_loss / len(patch_num_dist_list)

            loss3 = patch_size_loss - patch_num_loss

            mse_loss_ = mse_loss(recx,input)

            metric1 = torch.softmax((-patch_num_loss), dim=-1)
            # metric1 = -patch_num_loss
            metric2 = mse_loss_.mean(-1)
            # print(metric1.shape,metric2.shape)
            metric = metric1 * (cont_beta) + metric2 * (1-cont_beta)

            # metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            # metric = torch.softmax((-patch_num_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        print(self.thre_loader.__len__())
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            patch_num_dist_list,patch_size_dist_list,patch_num_mx_list,patch_size_mx_list,recx = self.model(input)

            if use_project_score:
                patch_num_loss, patch_size_loss = anomaly_score(patch_num_mx_list,patch_size_mx_list,win_size=win_size,train=0,temp=temperature)
            else:
                patch_num_loss, patch_size_loss = anomaly_score(patch_num_dist_list,patch_size_dist_list,win_size=win_size,train=0)
            patch_num_loss = patch_num_loss / len(patch_num_dist_list)
            patch_size_loss = patch_size_loss / len(patch_num_dist_list)

            loss3 = patch_size_loss - patch_num_loss

            mse_loss_ = mse_loss(recx,input)
            metric1 = torch.softmax((-patch_num_loss), dim=-1)
            # metric1 = -patch_num_loss
            metric2 = mse_loss_.mean(-1)
            metric = metric1 * (cont_beta) + metric2 * (1-cont_beta)

            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []

        test_data = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            patch_num_dist_list,patch_size_dist_list,patch_num_mx_list,patch_size_mx_list,recx = self.model(input)

            if use_project_score:
                patch_num_loss, patch_size_loss = anomaly_score(patch_num_mx_list,patch_size_mx_list,win_size=win_size,train=0,temp=temperature)
            else:
                patch_num_loss, patch_size_loss = anomaly_score(patch_num_dist_list,patch_size_dist_list,win_size=win_size,train=0)
            patch_num_loss = patch_num_loss / len(patch_num_dist_list)
            patch_size_loss = patch_size_loss / len(patch_num_dist_list)

            loss3 = patch_size_loss - patch_num_loss

            # metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            # metric = torch.softmax((-patch_num_loss ), dim=-1)
            mse_loss_ = mse_loss(recx,input)
            metric1 = torch.softmax((-patch_num_loss), dim=-1)
            # metric1 = -patch_num_loss
            metric2 = mse_loss_.mean(-1)
            metric = metric1 * (cont_beta) + metric2 * (1-cont_beta)

            cri = metric.detach().cpu().numpy()

            attens_energy.append(cri)
            test_labels.append(labels)
            test_data.append(input_data.cpu().numpy().reshape(-1,input_data.shape[-1]))
            
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)
        test_data = np.concatenate(test_data,axis=0)


        pred = (test_energy > thresh).astype(int)

        
        gt = test_labels.astype(int)

        # precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')

        pres,recs,thresholds = precision_recall_curve(gt, test_energy)
        f1 = 2 * (pres * recs) / (pres + recs)
        best_thre = thresholds[np.argmax(f1)]
        precision = pres[np.argmax(f1)]
        recall = recs[np.argmax(f1)]
        f_score = f1[np.argmax(f1)]
        print("WOPA Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(precision, recall, f_score))


        # print("WOPA Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(precision, recall, f_score))

        # res, thrd, pred2 = my_best_f1(test_energy,gt)
        # pred = (test_energy > best_thre).astype(int)

        if self.mode == 'test':
            logf = r'/share/home/202220143416/project/SimAD/PatchAD/logs'
            import pandas as pd

            matrix = [self.index]
            scores_simple = combine_all_evaluation_scores(pred, gt, test_energy, self.full_res)
            print('===========FULL DATA EVALUATION START===========')
            for key, value in scores_simple.items():
                matrix.append(value)
                print('{0:21} : {1:0.4f}'.format(key, value))
            print('===========FULL DATA EVALUATION END===========')

            matrix = [self.index]
            scores_simple = combine_all_evaluation_scores(pred, gt, test_energy, self.full_res)
            print('===========FULL DATA EVALUATION START===========')
            for key, value in scores_simple.items():
                matrix.append(value)
                print('{0:21} : {1:0.4f}'.format(key, value))
            print('===========FULL DATA EVALUATION END===========')

        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0:
                        break
                    else:
                        if pred[j] == 0:
                            pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)

        

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision, recall, f_score))
        
        return test_energy, gt, pred, thresh, test_data 
    

    @torch.no_grad()
    def analysis(self,from_file=1):
        if from_file:
            print('load model from file')
            self.model.load_state_dict(
                torch.load(
                    os.path.join(str(self.model_save_path), str(self.data_name) + '_checkpoint.pth')))
        self.model.eval()
        temperature = 1 #+ (self.patch_mx*10)
        win_size = self.win_size
        use_project_score = 0
        # (1) stastic on the train set
        attens_energy = []

        cont_beta = self.cont_beta

        mse_loss = nn.MSELoss(reduction='none')
        
        def entropy(probabilities):
            # 确保概率之和为1
            probabilities = probabilities / (probabilities.sum() + 1e-7)
            # 计算信息熵
            entropy_ = -torch.sum(probabilities * torch.log2(probabilities),dim=[1,2,3])
            return entropy_

        
        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []

        test_data = []
        W1 = []
        W2 = []

        Ws = {'Normal':{
            'w1':[],
            'w2':[]
        },'Anomaly':{
            'w1':[],
            'w2':[]
        }}

        eW1 = []
        eW2 = []

        eWs = {'Normal':{
            'w1':[],
            'w2':[]
        },'Anomaly':{
            'w1':[],
            'w2':[]
        }}
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            patch_num_dist_list,patch_size_dist_list,patch_num_mx_list,patch_size_mx_list,recx = self.model(input,del_inter=0,del_intra=0)
            T1,T2 = self.model.T1,self.model.T2


            patch_num_dist_list1,patch_size_dist_list1,patch_num_mx_list1,patch_size_mx_list1,recx_inter = self.model(input,del_inter=0,del_intra=1)
            patch_num_dist_list2,patch_size_dist_list2,patch_num_mx_list2,patch_size_mx_list2,recx_intra = self.model(input,del_inter=1,del_intra=0)


            weight = torch.stack([recx_inter,recx_intra],dim=0)
            weight = F.softmax(weight,dim=0)
            weight1 = torch.mean(weight[0],dim=-1)
            weight2 = torch.mean(weight[1],dim=-1)
            # weight3 = torch.mean(weight[2],dim=-1)

            weight1 = weight1.mean(dim=-1)
            weight2 = weight2.mean(dim=-1)
            # weight3 = weight3.mean()

            # print('========= Rec ============')
            
            # print(weight1,weight2,weight3)
            # print(weight2/weight1)
            # 信息熵
            T1e = torch.sigmoid(T1)
            T2e = torch.sigmoid(T2)
            T1e = entropy(T1e)
            T2e = entropy(T2e)
            weight_E = torch.stack([T1e,T2e], dim=0)
            weight_E = torch.softmax(weight_E, dim=0)
            weight_T1e = weight_E[0]
            weight_T2e = weight_E[1]

            eW1.append(weight_T1e)
            eW2.append(weight_T2e)

            
            T1c = T1.mean(dim=2)
            T2c = T2.mean(dim=2)
            weight_T = torch.stack([T1c,T2c],dim=0)
            weight_T = F.softmax(weight_T,dim=0)
            weight_T1 = torch.mean(weight_T[0],dim=2)
            weight_T2 = torch.mean(weight_T[1],dim=2)

            weight_T1 = weight_T1.mean(dim=[-1])
            weight_T2 = weight_T2.mean(dim=[-1])
            # print(weight_T1.shape)
            # print('========= Logi ============')
            # print(weight_T1,weight_T2)
            # print(weight_T2/weight_T1)

            # weight_T1 = weight1
            # weight_T2 = weight2
            
            W1.append(weight_T1)
            W2.append(weight_T2)

            T_label = torch.mean(labels.float(),dim=1).to(weight1.device).reshape(-1)
            # B

            # 只计算标签为1的
            for b in range(T_label.shape[0]):
                if T_label[b] > 0:
                    anom_w1 = weight_T1[b] * T_label[b]
                    anom_w2 = weight_T2[b] * T_label[b]
                    Ws['Anomaly']['w1'].append(anom_w1)
                    Ws['Anomaly']['w2'].append(anom_w2)

                    anom_w1 = weight_T1e[b] * T_label[b]
                    anom_w2 = weight_T2e[b] * T_label[b]
                    eWs['Anomaly']['w1'].append(anom_w1)
                    eWs['Anomaly']['w2'].append(anom_w2)


                else:
                    norm_w1 = weight_T1[b]
                    norm_w2 = weight_T2[b]
                    Ws['Normal']['w1'].append(norm_w1)
                    Ws['Normal']['w2'].append(norm_w2)

                    norm_w1 = weight_T1e[b]
                    norm_w2 = weight_T2e[b]
                    eWs['Normal']['w1'].append(norm_w1)
                    eWs['Normal']['w2'].append(norm_w2)


            

            if use_project_score:
                patch_num_loss, patch_size_loss = anomaly_score(patch_num_mx_list,patch_size_mx_list,win_size=win_size,train=0,temp=temperature)
            else:
                patch_num_loss, patch_size_loss = anomaly_score(patch_num_dist_list,patch_size_dist_list,win_size=win_size,train=0)
            patch_num_loss = patch_num_loss / len(patch_num_dist_list)
            patch_size_loss = patch_size_loss / len(patch_num_dist_list)

            loss3 = patch_size_loss - patch_num_loss

            # metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            # metric = torch.softmax((-patch_num_loss ), dim=-1)
            mse_loss_ = mse_loss(recx,input)
            metric1 = torch.softmax((-patch_num_loss), dim=-1)
            # metric1 = -patch_num_loss
            metric2 = mse_loss_.mean(-1)
            metric = metric1 * (cont_beta) + metric2 * (1-cont_beta)

            cri = metric.detach().cpu().numpy()

            attens_energy.append(cri)
            test_labels.append(labels)
            test_data.append(input_data.cpu().numpy().reshape(-1,input_data.shape[-1]))
            
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)
        test_data = np.concatenate(test_data,axis=0)


        W1 = torch.mean(torch.concat(W1))
        W2 = torch.mean(torch.concat(W2))

        Ws['Anomaly']['w1'] = torch.mean(torch.tensor(Ws['Anomaly']['w1']))
        Ws['Anomaly']['w2'] = torch.mean(torch.tensor(Ws['Anomaly']['w2']))
        Ws['Normal']['w1'] = torch.mean(torch.tensor(Ws['Normal']['w1']))
        Ws['Normal']['w2'] = torch.mean(torch.tensor(Ws['Normal']['w2']))

        eW1 = torch.mean(torch.concat(eW1))
        eW2 = torch.mean(torch.concat(eW2))

        eWs['Anomaly']['w1'] = torch.mean(torch.tensor(eWs['Anomaly']['w1']))
        eWs['Anomaly']['w2'] = torch.mean(torch.tensor(eWs['Anomaly']['w2']))
        eWs['Normal']['w1'] = torch.mean(torch.tensor(eWs['Normal']['w1']))
        eWs['Normal']['w2'] = torch.mean(torch.tensor(eWs['Normal']['w2']))
                                        
        print(W1,W2)
        total_W_r = W2/(W1+W2)
        e_total_W_r = eW2/(eW1+eW2)
        
        anom_r = Ws['Anomaly']['w2'] / Ws['Anomaly']['w1']
        norm_r = Ws['Normal']['w2'] / Ws['Normal']['w1']

        e_anom_r = eWs['Anomaly']['w2'] / eWs['Anomaly']['w1']
        e_norm_r = eWs['Normal']['w2'] / eWs['Normal']['w1']

        

        print('Anomaly',anom_r, e_anom_r)
        print('Normal',norm_r, e_norm_r)
        print('W2/W1 - R',(anom_r+norm_r), (e_anom_r + e_norm_r))

        w2_r = Ws['Normal']['w2'] / Ws['Anomaly']['w2']
        w1_r = Ws['Normal']['w1'] / Ws['Anomaly']['w1']

        ew2_r = eWs['Normal']['w2'] / eWs['Anomaly']['w2']
        ew1_r = eWs['Normal']['w1'] / eWs['Anomaly']['w1']
        print('W2',w2_r, ew2_r)
        print('W1',w1_r, ew1_r)
        print('W12',w2_r + w1_r, ew2_r + ew1_r)

        w_r2 = (Ws['Normal']['w1'] + Ws['Normal']['w2']) - (Ws['Anomaly']['w1'] + Ws['Anomaly']['w2'])
        ew_r2 = (eWs['Normal']['w1'] + eWs['Normal']['w2']) - (eWs['Anomaly']['w1'] + eWs['Anomaly']['w2'])
        print('NA - W2',w_r2, ew_r2)

        w_r = (Ws['Normal']['w2']) - (Ws['Anomaly']['w2'])
        ew_r = (eWs['Normal']['w2']) - (eWs['Anomaly']['w2'])
        print('NA - W',w_r, ew_r) #++


        print('----- Analysis -----')
        import pandas as pd
        logf = r'/share/home/202220143416/project/SimAD/PatchAD/logs'

        datas = {
            'total_W_r':total_W_r.item(),
            'Anomaly_W_r':anom_r.item(),
            'Norm_W_r':norm_r.item(),
            'W2/W1 - R':(anom_r+norm_r).item(),
            'W2':w2_r.item(),
            'W1':w1_r.item(),
            'W12':(w2_r + w1_r).item(),
            'NA - W2':w_r2.item(),
            'NA - W':w_r.item(),

            'E_total_W_r':e_total_W_r.item(),
            'E_Anomaly_W_r':e_anom_r.item(),
            'E_Norm_W_r':e_norm_r.item(),
            'E_W2/W1 - R':(e_anom_r+e_norm_r).item(),
            'E_W2':ew2_r.item(),
            'E_W1':ew1_r.item(),
            'E_W12':(ew2_r + ew1_r).item(),
            'E_NA - W2':ew_r2.item(),
            'E_NA - W':ew_r.item()
        }
        if not hasattr(self,'cnt'):
            self.cnt = 0

        df = pd.DataFrame(datas,index=[self.cnt],columns=datas.keys())
        self.cnt += 1
        df.to_csv(os.path.join(logf,f'{self.dataset}_E9_analysis001.csv'),mode='a',header=False)
        print('----- Analysis -----')
        # return 



