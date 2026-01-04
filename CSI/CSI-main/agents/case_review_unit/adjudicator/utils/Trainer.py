import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import *
from utils.metrics import *
import copy
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
class Trainer():
    def __init__(self, model, device, lr, dataloaders, save_param_path, writer, early_stop, epoches, model_name,
                 save_predict_result_path, scheduler_option=False, save_threshold=0.8, start_epoch=0):
        self.model = model
        self.device = device
        self.model_name = model_name
        self.dataloaders = dataloaders
        self.start_epoch = start_epoch
        self.num_epochs = epoches
        self.early_stop = early_stop
        self.save_threshold = save_threshold
        self.writer = writer
        self.scheduler_option = scheduler_option
        ensure_dir(save_param_path)
        self.save_param_path = save_param_path
        ensure_dir(save_predict_result_path)
        self.save_predict_result_path = save_predict_result_path
        self.lr = lr
        self.CEloss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-5)
        if scheduler_option:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=1,
                min_lr=1e-6,
                verbose=True)
        self.best_test_acc = 0.0
    def train(self):
        since = time.time()
        self.model.cuda()
        best_model_wts_test = copy.deepcopy(self.model.state_dict())
        best_f1_test = 0.0
        best_epoch_test = 0
        is_earlystop = False
        self.best_test_acc = 0.0
        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
            if is_earlystop:
                break
            print('-' * 50)
            print('Epoch {}/{}'.format(epoch, self.start_epoch + self.num_epochs - 1))
            print('-' * 50)
            self.model.train()
            print('-' * 10)
            print('TRAIN')
            print('-' * 10)
            running_loss = 0.0
            tpred = []
            tlabel = []
            for batch in tqdm(self.dataloaders['train']):
                self.optimizer.zero_grad()
                batch_data = batch
                for k, v in batch_data.items():
                    if k != 'vid':
                        if isinstance(v, torch.Tensor):
                            batch_data[k] = v.cuda()
                        else:
                            batch_data[k] = v
                labels = batch_data['label']
                output = self.model(**batch_data)
                loss = self.CEloss(output, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * labels.size(0)
                tpred.extend(torch.max(output, 1)[1].tolist())
                tlabel.extend(labels.tolist())
            epoch_loss = running_loss / len(self.dataloaders['train'].dataset)
            print('Train Loss: {:.4f} '.format(epoch_loss))
            results = metrics(tlabel, tpred)
            print(results)
            self.writer.add_scalar('Loss/train', epoch_loss, epoch)
            self.writer.add_scalar('Acc/train', results['acc'], epoch)
            self.writer.add_scalar('F1/train', results['f1'], epoch)
            self.model.eval()
            print('-' * 10)
            print('VAL')
            print('-' * 10)
            val_loss = 0.0
            val_tpred = []
            val_tlabel = []
            for batch in tqdm(self.dataloaders['val']):
                batch_data = batch
                for k, v in batch_data.items():
                    if k != 'vid':
                        if isinstance(v, torch.Tensor):
                            batch_data[k] = v.cuda()
                        else:
                            batch_data[k] = v
                labels = batch_data['label']
                with torch.no_grad():
                    output = self.model(**batch_data)
                    loss = self.CEloss(output, labels)
                val_loss += loss.item() * labels.size(0)
                val_tpred.extend(torch.max(output, 1)[1].tolist())
                val_tlabel.extend(labels.tolist())
            epoch_loss_val = val_loss / len(self.dataloaders['val'].dataset)
            print('Val Loss: {:.4f} '.format(epoch_loss_val))
            results_val = metrics(val_tlabel, val_tpred)
            print(results_val)
            if self.scheduler_option:
                self.scheduler.step(epoch_loss_val)
            if results_val['f1'] > best_f1_test:
                best_f1_test = results_val['f1']
                best_epoch_test = epoch
                best_model_wts_test = copy.deepcopy(self.model.state_dict())
                if best_f1_test > self.save_threshold:
                    save_path = os.path.join(self.save_param_path,
                        f"{self.model_name}_val_{best_epoch_test}_{best_f1_test:.4f}.pth")
                    torch.save(best_model_wts_test, save_path)
                    print("saved " + save_path)
            else:
                if epoch - best_epoch_test >= self.early_stop - 1:
                    is_earlystop = True
                    print("early stop at epoch " + str(epoch))
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print("Best model on val: epoch " + str(best_epoch_test) + "_" + str(best_f1_test))
        ckp_path = os.path.join(self.save_param_path,
            f"{self.model_name}_val_{best_epoch_test}_{best_f1_test:.4f}.pth")
        return ckp_path
    def test(self, ckp_path):
        try:
            self.model.load_state_dict(torch.load(ckp_path))
        except FileNotFoundError as e:
            print(f"Error: Checkpoint not found at {ckp_path}.")
            raise e
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise e
        since = time.time()
        self.model.cuda()
        self.model.eval()
        pred = []
        label = []
        vid = []
        for batch in tqdm(self.dataloaders['test']):
            with torch.no_grad():
                batch_data = batch
                for k, v in batch_data.items():
                    if k != 'vid':
                        if isinstance(v, torch.Tensor):
                            batch_data[k] = v.cuda()
                        else:
                            batch_data[k] = v
                labels = batch_data['label']
                output = self.model(**batch_data)
                label.extend(labels.tolist())
                pred.extend(torch.max(output, 1)[1].tolist())
                vid.extend(batch_data['vid'])
        time_elapsed = time.time() - since
        print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        result = pd.DataFrame({'vid': vid, 'label': label, 'pred': pred})
        result.to_csv(os.path.join(self.save_predict_result_path, f"{self.model_name}.csv"), sep='\t', index=False)
        results = metrics(label, pred)
        print(results)
        return results
class Inferencer():
    def __init__(self, model, device, model_name, dataset, dataloader, save_predict_result_path):
        self.model = model
        self.device = device
        self.model_name = model_name
        self.dataset = dataset
        self.dataloader = dataloader
        ensure_dir(save_predict_result_path)
        self.save_predict_result_path = save_predict_result_path
    def inference(self, ckp_path):
        try:
            self.model.load_state_dict(torch.load(ckp_path), strict=False)
        except FileNotFoundError as e:
            print(f"Error: Checkpoint not found at {ckp_path}.")
            raise e
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise e
        since = time.time()
        self.model.cuda()
        self.model.eval()
        label = []
        vid = []
        pred = []
        for batch in tqdm(self.dataloader):
            with torch.no_grad():
                batch_data = batch
                for k, v in batch_data.items():
                    if k != 'vid':
                        if isinstance(v, torch.Tensor):
                            batch_data[k] = v.cuda()
                        else:
                            batch_data[k] = v
                labels = batch_data['label']
                output = self.model(**batch_data)
                probs = torch.softmax(output, dim=1)[:, 1]
                label.extend(labels.tolist())
                pred.extend(probs.cpu().numpy().tolist())
                vid.extend(batch_data['vid'])
        time_elapsed = time.time() - since
        print('Inference complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        result = pd.DataFrame({'vid': vid, 'label': label, 'pred': pred})
        result.to_csv(os.path.join(self.save_predict_result_path, f"{self.model_name}.csv"), sep='\t', index=False)
        results = metrics(label, pred)
        print(results)
        return results