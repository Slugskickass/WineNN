from torch.utils.tensorboard import SummaryWriter
import torch
import time
import numpy as np
class RunManager():
    def __init__(self):

        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None

        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None

        self.network = None
        self.loader = None
        self.tb = None

    def begin_run(self, run, network, loader):
        self.run_start_time = time.time()
        self.run_params = run
        self.run_count += 1
        self.network = network
        self.loader = loader
        self.tb = SummaryWriter(comment=f'-{run}')
        characteristics, labels = next(iter(self.loader))

    def end_run(self):
        self.tb.close()
        self.epoch_count = 0

    def begin_epoch(self):
        self.epoch_start_time = time.time()
        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0

    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_num_correct / len(self.loader.dataset)

        self.tb.add_scalar('Loss', loss, self.epoch_count)
        self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)

    def _get_num_correct(self, preds, labels):
        preds = preds.detach().numpy()
        preds = preds.astype(int)
        labels = labels.detach().numpy()
        labels = labels.astype(int)
        test_case = (labels - preds)
        unique, counts = np.unique(test_case, return_counts=True)
        outer = counts[np.where(unique == 0)[0]]/np.sum(counts)
        if outer.size == 0:
            outer = np.zeros(1)
        return outer[0]


    def track_loss(self, loss, batch):
        self.epoch_loss += loss.item() * batch[0].shape[0]

    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self._get_num_correct(preds, labels)

    def inform(self, discrete_n):
        if self.epoch_count % discrete_n == 0:
            print(self.epoch_count, ' ', self.run_count)
# class RunManager():
#     def __init__(self):
#
#         self.epoch_count = 0
#         self.epoch_loss = 0
#         self.epoch_num_correct = 0
#         self.epoch_start_time = None
#
#         self.run_params = None
#         self.run_count = 0
#         self.run_data = []
#         self.run_start_time = None
#
#         self.network = None
#         self.loader = None
#         self.tb = None
#
#     def begin_run(self, run, network, loader):
#         self.run_start_time = time.time()
#         self.run_params = run
#         self.run_count += 1
#         self.network = network
#         self.loader = loader
#         self.tb = SummaryWriter(comment=f'-{run}')
#         characteristics, labels = next(iter(self.loader))
#
#     def end_run(self):
#         self.tb.close()
#         self.epoch_count = 0
#
#     def begin_epoch(self):
#         self.epoch_start_time = time.time()
#         self.epoch_count += 1
#         self.epoch_loss = 0
#         self.epoch_num_correct = 0
#
#     def end_epoch(self):
#         epoch_duration = time.time() - self.epoch_start_time
#         run_duration = time.time() - self.run_start_time
#
#         loss = self.epoch_loss / len(self.loader.dataset)
#         #accuracy = self.epoch_num_correct / len(self.loader.dataset)
#         accuracy = self.epoch_num_correct
#
#         self.tb.add_scalar('Loss', loss, self.epoch_count)
#         self.tb.add_scalar('Accuracy', accuracy, self.epoch_count)
#
#
#     def _get_num_correct(self, preds, labels):
#         return preds.argmax(dim=1).eq(labels).sum().item()
#
#     def track_loss(self, loss, batch):
#         self.epoch_loss += loss.item() #* batch[0].shape[0]
#
#     def track_num_correct(self, preds, labels):
#         self.epoch_num_correct += self._get_num_correct(preds, labels) / len(labels)
#
#     def inform(self, discrete_n):
#         if self.epoch_count % discrete_n == 0:
#             print(self.epoch_count, ' ', self.run_count)
