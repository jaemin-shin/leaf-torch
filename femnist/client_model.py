#import tensorflow as tf
import torch
from torch import nn
from model import Model
import numpy as np
from baseline_constants import ACCURACY_KEY, LOSS_KEY, INPUT_SIZE
from utils.model_utils import build_model
class ClientModel(Model):
    def __init__(self,seed,dataset,model_name,lr,num_classes):
        self.seed=seed
        self.model_name=model_name
        self.num_classes=num_classes
        self.dataset=dataset
        super(ClientModel,self).__init__(seed,lr)

    def create_model(self):
        model=build_model(self.dataset,self.model_name,self.num_classes)
        loss=nn.CrossEntropyLoss()
        optimizer=torch.optim.SGD(model.parameters(),lr=self.lr)
        return model,loss,optimizer

    def test(self, data, device):
        x_vecs = self.preprocess_x(data['x'].numpy()).to(device)
        labels = self.preprocess_y(data['y'].numpy()).type(torch.LongTensor).to(device)

        self.model.eval()

        output = self.model(x_vecs)
        b=output.argmax(axis=1)
        index=0
        acc=0
        for item in b:
            if item==labels[index]:
                acc+=1
            index+=1
        acc=acc/index
        loss = self.losses(output, labels).detach().cpu().numpy()
        return {ACCURACY_KEY: acc, LOSS_KEY: loss}

    def preprocess_x(self, raw_x_batch):
        return torch.from_numpy(raw_x_batch.reshape((-1, *INPUT_SIZE)))

    def preprocess_y(self, raw_y_batch):
        return torch.from_numpy(raw_y_batch)
