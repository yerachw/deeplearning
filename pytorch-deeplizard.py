import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#from IPython.display import display, clear_output
import pandas as pd
import time
import json

from itertools import product
from collections import namedtuple
from collections import OrderedDict

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        
    def forward(self, t):
       
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        
        return t

class RunBuilder():
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs

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
        
    def begin_run(self, run, network, loader, device):
        
        self.run_start_time = time.time()

        self.run_params = run
        self.run_count += 1
        
        self.network = network
        self.loader = loader
        self.tb = SummaryWriter(comment=f'-{run}')
        
        images, labels = next(iter(self.loader))
        grid = torchvision.utils.make_grid(images)

        self.tb.add_image('images', grid)
        self.tb.add_graph(
             self.network
            ,images.to(device)
        )
        
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
        
        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)
        
        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results['loss'] = loss
        results["accuracy"] = accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k,v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)
        
        print(f"{self.run_params} -- Run: {self.run_count}, Epoch: {self.epoch_count}, Loss: {loss}, Accuracy {accuracy}")
        #df = pd.DataFrame.from_dict(self.run_data, orient='columns')
        
        # clear_output(wait=True)
        # display(df)
        
    def track_loss(self, loss, batch):
        self.epoch_loss += loss.item() * batch[0].shape[0]
        
    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self._get_num_correct(preds, labels)
    
    def _get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()
    
    def save(self, fileName):
        
        pd.DataFrame.from_dict(
            self.run_data
            ,orient='columns'
        ).to_csv(f'{fileName}.csv')
        
        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)

train_set = torchvision.datasets.FashionMNIST(
    root='/home/ubuntu/data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)


params = OrderedDict(
    lr = [0.001]        #[.01, .001, .0001, .00001]
    ,batch_size = [100] # [100, 1000, 10000]
    ,shuffle = [True]
    ,epochs=[50]
    ,num_workers=[4]
)
layers = OrderedDict([
    ('conv1', nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5))
    ,('relu1', nn.ReLU())
    ,('MaxPool1', nn.MaxPool2d(kernel_size=2, stride=2))
    ,('conv2', nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5))
    ,('relu2', nn.ReLU())
    ,('MaxPool2', nn.MaxPool2d(kernel_size=2, stride=2))
    ,('flat', nn.Flatten(start_dim=1))
    ,('hidden1', nn.Linear(in_features=12*4*4, out_features=120))
    ,('relu3', nn.ReLU())
    ,('hidden2', nn.Linear(in_features=120, out_features=60))
    ,('relu4', nn.ReLU())
    ,('output', nn.Linear(60, 10))
])

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
m = RunManager()
for run in RunBuilder.get_runs(params):

    # network = nn.Sequential(
    #   nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
    # , nn.ReLU()
    # , nn.MaxPool2d(kernel_size=2, stride=2)
    # , nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
    # , nn.ReLU()
    # , nn.MaxPool2d(kernel_size=2, stride=2)
    # , nn.Flatten(start_dim=1)  
    # , nn.Linear(in_features=12*4*4, out_features=120)
    # , nn.ReLU()
    # , nn.Linear(in_features=120, out_features=60)
    # , nn.ReLU()
    # , nn.Linear(in_features=60, out_features=10)
    # ).to(device)
    network = nn.Sequential(layers).to(device)
    loader = DataLoader(train_set, batch_size=run.batch_size, shuffle=run.shuffle, num_workers=run.num_workers)
    optimizer = optim.Adam(network.parameters(), lr=run.lr)
    
    m.begin_run(run, network, loader, device)
    for epoch in range(run.epochs):
        m.begin_epoch()
        for batch in loader:            
            images = batch[0].to(device)
            labels = batch[1].to(device)
            preds = network(images) # Pass Batch
            loss = F.cross_entropy(preds, labels) # Calculate Loss
            optimizer.zero_grad() # Zero Gradients
            loss.backward() # Calculate Gradients
            optimizer.step() # Update Weights
            
            m.track_loss(loss, batch)
            m.track_num_correct(preds, labels)  
        m.end_epoch()
    m.end_run()
    torch.save(network.state_dict(), "model.pth")
m.save('results')
