import torch
import torch.nn.functional as F

from torch.nn import Linear
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, JumpingKnowledge

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger


class JKNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, mode='cat'):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.jump = JumpingKnowledge(mode=mode, channels=hidden_channels, num_layers=num_layers)
        if mode == 'cat':
            self.lin = Linear(num_layers * hidden_channels, out_channels)
        else:
            self.lin = Linear(hidden_channels, out_channels)

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

        self.jump.reset_parameters()
        self.lin.reset_parameters()


    def forward(self, x, adj_t):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs += [x]

        x = self.jump(xs)
        x = self.lin(x)

        return F.log_softmax(x, dim=-1)


dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=T.ToSparseTensor())
data = dataset[0]
data.adj_t = data.adj_t.to_symmetric()

split_idx = dataset.get_idx_split()

device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

data = data.to(device)
train_idx = split_idx['train'].to(device)


evaluator = Evaluator(name='ogbn-arxiv')
logger = Logger(10, None)

lr = 0.01
epochs = 500
num_layers = 5
hidden_channels = 128
log_steps = 1
dropout = 0.5
runs = 10
mode = 'max'

model = JKNet(data.num_features, hidden_channels, dataset.num_classes,
              num_layers, dropout, mode=mode).to(device)

def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc

for run in range(runs):
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, 1 + epochs):
        loss = train(model, data, train_idx, optimizer)
        result = test(model, data, split_idx, evaluator)
        logger.add_result(run, result)

        if epoch % log_steps == 0:
            train_acc, valid_acc, test_acc = result
            print(f'Run: {run + 1:02d}, '
                  f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * valid_acc:.2f}% '
                  f'Test: {100 * test_acc:.2f}%')

    logger.print_statistics(run)
logger.print_statistics()

# Sample running result
# All runs:
# Highest Train: 77.55 ± 0.09
# Highest Valid: 73.35 ± 0.07
# Final Train: 77.24 ± 0.21
# Final Test: 72.19 ± 0.21