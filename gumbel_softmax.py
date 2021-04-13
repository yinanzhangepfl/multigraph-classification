from math import ceil
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool, JumpingKnowledge

device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.to(device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax(logits, temperature, hard, decay_rate=1):

    y = logits + decay_rate*sample_gumbel(logits.size())
    y = F.softmax(y / temperature, dim=-1)
    if not hard:
        return y
    y_hard = (y == y.max(dim=2, keepdim=True)[0])
    y_hard = y_hard.type(torch.FloatTensor)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard
    
class Block_1hop(torch.nn.Module):
    # If we only connect up to 1-hop neighbors, jumping knowledge is always False.
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Block_1hop, self).__init__()

        self.conv1 = DenseSAGEConv(in_channels, out_channels)  

    def reset_parameters(self):
        self.conv1.reset_parameters()

    def forward(self, x, adj, mask=None, add_loop=True):
        x1 = F.relu(self.conv1(x, adj, mask, add_loop))
        x1 = F.normalize(x1, p=2, dim=-1)
        return x1

    
class Block_2hop(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Block_2hop, self).__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, adj, mask=None, add_loop=True):
        x1 = F.relu(self.conv1(x, adj, mask, add_loop))
        x1 = F.normalize(x1, p=2, dim=-1)
        x2 = F.relu(self.conv2(x1, adj, mask, add_loop))
        x2 = F.normalize(x2, p=2, dim=-1)
        return x2

    
class Block_3hop(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Block_3hop, self).__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()

    def forward(self, x, adj, mask=None, add_loop=True):
        x1 = F.relu(self.conv1(x, adj, mask, add_loop))
        x1 = F.normalize(x1, p=2, dim=-1)
        x2 = F.relu(self.conv2(x1, adj, mask, add_loop))
        x2 = F.normalize(x2, p=2, dim=-1)
        x3 = F.relu(self.conv3(x2, adj, mask, add_loop))
        x3 = F.normalize(x3, p=2, dim=-1)
        return x3
    
class DiffPool(torch.nn.Module):
    def __init__(self, dataset, num_layers=5, hidden=64, hop=2, num_patches=5, ratio=0.25, plot=False, dropout=False, ge=False, gs=False, total=20, hard=False, hard_train=False, aux_loss=False, agg='add', decay_rate=1):
        super(DiffPool, self).__init__()
        
        Block = [Block_1hop, Block_2hop, Block_3hop][hop-1]
        self.hidden = hidden
        self.num_patches = num_patches
        self.dropout = dropout
        self.plot = plot
        self.ge = ge
        self.gs = gs
        self.total = total
        self.hard = hard
        self.hard_train = hard_train
        self.aux_loss = aux_loss
        self.agg = agg
        self.decay_rate = decay_rate
        num_nodes = ceil(ratio * dataset[0].num_nodes)
        self.embed_block1 = Block(dataset.num_features, hidden, hidden)
        self.pool_block1 = Block(dataset.num_features, hidden, num_nodes)

        self.embed_blocks = torch.nn.ModuleList()
        self.pool_blocks = torch.nn.ModuleList()
        for i in range((num_layers // 2) - 1):
            num_nodes = ceil(ratio * num_nodes)
            self.embed_blocks.append(Block(hidden, hidden, hidden))
            self.pool_blocks.append(Block(hidden, hidden, num_nodes))
        
        self.pool_block_last = Block(hidden, hidden, 1)
        self.fc1 = Linear(self.total*hidden, 512)
        self.fc2 = Linear(512, 256)
        self.fc3 = Linear(256, self.num_patches*self.total)
        self.lin1 = Linear(hidden, dataset.num_classes)
        
        
    def reset_parameters(self):
        self.embed_block1.reset_parameters()
        self.pool_block1.reset_parameters()
        for embed_block, pool_block in zip(self.embed_blocks,
                                           self.pool_blocks):
            embed_block.reset_parameters()
            pool_block.reset_parameters()
        self.pool_block_last.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.lin1.reset_parameters()
        
       
    def reset_gumbel(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()

    
    def sample(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return F.relu(self.fc3(h2))
    
    
    def forward(self, data, temp):
        x, adj, mask = data.x, data.adj, data.mask

        s = self.pool_block1(x, adj, mask, add_loop=True)
        s_return = s.clone().detach()
        x = self.embed_block1(x, adj, mask, add_loop=True)
        x, adj, _, _ = dense_diff_pool(x, adj, s, mask)
        adj_return = adj.clone().detach()

        for i, (embed_block, pool_block) in enumerate(
                zip(self.embed_blocks, self.pool_blocks)):
            s = pool_block(x, adj)
            x = embed_block(x, adj)
            if i < len(self.embed_blocks) - 1:
                x, adj, _, _ = dense_diff_pool(x, adj, s)
                
        s = self.pool_block_last(x, adj)
        x, _, _, _ = dense_diff_pool(x, adj, s) 
        x = x.squeeze()

        # return graph embeddings
        if self.ge:
            return x
        
        num_patients = x.size(0)//self.total
        

        q = self.sample(x.reshape(num_patients, -1))
        q_y = q.view(num_patients, self.num_patches, self.total)
        q_y_return = q_y.clone().detach()
        z = gumbel_softmax(q_y, temp, self.hard_train, self.decay_rate)
        z_return = z.clone().detach()
        
        
        if self.aux_loss:
            z_reverse = torch.abs(z-1)
            complement = torch.ones(num_patients, self.total)
            for i in range(self.num_patches):
                complement = z_reverse[:, i, :]*complement
            discard_graphs = complement.reshape(num_patients*self.total, 1)*x
            discard_graphs = discard_graphs[discard_graphs.sum(dim=1) != 0]
        
        # multiply gumbel softmax variables with graph embeddinggs
        x = x.reshape(num_patients, self.total, self.hidden)
        if self.hard:
            with torch.no_grad():
                z_hard = (z == z.max(dim=2, keepdim=True)[0])
                z_hard = z_hard.type(torch.FloatTensor)
                x_hard = torch.bmm(z_hard, x)
        x = torch.bmm(z, x)
        

       # aggreator after linear layer
        x = self.lin1(x.reshape(num_patients*self.num_patches, -1))
        if self.agg == 'add':
            x = x.reshape(num_patients, self.num_patches, -1).sum(dim=1)
        elif self.agg == 'max':
            x = x.reshape(num_patients, self.num_patches, -1).max(dim=1)[0]
        elif self.agg == 'mean':
            x = x.reshape(num_patients, self.num_patches, -1).mean(dim=1)
        else:
            os._exit(1)
            
        
        if self.hard:
            with torch.no_grad():
                x_hard = x_hard.max(dim=1)[0]
                x_hard = self.lin1(x_hard)
            return (F.softmax(x, dim=-1), F.softmax(x_hard, dim=-1)), z_return, q_y_return
        
        if self.plot:
            return F.softmax(x, dim=-1), (s_return, adj_return), z_return

        if self.gs:
            return z_return
        
        return F.log_softmax(x, dim=-1) if not self.aux_loss else (F.log_softmax(x, dim=-1), discard_graphs, z)

    def __repr__(self):
        return self.__class__.__name__
