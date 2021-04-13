from math import ceil

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool, JumpingKnowledge


class Block_1hop(torch.nn.Module):
    # If we only connect up to 1-hop neighbors, jumping knowledge is always False.
    def __init__(self, in_channels, hidden_channels, out_channels, jp=False):
        super(Block_1hop, self).__init__()

        self.conv1 = DenseSAGEConv(in_channels, out_channels)  

    def reset_parameters(self):
        self.conv1.reset_parameters()

    def forward(self, x, adj, mask=None, add_loop=True):
        x1 = F.relu(self.conv1(x, adj, mask, add_loop))
        x1 = F.normalize(x1, p=2, dim=-1)
        return x1

    
class Block_2hop(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, jp=False):
        super(Block_2hop, self).__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, out_channels)
        self.jp = jp
        if self.jp:
            self.jump = JumpingKnowledge('cat')
            self.lin = Linear(hidden_channels + out_channels, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        if self.jp:
            self.lin.reset_parameters()

    def forward(self, x, adj, mask=None, add_loop=True):
        x1 = F.relu(self.conv1(x, adj, mask, add_loop))
        x1 = F.normalize(x1, p=2, dim=-1)
        x2 = F.relu(self.conv2(x1, adj, mask, add_loop))
        x2 = F.normalize(x2, p=2, dim=-1)
        if self.jp:
            return F.relu(self.lin(self.jump([x1, x2])))
        return x2

    
class Block_3hop(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, jp=False):
        super(Block_3hop, self).__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels)
        self.jp = jp
        if self.jp:
            self.jump = JumpingKnowledge('cat')
            self.lin = Linear(2*hidden_channels + out_channels, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        if self.jp:
            self.lin.reset_parameters()

    def forward(self, x, adj, mask=None, add_loop=True):
        x1 = F.relu(self.conv1(x, adj, mask, add_loop))
        x1 = F.normalize(x1, p=2, dim=-1)
        x2 = F.relu(self.conv2(x1, adj, mask, add_loop))
        x2 = F.normalize(x2, p=2, dim=-1)
        x3 = F.relu(self.conv3(x2, adj, mask, add_loop))
        x3 = F.normalize(x3, p=2, dim=-1)
        if self.jp:
            return F.relu(self.lin(self.jump([x1, x2, x3])))
        return x3


class DiffPool(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, jpgs=False, jp=False, hop=2, num_patches=5, \
                 ratio=0.25, plot=False, dropout=False, ge=False):
        super(DiffPool, self).__init__()
        
        Block = [Block_1hop, Block_2hop, Block_3hop][hop-1]
        self.num_patches = num_patches
        self.dropout = dropout
        self.plot = plot
        num_nodes = ceil(ratio * dataset[0].num_nodes)
        self.embed_block1 = Block(dataset.num_features, hidden, hidden, jpgs)
        self.pool_block1 = Block(dataset.num_features, hidden, num_nodes, jpgs)

        self.embed_blocks = torch.nn.ModuleList()
        self.pool_blocks = torch.nn.ModuleList()
        for i in range((num_layers // 2) - 1):
            num_nodes = ceil(ratio * num_nodes)
            self.embed_blocks.append(Block(hidden, hidden, hidden, jpgs))
            self.pool_blocks.append(Block(hidden, hidden, num_nodes, jpgs))
        
        self.pool_block_last = Block(hidden, hidden, 1, jpgs)
        self.jp = jp
        self.ge = ge
        if self.jp:
            self.jump = JumpingKnowledge(mode='cat')
            self.lin1 = Linear((len(self.embed_blocks) + 1) * hidden, hidden)
            self.lin2 = Linear(hidden, dataset.num_classes)
        else:
            # self.lin1 = Linear(num_patches*hidden, hidden)
            self.lin1 = Linear(hidden, dataset.num_classes)
        
    def reset_parameters(self):
        self.embed_block1.reset_parameters()
        self.pool_block1.reset_parameters()
        for embed_block, pool_block in zip(self.embed_blocks,
                                           self.pool_blocks):
            embed_block.reset_parameters()
            pool_block.reset_parameters()
        self.pool_block_last.reset_parameters()
        self.lin1.reset_parameters()
        if self.jp:
            self.lin2.reset_parameters()

    def forward(self, data):
        x, adj, mask = data.x, data.adj, data.mask

        s = self.pool_block1(x, adj, mask, add_loop=True)
        s_return = s.clone().detach()
        x = self.embed_block1(x, adj, mask, add_loop=True)
        if self.jp:
            xs = [x.mean(dim=1)]
        x, adj, _, _ = dense_diff_pool(x, adj, s, mask)
        adj_return = adj.clone().detach()

        for i, (embed_block, pool_block) in enumerate(
                zip(self.embed_blocks, self.pool_blocks)):
            s = pool_block(x, adj)
            x = embed_block(x, adj)
            if i < len(self.embed_blocks) - 1:
                if self.jp:
                    xs.append(x.mean(dim=1))
                x, adj, _, _ = dense_diff_pool(x, adj, s)
                
        s = self.pool_block_last(x, adj)
        x, _, _, _ = dense_diff_pool(x, adj, s)
        if self.jp:
            xs.append(x.squeeze())
            x = self.jump(xs)
            x = F.relu(self.lin1(x)) 
            # return graph embedding
            if self.ge:
                return x
            # !!! 
            x = x.squeeze().reshape(x.size(0)//self.num_patches, self.num_patches, -1).max(dim=1)[0]
            if self.dropout:
                x = F.dropout(x, p=0.2, training=self.training)
            x = self.lin2(x)
              
        else:
            # return graph embedding
            if self.ge:
                return x.squeeze()
            num_patients = x.size(0)//self.num_patches
            x = x.squeeze().reshape(num_patients, self.num_patches, -1).max(dim=1)[0]
            if self.dropout:
                x = F.dropout(x, p=0.2, training=self.training)
            x = self.lin1(x)
            
                

        if self.plot:
            return F.softmax(x, dim=-1), (s_return, adj_return)

        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
