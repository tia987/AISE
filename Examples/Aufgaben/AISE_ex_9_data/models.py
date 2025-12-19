"""
This file contains the models for the GNN-based models.
"""
import torch
import copy
#from torch_scatter import scatter
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.nn import GCNConv, GATConv


##########Utility Functions##########

class Activation(nn.Module):
    """
        Parameters:
        -----------
            x: torch.FloatTensor
                input tensor
        Returns:
        --------
            y: torch.FloatTensor
                output tensor, same shape as the input tensor, since it's element-wise operation
    """
    def __init__(self, activation:str):
        super().__init__()
        activation = activation.lower() # prevent potential typo
        if activation in ['sigmoid', 'tanh']:
            # prevent potential warning message
            self.activation_fn = getattr(torch, activation)
        elif activation == "swish":
            self.beta = nn.Parameter(torch.ones(1), requires_grad=True)
    
        elif activation == "identity":
            self.activation_fn = lambda x: x
        else:
            self.activation_fn = getattr(F, activation)
        self.activation = activation
    def forward(self, x):
        if self.activation == "swish":
            return x * torch.sigmoid(self.beta * x)
        elif self.activation == "gelu":
            return x * torch.sigmoid(1.702 * x)
        elif self.activation == "mish":
            return x * torch.tanh(F.softplus(x))
        else:
            return self.activation_fn(x)

class MLP(nn.Module):
    """MLP with configurable number of layers and activation function

        Parameters:
        -----------
            x: torch.FloatTensor
                node features [num_nodes, num_features]
        Returns:
        --------
            y: torch.FloatTensor
                node labels [num_nodes, num_classes]
    """
    def __init__(self, num_features, num_classes,
        num_hidden=64, num_layers=3, activation="relu", input_dropout=0., dropout=0., bn=False, res=False):
        super().__init__()
        self.layers     = nn.ModuleList([nn.Linear(num_features, num_hidden)])
        for i in range(num_layers-2):
            self.layers.append(nn.Linear(num_hidden, num_hidden))
        self.layers.append(nn.Linear(num_hidden, num_classes))
        self.activation = Activation(activation)
        self.input_dropout    = nn.Dropout(input_dropout) if input_dropout > 0 else Identity()
        self.dropout          = nn.Dropout(dropout) if dropout > 0 else Identity()
        if bn:
            self.bns = nn.ModuleList([nn.BatchNorm1d(num_features)])
            for i in range(num_layers-2):
                self.bns.append(nn.BatchNorm1d(num_hidden))
            self.bns.append(nn.BatchNorm1d(num_hidden))
        else:
            self.bns = None
        if res:
            self.linear = nn.Linear(num_features, num_classes)
        else:
            self.linear = None
        self.num_features = num_features
        self.num_classes  = num_classes
        self.reset_parameters()
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        if self.bns is not None:
            for bn in self.bns:
                bn.reset_parameters()
    def forward(self, x):
        input = x
        x = self.input_dropout(x)
        x = self.bns[0](x) if self.bns is not None else x
        for i,layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.bns[i](x) if self.bns is not None else x
        x = self.layers[-1](x)
        if self.linear is not None:
            x = x + self.linear(input)
        return x

class Identity(nn.Module):
    def forward(self, x):
        return x 

#---------------------------------#

def init_gnn_model(model, num_features, num_classes, kwargs):
    if model == "gcn":
        return GCN(num_features, num_classes, num_hidden=kwargs.n_hidden, num_layers=kwargs.n_layers, activation=kwargs.activation)
    elif model == "gat":
        return GAT(num_features, num_classes, num_hidden=kwargs.n_hidden, num_layers=kwargs.n_layers, num_heads=kwargs.num_heads, activation=kwargs.activation)
    elif model == "sign":
        return SIGN(num_features, num_classes, num_hidden=kwargs.n_hidden, num_layers=kwargs.n_layers, num_hops=kwargs.num_hops, activation=kwargs.activation)
    elif model == "mpnp":
        return MPNP(num_features, num_classes, num_hidden=kwargs.n_hidden, num_mp_layers=kwargs.num_mp_layers, activation=kwargs.activation)
    else:
        raise NotImplementedError(f"Unknown model {model}")


class IN(gnn.MessagePassing):
    """
    Interaction Network (IN) - Single Message Passing Layer
    
    Components:
    - Edge Update Function (message_net): computes messages on edges
    - Node Update Function (node_update_net): updates node features
    """
    def __init__(self, num_features, num_classes, num_hidden=64, activation="relu", 
                 edge_hidden=64, node_hidden=64):
        super().__init__(aggr='mean')

        # Edge Update Function: computes messages from edge features
        # Input: [x_i, x_j] (concatenated source and target node features)
        # Output: message of dimension edge_hidden
        self.edge_update_net = nn.Sequential(
            nn.Linear(2 * num_features, edge_hidden),
            nn.Linear(edge_hidden, edge_hidden)
        )
        
        # Node Update Function: updates node features using aggregated messages
        # Input: [x, aggregated_message] (node feature + aggregated neighbor messages)
        # Output: updated node features of dimension num_classes
        self.node_update_net = nn.Sequential(
            nn.Linear(num_features + edge_hidden, node_hidden),
            nn.Linear(node_hidden, num_classes)
        )
        
        self.activation = Activation(activation)
        self.norm = nn.InstanceNorm1d(num_features)
        self.num_features = num_features
        self.num_classes = num_classes
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.edge_update_net:
            module.reset_parameters()
        for module in self.node_update_net:
            module.reset_parameters()
        self.norm.reset_parameters()
    
    def forward(self, x, edge_index):
        x = self.propagate(edge_index, x=x)
        x = self.norm(x.permute(0,2,1)).permute(0,2,1)
        return x
    
    def message(self, x_i: torch.Tensor , x_j: torch.Tensor) -> torch.Tensor:
        """
        Edge Update Function: compute messages on edges
        Args:
            x_i: target node features
            x_j: source node features
        Returns:
            message: computed edge messages
        """
        message = self.edge_update_net[0](torch.cat([x_i, x_j], dim=-1))
        message = self.activation(message)
        message = self.edge_update_net[1](message)
        message = self.activation(message)
        return message
    
    def update(self, message: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Node Update Function: update node features using aggregated messages
        Args:
            message: aggregated messages from neighbors
            x: current node features
        Returns:
            updated node features
        """
        update = self.node_update_net[0](torch.cat([x, message], dim=-1))
        update = self.activation(update)
        update = self.node_update_net[1](update)
        update = self.activation(update)

        if self.num_features == self.num_classes:
            update = update + x  # Residual connection
        return update


class MPNP(nn.Module):
    """
    Message Passing Neural PDE (MPNP)
    
    Stacks multiple Interaction Network (IN) layers to perform 
    multiple rounds of message passing.
    
    Args:
        num_features: input feature dimension
        num_classes: output feature dimension
        num_hidden: hidden feature dimension
        num_mp_layers: number of message passing layers (stacked IN layers)
        activation: activation function
    """
    def __init__(self, num_features, num_classes,
                 num_hidden=64, num_mp_layers=3, activation="relu"):
        super().__init__()
        
        # Message Passing Layers: stack multiple IN layers
        self.mp_layers = nn.ModuleList()
        
        self.mp_layers.append(IN(num_features, num_hidden, num_hidden, activation))
        for i in range(num_mp_layers - 2):
            self.mp_layers.append(IN(num_hidden, num_hidden, num_hidden, activation))
        self.mp_layers.append(IN(num_hidden, num_classes, num_hidden, activation))
        
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_mp_layers = num_mp_layers
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mp_layers:
            layer.reset_parameters()
    
    def forward(self, x, edge_index):
        """
        Forward pass: apply message passing layers sequentially
        """
        for layer in self.mp_layers:
            x = layer(x, edge_index)
        return x


class GCN(nn.Module):
    def __init__(self, num_features, num_classes, 
        num_hidden  = 64, num_layers  = 3, activation  = "relu"):
        super().__init__()
        self.layers     = nn.ModuleList([GCNConv(num_features, num_hidden)])
        for i in range(num_layers-2):
            self.layers.append(GCNConv(num_hidden, num_hidden))
        self.layers.append(GCNConv(num_hidden, num_classes))
        self.activation = Activation(activation)
        self.num_features = num_features
        self.num_classes  = num_classes
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x, edge_index))
        x = self.layers[-1](x, edge_index)
        return x


class GAT(nn.Module):
    def __init__(self, num_features, num_classes,
        num_hidden  = 64, num_layers  = 3, num_heads=4, activation  = "relu"):
        super().__init__()
        self.layers     = nn.ModuleList([GATConv(num_features, num_hidden//num_heads, heads=num_heads)])
        for i in range(num_layers-2):
            self.layers.append(GATConv(num_hidden, num_hidden//num_heads, heads=num_heads))
        # self.layers.append(GATConv(num_hidden, num_classes, num_heads=1))
        self.layers.append(GATConv(num_hidden, num_classes, heads=1))
        self.activation = Activation(activation)
        self.num_features = num_features
        self.num_classes  = num_classes
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index):
        xs = []
        for i in range(x.shape[0]):
            x_i = x[i]
            for layer in self.layers[:-1]:
                x_i = self.activation(layer(x_i, edge_index))
            x_i = self.layers[-1](x_i, edge_index)
            xs.append(x_i)
        x = torch.stack(xs, dim=0)
        
        return x


class SIGN(nn.Module):
    def __init__(self, num_features, num_classes,
        num_hidden = 64, num_layers = 3, num_hops=3, activation = "relu"):
        super().__init__()
        self.props   = nn.ModuleList([gnn.SGConv(num_features, num_hidden, K=k) for k in range(1,num_hops+1)])
        self.braches = nn.ModuleList(
            [MLP(num_features, num_hidden, num_hidden, num_layers, activation, res=True)]+
            [MLP(num_hidden, num_hidden, num_hidden, num_layers-1, activation, res=True) for _ in range(num_hops)])
        self.merger  = MLP(num_hidden*(num_hops + 1), num_classes, num_hidden, num_layers, activation)
        self.num_features = num_features
        self.num_classes  = num_classes
        self.reset_parameters()
    def reset_parameters(self):
        for prop, branch in zip(self.props, self.braches):
            prop.reset_parameters()
            branch.reset_parameters()
        self.merger.reset_parameters()

    def forward(self, x, edge_index):
        """
            Parameters:
            -----------
                x: torch.FloatTensor [..., n_node, n_feature]
            Returns:
            --------
                y: torch.FloatTensor [..., n_node, n_class]
        """
        props = [x] + [prop(x, edge_index) for prop in self.props]
        branches = [branch(prop) for prop, branch in zip(props, self.braches)]
        x = torch.cat(branches, dim=-1)
        x = self.merger(x)
        return x