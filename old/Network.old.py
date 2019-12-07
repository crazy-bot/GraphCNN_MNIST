import torch.nn as nn
import torch

class BatchNorm1d_GNN(nn.BatchNorm1d):
    '''To make batch normalization over features'''

    def __init__(self, num_features):
        super(BatchNorm1d_GNN, self).__init__(num_features)

    def forward(self, x):
        return super(BatchNorm1d_GNN, self).forward(x.permute(0, 2, 1)).permute(0, 2, 1)


class GraphLayer(nn.Module):
    def __init__(self, in_features, out_features, bnorm=True, activation=nn.ReLU(True), n_relations=1):
        super(GraphLayer, self).__init__()
        self.bnorm = bnorm
        self.n_relations = n_relations
        self.K = 1
        fc = [nn.Linear(in_features * n_relations, out_features)]
        if bnorm:
            fc.append(BatchNorm1d_GNN(out_features))
        if activation is not None:
            fc.append(activation)
        # unpack items of fc list and build the network
        self.fc = nn.Sequential(*fc)

    def chebyshev_basis(self, L, X, K):
        # GCN
        assert K == 1, K
        # Performs a batch matrix-matrix product of matrices L and X
        # L: B,N,M  X: B,M,F  PROD: B,N,F
        return torch.bmm(L, X).unsqueeze(2)  # size B,N,1,F

    def laplacian_batch(self, A, add_identity=False):
        '''
        Computes normalized graph Laplacian transformed so that its eigenvalues are in range [-1, 1].
        A can be a multirelational matrix.
        '''
        B, N = A.shape[:2]
        if add_identity:
            A = A + torch.eye(N, device=A.get_device() if A.is_cuda else 'cpu').unsqueeze(0).unsqueeze(3)
        D = torch.sum(A, 1)  # nodes degree (B,N,R)
        D_hat = (D + 1e-5) ** (-0.5)
        L = D_hat.view(B, N, 1, -1) * A * D_hat.view(B, 1, N, -1)  # B,N,N,R
        return L if add_identity else -L  # for a valid Chebyshev basis

    def relation_fusion(self, x, A):
        y = []
        B, N = x.shape[:2]
        for rel in range(self.n_relations):
            y.append(self.chebyshev_basis(A[:, :, :, rel], x, self.K))  # B,N,K,C
            #print('y shape', y[0].shape)
        y = self.fc(torch.cat(y, 2).view(B, N, -1))  # B,N,F
        return y

    def forward(self, data):
        x, A, is_Laplacian = data[:3]

        B, N, C = x.shape
        if len(A.shape) == 3:
            A = A.unsqueeze(3)
        if not is_Laplacian:
            # can be done during preprocessing (except for the learnable graph)
            A = self.laplacian_batch(A, add_identity=True)
        #print('x,a shape', x.shape, A.shape)
        y = self.relation_fusion(x, A)

        return [y, A, True]  # return Laplacian to avoid computing it again

class GNN(nn.Module):
    '''
    Graph Neural Network class.
    '''

    def __init__(self,in_features=3, n_classes=10, filters=[32, 64, 512], n_relations=1):
        super(GNN, self).__init__()
        graph_layers = []
        for layer, f in enumerate(filters):
            graph_layers.append(GraphLayer(in_features if layer == 0 else filters[layer - 1], f,
                                           n_relations=n_relations))

        # stack og graph CNN layers
        self.graph_layers = nn.Sequential(*graph_layers)
        # classification layer
        self.fc = nn.Linear(filters[-1], n_classes)

    def forward(self, data):
        #print('graph layer parameters')
        x = self.graph_layers(data) # size 3,[y,A,is_laplacian]
        out = x[0]
        #print('graph layer output',out.shape)# B,N,out_features; B= batch_size, N= no of nodes,
        out = torch.max(out, dim=1)[0]  # Global MAX pooling
        final_out = self.fc(out)
        #print('final_out output', final_out.shape)
        return final_out # classification

