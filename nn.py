import os
from typing import Sequence, Tuple, Dict, List, Union, Optional

import dgl
import karateclub
import networkx as nx
import numpy as np
import torch as th
from dgl import DGLGraph
from dgl import DGLHeteroGraph
from dgl.nn import TAGConv, GraphConv
from scipy.sparse import coo_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.backends import cudnn
from torch.nn import Module


class BiGraphConv(GraphConv):
    def __init__(self, in_feats, out_feats, bias: bool = False, conv_cls=GraphConv, **conv_kwargs):
        super(BiGraphConv, self).__init__(in_feats, out_feats)
        self.lin = th.nn.Linear(2 * out_feats, out_feats, bias=bias)
        self.graph_conv_ = conv_cls(in_feats, out_feats, **conv_kwargs, bias=bias)
        self.out_feats = out_feats

    def forward(self, graph, feat):
        r"""

        Description
        -----------
        Compute bi-directional graph convolution, i.e. a linear combination of both directions.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        with graph.local_scope():
            graph_rev = dgl.reverse(graph)
            x = self.graph_conv_(graph, feat)
            x_rev = self.graph_conv_(graph_rev, feat)
            return self.lin(th.cat([x, x_rev], dim=-1))


class FwdBwdGraphConv(GraphConv):
    def __init__(self, in_feats, out_feats, bias: bool = True, conv_cls=GraphConv, **conv_kwargs):
        super(FwdBwdGraphConv, self).__init__(in_feats, out_feats)
        self.lin = th.nn.Linear(2 * out_feats, out_feats, bias=bias)
        self.fwd_graph_conv_ = conv_cls(in_feats, out_feats, **conv_kwargs)
        self.bwd_graph_conv_ = conv_cls(out_feats, out_feats, **conv_kwargs)
        self.out_feats = out_feats

    def forward(self, graph, feat):
        r"""

        Description
        -----------
        Compute bi-directional graph convolution, i.e. a linear combination of both directions.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        with graph.local_scope():
            graph_rev = dgl.reverse(graph)
            x_fwd = self.fwd_graph_conv_(graph, feat)
            x_bwd = self.bwd_graph_conv_(graph_rev, x_fwd)
            return self.lin(th.cat([x_fwd, x_bwd], dim=-1))


class TypedMaskedGraphConv(GraphConv):
    def __init__(self, in_feats, out_feats, num_types=135, type_arg: str = 'type', conv_cls=GraphConv, **conv_kwargs):
        super(TypedMaskedGraphConv, self).__init__(in_feats, out_feats)
        self.num_types = num_types
        self.node_type_embeddings_ = th.nn.Embedding(num_embeddings=self.num_types, embedding_dim=out_feats)
        self.graph_conv_ = conv_cls(in_feats, out_feats, **conv_kwargs)
        self.out_feats = out_feats
        self.type_arg = type_arg

    def forward(self, graph, feat):
        r"""

        Description
        -----------
        Compute node type masked graph convolution.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        with graph.local_scope():
            # node_types = graph._node_frames[0][self.type_arg].long()
            node_types = th.where(graph.ndata['type'])[1].long()
            mask = self.node_type_embeddings_(node_types)
            res = self.graph_conv_(graph, feat)
            return res * mask[:, :self.out_feats]


class GCN(Module):
    """
    Graph-Level Conv Net with pooling functions for aggregation.

    Parameters
    ----------

    layer_sizes:
        Sequence of depths (number of filters) for the GraphConv Layers. Consequently, this also
        determines the number of hidden layers.
    dropout:
        drop probability for dropout applied before pooling, set to `None` for no dropout.
    conv_cls:
        Convolution Module that is applied for feature extraction.
    output_size:
        Numer of outputs per graph - set to 0 or None for raw graph-level embeddings as output.
    conv_kwargs:
        Keyword arguments used for instantiation of each Conv layer.
    skip_connections:
        Use skip connections where applicable (i.e. between layers with matching sizes).
    batch_norm:
        Use batchnorm layers after each convolutional layer to reduce covariate shift for improved convergence.
    softmax:
        Use a softmax layer to produce prediction probabilities.
    dense_layer_sizes:
        Sequence of widths (number of neurons) for suprevised tasks.

    """

    def __init__(self, num_features: int, layer_sizes: Tuple[int] = (32, 32, 32, 32, 32),
                 output_size: Union[Sequence[int], int, None] = 1, dropout: float = 0.1,
                 conv_cls: type = BiGraphConv, skip_connections: bool = True, batch_norm: bool = True,
                 softmax: bool = False, dense_layer_sizes: Optional[Tuple[int]] = (),
                 node_type_embedding_size: Optional[int] = None, etypes: Sequence = None, **conv_kwargs):
        super().__init__()
        if conv_kwargs is None:
            conv_kwargs = {}
        self.conv_layers_ = th.nn.ModuleList()
        self.num_features = num_features
        self.dropout = dropout
        self.layer_sizes = layer_sizes
        self.conv_cls = conv_cls
        self._layer_ind_mapping = {}
        self.skip_indices_ = {}
        self.skip_connections = skip_connections
        self.softmax = softmax
        self.dense_layer_sizes = dense_layer_sizes
        self.output_layer_ = None
        self.node_type_embedding_size = node_type_embedding_size

        if dropout is not None and (dropout < 0 or dropout > 1):
            raise UserWarning("Choose None for no dropout or a float between 0 and 1.")

        if node_type_embedding_size is not None and node_type_embedding_size > 0:
            self.node_type_embeddings_ = th.nn.Embedding(num_embeddings=self.num_features,
                                                         embedding_dim=self.node_type_embedding_size)
            input_size = self.node_type_embedding_size
        else:
            input_size = self.num_features

        for layer_ind, size in enumerate(self.layer_sizes):
            is_last = layer_ind == len(self.layer_sizes) - 1

            if conv_cls in (dgl.nn.GATConv, dgl.nn.EdgeConv):
                conv_layer = conv_cls(input_size, size, **conv_kwargs)
            else:
                conv_layer = conv_cls(input_size, size, **conv_kwargs, bias=not batch_norm)
            self.conv_layers_.append(conv_layer)

            if batch_norm:
                self.conv_layers_.append(th.nn.BatchNorm1d(size))

            # no activation for last conv layer
            if not is_last:
                self.conv_layers_.append(th.nn.ReLU())

            actual_ind = len(self.conv_layers_) - 1
            self._layer_ind_mapping[layer_ind] = actual_ind

            if self.skip_connections and layer_ind >= 2 and size == self.layer_sizes[layer_ind - 2]:
                self.skip_indices_[actual_ind] = self._layer_ind_mapping[layer_ind - 2]
            input_size = size

        if self.dense_layer_sizes not in ((), None):
            self.dense_layers_ = th.nn.ModuleList()
            in_features = self.layer_sizes[-1]
            for layer_size in self.dense_layer_sizes:
                dense_layer = th.nn.Linear(in_features=in_features, out_features=layer_size)
                self.dense_layers_.append(
                    th.nn.Sequential(dense_layer, th.nn.ReLU(), th.nn.LayerNorm(layer_size)))
                in_features = layer_size

            output_input_size = self.dense_layer_sizes[-1]
        else:
            output_input_size = self.layer_sizes[-1]

        if self.dropout is not None and 0 < self.dropout <= 1:
            self.conv_layers_.append(th.nn.Dropout(self.dropout))

        if output_size is not None:
            self.output_layer_ = th.nn.Linear(output_input_size, output_size)
            if self.softmax:
                self.output_layer_ = th.nn.Sequential(self.output_layer_, th.nn.Softmax(dim=1))

    def _apply_output_layers(self, x):
        if self.dense_layer_sizes not in ((), None):
            for layer in self.dense_layers_:
                x = layer(x)

        if self.output_layer_ is None:
            return x
        return self.output_layer_(x)

    def forward(self, g: DGLGraph, features: th.Tensor, return_node_representations: bool = False,
                mask=None):
        layer_outputs = {}
        x = features

        if self.node_type_embedding_size is not None:
            x = th.sigmoid(self.node_type_embeddings_.forward(th.where(g.ndata['type'])[1].long()))

        for ind, layer in enumerate(self.conv_layers_):
            if ind in self.skip_indices_:
                x = x + layer_outputs[self.skip_indices_[ind]]

            x = layer(g, x) if isinstance(layer, self.conv_cls) else layer(x)
            if ind in self.skip_indices_.values():
                layer_outputs[ind] = x

        graph_sizes = tuple(g.batch_num_nodes().cpu().numpy())
        if len(graph_sizes) > 1:
            if mask is None or len(graph_sizes) != len(mask):
                raise UserWarning('Multiple graphs were passed to the model, but there is no mask specifying which node'
                                  'to pick from each graphs')

            x = th.split(x, graph_sizes) if len(g.batch_num_nodes()) > 1 else [x]
            x = th.vstack([graph_x[node] for graph_x, node in zip(x, mask)])
        else:
            if mask is not None:
                x = x[mask]

        if return_node_representations:
            return x

        return self._apply_output_layers(x)


class GCNEstimator(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, nb_epochs=100, shuffle_batches: bool = True, loss_criterion=th.nn.CrossEntropyLoss(),
                 optimizer_cls=th.optim.Adam, lr: float = 0.005, weight_decay: float = 0.01, verbose: bool = False,
                 device: Optional[str] = 'cuda', scoring: callable = accuracy_score,
                 random_state: Union[int, np.random.RandomState, int] = None, node_feat_attr: str = 'type',
                 lr_scheduler_cls=th.optim.lr_scheduler.ExponentialLR, lr_scheduler_kwargs=dict(gamma=0.98),
                 dense_layer_sizes=(64,), loss_weighting: bool = True, self_loops: bool = False, **model_kwargs):
        """
        Node level Res-GCN classifier.

        Parameters
        ----------

        scoring:
            Scoring function for validation scores and `self.score()`.
        loss_criterion:
            loss criterion implementing `torch.nn.modules._WeightedLoss` that is used for gradient descent.
        model_kwargs:
            Keyword Arguments passed to the P-GCN for instantiating `self.model_`.
        """
        self.device: str = device
        self.model_: Optional[GCN] = None
        self.training_history_ = []
        self.random_state, self.rng_ = random_state, None
        if model_kwargs is None:
            model_kwargs = {}
        self.optimizer_cls = optimizer_cls
        self.nb_epochs = nb_epochs
        self.lr = lr
        self.verbose = verbose
        self.model_: GCN
        self.optimizer_: optimizer_cls
        self.lr_scheduler_: th.optim.lr_scheduler
        self.training_history_: List[Dict[str, float]]
        self.loss_criterion = loss_criterion
        self.shuffle_batches = shuffle_batches
        self.model_kwargs = model_kwargs
        self.weight_decay = weight_decay
        self.scoring = scoring
        self.lr_scheduler_cls, self.lr_scheduler_kwargs = lr_scheduler_cls, lr_scheduler_kwargs
        self.dense_layer_sizes = dense_layer_sizes
        self.label_encoder_ = LabelEncoder()
        self.loss_weighting = loss_weighting
        self.classes_ = None
        self.self_loops = self_loops
        self.node_feat_attr = node_feat_attr

    def _get_model_outputs(self, graph: dgl.DGLGraph, features: np.ndarray, **kwargs):
        def _to_device_tensor(a) -> th.Tensor:
            tensor = th.Tensor(a) if isinstance(a, np.ndarray) else th.sparse.Tensor(a.toarray())
            return tensor.to(self.device) if self.device not in ('cpu', None) else tensor

        features = _to_device_tensor(features.astype(np.float32))
        return self.model_(graph.to(self.device), features, **kwargs)

    def _fit_node_type_enc(self, graph: Union[nx.DiGraph, Sequence[nx.DiGraph]]):
        if isinstance(graph, Sequence):
            all_node_types = set()
            for g in graph:
                node_types = set(list(nx.get_node_attributes(g, self.node_feat_attr).values()))
                all_node_types = all_node_types.union(node_types)
            node_types = np.array(list(all_node_types)).reshape(-1, 1)

        else:
            node_types = np.array(list(nx.get_node_attributes(graph, self.node_feat_attr).values())).reshape(-1, 1)

        self.node_type_enc_ = OneHotEncoder(handle_unknown='ignore').fit(node_types)

    def _preprocess_graph(self, graph: nx.DiGraph) -> Tuple[Union[DGLGraph, DGLHeteroGraph], np.ndarray]:
        node_types = np.array(list(nx.get_node_attributes(graph, self.node_feat_attr).values())).reshape(-1, 1)
        node_types = self.node_type_enc_.transform(node_types).todense()

        graph = dgl.from_networkx(graph)
        if self.self_loops:
            graph = dgl.add_self_loop(graph)
        return graph, node_types

    def _preprocess_input(self, graph: Union[nx.DiGraph, Sequence[nx.DiGraph]]) -> Tuple[
        Union[DGLGraph, DGLHeteroGraph], np.ndarray]:
        if isinstance(graph, Sequence):
            graphs, features = [], []
            for i, (g, feat) in enumerate(map(self._preprocess_graph, graph)):
                graphs.append(g)
                features.append(feat)

            return dgl.batch(graphs), np.vstack(features)
        return self._preprocess_graph(graph)

    def _set_random_state(self):
        def _get_rng(random_state):
            if random_state is None:
                return np.random.RandomState()
            if type(random_state) is np.random.RandomState:
                return random_state
            return np.random.RandomState(random_state)

        self.rng_ = _get_rng(self.random_state)
        if self.random_state is not None:
            th.manual_seed(self.random_state)
            cudnn.deterministic = True
            cudnn.benchmark = False
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


class KarateclubTransformer(BaseEstimator):
    def __init__(self, model=karateclub.DeepWalk(workers=6)):
        self.model = model

    def transform(self, graph, mask=None):
        self.model.fit(nx.convert_node_labels_to_integers(graph))
        embeddings = self.model.get_embedding()
        return embeddings if mask is None else embeddings[mask]

    def fit(self, *_):
        return self

    def fit_transform(self, graph, *_, mask=None):
        return self.transform(graph, mask=mask)


class MUSAETransformer(BaseEstimator):
    def __init__(self, model=karateclub.MUSAE(workers=6), type_attribute: str = 'type'):
        self.model = model
        self.type_attribute = type_attribute

    def transform(self, graph, mask=None):
        node_types = nx.get_node_attributes(graph, self.type_attribute)
        node_types = [node_types[n] for n in graph.nodes]
        node_types = OneHotEncoder().fit_transform(np.array(node_types).reshape(-1, 1))

        self.model.fit(nx.convert_node_labels_to_integers(graph), coo_matrix(node_types))
        embeddings = self.model.get_embedding()
        return embeddings if mask is None else embeddings[mask]

    def fit(self, *_):
        return self

    def fit_transform(self, graph, *_, mask=None):
        return self.transform(graph, mask=mask)
