from typing import Dict, List, Union, Optional

import dgl
import karateclub
import networkx as nx
import numpy as np
import torch as th
from dgl import DGLGraph
from networkx import NetworkXNotImplemented
from scipy import sparse as sp
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch import sigmoid
from torch.nn.functional import binary_cross_entropy_with_logits

from nn import GCN, GCNEstimator, KarateclubTransformer
from util import create_neg_graph


class CentralityTransformer(BaseEstimator):
    def __init__(self, centralities=(nx.in_degree_centrality, nx.out_degree_centrality,
                                     nx.eigenvector_centrality_numpy, nx.pagerank_scipy)):
        self.centralities = centralities

    def transform(self, graph, mask=None):
        if mask is not None:
            relevant_nodes = mask[mask].keys()
            nodes = [n for n in graph.nodes if n in relevant_nodes]
        else:
            nodes = graph.nodes

        centralities = [c(graph) for c in self.centralities]
        return np.array([[centrality[node] for centrality in centralities] for node in nodes])

    def fit(self, *_, **__):
        return self

    def fit_transform(self, graph, *_, mask=None):
        return self.transform(graph, mask=mask)


class NeighTypeVectorizer(BaseEstimator):
    def __init__(self):
        self.node_type_enc_ = None

    def _get_neigh_types(self, graph, mask=None):
        def _count_neigh_types(node) -> np.ndarray:
            node_indices = sp.find(adj[node])[1]
            neigh_types = node_types[node_indices]
            return np.sum(neigh_types, 0)

        if mask is not None:
            relevant_nodes = mask[mask].keys()
            nodes = [i for i, n in enumerate(graph.nodes) if n in relevant_nodes]
        else:
            nodes = range(len(graph))

        adj = nx.adj_matrix(graph.to_undirected())
        node_types = np.array(list(nx.get_node_attributes(graph, 'type').values())).reshape(-1, 1)
        node_types = self.node_type_enc_.transform(node_types).todense()
        return np.array(list(map(_count_neigh_types, nodes))).squeeze()

    def fit(self, graph, *_):
        node_types = np.array(list(nx.get_node_attributes(graph, 'type').values())).reshape(-1, 1)
        self.node_type_enc_ = OneHotEncoder().fit(node_types)
        return self

    def transform(self, graph, mask=None):
        return self._get_neigh_types(graph, mask=mask)


class LinkDecoderModule(th.nn.Module):
    def __init__(self, activation=sigmoid, method: str = 'dot', dense_layer_size: Optional[int] = None,
                 asymmetric: bool = True):
        super().__init__()
        self.activation = activation
        self.method = method.lower()
        self.asymmetric = asymmetric
        self.dense_layer = th.nn.Linear(dense_layer_size, 1) if dense_layer_size else None
        methods = ('cat', 'concat', 'ip', 'dot', 'mul', 'hadamard', 'l1', 'l2', 'avg')
        if self.method not in methods:
            raise UserWarning('Unknown method <' + method + '>. Please choose from: ' + ', '.join(methods) + '.')

    def forward(self, graph: dgl.DGLGraph, feat):
        res = None
        u, v = graph.edges()
        emb_u, emb_v = feat[u], feat[v]
        if self.asymmetric:
            cutoff = int(round(feat.shape[1] / 2))
            emb_u, emb_v = emb_u[:, :cutoff], emb_v[:, cutoff:]

        if self.method in ('cat', 'concat'):
            res = th.hstack([emb_u, emb_v])

        if self.method in ('ip', 'dot'):
            res = th.unsqueeze(th.sum(emb_u * emb_v, dim=1), dim=1)

        if self.method in ('mul', 'hadamard'):
            res = emb_u * emb_v

        if self.method == 'l1':
            res = th.abs(emb_u - emb_v)

        if self.method == 'l2':
            res = th.square(emb_u - emb_v)

        if self.method == 'avg':
            res = (emb_u + emb_v) / 2

        if self.method not in ('ip', 'dot'):
            res = self.dense_layer(res)

        if self.activation is not None:
            return self.activation(res)

        return res.squeeze()


class LinkPredictorModel(th.nn.Module):
    def __init__(self, gcn: GCN, decoder: LinkDecoderModule):
        """
        Model for link prediction that uses a (gcn) model to generate node features and a (dot product) decoder to
        predict links.

        Parameters
        ----------

        gcn:
            Model that generates node features, e.g. a learnable GCN with learnable weights. Could also be static model.
        decoder:
            Decoder that returns the likelihood of a link between node pairs.
        """
        super().__init__()
        self.gcn = gcn
        self.pred = decoder

    def forward(self, g: dgl.DGLGraph, neg_g: dgl.DGLGraph, node_features: th.Tensor):
        """
        Compute link likelihood scores for all edges present in the two provided graph.
        The edges in the negative graph are used as an indicator which pair of nodes to generate link likelihoods for.

        g:
            Graph with positive edges, e.g. actual graphs.
        neg_g:
            Graph with negatie edges, e.g. graph generated by sampling unconnected node pairs without.
        node_features:
            Initial node features, e.g. one-hot node types for heterogeneous nodes.
        """
        node_embeddings: th.Tensor = self.gcn(g, node_features)
        return self.pred(g, node_embeddings), self.pred(neg_g, node_embeddings)

    def predict(self, train_graph: dgl.DGLGraph, test_graph: dgl.DGLGraph, node_features: th.Tensor):
        """
        Compute link likelihood scores for all edges present in the test graph based on features computed on the
        train graph. For each node, its feature is based on its neigborhood etc. in the train graph. The edges in the
        test graph are used as an indicator which pair of nodes to generate link likelihoods for.

        train_graph:
            Train graph that the node features are computed on.
        test_graph:
            Test graph that contains one edge for each pair of nodes that we want a prediction for, e.g. graph
            created as the union of test postitives and test negatives.
        node_features:
            Initial node features, e.g. one-hot node types for heterogeneous nodes.
        """
        node_embeddings: th.Tensor = self.gcn(train_graph, node_features)
        return self.pred(test_graph, node_embeddings)


def cross_entropy_loss(pos_scores, neg_scores):
    probas = th.cat([pos_scores, neg_scores])
    target = th.cat([th.ones_like(pos_scores), th.zeros_like(neg_scores)])
    return binary_cross_entropy_with_logits(probas, target)


class LinkPredictor(GCNEstimator):
    def __init__(self, nb_epochs=200, loss_criterion=cross_entropy_loss, optimizer_cls=th.optim.Adam, lr: float = 0.01,
                 weight_decay: float = 0.02, verbose: bool = False, device: Optional[str] = 'cuda',
                 scoring: callable = average_precision_score, node_feat_attr: str = 'type',
                 random_state: Union[int, np.random.RandomState, int] = 42, neg_ratio: int = 1,
                 lr_scheduler_cls=th.optim.lr_scheduler.ExponentialLR, lr_scheduler_kwargs=dict(gamma=1.0),
                 dense_layer_sizes=(), decoder=LinkDecoderModule(), full_negative_sampling: bool = False,
                 validation_size: float = 0, **model_kwargs):
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
        super().__init__(nb_epochs=nb_epochs, loss_criterion=loss_criterion, optimizer_cls=optimizer_cls, lr=lr,
                         weight_decay=weight_decay, verbose=verbose, device=device, scoring=scoring,
                         random_state=random_state, lr_scheduler_cls=lr_scheduler_cls, node_feat_attr=node_feat_attr,
                         lr_scheduler_kwargs=lr_scheduler_kwargs, dense_layer_sizes=dense_layer_sizes,
                         self_loops=True, **model_kwargs)
        self.training_history_ = []
        self.random_state, self.rng_ = random_state, None
        if model_kwargs is None:
            model_kwargs = {}
        self.model_: LinkPredictorModel
        self.optimizer_: optimizer_cls
        self.lr_scheduler_: th.optim.lr_scheduler
        self.training_history_: List[Dict[str, float]]
        self.model_kwargs = model_kwargs
        self.weight_decay = weight_decay
        self.scoring = scoring
        self.lr_scheduler_cls, self.lr_scheduler_kwargs = lr_scheduler_cls, lr_scheduler_kwargs
        self.dense_layer_sizes = dense_layer_sizes
        self.label_encoder_ = LabelEncoder()
        self.neg_ratio = neg_ratio
        self.decoder = decoder
        self.full_negative_sampling = full_negative_sampling
        self.validation_size = validation_size

    def _full_neg_sampling(self, graph: dgl.DGLGraph, k: int) -> dgl.DGLGraph:
        u, v = graph.edges()
        adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
        adj_neg = 1 - adj.todense() - np.eye(graph.number_of_nodes())
        neg_u, neg_v = np.where(adj_neg != 0)

        neg_eids = self.rng_.choice(len(neg_u), graph.number_of_edges() * k)
        train_neg_u, train_neg_v = neg_u[neg_eids], neg_v[neg_eids]

        return dgl.graph((train_neg_u, train_neg_v), num_nodes=graph.number_of_nodes())

    def _rejection_neg_sampling(self, graph: dgl.DGLGraph, k: float) -> dgl.DGLGraph:
        num_pos = graph.number_of_edges()
        num_neg_edges = int(round(num_pos * (k + 1)))
        neg_ind = self.rng_.randint(0, graph.number_of_nodes() - 1, size=[num_neg_edges, 2])
        neg_ind = neg_ind[:int(round(num_pos * k))]
        # neg_ind = np.unique(neg_ind, axis=1)[:int(round(num_pos * k))]
        neg_u, neg_v = graph.nodes()[neg_ind.T[0]], graph.nodes()[neg_ind.T[1]]
        return dgl.graph((neg_u, neg_v), num_nodes=graph.number_of_nodes())

    def _construct_negative_graph(self, graph: dgl.DGLGraph, k: int, full_sampling: bool = False) -> dgl.DGLGraph:
        if full_sampling:
            return self._full_neg_sampling(graph, k)
        return self._rejection_neg_sampling(graph, k)

        src, dst = graph.edges()
        neg_src = src.repeat_interleave(k)
        neg_dst = th.randint(0, graph.num_nodes(), (len(src) * k,))
        return dgl.graph((neg_src, neg_dst), num_nodes=graph.num_nodes())

    def _get_node_features(self, g: nx.DiGraph, train=False) -> (dgl.DGLGraph, th.Tensor):
        node_feats = np.vstack(list(nx.get_node_attributes(g, self.node_feat_attr).values()))

        if len(node_feats[0]) == 1 and node_feats.dtype not in (float, np.float16, np.float32):
            # categorical node features
            node_feats = node_feats.reshape(-1, 1)
            if train:
                self.node_type_enc_ = OneHotEncoder(handle_unknown='ignore').fit(node_feats)
            node_feats: th.FloatTensor = self.node_type_enc_.transform(node_feats).todense()
        node_feats = th.Tensor(node_feats)

        graph: dgl.DGLGraph = dgl.from_networkx(g)
        graph.ndata[self.node_feat_attr] = node_feats.long()
        return graph, node_feats

    def fit(self, X: nx.DiGraph):
        self._set_random_state()

        graph, node_feats = self._get_node_features(X, train=True)

        gcn = GCN(num_features=node_feats.shape[-1], output_size=None, dense_layer_sizes=self.dense_layer_sizes,
                  **self.model_kwargs)
        self.model_ = LinkPredictorModel(gcn=gcn, decoder=self.decoder)
        self.model_.to(self.device)
        self.training_history_ = []
        self.optimizer_ = self.optimizer_cls(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.lr_scheduler_ = self.lr_scheduler_cls(optimizer=self.optimizer_, **self.lr_scheduler_kwargs)

        if self.validation_size is not None and self.validation_size > 0:
            train_ind, valid_ind = train_test_split(range(graph.number_of_edges()), test_size=self.validation_size,
                                                    random_state=self.random_state)
            graph_train, graph_valid = dgl.remove_edges(graph, valid_ind), dgl.remove_edges(graph, train_ind)
            graph_valid = dgl.add_self_loop(graph_valid)
        else:
            graph_train, graph_valid = graph, None

        graph_train = dgl.add_self_loop(graph_train)

        for epoch in range(self.nb_epochs):
            epoch_results = self._train_epoch(optimizer=self.optimizer_, lr_scheduler=self.lr_scheduler_, epoch=epoch,
                                              graph_train=graph_train, features_train=node_feats,
                                              graph_valid=graph_valid)
            self.training_history_.append(epoch_results)
        self.model_.eval()
        th.cuda.empty_cache()
        return self

    def _train_epoch(self, optimizer, lr_scheduler: th.optim.lr_scheduler, epoch: int, graph_train: DGLGraph,
                     features_train: th.Tensor, graph_valid: Optional[dgl.DGLGraph] = None) -> Dict[str, float]:
        self.model_.train()
        optimizer.zero_grad()

        negative_graph = self._construct_negative_graph(graph_train, self.neg_ratio,
                                                        full_sampling=self.full_negative_sampling)
        pos_score, neg_score = self.model_(graph_train.to(self.device), negative_graph.to(self.device),
                                           features_train.to(self.device))
        loss = self.loss_criterion(pos_score, neg_score)

        loss.backward()
        loss = float(loss.detach())
        optimizer.step()

        y = th.cat([th.ones_like(pos_score), th.zeros_like(neg_score)]).cpu()
        y_pred = th.cat([pos_score, neg_score]).cpu().detach()  # >= 0.5
        epoch_results = dict(epoch=epoch, lr=optimizer.param_groups[0]['lr'], loss=loss,
                             train_score=self.scoring(y, y_pred))
        if graph_valid is not None:
            negative_graph = self._construct_negative_graph(graph_valid, self.neg_ratio,
                                                            full_sampling=self.full_negative_sampling)
            pos_score, neg_score = self.model_(graph_valid.to(self.device), negative_graph.to(self.device),
                                               features_train.to(self.device))
            y_pred = th.cat([pos_score, neg_score]).cpu().detach()  # >= 0.5
            y_valid = th.cat([th.ones_like(pos_score), th.zeros_like(neg_score)]).cpu()
            epoch_results['valid_score'] = self.scoring(y_valid, y_pred)

        if self.verbose:
            fields = [k + (' {:03d}' if type(v) is int else ' {:.4f}').format(v) for k, v in epoch_results.items()]
            print(' | '.join(fields))

        lr_scheduler.step()
        self.model_.eval()
        return epoch_results

    def predict_proba(self, X: nx.DiGraph, X_test: nx.DiGraph) -> np.ndarray:
        graph, node_feats = self._get_node_features(X)
        test_graph: dgl.DGLGraph = dgl.from_networkx(X_test)

        with th.no_grad():
            model_outputs = self.model_.predict(train_graph=dgl.add_self_loop(graph).to(self.device),
                                                test_graph=test_graph.to(self.device),
                                                node_features=node_feats.to(self.device)).squeeze()
            th.cuda.empty_cache()
        return model_outputs.detach().cpu().numpy()

    def predict(self, X: nx.DiGraph, X_test: nx.DiGraph) -> np.ndarray:
        return self.predict_proba(X, X_test) >= .5

    def transform(self, X: nx.DiGraph) -> np.ndarray:
        graph, node_feats = self._get_node_features(X)

        with th.no_grad():
            model_outputs = self.model_.gcn.forward(dgl.add_self_loop(graph).to(self.device),
                                                    node_feats.to(self.device), return_node_representations=False)
            th.cuda.empty_cache()
        return model_outputs.detach().cpu().numpy()


class LinkDecoder(BaseEstimator):
    def __init__(self, activation: str = 'sigmoid', asymmetric: bool = True, method: str = 'l1',
                 clf: Optional[BaseEstimator] = RandomForestClassifier(n_jobs=6, oob_score=True)):
        self.activation = activation
        self.asymmetric = asymmetric
        self.method = method.lower()
        self.clf = clf

    def _decode(self, emb_u: np.ndarray, emb_v: np.ndarray) -> np.ndarray:
        if self.asymmetric:
            cutoff = int(round(emb_u.shape[1] / 2))
            emb_u, emb_v = emb_u[:, :cutoff], emb_v[:, cutoff:]

        if self.method in ('cat', 'concat'):
            return np.hstack([emb_u, emb_v])

        if self.method in ('ip', 'dot'):
            return np.sum(emb_u * emb_v, axis=1)

        if self.method in ('mul', 'hadamard'):
            return emb_u * emb_v

        if self.method == 'l1':
            return np.abs(emb_u - emb_v)

        if self.method == 'l2':
            return np.square(emb_u - emb_v)

        if self.method == 'avg':
            return (emb_u + emb_v) / 2

    def fit(self, emb_u, emb_v, y):
        if self.clf is not None and self.method not in ('dot', 'ip'):
            feat = self._decode(emb_u, emb_v)
            self.clf.fit(feat, y)

        return self

    def predict_proba(self, emb_u: np.ndarray, emb_v: np.ndarray) -> np.ndarray:
        res = self._decode(emb_u, emb_v)

        if self.clf is not None and self.method not in ('ip', 'dot'):
            return self.clf.predict_proba(res)[:, 1]

        return 1 / (1 + np.exp(-res)) if self.activation == 'sigmoid' else res


class LinkPredictionWrapper(BaseEstimator):
    def __init__(self, encoder=KarateclubTransformer(karateclub.DeepWalk(workers=6)), decoder=LinkDecoder()):
        self.decoder = decoder
        self.encoder = encoder

    def fit(self, X: nx.DiGraph, *_):
        if not isinstance(self.encoder, KarateclubTransformer):
            self.encoder.fit(X)

        X_adj = nx.adj_matrix(X)
        X_neg = create_neg_graph(X)
        indices = np.vstack([np.argwhere(X_adj), np.argwhere(nx.adj_matrix(X_neg))])
        y = np.asarray(X_adj[indices.T[0], indices.T[1]]).squeeze()

        node_embeddings = self.transform(X)
        u, v = indices.T
        self.decoder.fit(node_embeddings[u], node_embeddings[v], y)

        return self

    def transform(self, X: nx.DiGraph, mask=None) -> np.ndarray:
        return self.encoder.transform(X) if mask is None else self.encoder.transform(X, mask=mask)

    def predict_proba(self, X: nx.DiGraph, X_test: nx.DiGraph) -> np.ndarray:
        node_embeddings = self.transform(X)
        u, v = np.array(X_test.edges()).T
        u_emb, v_emb = node_embeddings[u], node_embeddings[v]
        return self.decoder.predict_proba(u_emb, v_emb)

    def predict(self, X: nx.DiGraph, X_test: nx.DiGraph) -> np.ndarray:
        return self.predict_proba(X, X_test) >= 0.5


class NetworkxLinkPredictor(BaseEstimator):
    def __init__(self, algorithm=nx.adamic_adar_index, activation: str = 'sigmoid', verbose: bool = False):
        self.algorithm = algorithm
        self.activation = activation
        self.verbose = verbose

    def fit(self, *_, **__):
        pass

    def predict_proba(self, X: nx.DiGraph, X_test: nx.DiGraph) -> np.ndarray:
        try:
            res = self.algorithm(X, ebunch=X_test.edges)
        except NetworkXNotImplemented:
            if self.verbose:
                print('Using undirected graph...')
            res = self.algorithm(nx.to_undirected(X), ebunch=X_test.edges)
        res = np.array(list(res))[:, -1]
        return 1 / (1 + np.exp(-res)) if self.activation == 'sigmoid' else res