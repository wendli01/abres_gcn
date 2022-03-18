from typing import Sequence
import networkx as nx
import numpy as np
import numpy.random
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from networkx import convert_node_labels_to_integers
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def generate_type_graph(instance_graph: nx.DiGraph, node_type_attr: str = 'type') -> nx.DiGraph:
    type_graph = nx.DiGraph()
    node_types = nx.get_node_attributes(instance_graph, node_type_attr)
    for node, node_type in node_types.items():
        for _, to_node in instance_graph.out_edges(node):
            to_node_type = node_types[to_node]
            if (node_type, to_node_type) not in type_graph.edges:
                type_graph.add_edge(node_type, to_node_type)
                type_graph[node_type][to_node_type]["weight"] = 1
            else:
                weight = type_graph[node_type][to_node_type]['weight']
                type_graph[node_type][to_node_type]["weight"] = weight + 1
    return type_graph


def create_neg_graph(X: nx.DiGraph, random_state: int = 42, negative_sampling_ratio: float = 1.0):
    rng = np.random.RandomState(random_state)
    adj = nx.adj_matrix(X)

    pos_ind = rng.permutation(np.argwhere(adj))
    num_pos = len(pos_ind)

    num_neg_edges = int(round(num_pos * (negative_sampling_ratio + 1)))
    neg_ind = np.unique(rng.randint(0, X.number_of_nodes() - 1, size=[num_neg_edges, 2]), axis=1)
    neg_ind = neg_ind[:int(round(num_pos * negative_sampling_ratio))]

    test_negatives = np.transpose([np.array(list(X.nodes))[neg_ind.T[0]], np.array(list(X.nodes))[neg_ind.T[1]]])
    X_neg = nx.create_empty_copy(X)
    for edge in test_negatives:
        X_neg.add_edge(*edge)

    return X_neg


def plot_type_scores(X: nx.DiGraph, type_scores):
    def get_type_level(n, primitive_types=('pmString', 'pmLabel', 'pmDate', 'pmNumber', 'pmDateTime')):
        return np.min([nx.shortest_path_length(type_graph, n, p_type) if
                       nx.has_path(type_graph, n, p_type) else np.infty for p_type in primitive_types])

    type_graph = generate_type_graph(X)
    type_levels = {n: get_type_level(n) for n in type_graph.nodes}
    type_scores_df = pd.DataFrame.from_records(np.hstack(type_scores))
    n_splits = len(type_scores_df['fold'].unique())
    score_counts = type_scores_df.groupby(['node_type']).score.count()
    type_scores_df = type_scores_df[type_scores_df.node_type.isin(score_counts.index[score_counts == 2 * n_splits])]
    type_scores_df['type_level'] = type_scores_df.node_type.map(type_levels)
    sns.catplot(data=type_scores_df, x='node_type', y='score', row='kind', kind='box', aspect=4, hue='type_level',
                order=type_scores_df.fillna(-1).groupby('node_type').score.mean().sort_values().index, dodge=False)
    plt.xticks(rotation=90)
    plt.show()


def cv_score_link_prediction(est, X: nx.DiGraph, n_splits: int = 10, random_state: int = 42, test_size: float = .2,
                             scoring=(average_precision_score, roc_auc_score), verbose: bool = False,
                             plot_type_error: bool = False, negative_sampling_ratio: float = 1.0,
                             plot_pr_curve: bool = False, remove_direction: bool = False, **kwargs):
    rng = np.random.RandomState(random_state)
    adj = nx.adj_matrix(X)

    # This is absolutely necessary if working with string node names, otherwise correct edge order is not guaranteed!
    X = convert_node_labels_to_integers(X)
    if isinstance(scoring, Sequence):
        primary_scoring = scoring[0]
    else:
        primary_scoring = scoring
        scoring = [scoring]

    scores, pr_curves, type_scores = [], [], []
    for i in range(n_splits):
        pos_ind = rng.permutation(np.argwhere(adj))
        num_pos = len(pos_ind)
        num_test_pos = int(round(test_size * num_pos))

        edges = np.transpose([np.array(list(X.nodes))[pos_ind.T[0]], np.array(list(X.nodes))[pos_ind.T[1]]])
        train_positives, test_positives = edges[num_test_pos:], edges[:num_test_pos]

        num_test_edges = int(round(num_test_pos * (negative_sampling_ratio + 1)))
        neg_ind = np.unique(rng.randint(0, X.number_of_nodes() - 1, size=[num_test_edges, 2]), axis=1)

        test_negatives = np.transpose([np.array(list(X.nodes))[neg_ind.T[0]], np.array(list(X.nodes))[neg_ind.T[1]]])
        test_indices = np.unique([*pos_ind[:num_test_pos], *neg_ind], axis=0)[:num_test_edges]
        test_edges = np.unique([*test_positives, *test_negatives], axis=0)[:num_test_edges]

        train_graph = nx.create_empty_copy(X, with_data=True)
        for edge in train_positives:
            train_graph.add_edge(*edge)

        test_graph: nx.DiGraph = nx.create_empty_copy(X, with_data=True)
        for edge in test_edges:
            test_graph.add_edge(*edge)

        if remove_direction:
            train_graph = nx.to_undirected(train_graph)

        est.fit(X=train_graph, **kwargs)

        y_pred = est.predict_proba(train_graph, test_graph)
        y_true = np.asarray(adj[test_indices.T[0], test_indices.T[1]]).squeeze()
        fold_scores = {scorer.__name__: scorer(y_true, y_pred) for scorer in scoring}
        scores.append(fold_scores)

        if verbose:
            print('\t'.join([scorer + ': ' + str(round(score, 3)) for scorer, score in fold_scores.items()]))

        if plot_type_error:
            node_types = nx.get_node_attributes(test_graph, 'type')
            u_types, v_types = np.array([[node_types[u], node_types[v]] for u, v in test_edges]).T
            source_scores = [
                {'kind': 'source', 'fold': i, 'node_type': t,
                 'score': primary_scoring(y_true[u_types == t], y_pred[u_types == t]) if np.sum(
                     u_types == t) > 10 else None} for t in np.unique(list(node_types.values()))]
            target_scores = [
                {'kind': 'target', 'fold': i, 'node_type': t,
                 'score': primary_scoring(y_true[v_types == t], y_pred[v_types == t]) if np.sum(
                     v_types == t) > 10 else None} for t in np.unique(list(node_types.values()))]
            type_scores.append(source_scores), type_scores.append(target_scores)

        if plot_pr_curve:
            pr_curves.append(precision_recall_curve(y_true, est.predict_proba(train_graph, test_graph)))

    if plot_type_error:
        plot_type_scores(X, type_scores)

    if plot_pr_curve:
        _, ax = plt.subplots(figsize=(7, 4), dpi=200)
        for i, (curve, score) in enumerate(zip(pr_curves, scores)):
            plt.plot(*curve[:2], label=f'Run {i + 1} (AP ={-round(np.trapz(*curve), 3)})')
        plt.xlabel('Recall'), plt.ylabel('Precision'), plt.grid(), plt.legend()
        plt.xlim(-0.02, 1.02), plt.ylim(-0.02, 1.02)
        ax.set_aspect('equal', adjustable='box')
        plt.show()

    scores_df = pd.DataFrame(scores)
    print(scores_df.agg([np.mean, np.std]).round(3).T)
    return scores_df.to_dict(orient='list')
