#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split


# In[2]:


import mindspore as ms
from mindspore import ops


# In[3]:


def encode_onehot(labels):
    """Convert label to onehot format.

    Parameters
    ----------
    labels : numpy.array
        node labels

    Returns
    -------
    numpy.array
        onehot labels
    """
    eye = np.eye(labels.max() + 1)
    onehot_mx = eye[labels]
    return onehot_mx


# In[4]:


def tensor2onehot(labels):
    """Convert label tensor to label onehot tensor.

    Parameters
    ----------
    labels : torch.LongTensor
        node labels

    Returns
    -------
    torch.LongTensor
        onehot labels tensor

    """

    eye = ops.eye(labels.max() + 1)
    onehot_mx = eye[labels]
    return onehot_mx


# In[5]:


def preprocess(adj, features, labels, preprocess_adj=False, preprocess_feature=False, onehot_feature=False, sparse=False, device='cpu'):
    """Convert adj, features, labels from array or sparse matrix to
    torch Tensor, and normalize the input data.

    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        the adjacency matrix.
    features : scipy.sparse.csr_matrix
        node features
    labels : numpy.array
        node labels
    preprocess_adj : bool
        whether to normalize the adjacency matrix
    preprocess_feature :
        whether to normalize the feature matrix
    sparse : bool
       whether to return sparse tensor
    device : str
        'cpu' or 'cuda'
    """

    if preprocess_adj:
        adj_norm = normalize_adj(adj)

    if preprocess_feature:
        features = normalize_feature(features)

    labels = ms.Tensor(labels)
    if sparse:
        adj = sparse_mx_to_torch_sparse_tensor(adj)
        features = sparse_mx_to_torch_sparse_tensor(features)
    else:
        if onehot_feature == True:
            features = ops.eye(features.shape[0])
        else:
            features = ms.Tensor(np.array(features.todense()))
        adj = ms.Tensor(adj.todense())

    return adj, features, labels


# In[6]:


def to_tensor(adj, features, labels=None):
    """Convert adj, features, labels from array or sparse matrix to
    torch Tensor.

    Parameters
    ----------
    adj : scipy.sparse.csr_matrix
        the adjacency matrix.
    features : scipy.sparse.csr_matrix
        node features
    labels : numpy.array
        node labels
    device : str
        'cpu' or 'cuda'
    """
    if sp.issparse(adj):
        adj = sparse_mx_to_torch_sparse_tensor(adj)
    else:
        adj = ms.Tensor(adj)
    if sp.issparse(features):
        features = sparse_mx_to_torch_sparse_tensor(features)
    else:
        features = ms.Tensor(np.array(features))

    if labels is None:
        return adj, features
    else:
        labels = ms.Tensor(labels)
        return adj, features, labels


# In[7]:


def normalize_feature(mx):
    """Row-normalize sparse matrix

    Parameters
    ----------
    mx : scipy.sparse.csr_matrix
        matrix to be normalized

    Returns
    -------
    scipy.sprase.lil_matrix
        normalized matrix
    """
    if type(mx) is not sp.lil.lil_matrix:
        mx = mx.tolil()
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


# In[8]:


def normalize_adj(mx):
    """Normalize sparse adjacency matrix,
    A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    Row-normalize sparse matrix

    Parameters
    ----------
    mx : scipy.sparse.csr_matrix
        matrix to be normalized

    Returns
    -------
    scipy.sprase.lil_matrix
        normalized matrix
    """

    if type(mx) is not sp.lil.lil_matrix:
        mx = mx.tolil()
    if mx[0, 0] == 0:
        mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    return mx


# In[10]:


def normalize_sparse_tensor(adj, fill_value=1):
    """Normalize sparse tensor. Need to import torch_scatter
    """
    edge_index = adj._indices()
    edge_weight = adj._values()
    num_nodes = adj.size(0)
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index
    
   
    deg = ms.Tenosr.scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    values = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    shape = adj.shape
    
    return ms.SparseTensor(edge_index, values,shape)


# In[11]:


def add_self_loops(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    loop_index = ops.arange(0, num_nodes, dtype=ms.long,
                              device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        loop_weight = edge_weight.new_full((num_nodes, ), fill_value)
        edge_weight = ops.cat([edge_weight, loop_weight], dim=0)

    edge_index = ops.cat([edge_index, loop_index], dim=1)

    return edge_index, edge_weight


# In[12]:


def normalize_adj_tensor(adj, sparse=False):
    """Normalize adjacency tensor matrix.
    """
    if sparse:
        # TODO if this is too slow, uncomment the following code,
        # but you need to install torch_scatter
        # return normalize_sparse_tensor(adj)
        adj = to_scipy(adj)
        mx = normalize_adj(adj)
        return sparse_mx_to_torch_sparse_tensor(mx)
    else:
        mx = adj + ops.eye(adj.shape[0])
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[ops.isinf(r_inv)] = 0.
        r_mat_inv = ops.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
    return mx


# In[13]:


def degree_normalize_adj(mx):
    """Row-normalize sparse matrix"""
    mx = mx.tolil()
    if mx[0, 0] == 0:
        mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


# In[14]:


def degree_normalize_sparse_tensor(adj, fill_value=1):
    """degree_normalize_sparse_tensor.
    """
    edge_index = adj._indices()
    edge_weight = adj._values()
    num_nodes = adj.size(0)

    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value, num_nodes)

    row, col = edge_index
    
    deg = ms.Tensor.scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-1)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    values = deg_inv_sqrt[row] * edge_weight
    shape = adj.shape
    return ms.SparseTensor(edge_index, values,shape)


# In[15]:


def degree_normalize_adj_tensor(adj, sparse=True):
    """degree_normalize_adj_tensor.
    """

    
    if sparse:
        adj = to_scipy(adj)
        mx = degree_normalize_adj(adj)
        return sparse_mx_to_torch_sparse_tensor(mx)
    else:
        mx = adj + ops.eye(adj.shape[0]).to(device)
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1).flatten()
        r_inv[ops.isinf(r_inv)] = 0.
        r_mat_inv = ops.diag(r_inv)
        mx = r_mat_inv @ mx
    return mx


# In[16]:


def accuracy(output, labels):
    """Return accuracy of output compared to labels.

    Parameters
    ----------
    output : torch.Tensor
        output from model
    labels : torch.Tensor or numpy.array
        node labels

    Returns
    -------
    float
        accuracy
    """
    if labels is int:
        labels = [labels]
    if type(labels) is not ms.Tensor:
        labels = ms.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


# In[17]:


def loss_acc(output, labels, targets, avg_loss=True):
    if type(labels) is not ms.Tensor:
        labels = ms.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()[targets]
    loss = ops.nll_loss(output[targets], labels[targets],
                      reduction='mean' if avg_loss else 'none')

    if avg_loss:
        return loss, correct.sum() / len(targets)
    return loss, correct


# In[18]:


def classification_margin(output, true_label):
    """Calculate classification margin for outputs.
    `probs_true_label - probs_best_second_class`

    Parameters
    ----------
    output: torch.Tensor
        output vector (1 dimension)
    true_label: int
        true label for this node

    Returns
    -------
    list
        classification margin for this node
    """

    probs = ops.exp(output)
    probs_true_label = probs[true_label].clone()
    probs[true_label] = 0
    probs_best_second_class = probs[probs.argmax()]
    return (probs_true_label - probs_best_second_class).item()


# In[19]:


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = ms.Tensor.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = ms.Tensor.from_numpy(sparse_mx.data)
    shape = osp.Size(sparse_mx.shape)
    return ms.SparseTensor(indices, values,shape)


# In[20]:


def to_scipy(tensor):
    """Convert a dense/sparse tensor to scipy matrix"""
    if is_sparse_tensor(tensor):
        values = tensor._values()
        indices = tensor._indices()
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)
    else:
        indices = tensor.nonzero().t()
        values = tensor[indices[0], indices[1]]
        return sp.csr_matrix((values.cpu().numpy(), indices.cpu().numpy()), shape=tensor.shape)


# In[21]:


'''这个对应函数没找到
def is_sparse_tensor(tensor):
    """Check if a tensor is sparse tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        given tensor

    Returns
    -------
    bool
        whether a tensor is sparse tensor
    """
    if tensor.layout == torch.sparse_coo:
        return True
    else:
        return False
'''


# In[22]:


def get_train_val_test(nnodes, val_size=0.1, test_size=0.8, stratify=None, seed=None):
    """This setting follows nettack/mettack, where we split the nodes
    into 10% training, 10% validation and 80% testing data

    Parameters
    ----------
    nnodes : int
        number of nodes in total
    val_size : float
        size of validation set
    test_size : float
        size of test set
    stratify :
        data is expected to split in a stratified fashion. So stratify should be labels.
    seed : int or None
        random seed

    Returns
    -------
    idx_train :
        node training indices
    idx_val :
        node validation indices
    idx_test :
        node test indices
    """

    assert stratify is not None, 'stratify cannot be None!'

    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(nnodes)
    train_size = 1 - val_size - test_size
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=None,
                                                   train_size=train_size + val_size,
                                                   test_size=test_size,
                                                   stratify=stratify)

    if stratify is not None:
        stratify = stratify[idx_train_and_val]

    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=None,
                                          train_size=(
                                              train_size / (train_size + val_size)),
                                          test_size=(
                                              val_size / (train_size + val_size)),
                                          stratify=stratify)

    return idx_train, idx_val, idx_test


# In[23]:


def get_train_test(nnodes, test_size=0.8, stratify=None, seed=None):
    """This function returns training and test set without validation.
    It can be used for settings of different label rates.

    Parameters
    ----------
    nnodes : int
        number of nodes in total
    test_size : float
        size of test set
    stratify :
        data is expected to split in a stratified fashion. So stratify should be labels.
    seed : int or None
        random seed

    Returns
    -------
    idx_train :
        node training indices
    idx_test :
        node test indices
    """
    assert stratify is not None, 'stratify cannot be None!'

    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(nnodes)
    train_size = 1 - test_size
    idx_train, idx_test = train_test_split(idx, random_state=None,
                                           train_size=train_size,
                                           test_size=test_size,
                                           stratify=stratify)

    return idx_train, idx_test


# In[24]:


def get_train_val_test_gcn(labels, seed=None):
    """This setting follows gcn, where we randomly sample 20 instances for each class
    as training data, 500 instances as validation data, 1000 instances as test data.
    Note here we are not using fixed splits. When random seed changes, the splits
    will also change.

    Parameters
    ----------
    labels : numpy.array
        node labels
    seed : int or None
        random seed

    Returns
    -------
    idx_train :
        node training indices
    idx_val :
        node validation indices
    idx_test :
        node test indices
    """
    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(len(labels))
    nclass = labels.max() + 1
    idx_train = []
    idx_unlabeled = []
    for i in range(nclass):
        labels_i = idx[labels == i]
        labels_i = np.random.permutation(labels_i)
        idx_train = np.hstack((idx_train, labels_i[: 20])).astype(np.int)
        idx_unlabeled = np.hstack(
            (idx_unlabeled, labels_i[20:])).astype(np.int)

    idx_unlabeled = np.random.permutation(idx_unlabeled)
    idx_val = idx_unlabeled[: len(idx_unlabeled)//2]
    idx_test = idx_unlabeled[len(idx_unlabeled)//2:]
    return idx_train, idx_val, idx_test


# In[25]:


def get_train_test_labelrate(labels, label_rate):
    """Get train test according to given label rate.
    """
    nclass = labels.max() + 1
    train_size = int(round(len(labels) * label_rate / nclass))
    print("=== train_size = %s ===" % train_size)
    idx_train, idx_val, idx_test = get_splits_each_class(
        labels, train_size=train_size)
    return idx_train, idx_test


# In[26]:


def get_splits_each_class(labels, train_size):
    """We randomly sample n instances for class, where n = train_size.
    """
    idx = np.arange(len(labels))
    nclass = labels.max() + 1
    idx_train = []
    idx_val = []
    idx_test = []
    for i in range(nclass):
        labels_i = idx[labels == i]
        labels_i = np.random.permutation(labels_i)
        idx_train = np.hstack(
            (idx_train, labels_i[: train_size])).astype(np.int)
        idx_val = np.hstack(
            (idx_val, labels_i[train_size: 2*train_size])).astype(np.int)
        idx_test = np.hstack(
            (idx_test, labels_i[2*train_size:])).astype(np.int)

    return np.random.permutation(idx_train), np.random.permutation(idx_val),         np.random.permutation(idx_test)


# In[27]:


def unravel_index(index, array_shape):
    rows = index // array_shape[1]
    cols = index % array_shape[1]
    return rows, cols


def get_degree_squence(adj):
    try:
        return adj.sum(0)
    except:
        #ts.sum没有对应函数ops。sum试一下
        return ops.sum(adj, dim=1).to_dense()


# In[28]:


def likelihood_ratio_filter(node_pairs, modified_adjacency, original_adjacency, d_min, threshold=0.004):
    """
    Filter the input node pairs based on the likelihood ratio test proposed by Zügner et al. 2018, see
    https://dl.acm.org/citation.cfm?id=3220078. In essence, for each node pair return 1 if adding/removing the edge
    between the two nodes does not violate the unnoticeability constraint, and return 0 otherwise. Assumes unweighted
    and undirected graphs.
    """

    original_degree_sequence = original_adjacency.sum(0)
    current_degree_sequence = modified_adjacency.sum(0)

    concat_degree_sequence = ops.cat(
        (current_degree_sequence, original_degree_sequence))

    # Compute the log likelihood values of the original, modified, and combined degree sequences.
    ll_orig, alpha_orig, n_orig, sum_log_degrees_original = degree_sequence_log_likelihood(
        original_degree_sequence, d_min)
    ll_current, alpha_current, n_current, sum_log_degrees_current = degree_sequence_log_likelihood(
        current_degree_sequence, d_min)

    ll_comb, alpha_comb, n_comb, sum_log_degrees_combined = degree_sequence_log_likelihood(
        concat_degree_sequence, d_min)

    # Compute the log likelihood ratio
    current_ratio = -2 * ll_comb + 2 * (ll_orig + ll_current)

    # Compute new log likelihood values that would arise if we add/remove the edges corresponding to each node pair.
    new_lls, new_alphas, new_ns, new_sum_log_degrees = updated_log_likelihood_for_edge_changes(node_pairs,
                                                                                               modified_adjacency, d_min)

    # Combination of the original degree distribution with the distributions corresponding to each node pair.
    n_combined = n_orig + new_ns
    new_sum_log_degrees_combined = sum_log_degrees_original + new_sum_log_degrees
    alpha_combined = compute_alpha(
        n_combined, new_sum_log_degrees_combined, d_min)
    new_ll_combined = compute_log_likelihood(
        n_combined, alpha_combined, new_sum_log_degrees_combined, d_min)
    new_ratios = -2 * new_ll_combined + 2 * (new_lls + ll_orig)

    # Allowed edges are only those for which the resulting likelihood ratio measure is < than the threshold
    allowed_edges = new_ratios < threshold

    if allowed_edges.is_cuda:
        filtered_edges = node_pairs[allowed_edges.cpu(
        ).numpy().astype(np.bool)]
    else:
        filtered_edges = node_pairs[allowed_edges.numpy().astype(np.bool)]

    allowed_mask = ops.zeros(modified_adjacency.shape)
    allowed_mask[filtered_edges.T] = 1
    allowed_mask += allowed_mask.t()
    return allowed_mask, current_ratio


# In[29]:


def degree_sequence_log_likelihood(degree_sequence, d_min):
    """
    Compute the (maximum) log likelihood of the Powerlaw distribution fit on a degree distribution.
    """

    # Determine which degrees are to be considered, i.e. >= d_min.
    D_G = degree_sequence[(degree_sequence >= d_min.item())]
    try:
        sum_log_degrees = ops.log(D_G).sum()
    except:
        sum_log_degrees = np.log(D_G).sum()
    n = len(D_G)

    alpha = compute_alpha(n, sum_log_degrees, d_min)
    ll = compute_log_likelihood(n, alpha, sum_log_degrees, d_min)
    return ll, alpha, n, sum_log_degrees


# In[30]:


def updated_log_likelihood_for_edge_changes(node_pairs, adjacency_matrix, d_min):
    # For each node pair find out whether there is an edge or not in the input adjacency matrix.

    edge_entries_before = adjacency_matrix[node_pairs.T]
    degree_sequence = adjacency_matrix.sum(1)
    D_G = degree_sequence[degree_sequence >= d_min.item()]
    sum_log_degrees = ops.log(D_G).sum()
    n = len(D_G)
    deltas = -2 * edge_entries_before + 1
    d_edges_before = degree_sequence[node_pairs]

    d_edges_after = degree_sequence[node_pairs] + deltas[:, None]

    # Sum the log of the degrees after the potential changes which are >= d_min
    sum_log_degrees_after, new_n = update_sum_log_degrees(
        sum_log_degrees, n, d_edges_before, d_edges_after, d_min)
    # Updated estimates of the Powerlaw exponents
    new_alpha = compute_alpha(new_n, sum_log_degrees_after, d_min)
    # Updated log likelihood values for the Powerlaw distributions
    new_ll = compute_log_likelihood(
        new_n, new_alpha, sum_log_degrees_after, d_min)
    return new_ll, new_alpha, new_n, sum_log_degrees_after


# In[31]:


def update_sum_log_degrees(sum_log_degrees_before, n_old, d_old, d_new, d_min):
    # Find out whether the degrees before and after the change are above the threshold d_min.
    old_in_range = d_old >= d_min
    new_in_range = d_new >= d_min
    d_old_in_range = d_old * old_in_range.float()
    d_new_in_range = d_new * new_in_range.float()

    # Update the sum by subtracting the old values and then adding the updated logs of the degrees.
    sum_log_degrees_after = sum_log_degrees_before - (ops.log(ops.clamp(d_old_in_range, min=1))).sum(1)         + (ops.log(ops.clamp(d_new_in_range, min=1))).sum(1)

    # Update the number of degrees >= d_min

    new_n = n_old - (old_in_range != 0).sum(1) + (new_in_range != 0).sum(1)
    new_n = new_n.float()
    return sum_log_degrees_after, new_n


# In[34]:


def compute_alpha(n, sum_log_degrees, d_min):
    try:
        alpha = 1 + n / (sum_log_degrees - n * ops.log(d_min - 0.5))
    except:
        alpha = 1 + n / (sum_log_degrees - n * np.log(d_min - 0.5))
    return alpha


def compute_log_likelihood(n, alpha, sum_log_degrees, d_min):
    # Log likelihood under alpha
    try:
        ll = n * ops.log(alpha) + n * alpha *             ops.log(d_min) + (alpha + 1) * sum_log_degrees
    except:
        ll = n * np.log(alpha) + n * alpha * np.log(d_min) +             (alpha + 1) * sum_log_degrees

    return ll


# In[33]:


def ravel_multiple_indices(ixs, shape, reverse=False):
    """
    "Flattens" multiple 2D input indices into indices on the flattened matrix, similar to np.ravel_multi_index.
    Does the same as ravel_index but for multiple indices at once.
    Parameters
    ----------
    ixs: array of ints shape (n, 2)
        The array of n indices that will be flattened.

    shape: list or tuple of ints of length 2
        The shape of the corresponding matrix.

    Returns
    -------
    array of n ints between 0 and shape[0]*shape[1]-1
        The indices on the flattened matrix corresponding to the 2D input indices.

    """
    if reverse:
        return ixs[:, 1] * shape[1] + ixs[:, 0]

    return ixs[:, 0] * shape[1] + ixs[:, 1]


# In[35]:


def reshape_mx(mx, shape):
    indices = mx.nonzero()
    return sp.csr_matrix((mx.data, (indices[0], indices[1])), shape=shape)


# HSIC Part.
def hsic_normalized_cca(x, y, sigma=5.0, ktype='gaussian'):
    m = int(x.size()[0])
    Kxc = kernelmat(x, sigma=sigma)
    Kyc = kernelmat(y, sigma=sigma, ktype=ktype)
    epsilon = 1E-5
    K_I = ops.eye(m)
    Kxc_i = ops.inverse(Kxc + epsilon * m * K_I)
    Kyc_i = ops.inverse(Kyc + epsilon * m * K_I)
    Rx = (Kxc.mm(Kxc_i))
    Ry = (Kyc.mm(Kyc_i))
    Pxy = ops.sum(ops.mul(Rx, Ry.t()))
    return Pxy


# In[43]:


def distmat(X):
    """ distance matrix
    """
    r = ops.sum(X * X, 1)
    r = r.view([-1, 1])
    a = ops.mm(X, ops.transpose(X, 0, 1))
    D = r.expand_as(a) - 2 * a + ops.transpose(r, 0, 1).expand_as(a)
    D = ops.abs(D)
    return D


# In[44]:


def sigma_estimation(X, Y):
    """ sigma from median distance
    """
    D = distmat(ops.cat([X, Y]))
    D = D.detach()
    Itri = np.tril_indices(D.shape[0], -1)
    Tri = D[Itri]
    med = np.median(Tri)
    if med <= 0:
        med = np.mean(Tri)
    if med < 1E-2:
        med = 1E-2
    return med


# In[45]:


def kernelmat(X, sigma, ktype='gaussian'):
    """ kernel matrix baker
    """
    m = int(X.size()[0])
    H = ops.eye(m) - (1. / m) * ops.ones([m, m])
    if ktype == "gaussian":
        Dxx = distmat(X)
        if sigma:
            variance = 2. * sigma * sigma * X.size()[1]
            # kernel matrices
            Kx = ops.exp(-Dxx / variance).type(ms.Tensor)
        else:
            try:
                sx = sigma_estimation(X, X)
                Kx = ops.exp(-Dxx / (2. * sx * sx)).type(ms.Tensor)
            except RuntimeError as e:
                raise RuntimeError("Unstable sigma {} with maximum/minimum input ({},{})".format(
                    sx, ops.max(X), ops.min(X)))

    elif ktype == "linear":
        Kx = ops.mm(X, X.T).type(ms.Tensor)

    elif ktype == 'IMQ':
        Dxx = distmat(X)
        Kx = 1 * ops.rsqrt(Dxx + 1)

    Kxc = ops.mm(Kx, H)

    return Kxc


# In[46]:


def pairwise_distances(x):
    # x should be two dimensional
    instances_norm = ops.sum(x ** 2, -1).reshape((-1, 1))
    return -2 * ops.mm(x, x.t()) + instances_norm + instances_norm.t()


def GaussianKernelMatrix(x, sigma=5):
    pairwise_distances_ = pairwise_distances(x)
    return ops.exp(-pairwise_distances_ / sigma)


def HSIC(x, y, s_x=1, s_y=1):
    m, _ = x.shape  # batch size
    K = GaussianKernelMatrix(x, s_x)
    L = GaussianKernelMatrix(y, s_y)
    H = ops.eye(m) - 1.0 / m * ops.ones((m, m))
    HSIC = ops.trace(ops.mm(L, ops.mm(
        H, ops.mm(K, H)))) / ((m - 1) ** 2)
    return HSIC


# In[47]:


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data +                     self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}







