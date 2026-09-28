"""
Message-Passing GNNs from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - edges_to_coo
def edges_to_coo(edge_list, num_nodes=None):
    # TODO: Convert a list of (src, dst) edge pairs into COO-format src/dst tensors.
    if isinstance(edge_list,torch.Tensor):
        edges=edge_list.long().reshape(-1,2)
    else:
        if len(edge_list)==0:
            edges=torch.zeros((0,2),dtype=torch.long)
        else:
            edges=torch.tensor(edge_list,dtype=torch.long).reshape(-1,2)
    if edges.numel()==0:
        src=torch.zeros(0,dtype=torch.long)
        dst=torch.zeros(0,dtype=torch.long)
    else:
        src=edges[:,0]
        dst=edges[:,1]
    if num_nodes is None:
        if edges.numel()==0:
            num_nodes=0
        else:
            num_nodes=int(edges.max().item())+1
    else:
        num_nodes=int(num_nodes)

    return src,dst,num_nodes

# Step 2 - add_self_loops
def add_self_loops(src, dst, num_nodes):
    """Append self-loop edges (i, i) for every node to COO edge indices.

    Args:
        src: LongTensor [E] source node indices.
        dst: LongTensor [E] destination node indices.
        num_nodes: int, number of nodes in the graph.

    Returns:
        src_out: LongTensor [E + num_nodes]
        dst_out: LongTensor [E + num_nodes]
    """
    # TODO: Append self-loop edges (i, i) for every node to the COO tensors
    num1=torch.arange(num_nodes,dtype=src.dtype,device=src.device)
    num2=torch.arange(num_nodes,dtype=dst.dtype,device=dst.device)
    src_out=torch.cat([src,num1],dim=0)
    dst_out=torch.cat([dst,num2],dim=0)
    return src_out,dst_out

# Step 3 - compute_node_degrees
def compute_node_degrees(src, dst, num_nodes, edge_weight=None):
    """Compute per-node in-degrees (optionally weighted) from COO edges.

    Args:
        src (LongTensor): Source node indices of shape [E].
        dst (LongTensor): Destination node indices of shape [E].
        num_nodes (int): Number of nodes N.
        edge_weight (FloatTensor, optional): Per-edge weights of shape [E].

    Returns:
        FloatTensor: In-degrees of shape [N].
    """
    # TODO: Compute per-node in-degrees by scattering onto destination nodes
    E = src.shape[0]

    # 1. 无权图：每条边贡献 1.0，长度是 E，不是 num_nodes
    if edge_weight is None:
        edge_weight = torch.ones(E, dtype=torch.float32, device=src.device)
    else:
        edge_weight = torch.as_tensor(edge_weight, dtype=torch.float32, device=src.device)

    # 2. 初始化累加器，孤立节点自动为 0
    degrees = torch.zeros(num_nodes, dtype=torch.float32, device=src.device)

    # 3. 按 dst 索引散射累加
    degrees.scatter_add_(0, dst.long(), edge_weight)

    return degrees
    
    
    
    pass

# Step 4 - symmetric_normalize_edge_weights
def symmetric_normalize_edge_weights(src, dst, num_nodes, edge_weight=None):
    """Compute symmetrically normalized edge weights w_ij / sqrt(d_i * d_j).

    Args:
        src (LongTensor): Source node indices of shape [E].
        dst (LongTensor): Destination node indices of shape [E].
        num_nodes (int): Number of nodes N.
        edge_weight (FloatTensor, optional): Per-edge weights of shape [E].
            Defaults to all ones (float32) when None.

    Returns:
        FloatTensor: Symmetrically normalized weights of shape [E].
    """
    # TODO: Compute symmetrically normalized edge weights for GCN-style propagation.
    degree=compute_node_degrees(src,dst,num_nodes,edge_weight)
    if edge_weight is None:
        edge_weight=torch.ones(len(src),dtype=torch.float32)
    i=0
    for s,d in zip(src,dst):
        if degree[s]==0 or degree[d]==0:
            edge_weight[i]=0
        else:
            edge_weight[i]=edge_weight[i]/torch.sqrt(degree[s]*degree[d])
        i=i+1
    return edge_weight

    
    
    
    pass

# Step 5 - gather_source_node_features
def gather_source_node_features(node_features, src):
    # TODO: Return edge-aligned source feature rows (E, F) from node_features.
    
     return node_features[src]

# Step 6 - scatter_sum_to_nodes
def scatter_sum_to_nodes(edge_features, dst, num_nodes):
    """Scatter-sum edge features onto destination nodes to produce per-node aggregated vectors.

    Args:
        edge_features: FloatTensor of shape (E, F) with one feature row per edge.
        dst: LongTensor of shape (E,) with destination node index for each edge.
        num_nodes: int, number of nodes N in the graph.

    Returns:
        FloatTensor of shape (N, F); row j is the sum of edge features with dst == j.
    """
    # TODO: Scatter-sum edge features onto destination nodes to produce per-node vectors
    f=edge_features.shape[1]
    result=torch.zeros(num_nodes,f,dtype=torch.float32)
    result.index_add_(0,dst,edge_features)
    
    return result
    pass

# Step 7 - scatter_mean_to_nodes
def scatter_mean_to_nodes(edge_features, dst, num_nodes):
    # TODO: Scatter-mean edge features onto destination nodes (sum then divide by in-degree).
    degree=compute_node_degrees(dst,dst,num_nodes)
    result=scatter_sum_to_nodes(edge_features,dst,num_nodes)
    degree=degree.clamp(min=1.0)
    degree=degree.unsqueeze(-1)
    return result/degree
   
    
    pass

# Step 8 - scatter_max_to_nodes
def scatter_max_to_nodes(edge_features, dst, num_nodes):
    # TODO: Scatter-max edge features onto destination nodes (elementwise max).
    f=edge_features.shape[1]
    result=torch.full((num_nodes,f),float('-inf'),dtype=edge_features.dtype)
    dst=dst.unsqueeze(-1).expand_as(edge_features) 
    result=result.scatter_reduce(0,dst,edge_features,reduce="amax", include_self=True)
    
    return result
    
    pass

# Step 9 - compute_messages
def compute_messages(node_features, src, dst, message_fn, edge_attr=None):
    """Build per-edge messages via gather + message_fn.

    Args:
        node_features: FloatTensor of shape (N, F).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        message_fn: callable(src_feats, dst_feats[, edge_attr]) -> messages.
        edge_attr: optional FloatTensor of shape (E, Fe).

    Returns:
        messages: FloatTensor of shape (E, M).
    """
    # TODO: Build per-edge messages by gathering features and applying message_fn
    src=gather_source_node_features(node_features,src)
    dst=gather_source_node_features(node_features,dst)
    if edge_attr is None:
        return message_fn(src,dst)
    else:
        return message_fn(src,dst,edge_attr)
    pass

# Step 10 - aggregate_messages
def aggregate_messages(messages, dst, num_nodes, aggr='sum'):
    """Aggregate edge messages onto destination nodes using sum, mean, or max.

    Args:
        messages: FloatTensor of shape (E, M) with one message vector per edge.
        dst: LongTensor of shape (E,) with destination node index for each edge.
        num_nodes: int, number of nodes N in the graph.
        aggr: str in {'sum', 'mean', 'max'} selecting the reduction.

    Returns:
        FloatTensor of shape (N, M); row j is the aggregated message for node j.
    """
    # TODO: Aggregate edge messages onto destination nodes via sum/mean/max...
    if aggr=='sum':
       result= scatter_sum_to_nodes(messages,dst,num_nodes)
    elif aggr=='mean':
        result=scatter_mean_to_nodes(messages,dst,num_nodes)
    elif aggr=='max':
        result=scatter_max_to_nodes(messages,dst,num_nodes)
    return result
    
    
    
    pass

# Step 11 - update_node_features (not yet solved)
# TODO: implement

# Step 12 - message_passing_layer (not yet solved)
# TODO: implement

# Step 13 - stack_message_passing_layers (not yet solved)
# TODO: implement

# Step 14 - gcn_renormalize_adjacency (not yet solved)
# TODO: implement

# Step 15 - gcn_linear_transform (not yet solved)
# TODO: implement

# Step 16 - gcn_layer_forward (not yet solved)
# TODO: implement

# Step 17 - init_gcn_parameters (not yet solved)
# TODO: implement

# Step 18 - gcn_stack_forward (not yet solved)
# TODO: implement

# Step 19 - gat_attention_logits (not yet solved)
# TODO: implement

# Step 20 - gat_masked_neighbor_softmax (not yet solved)
# TODO: implement

# Step 21 - gat_head_forward (not yet solved)
# TODO: implement

# Step 22 - merge_gat_heads (not yet solved)
# TODO: implement

# Step 23 - gat_layer_forward (not yet solved)
# TODO: implement

# Step 24 - init_gat_parameters (not yet solved)
# TODO: implement

# Step 25 - gat_stack_forward (not yet solved)
# TODO: implement

# Step 26 - global_mean_pool (not yet solved)
# TODO: implement

# Step 27 - global_sum_pool (not yet solved)
# TODO: implement

# Step 28 - global_max_pool (not yet solved)
# TODO: implement

# Step 29 - global_mean_max_pool (not yet solved)
# TODO: implement

# Step 30 - node_classification_head (not yet solved)
# TODO: implement

# Step 31 - graph_regression_head (not yet solved)
# TODO: implement

# Step 32 - generate_sbm_graph (not yet solved)
# TODO: implement

# Step 33 - build_node_classification_dataset (not yet solved)
# TODO: implement

# Step 34 - generate_molecule_like_graph (not yet solved)
# TODO: implement

# Step 35 - build_graph_regression_dataset (not yet solved)
# TODO: implement

# Step 36 - collate_graph_batch (not yet solved)
# TODO: implement

# Step 37 - cross_entropy_loss (not yet solved)
# TODO: implement

# Step 38 - mse_loss (not yet solved)
# TODO: implement

# Step 39 - accuracy_metric (not yet solved)
# TODO: implement

# Step 40 - mae_metric (not yet solved)
# TODO: implement

# Step 41 - gnn_train_step (not yet solved)
# TODO: implement

# Step 42 - train_node_classifier (not yet solved)
# TODO: implement

# Step 43 - train_graph_regressor (not yet solved)
# TODO: implement

# Step 44 - representation_similarity (not yet solved)
# TODO: implement

# Step 45 - oversmoothing_diagnostic (not yet solved)
# TODO: implement

# Step 46 - mpnn_gnn_experiment (not yet solved)
# TODO: implement

