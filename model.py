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

# Step 11 - update_node_features
def update_node_features(node_features, aggregated, update_fn):
    # TODO: Implement update_node_features to fuse each node's current state with its aggregated...
    return update_fn(node_features,aggregated)
    pass

# Step 12 - message_passing_layer
def message_passing_layer(node_features, src, dst, message_fn, update_fn, aggr='sum', edge_attr=None):
    """Run one full Gilmer MPNN step: message, aggregate, and update.

    Args:
        node_features: FloatTensor of shape (N, F).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        message_fn: callable(src_feats, dst_feats[, edge_attr]) -> messages (E, M).
        update_fn: callable(node_features, aggregated) -> updated (N, H).
        aggr: str in {'sum', 'mean', 'max'}.
        edge_attr: optional FloatTensor of shape (E, Fe).

    Returns:
        updated_features: FloatTensor of shape (N, H).
    """
    # TODO: compose message, aggregate, and update into one MPNN step
    mes=compute_messages(node_features,src,dst,message_fn,edge_attr)
    mes=aggregate_messages(mes,dst,len(node_features),aggr)
    return update_node_features(node_features,mes,update_fn)
    
    pass

# Step 13 - stack_message_passing_layers
def stack_message_passing_layers(node_features, src, dst, layers, edge_attr=None):
    """Apply a sequence of message-passing layer callables to produce deep node embeddings.

    Args:
        node_features: FloatTensor of shape (N, F).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        layers: list of callables, each
            layer(node_features, src, dst, edge_attr=None) -> Tensor (N, H_i).
        edge_attr: optional FloatTensor of shape (E, Fe).

    Returns:
        embeddings: FloatTensor of shape (N, H), final layer output.
        all_layer_outputs: list of FloatTensors, one per layer (N, H_i).
    """
    # TODO: Apply a sequence of MP layer callables; return final + intermediates
    h=node_features
    s=src
    d=dst
    out=[]
    for layer in layers:
        h=layer(h,s,d,edge_attr)
        out.append(h)
    return h,out

    
    pass

# Step 14 - gcn_renormalize_adjacency
def gcn_renormalize_adjacency(src, dst, num_nodes):
    """Apply Kipf-Welling renormalization: self-loops then symmetric norm.

    Args:
        src: LongTensor [E] source node indices.
        dst: LongTensor [E] destination node indices.
        num_nodes: int, number of nodes N.

    Returns:
        src_hat: LongTensor [E + N] sources after self-loops.
        dst_hat: LongTensor [E + N] destinations after self-loops.
        norm_weight: FloatTensor [E + N] symmetrically normalized weights.
    """
    # TODO: add self-loops then symmetrically normalize the adjacency...
    src,dst=add_self_loops(src,dst,num_nodes)
    norm_weight=symmetric_normalize_edge_weights(src,dst,num_nodes)
    return src,dst,norm_weight
    
    pass

# Step 15 - gcn_linear_transform
def gcn_linear_transform(node_features, weight, bias=None):
    """Apply the GCN linear feature transform X @ W (+ bias).

    Args:
        node_features: FloatTensor of shape (N, Fin).
        weight: FloatTensor of shape (Fin, Fout).
        bias: optional FloatTensor of shape (Fout).

    Returns:
        FloatTensor of shape (N, Fout).
    """
    # TODO: compute the matrix product and optionally add a bias vector
    if bias is not None:
        product= torch.matmul(node_features,weight)
        bias=torch.tensor(bias,dtype=torch.float32)
        return product+bias
    else :
        product=torch.matmul(node_features,weight)
        return product
    
    
    pass

# Step 16 - gcn_layer_forward
def gcn_layer_forward(node_features, src, dst, weight, bias=None, num_nodes=None, activation=None):
    """Forward pass of one GCN layer: renormalize, transform, propagate.

    Args:
        node_features: FloatTensor of shape (N, Fin).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        weight: FloatTensor of shape (Fin, Fout).
        bias: optional FloatTensor of shape (Fout,).
        num_nodes: optional int N; defaults to node_features.shape[0].
        activation: optional callable applied to the output.

    Returns:
        FloatTensor of shape (N, Fout).
    """
    # TODO: Forward pass of one GCN layer: renormalize, transform, propagate...
    N = num_nodes if num_nodes is not None else node_features.shape[0]

    src_hat, dst_hat, norm_weight = gcn_renormalize_adjacency(src, dst, N)

    h = gcn_linear_transform(node_features, weight)  # (N, Fout)

    h_src = gather_source_node_features(h, src_hat)  # (E+N, Fout)
    h_src = h_src * norm_weight.unsqueeze(-1)

    out = scatter_sum_to_nodes(h_src, dst_hat, N)  # (N, Fout)

    if bias is not None:
        out = out + bias

    if activation is not None:
        out = activation(out)

    return out
    pass

# Step 17 - init_gcn_parameters
def init_gcn_parameters(in_dim, out_dim, with_bias=True, seed=None):
    # TODO: Initialize GCN weight (and optional bias) with Glorot-style uniform...
    
    
    if seed is not None:
        torch.manual_seed(seed)
    a=torch.sqrt(torch.tensor(6.0/(in_dim+out_dim),dtype=torch.float32))
    weight=torch.empty(in_dim,out_dim,dtype=torch.float32).uniform_(-a,a)
    
    params={'weight':weight}
    if with_bias:
         params['bias'] = torch.zeros(out_dim, dtype=torch.float32)
    return params
    
    pass

# Step 18 - gcn_stack_forward
def gcn_stack_forward(node_features, src, dst, param_list, activations=None, num_nodes=None):
    """Run a stack of GCN layers to produce deep node embeddings.

    Args:
        node_features: FloatTensor of shape (N, F0).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        param_list: list of dicts, each with 'weight' (Fin, Fout) and optional 'bias' (Fout,).
        activations: optional list of callables or None, one per layer.
        num_nodes: optional int N; defaults to node_features.shape[0].

    Returns:
        embeddings: FloatTensor of shape (N, FL), the final layer output.
        all_layer_outputs: list of FloatTensor outputs after each layer.
    """
    # TODO: Run a stack of GCN layers to produce deep node embeddings
    all_layer_outputs=[]
    if num_nodes is None:
        num_nodes = node_features.shape[0]
    if activations is None:
        activations = [None] * len(param_list)
    for param,activation in zip(param_list,activations):
        bias = param.get('bias', None)
        node_features = gcn_layer_forward(
            node_features,
            src,
            dst,
            param['weight'],
            bias=bias,
            num_nodes=num_nodes,
            activation=activation,
        )
        all_layer_outputs.append(node_features)
    
        
    return node_features,all_layer_outputs
    pass

# Step 19 - gat_attention_logits
def gat_attention_logits(node_features, src, dst, attn_src, attn_dst, weight):
    """Compute unnormalized GAT attention logits and transformed features.

    Args:
        node_features: FloatTensor of shape (N, Fin).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        attn_src: FloatTensor of shape (Fout,) source attention vector.
        attn_dst: FloatTensor of shape (Fout,) destination attention vector.
        weight: FloatTensor of shape (Fin, Fout) shared linear transform.

    Returns:
        logits: FloatTensor of shape (E,) unnormalized attention scores.
        transformed: FloatTensor of shape (N, Fout) linearly transformed nodes.
    """
    # TODO: return per-edge LeakyReLU attention logits and transformed features
    h=gcn_linear_transform(node_features,weight)
    
    result=(h[src]*attn_src).sum(dim=-1)+(h[dst]*attn_dst).sum(dim=-1)


    
    logits=torch.nn.functional.leaky_relu(result,negative_slope=0.2)
    return logits,h
    pass

# Step 20 - gat_masked_neighbor_softmax
def gat_masked_neighbor_softmax(logits, dst, num_nodes):
    """Numerically stable softmax of attention logits over each dest node's neighbors.

    Args:
        logits: FloatTensor of shape (E,) with one unnormalized attention logit per edge.
        dst: LongTensor of shape (E,) with destination node index for each edge.
        num_nodes: int, number of nodes N in the graph.

    Returns:
        FloatTensor of shape (E,) with attention coefficients that sum to 1 over
        each destination's incoming edges.
    """
    # TODO: Numerically stable softmax of attention logits over each dest node's neighbors
    # 1. 把 (E,) 抬成 (E, 1)，好让 scatter 工具按节点维度处理
    logits_2d = logits.unsqueeze(-1)                     # (E, 1)

    # 2. 每个目标节点上的 max：m_j，形状 (N, 1)
    max_per_node = scatter_max_to_nodes(logits_2d, dst, num_nodes)   # (N, 1)

    # 3. 用 dst 把 m_j 广播回每条边，减去后做 exp
    shifted = logits_2d - max_per_node[dst]              # (E, 1)
    exp_shifted = torch.exp(shifted)                     # (E, 1)

    # 4. 每个目标节点的分母 Z_j = sum_k exp(e_kj - m_j)，形状 (N, 1)
    denom = scatter_sum_to_nodes(exp_shifted, dst, num_nodes)        # (N, 1)

    # 5. 每条边除以自己目标节点的分母
    alpha = exp_shifted / denom[dst]                     # (E, 1)

    # 6. 拍回 (E,)
    return alpha.squeeze(-1)
    pass

# Step 21 - gat_head_forward
def gat_head_forward(node_features, src, dst, weight, attn_src, attn_dst, bias=None, num_nodes=None, activation=None):
    """Forward pass of a single GAT attention head.

    Args:
        node_features: FloatTensor of shape (N, Fin).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        weight: FloatTensor of shape (Fin, Fout) shared linear transform.
        attn_src: FloatTensor of shape (Fout,) source attention vector.
        attn_dst: FloatTensor of shape (Fout,) destination attention vector.
        bias: optional FloatTensor of shape (Fout,).
        num_nodes: optional int N; inferred from node_features if None.
        activation: optional callable applied to the head output.

    Returns:
        head_out: FloatTensor of shape (N, Fout).
        attn_coeffs: FloatTensor of shape (E,) attention coefficients.
    """
    # TODO: Forward pass of a single GAT attention head: transform, coeffs, aggregate...
    if num_nodes is None:
        num_nodes=len(node_features)
    
    logit,transformed=gat_attention_logits(node_features,src,dst,attn_src,attn_dst,weight)
    attn_coeffs=gat_masked_neighbor_softmax(logit,dst,num_nodes)
    h_src= gather_source_node_features(transformed,src)    
    weight_mes=h_src*attn_coeffs.unsqueeze(-1)
    head_out=scatter_sum_to_nodes(weight_mes,dst,num_nodes)
    if bias is not None:
        head_out=head_out+bias
    if activation is not None:
        head_out=activation(head_out)
    return head_out,attn_coeffs
    
    pass

# Step 22 - merge_gat_heads
def merge_gat_heads(head_outputs, mode='concat'):
    # TODO: Merge multi-head GAT outputs into one node-feature tensor.
    if isinstance(head_outputs,(list,tuple)):
        if mode=="concat":
            return torch.cat(head_outputs,dim=-1)
        else:
            head_outputs=torch.stack(head_outputs,dim=0)
            return head_outputs.mean(dim=0)
    else:
        h,n,f=head_outputs.shape
        if mode=='concat':
            return head_outputs.permute(1,0,2).reshape(n,h*f)
        else:
            return head_outputs.mean(dim=0)
    pass

# Step 23 - gat_layer_forward
def gat_layer_forward(node_features, src, dst, head_params, merge_mode='concat', num_nodes=None, activation=None):
    """Multi-head GAT layer: run each head, merge, optional activation.

    Args:
        node_features: FloatTensor (N, Fin).
        src: LongTensor (E,) source indices.
        dst: LongTensor (E,) destination indices.
        head_params: list of dicts with keys weight, attn_src, attn_dst,
            and optional bias for each head.
        merge_mode: 'concat' or 'mean'.
        num_nodes: optional int N; inferred from node_features if None.
        activation: optional callable applied after merging heads.

    Returns:
        out: FloatTensor (N, F_merged).
        all_attn: list of FloatTensor (E,) attention coeffs per head.
    """
    # TODO: run each head, merge outputs, apply optional nonlinearity...
    all_attn=[]
    head_out=[]
    for params in head_params:
        bias=params.get('bias',None)
        head,coeffs=gat_head_forward(node_features,src,dst,params['weight'],params['attn_src'],params['attn_dst'],bias,num_nodes)
        all_attn.append(coeffs)
        head_out.append(head)

    out=merge_gat_heads(head_out,merge_mode)
    if activation is not None:
        out =activation(out)

    return out,all_attn

    
    
    
    
    
    
    pass

# Step 24 - init_gat_parameters
def init_gat_parameters(in_dim, out_dim, num_heads=1, with_bias=True, seed=None):
    # TODO: Initialize multi-head GAT parameters with Glorot-style initialization.
    
    torch.manual_seed(seed)
    out=[]
    for num in range(num_heads):
        dict={}
        a=torch.sqrt(torch.tensor(6/(in_dim+out_dim)))
        weight=torch.empty(in_dim,out_dim).uniform_(-a,a)
        weight.requires_grad_(True)
        a=torch.sqrt(torch.tensor(6/(1+out_dim)))
        attn_src=torch.empty(out_dim).uniform_(-a,a)
        attn_src.requires_grad_(True)
        attn_dst=torch.empty(out_dim).uniform_(-a,a)
        attn_dst.requires_grad_(True)
        dict["weight"]=weight
        dict['attn_src']=attn_src
        dict['attn_dst']=attn_dst

        bias=None
        if with_bias:
            bias=torch.zeros(out_dim,requires_grad=True)
            dict['bias']=bias
        out.append(dict)
    return out
    pass

# Step 25 - gat_stack_forward
def gat_stack_forward(node_features, src, dst, layer_param_list, merge_modes=None, activations=None, num_nodes=None):
    """Run a stack of multi-head GAT layers.

    Args:
        node_features: FloatTensor (N, F0).
        src: LongTensor (E,) source indices.
        dst: LongTensor (E,) destination indices.
        layer_param_list: list of length L; each entry is a head_params list
            for gat_layer_forward.
        merge_modes: optional list of L merge mode strings ('concat' or 'mean').
            Defaults to 'concat' for every layer.
        activations: optional list of L callables or None. Defaults to no
            activation for every layer.
        num_nodes: optional int N; inferred from node_features if None.

    Returns:
        embeddings: FloatTensor (N, FL) final layer output.
        all_layer_outputs: list of L FloatTensors, the output after each layer.
    """
    # TODO: Run a stack of multi-head GAT layers for deep node embeddings.
    if merge_modes is None:
        merge_modes=[None for i in range(len(layer_param_list))]
    if activations is None:
        activations=[None for i in range(len(layer_param_list))]
    num_nodes=len(node_features)
    all_layer_outputs=[]
    for param,merge,activation in zip(layer_param_list,merge_modes,activations):
        node_features,_=gat_layer_forward(node_features,src,dst,param,merge,num_nodes,activation)
        all_layer_outputs.append(node_features)
    return node_features,all_layer_outputs

    
    
    pass

# Step 26 - global_mean_pool
def global_mean_pool(node_features, batch_index, num_graphs=None):
    """Globally mean-pool node features into one graph-level vector per graph.

    Args:
        node_features: FloatTensor of shape (N, F) with one feature row per node.
        batch_index: LongTensor of shape (N,) mapping each node to a graph id in
            {0, ..., B-1}.
        num_graphs: Optional int B. If None, inferred as batch_index.max() + 1.

    Returns:
        FloatTensor of shape (B, F); row b is the mean of node features with
        batch_index == b.
    """
    # TODO: Mean-pool node features into one graph-level vector per graph...
    if num_graphs is None:
        num_graphs=batch_index.max()+1
    result=scatter_mean_to_nodes(node_features,batch_index,num_graphs)
    return result



    pass

# Step 27 - global_sum_pool
def global_sum_pool(node_features, batch_index, num_graphs=None):
    """Globally sum-pool node features into one graph-level vector per graph.

    Args:
        node_features: FloatTensor of shape (N, F) with one row per node.
        batch_index: LongTensor of shape (N,) mapping each node to a graph id
            in 0 .. B-1.
        num_graphs: optional int B. If None, inferred as max(batch_index) + 1.

    Returns:
        FloatTensor of shape (B, F); row g is the sum of node features with
        batch_index == g.
    """
    # TODO: sum-pool node features into one graph-level vector per graph
    if num_graphs is None:
        num_graphs=batch_index.max()+1
    return scatter_sum_to_nodes(node_features,batch_index,num_graphs)
    pass

# Step 28 - global_max_pool
def global_max_pool(node_features, batch_index, num_graphs=None):
    # TODO: Globally max-pool node features into one graph-level vector per graph.
    
    if num_graphs is None:
        num_graphs=batch_index.max()+1
    return scatter_max_to_nodes(node_features,batch_index,num_graphs)
    
    pass

# Step 29 - global_mean_max_pool
def global_mean_max_pool(node_features, batch_index, num_graphs=None):
    """Concatenate global mean and max pooled features into a 2F-dim graph vector.

    Args:
        node_features: FloatTensor of shape (N, F).
        batch_index: LongTensor of shape (N,) with graph ids in {0, ..., B-1}.
        num_graphs: Optional int B. If None, inferred as batch_index.max() + 1.

    Returns:
        FloatTensor of shape (B, 2F); each row is [mean_pool || max_pool].
    """
    # TODO: Concatenate global mean and max pooled features into a 2F-dim vector...
    mean_=global_mean_pool(node_features,batch_index,num_graphs)
    max_=global_max_pool(node_features,batch_index,num_graphs)
    return torch.cat([mean_,max_],dim=-1)
    
    
    pass

# Step 30 - node_classification_head
def node_classification_head(node_embeddings, weight, bias=None):
    # TODO: Map node embeddings to per-node class logits via a linear head...
   
    result=torch.matmul(node_embeddings,weight)
    if bias is None:
        return result
    return result+bias
   
    pass

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

