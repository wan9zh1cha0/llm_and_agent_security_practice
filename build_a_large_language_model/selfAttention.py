import torch
import torch.nn as nn

# self-attention refers to each pair of embeddings in the input vector paying attention to their relationship
# simplified self-attention
inputs = torch.tensor(
    [[0.43, 0.15, 0.89],
     [0.55, 0.87, 0.66],
     [0.57, 0.85, 0.64],
     [0.22, 0.58, 0.33],
     [0.77, 0.25, 0.10],
     [0.05, 0.80, 0.55]]
) # 6 embeddings, each embedding has 3 dimensions
attn_scores = inputs @ inputs.T # compute the dot product between each pair of embeddings, in order to get the distance between embeddings, therefore affecting the attention value
attn_weights = torch.softmax(attn_scores, dim=-1) # dim=-1 means normalize along the last dimension
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)

# self-attention with trainable weights
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x): # x is the input tensor of this layer
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1) # Adjust the weights to ensure the training effect
        context_vec = attn_weights @ values
        return(context_vec)


torch.manual_seed(123) # set random seed for reproducibility
d_in = inputs.shape[1] # number of columns, i.e., length of embedding vector
d_out = 2 # usually set equal to d_in
sa_v1 = SelfAttention_v1(d_in, d_out) #create object
print(sa_v1(inputs)) # pytorch grammer sugar, __call__ calls forward function

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias) # this module has optimized weight initialization
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    
    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vecs = attn_weights @ values
        return context_vecs

torch.manual_seed(123)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))

# v1, v2 contrast test

# causal attention (masked attention)
keys = sa_v2.W_key(inputs) # 2*6
queries = sa_v2.W_query(inputs)
attn_scores = queries @ keys.T # 6*6
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1) # to avoid large differences in softmax values due to high dimensionality, which would lead to small gradients for large value elements and affect training performance
context_length = attn_scores.shape[0] # =6
mask_simple = torch.tril(torch.ones(context_length, context_length))
masked_simple = attn_weights * mask_simple
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)

#print("TESTING\nFIRST test:")
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
#print("attn_scores", attn_scores)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
#print("masked attn_scores", masked)
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=-1)
#print("causal attn_weights", attn_weights)

# print(masked_simple_norm - attn_weights)

# introduce dropout to avoid overfitting, here dropout is applied before multiplying values
#torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
#attn_weights = dropout(attn_weights)
#print("dropout attn_weights", attn_weights)
values = sa_v2.W_value(inputs)
#print("qkv\n", queries, keys, values)
#print("values", values)
print("context vector", dropout(attn_weights) @ values)

#print("CLASS test:")
class CasualAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.d_in = d_in
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias) # bias refers to whether to add a bias term in the linear layer
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) 

    def forward(self, x):
        b, num_tokens, d_in = x.shape # batched input tensor has three dimensions, number of input tensors per batch, number of embeddings per input, number of dimensions per embedding
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        #print("qkv\n", queries, keys, values)
        attn_scores = queries @ keys.transpose(1,2) # only transpose each input tensor internally
        #print("attn_scores", attn_scores)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        #print("masked attn_scores", attn_scores)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        #print("causal attn_weights", attn_weights)
        attn_weights = self.dropout(attn_weights)
        #print("dropout attn_weights", attn_weights)
        #print("values", values)
        context_vec = attn_weights @ values
        return context_vec

batch = torch.stack((inputs, inputs), dim=0)
torch.manual_seed(123)
ca = CasualAttention(3, 2, 6, 0.0)
context_vecs = ca(batch)
print("context vector", context_vecs)

# multi-head attention
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([CasualAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

torch.manual_seed(123)
context_length = batch.shape[1]
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, 2)
context_vecs = mha(batch)
print(context_vecs)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out % num_heads != 0"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) # add a linear layer to combine outputs from different heads
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) # same as line 77, automatically move buffers to appropriate CPU or GPU

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) # essentially equivalent to having nn.Linear produce num_heads weight matrices and concatenate them
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        #print("qkv\n", queries, keys, values)

        attn_scores = queries @ keys.transpose(2, 3)
        #print("attn_scores\n", attn_scores)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        #print("masked attn_scores\n", attn_scores)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        #print("attn_weights\n", attn_weights)
        attn_weights = self.dropout(attn_weights) # when dropout=0.0, this line still has effect (new: maybe because didn't write self last time.)
        #print("dropout\n", attn_weights)
        context_vecs = (attn_weights @ values).transpose(1, 2)

        context_vecs = context_vecs.contiguous().view(b, num_tokens, self.d_out)

        context_vecs = self.out_proj(context_vecs)
        return context_vecs
    
torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, dropout=0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)