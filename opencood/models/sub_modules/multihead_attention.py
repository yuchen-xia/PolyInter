import torch.nn as nn
import torch
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear layers for query, key, and value
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        # self.v_linear = nn.Linear(embed_dim, embed_dim)
        
        # Output linear transformation
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor for dot product
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]).to("cuda"))
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = value
        
        # Split into multiple heads (batch_size, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        V = value.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        # Scaled dot-product attention
        attention_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Attention output
        attention_output = torch.matmul(attention_weights, V)
        
        # Concatenate heads and pass through final linear layer
        attention_output = attention_output.squeeze(1)
        # output = self.out_linear(attention_output)
        
        return attention_output, attention_weights