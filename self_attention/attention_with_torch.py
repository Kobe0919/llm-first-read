import torch.nn as nn
import torch

class SelfAttention(nn.Module):  # There're lots of benefits inheriting from nn.Module
    def __init__(self, d_in, d_out):  # Self: the instance being created
        super().__init__()
        self.d_out= d_out  # d_out is instance variable
                            # d_in is local variable in __init__() method
        # self.W_q= nn.Parameter(torch.rand(d_in, d_out))  # nn.Parameter: automatically registered as model parameter
        # self.W_k= nn.Parameter(torch.rand(d_in, d_out))
        # self.W_v= nn.Parameter(torch.rand(d_in, d_out))

        '''
        In Deep Learning, linear transformation is represented as y = Wx + b
        In PyTorch, nn.Linear is a module that compute y = x*W^T + b ((Wx)^T = x^T*W^T)
        So, It create A with shape (out_features, in_features)
        '''

        self.W_q = nn.Linear(d_in, d_out, bias=False)  
        self.W_k = nn.Linear(d_in, d_out, bias=False)
        self.W_v = nn.Linear(d_in, d_out, bias=False)

    
    def forward(self, x):  # The only method needed to be defined
                         # Builds the computation graph dynamically
        # queries= x @ self.W_q
        # keys= x @ self.W_k
        # values= x @ self.W_v
        
        '''
        Originally, only the method can be called,
        But, python introduces __call__() so that we can call the instance directly  
        '''

        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)
        
        atte_scores = queries @ keys.T
        attn_weights = torch.softmax(
            atte_scores / (self.d_out ** 0.5), dim=-1
        )
        
        context_vec = attn_weights @ values
        return context_vec