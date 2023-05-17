
import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    '''
    one head of self-attention
    '''
    def __init__(self, head_size):
        super().__init__()
        # head_size = n_embd // n_head = 64
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # 生成下三角矩阵
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape # B:batch size, T: block_size? C: n_embd?
        k = self.key(x)
        q = self.query(x)
        weight = q @ k.transpose(-2, -1) # B, T, T
        # 归一化
        weight = weight * k.shape[-1]**-0.5 # B, T, T
        # 将weight右上角全部设置为-inf
        weight = weight.masked_fill(self.tril[:T, :T]==0, float('-inf')) # B, T, T
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight) # B, T, T
        # v是目标，weight是权重，最后执行加权聚合
        v = self.value(x) # B, T, hs
        out = weight @ v # B, T, hs
        return out # B, T, hs
    
class MultiHeadAttention(nn.Module):
    '''
    multiple heads of self-attention in parallel
    '''
    # 这里有点比较有趣，n_embd ≈ head_size * num_heads！
    # 其实就是将n_embd拆分成num_heads份
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # B, T, hs*num_heads
        out = self.dropout(self.proj(out)) # B, T, n_embd
        return out # B, T, n_embd
    
class FeedForward(nn.Module):
    '''a simple linear layer followed by a non-linearity'''
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x) # B, T, n_embd

# Block整合以上所有类
class Block(nn.Module):
    '''
    Transformer blocks: communication followed by computation
    '''
    def __init__(self, n_embd, n_head):
        super().__init__()
        # head_size, hs在此处定义
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # B, T, n_embd
        x = x + self.ffwd(self.ln2(x)) # B, T, n_embd
        return x # B, T, n_embd
    
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        # 最后需要输出vocab_size长度的向量，用于做选择
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # 最好指定权重的初始化方法；apply方法会递归地将函数应用于模型的模块及子模块
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx输入尺寸是batch_size, block_size。本身其实是数值
        # 输出是batch_size, block_size, n_embd
        tok_emb = self.token_embedding_table(idx) # B, T, C
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # T, C
        # 其实我不是很理解这里tok+pos信息的操作，理论上是附带上每个单词的位置信息
        x = tok_emb + pos_emb # B, T, C
        x = self.blocks(x) # B, T, C
        x = self.ln_f(x) # B, T, C
        logits = self.lm_head(x) # B, T, vocab_size

        if targets is None:
            # 执行推理
            loss = None
        else:
            # 执行训练
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # 切片，选出的idx第二个维度范围在(1, block_size)之间
            idx_cond = idx[:, -block_size:]
            # 执行推理
            logits, _ = self(idx_cond) # B, T, vocab_size
            # 选取生成的结果中最后一个单词
            logits = logits[:, -1, :] # B, vocab_size
            probs = F.softmax(logits, dim=-1) # B, vocab_size
            # 根据权重logits选取logits的下标
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # 将生成的结果添加到idx的最后
            idx = torch.cat((idx, idx_next), dim=-1) # (B, T+1)
        return idx

if __name__ == '__main__':
    batch_size = 64
    block_size = 256 # 训练时句子长度
    max_iters = 5000
    eval_interval = 500
    learning_rate = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 200
    n_embd = 384
    n_head = 6
    n_layer = 6
    dropout = 0.2

    # 读取文本信息
    txt_file = r'input.txt'
    with open(txt_file, 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {}
    itos = {}
    for i, c in enumerate(chars):
        stoi[c] = i
        itos[i] = c
    encode = lambda s:[stoi[c] for c in s]
    decode = lambda l:''.join([itos[i] for i in l])

    # 分出训练和测试数据集
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data
    val_data = data[n:]

    # data loading
    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size, ))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y
    
    # initialize model
    model = GPTLanguageModel()
    model = model.to(device)
    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters())/1e6, 'Model parameters')

    # initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out
    
    # train process:
    for iter in range(max_iters):
        if iter%eval_interval == 0:
            # 验证损失
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        x, y = get_batch('train')
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))