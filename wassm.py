import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import os
import re
from collections import Counter, defaultdict
import json

class Config:
    vocab_size = 8192
    min_frequency = 2
    bpe_vocab_path = 'bpe_vocab_v1_streaming.json'

    # Model and Training Settings
    batch_size = 6
    block_size = 512
    n_embd = 384
    n_layer = 6
    n_head = 6
    dropout_rate = 0.1
    num_epochs = 1
    learning_rate = 5e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Generation and Checkpointing
    gen_top_k = 50
    gen_temperature = 1.0
    checkpoint_path = 'training_checkpoint_v1_streaming.pth'
    max_context_len = block_size * 2

class BPETokenizer:
    def __init__(self, config):
        self.config = config
        self.PAD, self.UNK, self.BOS, self.EOS = "<PAD>", "<UNK>", "<BOS>", "<EOS>"
        self.vocab = {self.PAD: 0, self.UNK: 1, self.BOS: 2, self.EOS: 3}
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.merges = {}
        self.next_id = len(self.vocab)
        self.pattern = re.compile(r'(\s+|\S+)')
    def byte_pair_encoding(self, text):
        print("Training V1 BPE tokenizer...")
        word_freqs = Counter(self.pattern.findall(text))
        all_chars = set("".join(word_freqs.keys()))
        for char in sorted(all_chars):
            if char not in self.vocab: self.vocab[char] = self.next_id; self.id_to_token[self.next_id] = char; self.next_id += 1
        splits = {word: list(word) for word in word_freqs.keys()}
        num_merges = self.config.vocab_size - len(self.vocab)
        for i in range(num_merges):
            pair_freqs = defaultdict(int)
            for word, freq in word_freqs.items():
                chars = splits[word]
                for j in range(len(chars) - 1): pair_freqs[(chars[j], chars[j+1])] += freq
            if not pair_freqs: break
            best_pair, freq = max(pair_freqs.items(), key=lambda x: x[1])
            if freq < self.config.min_frequency: break
            new_token = "".join(best_pair)
            if new_token not in self.vocab: self.vocab[new_token] = self.next_id; self.id_to_token[self.next_id] = new_token; self.next_id += 1
            self.merges[best_pair] = new_token
            for word in word_freqs:
                parts = splits[word]; j = 0
                while j < len(parts) - 1:
                    if parts[j] == best_pair[0] and parts[j+1] == best_pair[1]: parts[j] = new_token; del parts[j+1]
                    else: j += 1
        print(f"BPE training complete. Final vocab size: {len(self.vocab)}")
    def _tokenize_word(self, word):
        chars = list(word)
        while len(chars) > 1:
            found_merge = False
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i+1])
                if pair in self.merges: chars[i] = self.merges[pair]; del chars[i+1]; found_merge = True; break
            if not found_merge: break
        return chars
    def encode(self, text):
        tokens = [self.vocab[self.BOS]]
        for match in self.pattern.finditer(text):
            for token in self._tokenize_word(match.group(0)): tokens.append(self.vocab.get(token, self.vocab[self.UNK]))
        tokens.append(self.vocab[self.EOS])
        return tokens
    def decode(self, ids): return "".join([self.id_to_token.get(i, self.UNK) for i in ids])
    def save(self, path):
        serializable_merges = {"|".join(k): v for k, v in self.merges.items()}
        with open(path, 'w', encoding='utf-8') as f: json.dump({"vocab": self.vocab, "merges": serializable_merges}, f, ensure_ascii=False)
    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f: data = json.load(f)
        self.vocab = data["vocab"]; self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.merges = {tuple(k.split("|")): v for k, v in data["merges"].items()}; self.next_id = max(self.vocab.values()) + 1

class TextDataset(Dataset):
    def __init__(self, data_tensor, block_size):
        self.data, self.block_size = data_tensor, block_size
    def __len__(self): return len(self.data) - self.block_size
    def __getitem__(self, idx): return self.data[idx:idx+self.block_size], self.data[idx+1:idx+self.block_size+1]

class LinearCausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__(); assert config.n_embd % config.n_head == 0
        self.n_head, self.head_size = config.n_head, config.n_embd // config.n_head
        self.query, self.key, self.value = (nn.Linear(config.n_embd, config.n_embd, bias=False) for _ in range(3))
        self.proj = nn.Linear(config.n_embd, config.n_embd); self.dropout = nn.Dropout(config.dropout_rate)
    def feature_map(self, x): return F.elu(x) + 1.0
    def forward(self, x, state=None):
        B, T, C = x.shape
        q, k, v = self.query(x), self.key(x), self.value(x)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q_prime, k_prime = self.feature_map(q), self.feature_map(k)
        if state is None:
            prev_kv_context = torch.zeros(B, self.n_head, self.head_size, self.head_size, device=x.device)
            prev_k_sum = torch.zeros(B, self.n_head, self.head_size, device=x.device)
        else:
            prev_kv_context, prev_k_sum = state
        prev_kv_context, prev_k_sum = prev_kv_context.unsqueeze(2), prev_k_sum.unsqueeze(2)
        current_kv_context_cumulative = torch.cumsum(torch.einsum('bhsd,bhsv->bhsdv', k_prime, v), dim=2)
        current_k_sum_cumulative = torch.cumsum(k_prime, dim=2)
        kv_context_cumulative = prev_kv_context + current_kv_context_cumulative
        k_sum_cumulative = prev_k_sum + current_k_sum_cumulative
        numerator = torch.einsum('bhsd,bhsdv->bhsv', q_prime, kv_context_cumulative)
        denominator = torch.einsum('bhsd,bhsd->bhs', q_prime, k_sum_cumulative).unsqueeze(-1) + 1e-6
        y = (numerator / denominator).transpose(1, 2).contiguous().view(B, T, C)
        new_state = (kv_context_cumulative[:, :, -1], k_sum_cumulative[:, :, -1])
        return self.dropout(self.proj(y)), new_state

class HybridConvAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = (nn.LayerNorm(config.n_embd) for _ in range(3))
        self.kernel_size = 3
        self.conv = nn.Conv1d(config.n_embd, 2*config.n_embd, self.kernel_size, padding=0)
        self.activation = nn.GLU(dim=1)
        self.attn = LinearCausalAttention(config)
        self.mlp = nn.Sequential(nn.Linear(config.n_embd, 4 * config.n_embd), nn.GELU(), nn.Linear(4 * config.n_embd, config.n_embd), nn.Dropout(config.dropout_rate))
        self.dropout = nn.Dropout(config.dropout_rate)
    def forward(self, x, layer_cache=None):
        attn_state, conv_state = (None, None) if layer_cache is None else layer_cache
        res = x
        x_norm = self.ln1(x).permute(0, 2, 1)
        if conv_state is None: x_padded = F.pad(x_norm, (self.kernel_size - 1, 0))
        else: x_padded = torch.cat([conv_state, x_norm], dim=2)
        new_conv_state = x_padded[:, :, -(self.kernel_size - 1):]
        x_act = self.activation(self.conv(x_padded)).permute(0, 2, 1)
        x = res + self.dropout(x_act)
        attn_output, new_attn_state = self.attn(self.ln2(x), state=attn_state)
        x = x + attn_output
        x = x + self.mlp(self.ln3(x))
        return x, (new_attn_state, new_conv_state)

class LanguageModel(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__(); self.config = config
        self.token_embedding = nn.Embedding(vocab_size, config.n_embd)
        self.pos_embedding = nn.Embedding(config.max_context_len, config.n_embd)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.blocks = nn.ModuleList([HybridConvAttentionBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight
    def forward(self, idx, targets=None, cache=None, past_len=0):
        B, T = idx.shape; assert past_len + T <= self.config.max_context_len
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(past_len, past_len + T, dtype=torch.long, device=idx.device)
        pos_emb = self.pos_embedding(pos); x = self.dropout(tok_emb + pos_emb)
        if cache is None: cache = [None] * self.config.n_layer
        new_cache = []
        for i, block in enumerate(self.blocks): x, layer_cache = block(x, layer_cache=cache[i]); new_cache.append(layer_cache)
        x = self.ln_f(x); logits = self.lm_head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss, new_cache
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, top_k=50, temperature=1.0):
        self.eval(); _, _, cache = self(idx, past_len=0); current_token = idx[:, -1:]
        for _ in range(max_new_tokens):
            current_len = idx.size(1)
            logits, _, cache = self(current_token, cache=cache, past_len=current_len - 1)
            logits = logits[:, -1, :] / temperature
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
            k_logits = torch.full_like(logits, -float('inf')).scatter_(1, top_k_indices, top_k_logits)
            probs = F.softmax(k_logits, dim=-1)
            current_token = torch.multinomial(probs, 1)
            idx = torch.cat((idx, current_token), dim=1)
        self.train(); return idx

if __name__ == '__main__':
    cfg = Config()
    print(f"Using device: {cfg.device}")

    mode = input("Enter mode (training/prediction): ").strip().lower()

    if mode == 'training':
        # --- TRAINING MODE ---
        tokenizer = BPETokenizer(cfg)
        if os.path.exists(cfg.bpe_vocab_path):
            print(f"Loading existing V1 BPE vocabulary from {cfg.bpe_vocab_path}")
            tokenizer.load(cfg.bpe_vocab_path)
        else:
            try:
                with open('input.txt', 'r', encoding='utf-8') as f: text = f.read()
                tokenizer.byte_pair_encoding(text)
                tokenizer.save(cfg.bpe_vocab_path)
            except FileNotFoundError:
                print("Error: 'input.txt' not found. Cannot train a new tokenizer.")
                exit()
        vocab_size = len(tokenizer.vocab)
        print(f"V1 BPE tokenizer ready with vocab size {vocab_size}")

        model = LanguageModel(cfg, vocab_size).to(cfg.device)
        print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

        start_epoch, global_step = 1, 0
        if os.path.exists(cfg.checkpoint_path):
            try:
                checkpoint = torch.load(cfg.checkpoint_path, map_location=cfg.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch, global_step = checkpoint.get('epoch', 1), checkpoint.get('global_step', 0)
                print(f"Resuming from epoch {start_epoch} at step {global_step}")
            except Exception as e:
                print(f"Could not load checkpoint due to an error, starting from scratch. Error: {e}")
                
        with open('input.txt', 'r', encoding='utf-8') as f: text = f.read()
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        n = int(0.9 * len(data))
        train_dataset, val_dataset = TextDataset(data[:n], cfg.block_size), TextDataset(data[n:], cfg.block_size)
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

        @torch.no_grad()
        def evaluate(loader):
            model.eval(); total_loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(cfg.device), yb.to(cfg.device)
                _, loss, _ = model(xb, yb); total_loss += loss.item()
            model.train(); return total_loss / len(loader)

        print("\n--- Starting Training ---")
        model.train()
        for epoch in range(start_epoch, cfg.num_epochs + 1):
            for batch_idx, (xb, yb) in enumerate(train_loader):
                xb, yb = xb.to(cfg.device), yb.to(cfg.device)
                optimizer.zero_grad(set_to_none=True)
                _, loss, _ = model(xb, yb); loss.backward(); optimizer.step(); global_step += 1
                if global_step % 100 == 0: print(f"Epoch {epoch} | Step {global_step} | Loss {loss.item():.4f}")
                if global_step > 0 and global_step % 500 == 0:
                    print("-" * 50 + f"\n--- Generating text at step {global_step} ---")
                    context = torch.tensor([[tokenizer.vocab[tokenizer.BOS]]], dtype=torch.long, device=cfg.device)
                    generated_ids = model.generate(context, 100, top_k=cfg.gen_top_k, temperature=cfg.gen_temperature)[0].tolist()
                    print(tokenizer.decode(generated_ids)); print("-" * 50)
                    model.train()
            val_loss = evaluate(val_loader)
            print("=" * 60 + f"\nEpoch {epoch} Complete | Validation Loss: {val_loss:.4f}\n" + "=" * 60)
            torch.save({'epoch': epoch + 1, 'global_step': global_step, 'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict()}, cfg.checkpoint_path)
        
        print("Training finished.")

    elif mode == 'prediction':
        # --- PREDICTION MODE ---
        if not os.path.exists(cfg.bpe_vocab_path):
            print(f"Error: BPE vocabulary not found at '{cfg.bpe_vocab_path}'. Please run in 'training' mode first to create it.")
            exit()
        if not os.path.exists(cfg.checkpoint_path):
            print(f"Error: Training checkpoint not found at '{cfg.checkpoint_path}'. Please run in 'training' mode first to create a model.")
            exit()
        
        print(f"Loading BPE vocabulary from {cfg.bpe_vocab_path}")
        tokenizer = BPETokenizer(cfg)
        tokenizer.load(cfg.bpe_vocab_path)
        vocab_size = len(tokenizer.vocab)
        print(f"Tokenizer ready with vocab size {vocab_size}")

        model = LanguageModel(cfg, vocab_size).to(cfg.device)
        print(f"Loading model from checkpoint: {cfg.checkpoint_path}")
        checkpoint = torch.load(cfg.checkpoint_path, map_location=cfg.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters loaded successfully.")

        print("\n--- Starting Prediction Mode (type 'exit' or 'quit' to end) ---")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            # Encode the user's input, removing the final EOS token to allow for generation
            prompt_tokens = tokenizer.encode(user_input)[:-1]
            context = torch.tensor([prompt_tokens], dtype=torch.long, device=cfg.device)

            print("AI: ", end='', flush=True)
            
            # Generate a response from the model
            generated_sequence_ids = model.generate(context, max_new_tokens=150, top_k=cfg.gen_top_k, temperature=cfg.gen_temperature)[0].tolist()
            
            # Extract only the newly generated tokens
            response_ids = generated_sequence_ids[len(prompt_tokens):]

            # Stop decoding at the first End-Of-Sentence token if one is generated
            try:
                eos_index = response_ids.index(tokenizer.vocab[tokenizer.EOS])
                response_ids = response_ids[:eos_index]
            except ValueError:
                # No EOS token was generated, so we'll use the full response
                pass
            
            response_text = tokenizer.decode(response_ids)
            print(response_text)

    else:
        print("Invalid mode. Please enter 'training' or 'prediction'.")