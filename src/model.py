import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedder(nn.Module):
    '''
    word_id to vector_id
    '''

    def __init__(self, text_embedding_vectors):
        super(Embedder, self).__init__()

        # 日本語の学習済みを用意する
        self.embeddings = nn.Embedding.from_pretrained(
            embeddings=text_embedding_vectors, freeze=True
        )

    def forward(self, x):
        x_vec = self.embeddings(x)
        return x_vec


class PositionalEncoder(nn.Module):
    '''
    単語の位置を示すベクトル情報を付加する
    '''

    def __init__(self, vec_dim, max_seq_len):
        super(PositionalEncoder, self).__init__()

        self.vec_dim = vec_dim  # 単語ベクトルの次元数

        pe = torch.zeros(max_seq_len, vec_dim)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        pe = pe.to(device)

        for pos in range(max_seq_len):
            for i in range(0, vec_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2*i)/vec_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2*i)/vec_dim)))

        self.pe = pe.unsqueeze(0)  # 　ミニバッチの次元を追加

        self.pe.requires_grad = False  # 　勾配計算しない

    def forward(self, x):
        ret = math.sqrt(self.vec_dim)*x + self.pe
        return ret


class Attention(nn.Module):
    def __init__(self, d_model=300):
        super().__init__()

        # 1dconv(本来は)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

        self.d_k = d_model

    def forward(self, q, k, v, mask):
        key = self.key(k)
        query = self.query(q)
        value = self.value(v)

        # 内積をとりqureyとkeyの関連度を計算する。
        weights = torch.matmul(query, key.transpose(1, 2)
                               ) / math.sqrt(self.d_k)

        # softmaxで0をとるため
        mask = mask.unsqueeze(1)
        weights = weights.masked_fill(mask == 0, -1e9)

        # softmaxで正規化(attention weight 0 ~ 1)
        normlized_weights = F.softmax(weights, dim=-1)  # attention weight
        output = torch.matmul(normlized_weights, value)

        output = self.out(output)

        return output, normlized_weights


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=1024, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout(F.relu(x))
        x = self.linear_2(x)
        return x


class TransfomerBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()

        # layernormlize
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)

        self.attn = Attention(d_model)

        self.ff = FeedForward(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x_normlized = self.norm_1(x)
        output, normalized_weights = self.attn(
            x_normlized, x_normlized, x_normlized, mask)

        x2 = x + self.dropout_1(output)

        x_normlized2 = self.norm_2(x2)

        output = x2 + self.dropout_2(self.ff(x_normlized2))

        return output, normalized_weights  # [output, attention weights]


class ClassificationHead(nn.Module):
    def __init__(self, d_model=300, out_dim=2):
        super().__init__()

        self.linear = nn.Linear(d_model, out_dim)

        # 重みを正規分布で初期化
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, x):
        x0 = x[:, 0, :]  # [batch_size, text_len, depth]
        out = self.linear(x0)
        return out


class TransformerClassification(nn.Module):
    def __init__(self, text_embedding_vectors,
                 d_model=300, max_seq_len=256, out_dim=2):
        super().__init__()

        self.net1 = Embedder(text_embedding_vectors)
        self.net2 = PositionalEncoder(vec_dim=d_model, max_seq_len=max_seq_len)
        self.net3_1 = TransfomerBlock(d_model=d_model)
        self.net3_2 = TransfomerBlock(d_model=d_model)
        self.net4 = ClassificationHead(d_model=d_model, out_dim=out_dim)

    def forward(self, x, mask):
        x1 = self.net1(x)  # 単語をベクトルに
        x2 = self.net2(x1)  # Positon情報を足し算
        x3_1, normlized_weights_1 = self.net3_1(
            x2, mask)  # Self-Attentionで特徴量を変換
        x3_2, normlized_weights_2 = self.net3_2(
            x3_1, mask)  # Self-Attentionで特徴量を変換
        x4 = self.net4(x3_2)  # 最終出力の0単語目を使用して、分類0-1のスカラーを出力
        return x4, normlized_weights_1, normlized_weights_2
