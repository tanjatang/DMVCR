# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from models.multiatt.net_utils import FC, MLP, LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch, math


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self,dim1, dim2):
        super(MHAtt, self).__init__()

        self.hidden_size = dim1#  512
        self.dropout = 0.1
        self.multi_head = 8
        self.hidden_size_head = int(self.hidden_size / self.multi_head)
        self.linear_v = nn.Linear(dim2, self.hidden_size )
        self.linear_k = nn.Linear(dim2, self.hidden_size)
        self.linear_q = nn.Linear( self.hidden_size,self.hidden_size)
        self.linear_merge = nn.Linear(self.hidden_size,self.hidden_size)

        self.dropout = nn.Dropout(self.dropout)

    def forward(self, v, k, q, mask):

        n_batches = q.size(0)
        # print(self.hidden_size,'llllllllllll',v.size(), 'v1vv0000', k.size(), 'kkkkkk00000', q.shape)

        v = self.linear_v(v.cpu()).view(
            n_batches,
            -1,
            self.multi_head ,
            self.hidden_size_head
        ).transpose(1, 2)

        k = self.linear_k(k.cpu()).view(
            n_batches,
            -1,
            self.multi_head,
            self.hidden_size_head
        ).transpose(1, 2)

        q = self.linear_q(q.cpu()).view(
            n_batches,
            -1,
            self.multi_head,
            self.hidden_size_head
        ).transpose(1, 2)


        atted = self.att(v, k, q, mask)
        # print(v.size(), 'v1vv11111', k.size(), 'kkkkkk11111111111', q.shape)

        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.hidden_size
        )
        # print(atted.shape,'aaaaaaaaaaaaaaaa')
        atted = self.linear_merge(atted.cpu())
        # print('mmmmmmmmmmmmmmmmmmmmmmmmm', atted.size())
        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
#----------------------------------------------------------------------
#-----------------------------这里mask处理需要再想想_--------------------
#----------------------------------------------------------------------

            scores = scores.masked_fill(mask.cpu(), -1e9)


        att_map = F.softmax(scores, dim=-1)

        att_map = self.dropout(att_map)

        return torch.matmul(att_map.cuda(), value.cuda())


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, hidden_size):
        super(FFN, self).__init__()
        self.hidden_size = hidden_size

        self.mlp = MLP(
            in_size= self.hidden_size,
            mid_size= self.hidden_size*4,
            out_size= self.hidden_size,
            dropout_r= 0.1,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self,dim):
        super(SA, self).__init__()

        self.hidden_size = dim

        self.mhatt = MHAtt(self.hidden_size,self.hidden_size)
        self.ffn = FFN(self.hidden_size)

        self.dropout1 = nn.Dropout(0.1)
        self.norm1 = LayerNorm(self.hidden_size)

        self.dropout2 = nn.Dropout(0.1)
        self.norm2 = LayerNorm(self.hidden_size)

    def forward(self, x, x_mask):
        # print('sxsxsxsxsxssxx:',x.size())
        x = x.cpu()
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))
        # print('SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS')

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self,dim1,dim2):
        super(SGA, self).__init__()
        self.hidden_size = dim1

        self.mhatt1 = MHAtt(self.hidden_size,self.hidden_size)
        self.mhatt2 = MHAtt(self.hidden_size,dim2)
        self.ffn = FFN(self.hidden_size )


        self.dropout1 = nn.Dropout(0.1)
        self.norm1 = LayerNorm(self.hidden_size)

        self.dropout2 = nn.Dropout(0.1)
        self.norm2 = LayerNorm(self.hidden_size)

        self.dropout3 = nn.Dropout(0.1)
        self.norm3 = LayerNorm(self.hidden_size)

    def forward(self, x, y, x_mask, y_mask):
        '''这里的xy反的', x: 图片， y: 文字, '''

        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxx:',x.size())
        x = x.cpu()
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        # print('gggggggggggggggggggggggggggggggg')
        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self):
        super(MCA_ED, self).__init__()
        self.layer = 1
        self.enc_list = nn.ModuleList([SA(768) for _ in range(self.layer)])
        self.dec_list = nn.ModuleList([SGA(512,768) for _ in range(self.layer)])
        # self.map = nn.Linear(768, 512)  # map , 不知道。。。。。。。。。。

    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        # print('kkkkkkkkkkkkkkkkk')
        for enc in self.enc_list:
            x = enc(x, x_mask)
        # x_map = self.map(x)           #  先把文字映射到了图片维度

        for dec in self.dec_list:
            y = dec(y, x, y_mask, x_mask)


        return x, y


# ------------------------------------------------
# ---- MAC Layers Cascaded by Stacking ----
# -----------------------------------------------

class MCA_SK(nn.Module):
    def __init__(self):
        super(MCA_SK, self).__init__()
        self.num_unit = 1

        self.SA = SA(768)
        self.SGA = SGA(512,768)
        self.map = nn.Linear(768,512)# map , 不知道。。。。。。。。。。

    def MCA_layer(self, x, y, x_mask, y_mask):
        # X： 文字, Y: image

        x = self.SA(x, x_mask)
        # print('lllllllllllllllllllllllllllll', x.size())
        # x_map = self.map(x)
        y = self.SGA(y, x, y_mask, x_mask)#   map contex to image dimention

        return x, y

    def forward(self, x, y, x_mask, y_mask):
        # stack layers

        for l in range(self.num_unit):
            x, y = self.MCA_layer(x, y, x_mask, y_mask)
        return x, y

#----------------------------------------------------------------------------------------
#----------------------Att reduce module-------------------------------------------------
#-----------------------------------------------------------------------------------------

class AttFlat(nn.Module):
    def __init__(self, hidden_size):
        super(AttFlat, self).__init__()
        self.hidden_size = hidden_size
        self.flat_mlp_size = 512
        self.flat_glimpses = 1
        self.drop_out = 0.1
        self.flat_out_size = 1024
        self.mlp = MLP(
            in_size = self.hidden_size,
            mid_size = self.flat_mlp_size,
            out_size = self.flat_glimpses,
            dropout_r = self.drop_out,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            self.hidden_size * self.flat_glimpses,
            self.flat_out_size
        )

    def forward(self, x, x_mask):
        # print('pppppppppppppppppppppppppppppp',x.shape)

        att = self.mlp(x)

#---------------------------------------------------------------------------------
#----------------------mask mask mask --------------------------------------------
#---------------------------------------------------------------------------------

        # print('kkkkkkkkkkkkkkkkkkkkkk',att.size(),'lllll',x_mask.squeeze(1).squeeze(1).unsqueeze(2).size())

        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2).cpu(),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.flat_glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted
