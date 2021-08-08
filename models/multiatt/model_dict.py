"""
Let's get the relationships yo
"""

from typing import Dict, List, Any

import torch

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.nn.parallel
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TextFieldEmbedder, Seq2SeqEncoder, FeedForward, InputVariationalDropout, TimeDistributed
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.modules.matrix_attention import BilinearMatrixAttention
from utils.detector import SimpleDetector
from allennlp.nn.util import masked_softmax, weighted_sum, replace_masked_values
from allennlp.nn import InitializerApplicator
import os
# import pickle
import pickle
import ipdb
#######################################3
from PIL import Image
import pylab
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import pyplot
from matplotlib.patches import Rectangle


#######################################
SAVE_ROOT = "/phys/ssd/tangxueq/tmp/vcr/vcrimage/rationale"

@Model.register("MultiHopAttentionQA")
class AttentionQA(Model):
    def __init__(self,

                 vocab: Vocabulary,
                 span_encoder: Seq2SeqEncoder,
                 reasoning_encoder: Seq2SeqEncoder,
                 input_dropout: float = 0.3,
                 hidden_dim_maxpool: int =1024,
                 class_embs: bool=True,
                 reasoning_use_obj: bool=True,
                 reasoning_use_answer: bool=True,
                 reasoning_use_question: bool=True,
                 pool_reasoning: bool = True,
                 pool_answer: bool = True,
                 pool_question: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 ):
        super(AttentionQA, self).__init__(vocab)

        self.detector = SimpleDetector(pretrained=True, average_pool=True, semantic=class_embs, final_dim=512)
        ###################################################################################################
        INPUT_SIZE = 1536
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=1536,
            num_layers=2,  # hidden_layer的数目
            batch_first=True,  # 输入数据的维度一般是（batch, time_step, input)，该属性表征batch是否放在第一个维度
        )






        self.rnn_input_dropout = TimeDistributed(InputVariationalDropout(input_dropout)) if input_dropout > 0 else None

        self.span_encoder = TimeDistributed(span_encoder)
        self.span_reshape = TimeDistributed(torch.nn.Linear(512, 768))
        # self.out = nn.Linear(50, 1)
        self.reasoning_encoder = TimeDistributed(reasoning_encoder)

        self.span_attention = BilinearMatrixAttention(
            matrix_1_dim=span_encoder.get_output_dim(),
            matrix_2_dim=span_encoder.get_output_dim(),
        )

        self.obj_attention = BilinearMatrixAttention(
            matrix_1_dim=span_encoder.get_output_dim(),
            matrix_2_dim=self.detector.final_dim,
        )

        self.reasoning_use_obj = reasoning_use_obj
        self.reasoning_use_answer = reasoning_use_answer
        self.reasoning_use_question = reasoning_use_question
        self.pool_reasoning = pool_reasoning
        self.pool_answer = pool_answer
        self.pool_question = pool_question

        #[96,4,50,1024]
        dim = 768*2 #sum([d for d, to_pool in [(reasoning_encoder.get_output_dim(), self.pool_reasoning),
                                        # (span_encoder.get_output_dim(), self.pool_answer),
                                        # (span_encoder.get_output_dim(), self.pool_question)] if to_pool])

        self.final_mlp = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(dim, hidden_dim_maxpool),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(hidden_dim_maxpool, 4),
        )
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)
        self.count = 0
        ##############################################################################
        # self.memory_cell_path = getattr(opt, 'memory_cell_path', '0')
        self.memory_cell_path = '/home/tangxueq/MA_tang/r2c/models/saves/memory_cell.npz'

        if os.path.isfile(self.memory_cell_path):
            print('load memory_cell from {0}'.format(self.memory_cell_path))
            # memory_init = np.load(self.memory_cell_path)['memory_cell']
            memory_init = np.load(self.memory_cell_path)['memory_cell'][()]

            # print(type(memory_init),"hhhhhhhhhhhhhhhhhhhhhhhhhhh")
        else:
            print('create a new memory_cell')
            memory_init = np.random.rand(5000, 768*2) / 100

        memory_init = np.float32(memory_init)


        self.memory_cell = torch.from_numpy(memory_init).cuda().requires_grad_()
        print(self.memory_cell)






        self.ssg_mem = Memory_cell2()

    def _collect_obj_reps(self, span_tags, object_reps):
        """
        Collect span-level object representations
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :return:
        """
        span_tags_fixed = torch.clamp(span_tags, min=0)  # In case there were masked values here
        row_id = span_tags_fixed.new_zeros(span_tags_fixed.shape)
        row_id_broadcaster = torch.arange(0, row_id.shape[0], step=1, device=row_id.device)[:, None]

        # Add extra diminsions to the row broadcaster so it matches row_id
        leading_dims = len(span_tags.shape) - 2
        for i in range(leading_dims):
            row_id_broadcaster = row_id_broadcaster[..., None]
        row_id += row_id_broadcaster
        return object_reps[row_id.view(-1), span_tags_fixed.view(-1)].view(*span_tags_fixed.shape, -1)

    def embed_span(self, span, span_tags, span_mask, object_reps):
        """
        :param span: Thing that will get embed and turned into [batch_size, ..leading_dims.., L, word_dim]
        :param span_tags: [batch_size, ..leading_dims.., L]
        :param object_reps: [batch_size, max_num_objs_per_batch, obj_dim]
        :param span_mask: [batch_size, ..leading_dims.., span_mask
        :return:
        """
        retrieved_feats = self._collect_obj_reps(span_tags, object_reps)

        span_rep = torch.cat((span['bert'], retrieved_feats), -1)
        # add recurrent dropout here
        if self.rnn_input_dropout:
            span_rep = self.rnn_input_dropout(span_rep)

        x = self.span_encoder(span_rep, span_mask)
        x = self.span_reshape(x)

        return x, retrieved_feats

    def padding(self,q_rep,a_rep):
        # max_len = max(max([i.size(2) for i in q_rep]), max([i.size(2) for i in a_rep]))
        max_len = max(a_rep.size(2),q_rep.size(2))
        a1, b1, c1, d1 = a_rep.size()
        a2, b2, c2, d2 = q_rep.size()
        padding_a = torch.zeros(a1, b1, max_len - c1, d1).float().cuda()
        padding_q = torch.zeros(a2, b2, max_len - c2, d2).float().cuda()


        q_rep_new = torch.cat((q_rep, padding_q), dim=2)
        a_rep_new = torch.cat((a_rep, padding_a), dim=2)

        qa_rep = torch.cat((q_rep_new, a_rep_new), dim=3)  # [batch_size, 8, seq_len, 512]

        return qa_rep

    def Dictionary(self,h,M): #[96,4,50,768*2] M[5000,1536]
        h_size = h.size()  # 96*4*50*1536
        h = h.view(-1, h_size[3])  # (96*4*50)*(1536)
        att = torch.mm(h, torch.t(M))  # [(96*4*50),(1536)] * [1536,5000]
        att = F.softmax(att, dim=1)  # [96*4*50,5000]

        att_res = torch.mm(att, M)  # [96*4*50,5000]*[5000,1536]    ->   #[96*4*50,1536]
        att_res = att_res.view([96, -1, h_size[3]])  #[96*4,50,1536]
        return att_res

    def forward(self,

                training_mode,
                # images: torch.Tensor,
                obj_reps: Dict[str, torch.Tensor],
                objects: torch.LongTensor,
                segms: torch.Tensor,
                boxes: torch.Tensor,
                box_mask: torch.LongTensor,
                question: Dict[str, torch.Tensor],
                question_tags: torch.LongTensor,
                question_mask: torch.LongTensor,
                answers: Dict[str, torch.Tensor],
                answer_tags: torch.LongTensor,
                answer_mask: torch.LongTensor,
                metadata: List[Dict[str, Any]] = None,
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        """
        :param images: [batch_size, 3, im_height, im_width]
        :param objects: [batch_size, max_num_objects] Padded objects
        :param boxes:  [batch_size, max_num_objects, 4] Padded boxes
        :param box_mask: [batch_size, max_num_objects] Mask for whether or not each box is OK
        :param question: AllenNLP representation of the question. [batch_size, num_answers, seq_length]
        :param question_tags: A detection label for each item in the Q [batch_size, num_answers, seq_length]
        :param question_mask: Mask for the Q [batch_size, num_answers, seq_length]
        :param answers: AllenNLP representation of the answer. [batch_size, num_answers, seq_length]
        :param answer_tags: A detection label for each item in the A [batch_size, num_answers, seq_length]
        :param answer_mask: Mask for the As [batch_size, num_answers, seq_length]
        :param metadata: Ignore, this is about which dataset item we're on
        :param label: Optional, which item is valid
        :return: shit
        """
        # Trim off boxes that are too long. this is an issue b/c dataparallel, it'll pad more zeros that are
        # not needed
        # print("question:\n", len(question), "\n")
        # print(question['bert'].size())
        # print("answers:\n", len(answers), "\n")
        # print(answers['bert'].size())


        max_len = int(box_mask.sum(1).max().item())
        objects = objects[:, :max_len]
        box_mask = box_mask[:, :max_len]
        boxes = boxes[:, :max_len]
        segms = segms[:, :max_len]



        for tag_type, the_tags in (('question', question_tags), ('answer', answer_tags)):
            if int(the_tags.max()) > max_len:
                raise ValueError("Oh no! {}_tags has maximum of {} but objects is of dim {}. Values are\n{}".format(
                    tag_type, int(the_tags.max()), objects.shape, the_tags
                ))

        # obj_reps = self.detector(images=images, boxes=boxes, box_mask=box_mask, classes=objects, segms=segms)

        q_rep, q_obj_reps = self.embed_span(question, question_tags, question_mask, obj_reps['obj_reps'])
        a_rep, a_obj_reps = self.embed_span(answers, answer_tags, answer_mask, obj_reps['obj_reps'])

        # print(q_rep.size(),'lllllllllllllllllllllllllllllllllllllllllll')

##################################################################################################################
        '''try idea to create new model with dictionary'''

        '''padding q_rep and a_rep with 0'''
        # max_len = max(max([i.size(2) for i in question['bert']]),max([i.size(2) for i in answers['bert']]))
        # a, b, c, d = a_rep.size()
        # padding = torch.zeros(a,b,max_len,d).float()
        # q_rep_new = torch.cat((q_rep,padding),dim = 2)
        # a_rep_new = torch.cat((a_rep,padding), dim = 2)


        if training_mode == 0:
            qa_rep = self.padding(question['bert'],answers['bert'])  #[batch_size, 8, seq_len, 512]

        elif training_mode == 1:
            qa_rep = self.padding(q_rep,a_rep)
            print(qa_rep.size(),'qa_rep with image information!')

        #
        # #clip
        # a1, a2, a3, a4 = qa_rep.size()
        # if a3 >= 50:
        #     qa_rep = qa_rep[:, :, 0:50, :].contiguous()
        #
        # else:
        #     padding_i = torch.zeros(a1, a2, 50 - a3, a4).cuda()
        #     qa_rep = torch.cat((qa_rep, padding_i), dim=2)
        # # print(qa_rep.size(),'qa_rep.size()')
##################################################################

        qa_rep_res = self.Dictionary(qa_rep, self.memory_cell) # [96* 4, 50, 1536]
        # print(qa_rep_res.size(),'qa_rep_res.size()')

        # r_out, (h_n, h_c) = self.rnn(qa_rep_res,None)

        output, (h_n, h_c) = self.rnn(qa_rep_res)
        output = output[:,-1,:] #取最后一层输出
         #output[384,64]
        # output = output.reshape([96,-1,1536])
        # print(output.size(), 'output,sssssssssssssssss')
        # raise InterruptedError
        logits = self.final_mlp(output)#.squeeze(2)


        class_probabilities = F.softmax(logits, dim=-1)


        output_dict = {"label_logits": logits, "label_probs": class_probabilities,
                       'cnn_regularization_loss': obj_reps['cnn_regularization_loss'],
                       # Uncomment to visualize attention, if you want
                       # 'qa_attention_weights': qa_attention_weights,
                       # 'atoo_attention_weights': atoo_attention_weights,
                       }


        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            self._accuracy(logits, label)
            output_dict["loss"] = loss[None]



        return output_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}


class Memory_cell(nn.Module):
    def __init__(self, opt):
        """
        a_i = W^T*tanh(W_h*h + W_M*m_i)
        a_i: 1*1
        W: V*1
        W_h: V*R
        h: R*1
        W_M: V*R
        m_i: R*1
        M=[m_1,m_2,...,m_K]^T: K*R
        att = softmax(a): K*1
        h_out = M*att

        :param opt:
        """
        super(Memory_cell, self).__init__()
        self.R = opt.rnn_size
        self.V = opt.att_hid_size

        self.W = nn.Linear(self.V,1)
        self.W_h = nn.Linear(self.R, self.V)
        self.W_M = nn.Linear(self.R, self.V)

    def forward(self, h, M):
        M_size = M.size() #K*R
        h_size = h.size() #N*R

        att_h = self.W_h(h) #h:N*R att_h:N*V
        att_h = att_h.unsqueeze(1).expand([h_size[0],M_size[0],self.V]) #N*K*V

        M_expand = M.unsqueeze(0).expand([h_size[0],M_size[0],self.R]) #N*K*R
        att_M = self.W_M(M_expand) #N*K*V

        dot = att_h + att_M #N*K*V
        dot = F.tanh(att_M) #N*K*V
        dot = dot.view(-1, self.V)   #(N*K)*V
        dot = self.W(dot)   #N*K*1
        dot = dot.view(-1, M_size[0]) #N*K

        att = F.softmax(dot, dim=1) #N*K
        att_max = torch.max(att,dim=1)
        max_index = torch.argmax(att,dim=1)
        att_res = torch.bmm(att.unsqueeze(1), M_expand) # N*1*K, N*K*R->N*1*R
        att_res = att_res.squeeze(1) #N*R
        return att_res

class Memory_cell2(nn.Module):
    def __init__(self):
        """
        a_i = h^T*m_i
        a_i: 1*1

        h: R*1
        m_i: R*1
        M=[m_1,m_2,...,m_K]^T: K*R
        att = softmax(a): K*1
        h_out = M*att: N*R

        :param opt:
        """
        super(Memory_cell2, self).__init__()
        self.R = 512
        self.V = 512

        self.W = nn.Linear(self.V, 1)

    def forward(self, h, M): #   h: (b,N,T,R) ,   M_size = M.size()  # K*R
        M_size = M.size()  # K*R

        h_size = h.size()  # b*N*T*R
        h = h.view(-1,h_size[3]) # b*(N*T*R)
        att = torch.mm(h, torch.t(M)) #b*(N*T)*K
        att = F.softmax(att, dim=1) #b*(N*T)*K

        att_res = torch.mm(att, M)  #    b*(N*T)*K * K*R-> b*(N*T)*R
        att_res = att_res.view([h_size[0], h_size[1], h_size[2],h_size[3]])
        return att_res

class TopDownCore_mem(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCore_mem, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        #self.att_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)
        self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) # we, fc, h^2_t-1
        #self.att_lstm = nn.LSTMCell(opt.input_encoding_size + opt.rnn_size, opt.rnn_size) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size) # h^1_t, \hat v
        self.attention = Attention(opt)

    def forward(self, xt, fc_feats, att_feats, p_att_feats, state, att_masks):
        prev_h = state[0][-1]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)
        #att_lstm_input = torch.cat([prev_h, xt], 1)
        #att_lstm_input = torch.cat([fc_feats, xt], 1)

        # state[0][0] means the hidden state c in first lstm
        # state[1][0] means the cell h in first lstm
        # state[0] means hidden state and state[1] means cell, state[0][i] means
        # the i-th layer's hidden state
        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        lang_lstm_input = torch.cat([att, h_att], 1)
        # lang_lstm_input = torch.cat([att, F.dropout(h_att, self.drop_prob_lm, self.training)], 1) ?????

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state


class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // att_feats.size(-1)
        att = p_att_feats.view(-1, att_size, self.att_hid_size)

        att_h = self.h2att(h)  # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)  # batch * att_size * att_hid_size
        dot = att + att_h  # batch * att_size * att_hid_size
        dot = F.tanh(dot)  # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)  # (batch * att_size) * 1
        dot = dot.view(-1, att_size)  # batch * att_size

        weight = F.softmax(dot, dim=1)  # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True)  # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, att_feats.size(-1))  # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size

        return att_res

