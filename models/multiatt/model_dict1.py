"""
Let's get the relationships yo
"""

from typing import Dict, List, Any
import math
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
                 # lstm_encoder: Seq2SeqEncoder,
                 input_dropout: float = 0.5,
                 hidden_dim_maxpool: int =64,
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
################################################################################################################################
        path = os.path.join('/home/tangxueq/tmp', "memory_cell80080.npz")
        self.memory_cell = torch.nn.Parameter(torch.Tensor(self.memory_cell_load(path)))
        # for k,v in torch.load('/home/tangxueq/tmp/memory_cell800.th').items():
        #     print(k)
        # self.memory_cell.load_state_dict(torch.load('/home/tangxueq/tmp/memory_cell800.th'))
        # for p in self.detector.parameters():
        #     p.requires_grad = False
        print(self.memory_cell)


###############################################################################################################################
        # if os.path.isfile(self.memory_cell_path):
        #     print('load memory_cell from {0}'.format(self.memory_cell_path))
        #     memory_init = np.load(self.memory_cell_path)['memory_cell'][()]
        # else:
        #     print('create a new memory_cell')
        #     memory_init = np.random.rand(1000, 1024) / 100
        # memory_init = np.float32(memory_init)
        # self.memory_cell = torch.from_numpy(memory_init).cuda().requires_grad_()


#################################################################################################################################

        self.detector = SimpleDetector(pretrained=True, average_pool=True, semantic=class_embs, final_dim=512)
        self.detector.load_state_dict(torch.load('/home/tangxueq/tmp/detector.th'))
        for p in self.detector.parameters():
            p.requires_grad = False
        ###################################################################################################
        INPUT_SIZE =512 #768
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=2,  # hidden_layer的数目
            batch_first=True,  # 输入数据的维度一般是（batch, time_step, input)，该属性表征batch是否放在第一个维度
            # bidirectional=True

        )
############################################################################################################################3

        self.linear = torch.nn.Linear(512,256)
        self.mask_linear = torch.nn.Linear(1,512)

        self.encoder_layer_vc = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder_vc = nn.TransformerEncoder(self.encoder_layer_vc, num_layers=3)
        #
        self.AvgPool =nn.AdaptiveAvgPool1d(1)

        # self.encoder_layer_cv = nn.TransformerEncoderLayer(d_model=768, nhead=8)
        # self.transformer_encoder_cv = nn.TransformerEncoder(self.encoder_layer_cv, num_layers=2)

        # self.image_AttFlat = AttFlat(512)
        # self.qa_AttFlat = AttFlat(768)
        #
        # self.proj_norm = LayerNorm(1024)
        # self.proj = nn.Linear(1024, 4)
############################################################################################################################

        self.norm = torch.nn.BatchNorm1d(512)

        self.lstm_norm = torch.nn.BatchNorm1d(10000)
        self.rnn_input_dropout = TimeDistributed(InputVariationalDropout(input_dropout)) if input_dropout > 0 else None

        self.span_encoder = TimeDistributed(span_encoder)
        self.span_reshape = TimeDistributed(torch.nn.Linear(512, 768))
        self.qao_reshape = torch.nn.Linear(512,768)

        # self.out = nn.Linear(50, 1)
        # self.reasoning_encoder = TimeDistributed(lstm_encoder)

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
        dim = 64#768*2 #sum([d for d, to_pool in [(reasoning_encoder.get_output_dim(), self.pool_reasoning),
                                        # (span_encoder.get_output_dim(), self.pool_answer),
                                        # (span_encoder.get_output_dim(), self.pool_question)] if to_pool])
        # self.proj_norm = LayerNorm(256)
        # self.final_mlp = torch.nn.Sequential(
        #     torch.nn.Flatten(),
        #     torch.nn.Dropout(input_dropout, inplace=False),
        #     torch.nn.Linear(dim, hidden_dim_maxpool),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Dropout(input_dropout, inplace=False),
        #     torch.nn.Linear(hidden_dim_maxpool, 4),
        # )


        # self.final_mlp = torch.nn.Sequential(
        #     torch.nn.Flatten(),
        #     torch.nn.Dropout(input_dropout, inplace=False),
        #     torch.nn.Linear(dim, hidden_dim_maxpool),
        #     torch.nn.LeakyReLU(inplace=True),
        #     # torch.nn.Dropout(input_dropout, inplace=False),
        #     torch.nn.Linear(hidden_dim_maxpool, 4),
        # )
        # self.final_mlp = torch.nn.Sequential(
        #     torch.nn.Flatten(),
        #     torch.nn.Dropout(input_dropout, inplace=False),
        #     torch.nn.Linear(dim,4))

        #------------------------------------------------------------------

        self.final_mlp = torch.nn.Sequential(
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(dim, hidden_dim_maxpool),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(hidden_dim_maxpool, 1),
        )


        #----------------------------------------------------------------------


        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
      #####################################################################


        # for n, p in self.named_parameters():
        #     if n == "memory_cell":
        #         p.requires_grad = True
        #     else:
        #         p.requires_grad = False
        # # for p in self.parameters() :
        # #
        # #     p.requires_grad = False
        #
        # for p in self.detector.parameters():
        #     p.requires_grad = True


        initializer(self)

        ##############################################################################
        # self.memory_cell_path = getattr(opt, 'memory_cell_path', '0')


    def memory_cell_load(self,path):
        if os.path.isfile(path):
            print('load memory_cell from {0}'.format(path))
            memory_init = np.load(path)['memory_cell']
            # memory_init = torch.jit.load(path)['memory_cell'][()]
            memory_init = torch.from_numpy(memory_init).cuda().requires_grad_()
        else:
            print('create a new memory_cell')
            # memory_init = np.random.rand(10000, 1024)/ 100
            # memory_init = torch.random(10000,1024)/100
            memory_init = torch.rand(10000, 64)/100
        # memory_init = memory_init.long()

        # self.memory_cell = torch.from_numpy(memory_init).cuda().requires_grad_()

        return memory_init


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

        # x = self.span_encoder(span_rep, span_mask)
        # x = self.span_reshape(x)

        return self.span_encoder(span_rep, span_mask), retrieved_feats

    def padding(self,q_rep,a_rep):
        # max_len = max(max([i.size(2) for i in q_rep]), max([i.size(2) for i in a_rep]))
        max_len = max(a_rep.size(2),q_rep.size(2))
        a1, b1, c1, d1 = a_rep.size()
        a2, b2, c2, d2 = q_rep.size()
        padding_a = torch.zeros(a1, b1, max_len - c1, d1).float().cuda()
        padding_q = torch.zeros(a2, b2, max_len - c2, d2).float().cuda()


        q_rep_new = torch.cat((q_rep, padding_q), dim=2)
        a_rep_new = torch.cat((a_rep, padding_a), dim=2)

        qa_rep = torch.cat((q_rep_new, a_rep_new), dim=3)  # [batch_size, 8, seq_len, 1536]

        return qa_rep

    def Dictionary(self,h,M): #[96*4,768] M[10000,768]
        h_size = h.size()  # [96*4,768]
        # h = h.view(-1, h_size[3])  # [(96*4),768]
        att = torch.mm(h, torch.t(M))  # [(96*4*50),768] * [768,10000]
        att = F.softmax(att, dim=1)  # [96*4*50,10000]

        att_res = torch.mm(att, M)  # [96*4,10000]*[10000,768]    ->   #[96*4,768]
        # att_res = att_res.view([-1, h_size[1]*h_size[2], h_size[3]])  #[96*4,768]
        return att_res

    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)

    def forward(self,

                # training_mode,
                images: torch.Tensor,
                # obj_reps: Dict[str, torch.Tensor],
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

        obj_reps = self.detector(images=images, boxes=boxes, box_mask=box_mask, classes=objects, segms=segms)

##################################################################################################################
        '''try idea to create new model with dictionary'''

        '''padding q_rep and a_rep with 0'''
        # max_len = max(max([i.size(2) for i in question['bert']]),max([i.size(2) for i in answers['bert']]))
        # a, b, c, d = a_rep.size()
        # padding = torch.zeros(a,b,max_len,d).float()
        # q_rep_new = torch.cat((q_rep,padding),dim = 2)
        # a_rep_new = torch.cat((a_rep,padding), dim = 2)
        # print(question['bert'].size())


        #
        # if training_mode == 0:
        #     # qa_rep = self.padding(question['bert'],answers['bert'])  #[batch_size, 8, seq_len, 768]
        #     qa_rep = torch.cat((question['bert'],answers['bert']),dim = 2)


        # elif training_mode == 1:
        q_rep, q_obj_reps = self.embed_span(question, question_tags, question_mask, obj_reps['obj_reps'])
        a_rep, a_obj_reps = self.embed_span(answers, answer_tags, answer_mask, obj_reps['obj_reps'])


        ##########################################################################3

        qa_similarity = self.span_attention(
            q_rep.view(q_rep.shape[0] * q_rep.shape[1], q_rep.shape[2], q_rep.shape[3]),
            a_rep.view(a_rep.shape[0] * a_rep.shape[1], a_rep.shape[2], a_rep.shape[3]),
        ).view(a_rep.shape[0], a_rep.shape[1], q_rep.shape[2], a_rep.shape[2])

        qa_attention_weights = masked_softmax(qa_similarity, question_mask[..., None], dim=2)  # formula
        attended_q = torch.einsum('bnqa,bnqd->bnad', (qa_attention_weights, q_rep))

        # Have a second attention over the objects, do A by Objs
        # [batch_size, 4, answer_length, num_objs]
        atoo_similarity = self.obj_attention(a_rep.view(a_rep.shape[0], a_rep.shape[1] * a_rep.shape[2], -1),
                                             obj_reps['obj_reps']).view(a_rep.shape[0], a_rep.shape[1],
                                                                        a_rep.shape[2],
                                                                        obj_reps['obj_reps'].shape[1])
        atoo_attention_weights = masked_softmax(atoo_similarity, box_mask[:, None, None])
        attended_o = torch.einsum('bnao,bod->bnad', (atoo_attention_weights, obj_reps['obj_reps']))

        # print('qa_similarity.size(): ', qa_similarity.size(),'\n', 'attended_q.size(): ', attended_q.size(),'\n',
        #       'qa_attention_weights.size(): ', qa_attention_weights.size(),'\n','atoo_attention_weights.size() :',atoo_attention_weights.size(),'\n','attended_o.size() ',attended_o.size())

        # reasoning_inp = torch.cat([x for x, to_pool in [(a_rep, self.reasoning_use_answer),
        #                                                 (attended_o, self.reasoning_use_obj),
        #                                                 (attended_q, self.reasoning_use_question)]
        #                            if to_pool], -1)
        #
        #






        qa_rep = torch.cat((attended_q,attended_o),dim = 2)
        # reasoning_inp = torch.cat([x for x, to_pool in [(a_rep, self.reasoning_use_answer),
        #                                                 (attended_o, self.reasoning_use_obj),
        #                                                 (attended_q, self.reasoning_use_question)]
        #                            if to_pool], -1)
        #
        # print(qa_rep.size(),'reasoning_inp.size()')  [96,4,29,1536]

        # qa_rep = self.qao_reshape(qao_rep)

        if self.rnn_input_dropout is not None:
            qa_rep = self.rnn_input_dropout(qa_rep)

            #################################################################################


        #clip
        # a1, a2, a3, a4 = qa_rep.size()
        # if a3 >= 50:
        #     qa_rep = qa_rep[:, :, 0:50, :].contiguous()
        #
        # else:
        #     padding_i = torch.zeros(a1, a2, 50 - a3, a4).cuda()
        #     qa_rep = torch.cat((qa_rep, padding_i), dim=2)
        # # print(qa_rep.size(),'qa_rep.size()')
    ##################################################################
        qa_rep_size = qa_rep.size()

        qa_rep = qa_rep.view(qa_rep_size[0]*qa_rep_size[1],qa_rep_size[2],qa_rep_size[3])  #qa_rep[96*4,50,768]

    #########################################################################################################

        output, (h_n, h_c) = self.lstm(qa_rep)   # self.reasoning_encoder(qa_rep) #           #    #
        output = output[:, -1, :]  # 取最后一层输出  [96*4,1024]
    ###########################################################################################################
        # # '''这一段是transformer Encoder实现'''
        # #
        # transformer = self.transformer_encoder_vc(qa_rep)

    # #####################################################################################################3
    #     '''这是average pooling'''
    #     output = torch.transpose(transformer,1,2).contiguous()
    #
    #     output = self.AvgPool(output)
    #     output = torch.transpose(output, 1, 2).contiguous()
    #     output = output.squeeze()#[96 x 4, 512]
    #
    #     # output = self.linear(output)
    ##########################################################################################################
        # qa_feature_mask = self.make_mask(transformer)
        # # qa_image_mask = self.make_mask(qa_visual_context)
        #
        # # qa_image_res = self.image_AttFlat(qa_visual_context, qa_image_mask)  # [b,dim]
        # qa_feature_res = self.image_AttFlat(transformer, qa_feature_mask)
        # qa_feature_res = qa_feature_res.view(-1, 4, 1024)
        # # qa_image_res = torch.unsqueeze(qa_image_res, 1)
        # # qa_image_res = qa_image_res.repeat(1, 4, 1)
        #
        # qa_proj_feat = qa_feature_res  # + qa_image_res
        # proj_feat = self.proj_norm(qa_proj_feat)
        # output_res = proj_feat

    #
    # #######################################################################################################
        '''LSTM'''

        # qa_feature ,(h_n, h_c)= self.lstm(qa_rep)
        # qa_feature = qa_feature[:,-1,:]


    ##############################################################################################################
        # '''这一段是 DICT'''
        #
        # output = self.linear(output)
        att = output @ self.memory_cell.T

        att = self.lstm_norm(att)


        att = F.softmax(att, dim=1)  # [96*4,10000]


        output_res = att @ self.memory_cell

        # print('output_res:', output_res.size())

    #########################################################
        output_res = output_res.view(-1,4,64)
        # output_res = self.proj_norm(output_res)
        logits = self.final_mlp(output_res).squeeze(-1)#      [94,4,1024]
        # print(logits.size(),'kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk')
        class_probabilities = F.softmax(logits,dim = -1)

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

class AttFlat(nn.Module):
    def __init__(self, hidden_size):
        super(AttFlat, self).__init__()
        self.hidden_size = hidden_size
        self.flat_mlp_size = 512
        self.flat_glimpses = 1
        self.drop_out = 0.1
        self.flat_out_size = 1024
        self.mlp = MLP(
            in_size=self.hidden_size,
            mid_size=self.flat_mlp_size,
            out_size=self.flat_glimpses,
            dropout_r=self.drop_out,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            self.hidden_size * self.flat_glimpses,
            self.flat_out_size
        )

        self.lstm = nn.LSTM(
            input_size=self.flat_glimpses,
            hidden_size=self.hidden_size,
            num_layers=2,  # hidden_layer的数目
            batch_first=True,  # 输入数据的维度一般是（batch, time_step, input)，该属性表征batch是否放在第一个维度

        )
        ####################################################

    def forward(self, x, x_mask):
        att = self.mlp(x)

        # ---------------------------------------------------------------------------------
        # ----------------------mask mask mask --------------------------------------------
        # ---------------------------------------------------------------------------------

        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )

        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.flat_glimpses):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        # print(x_atted.shape,x.shape)
        x_atted = self.linear_merge(x_atted)

        return x_atted

class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x

class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))

class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2





