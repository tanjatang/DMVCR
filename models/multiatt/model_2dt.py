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
                 # lstm_encoder: Seq2SeqEncoder,
                 input_dropout: float = 0.45,
                 hidden_dim_maxpool: int =256,
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
        self.dic_size = 200

        path = '/home/tangxueq/MA_tang/r2c/models/saves/memory_cell'
        m1 = os.path.join(path, 'memory_cell1'+ '.npz')
        m2 = os.path.join(path, 'memory_cell2' + '.npz')
        self.memory_cell1, self.memory_cell2 = self.memory_cell_load(m1,m2)
        self.memory_cell1 = torch.nn.Parameter(torch.Tensor(self.memory_cell1))
        self.memory_cell2 = torch.nn.Parameter(torch.Tensor(self.memory_cell2))

        torch.save(self.memory_cell1,m1)
        print(self.memory_cell1)



        # self.memory_cell2 = torch.nn.Parameter(torch.Tensor(self.memory_cell_load(m1,m2)))
        torch.save(self.memory_cell2,m2)
        print(self.memory_cell2)


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
        INPUT_SIZE =768 #768
        self.Tlstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=1024,
            num_layers=2,  # hidden_layer的数目
            batch_first=True,  # 输入数据的维度一般是（batch, time_step, input)，该属性表征batch是否放在第一个维度
            dropout = input_dropout
        )

        self.Vlstm = nn.LSTM(
            input_size=512,
            hidden_size=512,
            num_layers=2,  # hidden_layer的数目
            batch_first=True,  # 输入数据的维度一般是（batch, time_step, input)，该属性表征batch是否放在第一个维度
            dropout= input_dropout
        )


        ######################################################################################

        self.linear_v = torch.nn.Linear(512,64)
        self.linear_c = torch.nn.Linear(768,64)



        self.encoder_layer_vc = nn.TransformerEncoderLayer(d_model=768, nhead=8)
        self.transformer_encoder_vc = nn.TransformerEncoder(self.encoder_layer_vc, num_layers=3)

        self.encoder_layer_cv = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer_encoder_cv = nn.TransformerEncoder(self.encoder_layer_cv, num_layers=3)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        ######################################################################################

        self.dropout = nn.Dropout(input_dropout)

        self.lstm_norm = torch.nn.BatchNorm1d(self.dic_size)
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
        dim = 4*(64*2)#768*2 #sum([d for d, to_pool in [(reasoning_encoder.get_output_dim(), self.pool_reasoning),
                                        # (span_encoder.get_output_dim(), self.pool_answer),
                                        # (span_encoder.get_output_dim(), self.pool_question)] if to_pool])

        # self.final_mlp = torch.nn.Sequential(
        #     torch.nn.Flatten(),
        #     torch.nn.Dropout(input_dropout, inplace=False),
        #     torch.nn.Linear(dim, hidden_dim_maxpool),
        #     torch.nn.ReLU(inplace=True),
        #     torch.nn.Dropout(input_dropout, inplace=False),
        #     torch.nn.Linear(hidden_dim_maxpool, 4),
        # )
        self.final_mlp = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(dim,4))


        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
      #####################################################################


        initializer(self)

        ##############################################################################
        # self.memory_cell_path = getattr(opt, 'memory_cell_path', '0')


    def memory_cell_load(self,m1,m2):
        if os.path.isfile(m1) and os.path.isfile(m2):
            print('load memory_cell from {0}'.format(m1),'{}'.format(m2))

            # memory_init = np.load(self.memory_cell_path)['memory_cell']
            memory_init1 = torch.load(m1)#['memory_cell1']#[()]


            memory_init2 = torch.load(m2)#['memory_cell2']#[()]
        else:
            print('create a new memory_cell')
            # memory_init = np.random.rand(10000, 1024)/ 100
            # memory_init = torch.random(10000,1024)/100
            memory_init1 = torch.randn(self.dic_size, 64)/100  #randn
            memory_init2 = torch.randn(self.dic_size, 64) / 100
        memory_init1 = memory_init1.float()
        memory_init2 = memory_init2.float()
        # self.memory_cell = torch.from_numpy(memory_init).cuda().requires_grad_()

        return memory_init1, memory_init2


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
        # h_size = h.size()  # [96*4,768]
        # # h = h.view(-1, h_size[3])  # [(96*4),768]
        # att = torch.mm(h, torch.t(M))  # [(96*4*50),768] * [768,10000]
        # att = F.softmax(att, dim=1)  # [96*4*50,10000]
        #
        # att_res = torch.mm(att, M)  # [96*4,10000]*[10000,768]    ->   #[96*4,768]
        # # att_res = att_res.view([-1, h_size[1]*h_size[2], h_size[3]])  #[96*4,768]

        dim = h.size(1)

        att = h @ M.T
        att = self.lstm_norm(att)

        att = F.softmax(att, dim=1)  # [96*4,10000]
       #############################################################3

        att = self.dropout(att)
        ################################################################
        # print('att1:', att.backward())
        # print('att1:', att.backward())
        # print("att_res :",att.grad)
        # output_res = torch.mm(att, self.memory_cell)  # [96*4,10000]*[10000,768]    ->   #[96*4,768]
        Toutput_res = att @ M
        # Toutput_res = Toutput_res.view(-1, 4, 1024)


        return Toutput_res

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




            # qa_rep = self.padding(question['bert'],answers['bert'])  #[batch_size, 8, seq_len, 768]
        Tqa_rep = torch.cat((question['bert'],answers['bert']),dim = 2)



        q_rep, q_obj_reps = self.embed_span(question, question_tags, question_mask, obj_reps['obj_reps'])
        a_rep, a_obj_reps = self.embed_span(answers, answer_tags, answer_mask, obj_reps['obj_reps'])

        # print(qa_rep.size(), 'qa_rep with image information!')
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

        Vqa_rep = torch.cat((attended_q,attended_o),dim = 2)
        # reasoning_inp = torch.cat([x for x, to_pool in [(a_rep, self.reasoning_use_answer),
        #                                                 (attended_o, self.reasoning_use_obj),
        #                                                 (attended_q, self.reasoning_use_question)]
        #                            if to_pool], -1)

        # print(reasoning_inp.size(),'reasoning_inp.size()')  [96,4,29,1536]

        # qa_rep = self.qao_reshape(qao_rep)

        if self.rnn_input_dropout is not None:
            Tqa_rep = self.rnn_input_dropout(Tqa_rep)
            Vqa_rep = self.rnn_input_dropout(Vqa_rep)


##################################################################
        Tqa_rep_size = Tqa_rep.size()

        Tqa_rep = Tqa_rep.view(Tqa_rep_size[0]*Tqa_rep_size[1],Tqa_rep_size[2],Tqa_rep_size[3])  #qa_rep[96*4,50,768]
##############################################################################



        Toutput = self.transformer_encoder_vc(Tqa_rep)

        # Toutput, (h_n, h_c) = self.Tlstm(Tqa_rep)   # self.reasoning_encoder(qa_rep) #           #    #

        # output = self.lstm_norm(output)
        Toutput = torch.transpose(Toutput,1,2).contiguous()  # 取最后一层输出  [96*4,786]
        Toutput = self.avg_pool(Toutput).squeeze()
        Toutput = self.linear_c(Toutput)

        Toutput_res = self.Dictionary(Toutput,self.memory_cell1) #[96,4,786]
        Toutput_res = Toutput_res.view(Tqa_rep_size[0],4,-1)



        Vqa_rep_size = Vqa_rep.size()
        Vqa_rep = Vqa_rep.view(Vqa_rep_size[0]*Vqa_rep_size[1],Vqa_rep_size[2],Vqa_rep_size[3])

    ###################################################################

        Voutput = self.transformer_encoder_cv(Vqa_rep)

        Voutput = torch.transpose(Voutput, 1, 2).contiguous()  # 取最后一层输出  [96*4,786]
        Voutput = self.avg_pool(Voutput).squeeze()


        # Voutput, (h,c) = self.Vlstm(Vqa_rep)
        # Voutput = Voutput[:,-1, :]

        # Voutput = torch.transpose(Voutput, 1, 2).contiguous()
        Voutput = self.linear_v(Voutput)

        Voutput_res = self.Dictionary(Voutput,self.memory_cell2)
        Voutput_res = Voutput_res.view( Vqa_rep_size[0],4,-1)

        output_res = torch.cat((Toutput_res,Voutput_res),dim = -1)



#########################################################################################################

        logits = self.final_mlp(output_res)#.squeeze(2       [94,4,1024]
        # print(logits)
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




