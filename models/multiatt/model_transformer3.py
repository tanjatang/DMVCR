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
                 input_dropout: float = 0.5,
                 hidden_dim_maxpool: int = 512, #1024,
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
        self.dict_size = 200
        path = '/home/tangxueq/MA_tang/r2c/models/saves/memory_cell1000.npz'
        self.memory_cell1 = torch.nn.Parameter(torch.Tensor(self.memory_cell_load(path, self.dict_size, 768)))
        print(self.memory_cell1)

        path = '/home/tangxueq/MA_tang/r2c/models/saves/memory_cell1000.npz'
        self.memory_cell2 = torch.nn.Parameter(torch.Tensor(self.memory_cell_load(path, self.dict_size, 768)))
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
        ###################################################################################################

        self.vclstm = nn.LSTM(
            input_size=768,
            hidden_size=512,#1024,
            num_layers=2,  # hidden_layer的数目
            batch_first=True,  # 输入数据的维度一般是（batch, time_step, input)，该属性表征batch是否放在第一个维度
            dropout= input_dropout
        )

        self.cvlstm = nn.LSTM(
            input_size=768,
            hidden_size=512,#1024,
            num_layers=2,  # hidden_layer的数目
            batch_first=True,  # 输入数据的维度一般是（batch, time_step, input)，该属性表征batch是否放在第一个维度
            dropout=input_dropout
        )

        self.VisualAttention = VisualAttention(512, 768)
        self.ContextAttention = ContextAttention(512, 768)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

###############################################################################################33
        self.encoder_layer_vc = nn.TransformerEncoderLayer(d_model=768, nhead=8)
        self.transformer_encoder_vc = nn.TransformerEncoder(self.encoder_layer_vc, num_layers=6)

        self.encoder_layer_cv = nn.TransformerEncoderLayer(d_model=768, nhead=8)
        self.transformer_encoder_cv = nn.TransformerEncoder(self.encoder_layer_cv, num_layers=6)
        self.dropout = nn.Dropout(input_dropout)
######################################################################################################
        self.image_AttFlat = AttFlat(512)
        self.qa_AttFlat = AttFlat(768)

        self.proj_norm = LayerNorm(1024)
        self.proj = nn.Linear(1024, 4)

        self.lstm_norm = torch.nn.BatchNorm1d(self.dict_size)

        self.rnn_input_dropout = TimeDistributed(InputVariationalDropout(input_dropout)) if input_dropout > 0 else None

        self.span_encoder = TimeDistributed(span_encoder)
        self.span_reshape = TimeDistributed(torch.nn.Linear(512, 768))
        self.qao_reshape = torch.nn.Linear(512,768)

        self.reshape = nn.Linear(512, 768*4)#
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
        dim = (768*2) * 4     #3584##768*2 #sum([d for d, to_pool in [(reasoning_encoder.get_output_dim(), self.pool_reasoning),
                                        # (span_encoder.get_output_dim(), self.pool_answer),
                                        # (span_encoder.get_output_dim(), self.pool_question)] if to_pool])

        self.final_mlp = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(dim, hidden_dim_maxpool),
            torch.nn.RReLU(inplace=True),
            torch.nn.Dropout(input_dropout, inplace=False),
            torch.nn.Linear(hidden_dim_maxpool, 4),
        )
        # self.final_mlp = torch.nn.Sequential(
        #     torch.nn.Flatten(),
        #     torch.nn.Linear(4*1024,4))


        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
      #####################################################################


        initializer(self)

        ##############################################################################


    def memory_cell_load(self,path, dict_size, dim):
        if os.path.isfile(path):
            print('load memory_cell from {0}'.format(path))
            # memory_init = np.load(self.memory_cell_path)['memory_cell']
            memory_init = torch.load(path)['memory_cell'][()]
        else:
            print('create a new memory_cell')
            # memory_init = np.random.rand(10000, 1024)/ 100
            # memory_init = torch.random(10000,1024)/100
            memory_init = torch.randn(dict_size, dim)/100
        memory_init = memory_init.float()

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
        dim = M.size(1)
        att = h @ M.T

        att = self.lstm_norm(att)
        att = F.softmax(att, dim=1)  # [96*4,10000]

#########################################################################
        att = self.dropout(att)
        ################################################################

        Toutput_res = att @ M
        Toutput_res = Toutput_res.view(-1, dim)

        return Toutput_res


    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)

    def forward(self,

                # training_mode,
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

##################################################################################################################
        '''try idea to create new model with dictionary'''

        '''padding q_rep and a_rep with 0'''

        obj_features = obj_reps['obj_reps'] #[b, num, 512]
        question_features = question['bert'] #[b, 4, len, 768]
        answers_features = answers['bert']  #[b, 4, len, 768]
        # for i in range(question_features.size(1)):
        #     q_text = question_features[:,i,:,:]
        #     a_text = answers_features[:,i,:,:]
        #
        #     #refine visual feature with context
        #     q_vc = self.VisualAttention(obj_features,q_text)
        #     a_vc = self.VisualAttention(obj_features,a_text)
        #
        #
        #     # refine context feature with visual
        #     q_cv = self.ContextAttention(obj_features,q_text,4)
        #     a_cv = self.ContextAttention(obj_features,a_text,4)
        #
        # qa_vc = torch.cat((q_vc,a_vc),dim = 1) #[b,num,512]
        # qa_cv = torch.cat((q_cv, a_cv), dim=1) #[b,len,768]
        #
        #
        # #encoder qa commonsense
        # vc_qa_rep, (h_vc, h_vc) = self.vclstm(qa_vc)  # [b,1024]
        # cv_qa_rep, (h_cv, h_cv) = self.cvlstm(qa_cv)  # [b ,1024]
        #
        # vc_qa_rep = vc_qa_rep[:,-1,:]
        # cv_qa_rep = cv_qa_rep[:,-1,:]
        #
        # vc_qa_res = self.Dictionary(vc_qa_rep, self.memory_cell1)  # [96, 1024]
        # cv_qa_res = self.Dictionary(cv_qa_rep, self.memory_cell2)  # #[b,1024]
        # output_res = torch.cat((vc_qa_res, cv_qa_res), dim=-1)  # [96, 1024*2]




        #refine visual feature with context
        q_visual_context = self.VisualAttention(obj_features,question_features) #[b, num, 512]
        a_visual_context = self.VisualAttention(obj_features,answers_features) #[b, num, 512]
        qa_visual_context = torch.cat((q_visual_context, a_visual_context), dim = 1)  #[b, num, 512]

        #refine context features with visual information
        q_context_visual = self.ContextAttention(obj_features, question_features,4)#[b, 4, len, 768]
        a_context_visual = self.ContextAttention(obj_features, answers_features, 4)#[b, 4, len, 768]
        qa_context_visual = torch.cat((q_context_visual, a_context_visual), dim = 2) #[b, 4, len, 768]
        batch_size, text_dim = qa_context_visual.size(0), qa_context_visual.size(3)
        qa_context_visual_inp = qa_context_visual.view(batch_size*4, -1, text_dim) #[b x 4, len, 768]
        ####################################################################################################
        # qa_context_visual_inp = self.transformer_encoder_cv(qa_context_visual_inp)



################################################################################################################
        # qa_feature_mask = self.make_mask(qa_context_visual_inp)
        # # qa_image_mask = self.make_mask(qa_visual_context)
        #
        # # qa_image_res = self.image_AttFlat(qa_visual_context, qa_image_mask)  # [b,dim]
        # qa_feature_res = self.qa_AttFlat(qa_context_visual_inp, qa_feature_mask)
        # qa_feature_res = qa_feature_res.view(-1, 4, 1024)
        # # qa_image_res = torch.unsqueeze(qa_image_res, 1)
        # # qa_image_res = qa_image_res.repeat(1, 4, 1)
        #
        # qa_proj_feat = qa_feature_res #+ qa_image_res
        # proj_feat = self.proj_norm(qa_proj_feat)
        # output_res = proj_feat






        
        # Encoder information lstm
        # vc_qa_rep, (h_vc, h_vc) = self.vclstm(qa_visual_context)    #[b,1024]
        # cv_qa_rep, (h_cv, h_cv) = self.cvlstm(qa_context_visual_inp) #[b x 4,1024]
        #
        # vc_qa_rep = vc_qa_rep[:, -1, :]  # 取最后一层输出  [96,1024]
        # cv_qa_rep = cv_qa_rep[:, -1, :]  #[b x 4,1024]
        #
        # vc_qa_res = self.Dictionary(vc_qa_rep, self.memory_cell1)  # [96, 1024]
        # cv_qa_res = self.Dictionary(cv_qa_rep, self.memory_cell2)  # #[b x 4,1024]
        # cv_qa_res = cv_qa_res.view(batch_size,4,-1)
        #
        # vc_qa_res = torch.unsqueeze(vc_qa_res,1)   # [96, 1, 1024]
        # vc_qa_res = vc_qa_res.repeat(1,4,1)            # [96, 4, 1024]

        # Encoder information transformer
        cv_qa = self.transformer_encoder_cv(qa_context_visual_inp) #[b*4,len,dim]
        cv_qa = torch.transpose(cv_qa,1,2).contiguous()            #[b*4, dim, len]
        cv_qa_rep = self.avg_pool(cv_qa)
        cv_qa_res = self.Dictionary(cv_qa_rep.squeeze(), self.memory_cell1) #[b*4, 768]
        cv_qa_res = cv_qa_res.view(batch_size,4,-1)




        #############################global average pooling
        # vc_qa_rep = self.avg_pool(qa_visual_context).squeeze(1)
        # qa_context_visual_inp = torch.transpose(qa_context_visual_inp,1,2).contiguous()
        #
        # cv_qa_rep = self.avg_pool(qa_context_visual_inp).squeeze()

        #
        # vc_qa_res = self.Dictionary(vc_qa_rep,self.memory_cell1)
        # cv_qa_res = self.Dictionary(cv_qa_rep,self.memory_cell2)
        #

        # cv_qa_res = cv_qa_res.view(batch_size, 4, -1)
        # vc_qa_res = torch.unsqueeze(vc_qa_rep, 1)  # [96, 1, 1024]
        # vc_qa_res = vc_qa_res.repeat(1, 4, 1)  # [96, 4, 1024]

        # print(vc_qa_res.size(),cv_qa_res.size(),'kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk')
        # output_res = torch.cat((vc_qa_res, cv_qa_res), dim=-1)  # [96, 4, 1024*2]


        ####################################################################
        #纯文字字典
        qa_features = torch.cat((question_features,answers_features),dim = 2)
        qa_features = qa_features.view(batch_size*4, -1, 768)

        # qa_rep, (h_cv, h_cv) = self.vclstm(qa_features)
        # qa_rep = qa_rep[:,-1,:]
        qa_trans = self.transformer_encoder_vc(qa_features)
        qa_trans = torch.transpose(qa_trans,1,2).contiguous()
        qa_rep = self.avg_pool(qa_trans)
        qa_res = self.Dictionary(qa_rep.squeeze(), self.memory_cell2)
        qa_res = qa_res.view(batch_size, 4, -1)

        ###############################################################

        output_res = torch.cat((cv_qa_res, qa_res),dim = 2)


###########################################################################################
        logits = self.final_mlp(output_res)#.squeeze(2       [94,1024*2]
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



#####################################################################################################

def conv1x1(in_planes, out_planes):
    "1x1 convolution with padding"
    # print('N:',out_planes)
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=False)

def func_attention(query, context, gamma1):
    """
    query: batch x ndf x queryL   ,text
    context: batch x ndf x n (sourceL=ihxiw)图
    mask: batch_size x sourceL
    """
    batch_size, queryL = query.size(0), query.size(2)
    num = context.size(2)
    sourceL = num

    # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query)

    # --> batch*sourceL x queryL
    attn = attn.view(batch_size*sourceL, queryL)
    attn = nn.Softmax()(attn)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size*queryL, sourceL)

    attn = attn * gamma1

    # bnorm = torch.nn.BatchNorm1d(sourceL)
    # attn = bnorm(attn.cpu())

    attn = nn.Softmax()(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT.cuda())

    return weightedContext, attn.view(batch_size, -1, num)


class VisualAttention(nn.Module):
    def __init__(self, idf, cdf):
        super(VisualAttention, self).__init__()
        # print("N1:",N)
        self.conv_context = nn.Linear(cdf, idf) #conv1x1(cdf, idf)

        # self.conv_context3 = conv1x1(cdf, 128*128)
        self.sm = nn.Softmax()
        self.dropout = nn.Dropout(0.5)



    def forward(self, obj, text):
        """
                    obj: batch x numn x idf (idf = 512)
                    text: batch x 4 x len x cdf (cdf = 768)
                """

        ## reshape text
        batch_size, text_dim = text.size(0), text.size(3)
        batch_size= text.size(0)
        num, visual_dim = obj.size(1), obj.size(2)
        text = self.conv_context(text) #[batch, 4, len, 512]

        text = text.view(batch_size, -1, visual_dim)      #[batch, 4 x len, 512]
        text = torch.transpose(text,1,2).contiguous()   #[batch, 512, 4 x len]


        #calculate attn

        attn_text = torch.bmm(obj, text) # [batch, num, 512] [batch, 512, 4 x len] -> [batch, num, 4 x len]
        attn_text = attn_text.view(batch_size * num, -1) #[batch x num, 4 x len]
        attn_text = self.sm(attn_text)                   #[batch x num, 4 x len]
        attn_text = attn_text.view(batch_size, num, -1)  #[batch, num, 4 x len]
        attn_text = torch.transpose(attn_text, 1, 2).contiguous()     #[batch, 4 x len, num]

        attn_text = self.dropout(attn_text)

        #calculate refine visual features with context
        weightedText = torch.bmm(text,attn_text)          #[batch, 512, 4 x len]  [batch, 4 x len, num] -> [batch, 512, num]
        weightedText = torch.transpose(weightedText,1, 2).contiguous() #[batch, num, 512]

        return weightedText

class ContextAttention(nn.Module):
    def __init__(self, idf, cdf):
        super(ContextAttention, self).__init__()

        self.conv_context = nn.Linear(idf, cdf)  # conv1x1(cdf, idf)

        self.sm = nn.Softmax()
        self.dropout = nn.Dropout(0.5)


    def forward(self, obj, text, gamma):
        """
                    obj: batch x numn x idf (idf = 512)
                    text: batch x 4 x len x cdf (cdf = 768)
                """

        ## reshape text
        batch_size, text_dim = text.size(0), text.size(3)
        batch_size = text.size(0)
        num, obj_dim = obj.size(1), obj.size(2)
        text = text.view(batch_size, -1, text_dim)  # [batch, 4 x len, 768]
        text = torch.transpose(text, 1, 2).contiguous()  # [batch, 768, 4 x len]

        #mapping object features to text dimention
        objects = self.conv_context(obj)  # [batch, num, 768]

        # calculate attn
        attn_v = torch.bmm(objects, text)  # [batch, num, 768] [batch, 768, 4 x len] -> [batch, num, 4 x len]
        attn_v = attn_v.view(batch_size * num, -1)  # [batch x num, 4 x len]
        attn_v = attn_v * gamma
        attn_v = self.sm(attn_v)  # [batch x num, 4 x len]
        attn_v = attn_v.view(batch_size, num, -1)  # [batch, num, 4 x len]

        attn_v = self.dropout(attn_v)

        objectsT = torch.transpose(objects, 1, 2).contiguous()  # [batch, 768, num]

        # calculate refine context features with visual information
        weightedText = torch.bmm(objectsT, attn_v)  # [batch, 768, num]  [batch, num, 4 x len] -> [batch, 768, 4 x len]
        weightedText = torch.transpose(weightedText, 1, 2).contiguous()  # [batch, 4 x len, 768]
        weightedText = weightedText.view(batch_size, 4, -1, text_dim)     # [batch, 4 , len, 768]

        return weightedText





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

        self.lstm = nn.LSTM(
            input_size=self.flat_glimpses,
            hidden_size=self.hidden_size,
            num_layers=2,  # hidden_layer的数目
            batch_first=True,  # 输入数据的维度一般是（batch, time_step, input)，该属性表征batch是否放在第一个维度

        )
        ####################################################

    def forward(self, x, x_mask):


        att = self.mlp(x)

#---------------------------------------------------------------------------------
#----------------------mask mask mask --------------------------------------------
#---------------------------------------------------------------------------------

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




























