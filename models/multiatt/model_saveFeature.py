"""
Let's get the relationships yo
"""

from typing import Dict, List, Any

import torch
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
# SAVE_ROOT = "/localstorage/tangxueq/tmp/"
SAVE_ROOT = "/phys/ssd/tangxueq/tmp/data/TXT/answer"
# SAVE_ROOT = "/data/scene_understanding/tangxueq/VCR/answer"

@Model.register("MultiHopAttentionQA")
class AttentionQA(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 span_encoder: Seq2SeqEncoder,
                 reasoning_encoder: Seq2SeqEncoder,
                 input_dropout: float = 0.3,
                 hidden_dim_maxpool: int = 1024,
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

       #-------------------------------------------------------------------------------------------
        with torch.no_grad():

        #------------------------------------------------------------------------------------------


            self.detector = SimpleDetector(pretrained=True, average_pool=True, semantic=class_embs, final_dim=512)#768
            ###################################################################################################



            self.rnn_input_dropout = TimeDistributed(InputVariationalDropout(input_dropout)) if input_dropout > 0 else None

            self.span_encoder = TimeDistributed(span_encoder)
            # self.span_reshape = TimeDistributed(torch.nn.Linear(512, 768))

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
            dim = sum([d for d, to_pool in [(reasoning_encoder.get_output_dim(), self.pool_reasoning),
                                            (span_encoder.get_output_dim(), self.pool_answer),
                                            (span_encoder.get_output_dim(), self.pool_question)] if to_pool])

            self.final_mlp = torch.nn.Sequential(
                torch.nn.Dropout(input_dropout, inplace=False),
                torch.nn.Linear(dim, hidden_dim_maxpool),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(input_dropout, inplace=False),
                torch.nn.Linear(hidden_dim_maxpool, 1),
            )
            self._accuracy = CategoricalAccuracy()
            self._loss = torch.nn.CrossEntropyLoss()




            initializer(self)
            # self.count = 0

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
        retrieved_feats = self._collect_obj_reps(span_tags, object_reps) #object_reps[96,23,768]   #retrieved_feats[96,4,13,768]

        span_rep = torch.cat((span['bert'], retrieved_feats), -1)
        # print(span_rep.size(),'span_rep.size()')                    #span_rep[96,4,13,768*2]
        # print(retrieved_feats.size(), 'retrieved_feats.size()')
        # print(span_mask.size(), 'span_mask.size()')                    #span_mask[96,4,13]

        # add recurrent dropout here
        if self.rnn_input_dropout:
            span_rep = self.rnn_input_dropout(span_rep)
            # print(span_rep.size(), 'span_rep.size() 1')



        # x = self.span_encoder(span_rep, span_mask)
        # x = self.span_reshape(x)
        #   #[96,4,13,768]
        # print(x.size(),'x.size()')
        #
        #
        # return x, retrieved_feats
        return self.span_encoder(span_rep, span_mask), retrieved_feats


    def forward(self,
                images: torch.Tensor,
                # obj_reps: torch.Tensor,
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
                label: torch.LongTensor = None

                ) -> Dict[str, torch.Tensor]:
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

        # print('//////////',images.size())

        for tag_type, the_tags in (('question', question_tags), ('answer', answer_tags)):
            if int(the_tags.max()) > max_len:
                raise ValueError("Oh no! {}_tags has maximum of {} but objects is of dim {}. Values are\n{}".format(
                    tag_type, int(the_tags.max()), objects.shape, the_tags
                ))

        obj_reps = self.detector(images=images, boxes=boxes, box_mask=box_mask, classes=objects, segms=segms)
        # print('//////////////////////////',obj_reps,'ppppppppppppppppppppppppppppp')
#################################################################################################################


###########################################################################################################3


        #########################################################################################################################
        # print("obj_reps:\n", type(obj_reps), len(obj_reps), obj_reps['obj_reps_raw'].size(),obj_reps['obj_reps'].size(), obj_reps['obj_logits'].size(),obj_reps['obj_labels'].size(),obj_reps['cnn_regularization_loss'].size(),"\n")
        # print(obj_reps)
        # raise InterruptedError("Error")
        # ipdb.set_trace()


        # print(obj_reps['obj_reps'].size())
        # print(obj_reps)
        # raise InterruptedError
        # Now get the question representations
        # print(question['bert'].size(),'question[bert].size()')
        # print(obj_reps['obj_reps'].size(),"obj_reps['obj_reps']")

        q_rep, q_obj_reps = self.embed_span(question, question_tags, question_mask, obj_reps['obj_reps'])
        # print(q_rep.size(), 'q_rep.size()')
        a_rep, a_obj_reps = self.embed_span(answers, answer_tags, answer_mask, obj_reps['obj_reps'])


        # raise InterruptedError

        # print('question; ',question['bert'].size(),'\n','answers: ',answers['bert'].size(),'\n','a_rep.size(): ',a_rep.size(),'\n','a_obj_reps.size(): ',a_obj_reps.size(),'\n','q_rep.size(): ', q_rep.size(), '\n','q_obj_reps.size(): ',q_obj_reps.size())


        ####################################
        # Perform Q by A attention
        # [batch_size, answer number, question_length, answer_length]
        qa_similarity = self.span_attention(
            q_rep.view(q_rep.shape[0] * q_rep.shape[1], q_rep.shape[2], q_rep.shape[3]),
            a_rep.view(a_rep.shape[0] * a_rep.shape[1], a_rep.shape[2], a_rep.shape[3]),
        ).view(a_rep.shape[0], a_rep.shape[1], q_rep.shape[2], a_rep.shape[2])

        qa_attention_weights = masked_softmax(qa_similarity, question_mask[..., None], dim=2) #formula
        attended_q = torch.einsum('bnqa,bnqd->bnad', (qa_attention_weights, q_rep))


        # Have a second attention over the objects, do A by Objs
        # [batch_size, 4, answer_length, num_objs]
        atoo_similarity = self.obj_attention(a_rep.view(a_rep.shape[0], a_rep.shape[1] * a_rep.shape[2], -1),
                                             obj_reps['obj_reps']).view(a_rep.shape[0], a_rep.shape[1],
                                                            a_rep.shape[2], obj_reps['obj_reps'].shape[1])
        atoo_attention_weights = masked_softmax(atoo_similarity, box_mask[:,None,None])
        attended_o = torch.einsum('bnao,bod->bnad', (atoo_attention_weights, obj_reps['obj_reps']))

        # print('qa_similarity.size(): ', qa_similarity.size(),'\n', 'attended_q.size(): ', attended_q.size(),'\n',
        #       'qa_attention_weights.size(): ', qa_attention_weights.size(),'\n','atoo_attention_weights.size() :',atoo_attention_weights.size(),'\n','attended_o.size() ',attended_o.size())



        reasoning_inp = torch.cat([x for x, to_pool in [(a_rep, self.reasoning_use_answer),
                                                           (attended_o, self.reasoning_use_obj),
                                                           (attended_q, self.reasoning_use_question)]
                                      if to_pool], -1)

        # print('reasoning_inp.size(): ', reasoning_inp.size())

        if self.rnn_input_dropout is not None:
            reasoning_inp = self.rnn_input_dropout(reasoning_inp)
        reasoning_output = self.reasoning_encoder(reasoning_inp, answer_mask)#
        # print('reasoning_output.size(): ', reasoning_output.size(),'\n','answer_mask: ',answer_mask.size())

        ###########################################
        things_to_pool = torch.cat([x for x, to_pool in [(reasoning_output, self.pool_reasoning),
                                                         (a_rep, self.pool_answer),
                                                         (attended_q, self.pool_question)] if to_pool], -1)
        # print('things_to_pool.size(): ', things_to_pool.size())

        pooled_rep = replace_masked_values(things_to_pool,answer_mask[...,None], -1e7).max(2)[0]#  things_to_pool :[96,4,29,1536]     answer_mask [96, 4, 29]
        # print('pooled_rep.size(): ', pooled_rep.size())
        # pooled_rep = pooled_rep[0]
        logits = self.final_mlp(pooled_rep).squeeze(2)


        # print('pooled_rep.size(): ', pooled_rep.size(),'\n','answer_mask: ',answer_mask.size())   #   pooled_rep :[96,4,1536]

        # raise InterruptedError
        ###########################################

        class_probabilities = F.softmax(logits, dim=-1)

        # print(class_probabilities,'class_probabilities')
        # output_dict = {"label_logits": logits, "label_probs": class_probabilities,
        #                'cnn_regularization_loss': obj_reps['cnn_regularization_loss'],
        #                # Uncomment to visualize attention, if you want
        #                # 'qa_attention_weights': qa_attention_weights,
        #                # 'atoo_attention_weights': atoo_attention_weights,
        #                }

        output_dict = {"label_logits": logits, "label_probs": class_probabilities,
                      'cnn_regularization_loss': obj_reps['cnn_regularization_loss'],
                       # Uncomment to visualize attention, if you want
                       # 'qa_attention_weights': qa_attention_weights,
                       # 'atoo_attention_weights': atoo_attention_weights,
                       }


        if label is not None:
            # print(label.long().view(-1),'label')
            loss = self._loss(logits, label.long().view(-1))
            self._accuracy(logits, label)
            output_dict["loss"] = loss[None]
            # print(output_dict['loss'],'loss')
       ####################---------------------------------------------------------------------------------------------------------

#---------------------------extract feature------------------------------------------------------------------------------------



        if self.training:


            str = metadata[0]['img_fn']
            img_fn = str.split('/')
            # # l = os.path.join(SAVE_ROOT, "train", "{}".format(metadata[0]['movie']),"{}".format(img_fn[0]))
            # # if not os.path.exists(l):
            # #     os.makedirs(l)
            #
            #
            # data = {
            #     # "detector":{
            #         # 'output_dict': output_dict,
            #         # 'obj_reps': {"feature_map":obj_reps['feature_map'].cpu(),"obj_reps_raw":obj_reps['obj_reps_raw'].cpu(),"obj_reps":obj_reps['obj_reps'].cpu(),'obj_logits':obj_reps['obj_logits'].cpu(),
            #         #              "obj_labels":obj_reps["obj_labels"].cpu(),'cnn_regularization_loss':obj_reps['cnn_regularization_loss'].cpu()},
            #         # 'obj_reps': {
            #         #     "obj_reps_raw": obj_reps['obj_reps_raw'].detach().cpu(), "obj_reps": obj_reps['obj_reps'].detach().cpu(),
            #         #     'obj_logits': obj_reps['obj_logits'].detach().cpu(),
            #         #     "obj_labels": obj_reps["obj_labels"].detach().cpu(),
            #         #     'cnn_regularization_loss': obj_reps['cnn_regularization_loss'].detach().cpu()},
            #         'images':images.detach().data.cpu().numpy(),
            #         # 'label_logits': logits.detach().cpu().numpy(),
            #         # 'label_probs': class_probabilities.detach().cpu().numpy(),
            #         # 'cnn_regularization_loss': obj_reps['cnn_regularization_loss'].detach().cpu().numpy(),
            #         'objects': objects.detach().data.cpu().numpy(),
            #         'segms': segms.detach().data.cpu().numpy(),
            #         'boxes': boxes.detach().data.cpu().numpy(),
            #         'box_mask': box_mask.detach().data.cpu().numpy(),
            #         'question': question['bert'].detach().data.cpu().numpy(),
            #         'question_tags': question_tags.detach().data.cpu().numpy(),
            #         'question_mask': question_mask.detach().data.cpu().numpy(),
            #         'answers': answers['bert'].detach().data.cpu().numpy(),
            #         'answer_tags': answer_tags.detach().data.cpu().numpy(),
            #         'answer_mask': answer_mask.detach().data.cpu().numpy(),
            #         'metadata': metadata,
            #         'label': label.detach().data.cpu().numpy()
            #     # },
            #     # "accuracy": self.get_metrics()['accuracy'],
            #
            #
            # }
            with open(os.path.join(SAVE_ROOT, "train","{}_{}_{}.npz".format(img_fn[0],img_fn[1],metadata[0]['question_number'])), "wb") as file:
            # # # with open(os.path.join(SAVE_ROOT, "train","{}.txt".format(self.count)), "wb")  as file:
            # #
            # #     pickle.dump(data, file)
            #     np.savez(file,data)
                np.savez(file, images=images.detach().data.cpu().numpy(), objects=objects.detach().data.cpu().numpy(),
                         segms=segms.detach().data.cpu().numpy(), boxs=boxes.detach().data.cpu().numpy(),
                         question=question['bert'].detach().data.cpu().numpy(),
                         box_mask=box_mask.detach().data.cpu().numpy(),
                         question_tags=question_tags.detach().data.cpu().numpy(),
                         question_mask=question_mask.detach().data.cpu().numpy(),
                         answers=answers['bert'].detach().data.cpu().numpy(),
                         answer_tags=answer_tags.detach().data.cpu().numpy(),
                         answer_mask=answer_mask.detach().data.cpu().numpy(), metadata=metadata,
                         label=label.detach().data.cpu().numpy(), )




        else:

            str = metadata[0]['img_fn']
            img_fn = str.split('/')
            # # l = os.path.join(SAVE_ROOT, "val", "{}".format(metadata[0]['movie']), "{}".format(img_fn[0]))
            # # if not os.path.exists(l):
            # #     os.makedirs(l)
            #
            #  data = {
            #     # "detector": {
            #         # 'output_dict': output_dict,
            #         # 'obj_reps': {"feature_map":obj_reps['feature_map'].cpu(),"obj_reps_raw":obj_reps['obj_reps_raw'].cpu(),"obj_reps":obj_reps['obj_reps'].cpu(),'obj_logits':obj_reps['obj_logits'].cpu(),
            #         #              "obj_labels":obj_reps["obj_labels"].cpu(),'cnn_regularization_loss':obj_reps['cnn_regularization_loss'].cpu()},
            #
            #         # 'obj_reps': {
            #         #              "obj_reps_raw": obj_reps['obj_reps_raw'].detach().cpu(), "obj_reps": obj_reps['obj_reps'].detach().cpu(),
            #         #              'obj_logits': obj_reps['obj_logits'].detach().cpu(),
            #         #              "obj_labels": obj_reps["obj_labels"].detach().cpu(),
            #         #              'cnn_regularization_loss': obj_reps['cnn_regularization_loss'].detach().cpu()},
            #         'images': images.detach().data.cpu().numpy(),
            #
            #         # 'label_logits': logits.cpu().numpy(),
            #         # 'label_probs': class_probabilities.cpu().numpy(),
            #         # 'cnn_regularization_loss': obj_reps['cnn_regularization_loss'].cpu().numpy(),
            #         'objects': objects.detach().data.cpu().numpy(),
            #         'segms': segms.detach().data.cpu().numpy(),
            #         'boxes': boxes.detach().data.cpu().numpy(),
            #         'box_mask': box_mask.detach().data.cpu().numpy(),
            #         'question': question['bert'].detach().data.cpu().numpy(),
            #         'question_tags': question_tags.detach().data.cpu().numpy(),
            #         'question_mask': question_mask.detach().data.cpu().numpy(),
            #         'answers': answers['bert'].detach().data.cpu().numpy(),
            #         'answer_tags': answer_tags.detach().data.cpu().numpy(),
            #         'answer_mask': answer_mask.detach().data.cpu().numpy(),
            #         'metadata': metadata,
            #         'label': label.detach().data.cpu().numpy(),
            #     # },
            #     # "accuracy": self.get_metrics()['accuracy'],
            #
            #  }
            with open(os.path.join(SAVE_ROOT, "val","{}_{}_{}.npz".format(img_fn[0],img_fn[1],metadata[0]['question_number'])), "wb") as file:
            # with open(os.path.join(SAVE_ROOT, "val","{}.txt".format(self.count)), "wb")  as file:


                # pickle.dump(data, file)
                np.savez(file, images=images.detach().data.cpu().numpy(),objects=objects.detach().data.cpu().numpy(),segms=segms.detach().data.cpu().numpy(),boxs=boxes.detach().data.cpu().numpy(),
                         question = question['bert'].detach().data.cpu().numpy(),box_mask=box_mask.detach().data.cpu().numpy(),question_tags=question_tags.detach().data.cpu().numpy(),question_mask=question_mask.detach().data.cpu().numpy(),
                         answers = answers['bert'].detach().data.cpu().numpy(), answer_tags=answer_tags.detach().data.cpu().numpy(),answer_mask=answer_mask.detach().data.cpu().numpy(),metadata=metadata,label=label.detach().data.cpu().numpy(),)

        # self.count += 1

      #################----------------------------------------------------------------------------------------------------
#-----------------------------------------end extract feature--------------------------------------------------------------------------------------------

        return output_dict


    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self._accuracy.get_metric(reset)}
