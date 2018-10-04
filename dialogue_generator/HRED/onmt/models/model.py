""" Onmt NMT Model base class definition """
import torch
import torch.nn as nn


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self, encoder, decoder, sentence_encoder, multigpu=False, sen_seg = -1, pad_idx = -1, gpu = True):
        self.multigpu = multigpu
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sentence_encoder = sentence_encoder # @HRED
        self.sen_seg = sen_seg # @HRED
        self.pad_idx = pad_idx # @HRED
        self.gpu = gpu # @HRED

    def word_level(self, src):
        # Seg sentence
        src_list = []; src_length = []
        batch_size = src.size()[1]
        sen_length = src.size()[0]
        for i in range(0, batch_size):
            example = []
            sentence = []
            for j in range(0, sen_length):
                val = int(src[j][i][0])
                if val == self.sen_seg:
                    sentence.append(val)
                    example.append(sentence)
                    sentence = []
                elif val != self.pad_idx:
                    sentence.append(val)
            if len(sentence) != "":
                example.append(sentence)
            src_list.append(example)

        # Padding sentence
        max_sen = max([len(src_list[i]) for i in range(0, batch_size)])
        for i in range(0, batch_size):
            cur_sen = len(src_list[i])
            for j in range(cur_sen, max_sen):
                src_list[i].append([])

        for i in range(0, batch_size):
            # max is trick: "Length of all samples has to be greater than 0, but found an element in 'lengths' that is <= 0"
            # src_length.append([len(src_list[i][j]) for j in range(0, len(src_list[i]))])
            src_length.append([max(len(src_list[i][j]), 1) for j in range(0, len(src_list[i]))])

        # Padding term
        for i in range(0, max_sen):
            max_term = max([len(src_list[j][i]) for j in range(0, batch_size)])
            for j in range(0, batch_size):
                cur_term = len(src_list[j][i])
                for z in range(cur_term, max_term):
                    src_list[j][i].append(self.pad_idx)

        term_list = []; length_list = []
        for i in range(0, max_sen):
            term_list.append([src_list[j][i] for j in range(0, batch_size)])
            length_list.append([src_length[j][i] for j in range(0, batch_size)])

        enc_final_list = []
        memory_bank_list = []
        for i in range(0, max_sen):
            # sort
            terms = term_list[i]
            lengths = length_list[i]
            sorted_idx = sorted(range(len(lengths)), key=lambda k: lengths[k], reverse = True)
            sort_terms = [terms[idx] for idx in sorted_idx]
            sort_lengths = [lengths[idx] for idx in sorted_idx]
            # encode
            if self.gpu:
                ssrc = torch.cuda.LongTensor(sort_terms)
                lengths = torch.cuda.LongTensor(sort_lengths)
            else:
                ssrc = torch.LongTensor(sort_terms)
                lengths = torch.LongTensor(sort_lengths)
            ssrc = torch.reshape(ssrc, (ssrc.size()[1], -1, 1))
            enc_final, memory_bank = self.encoder(ssrc, lengths)
            # sort back
            # memory_bank
            reverse_idx = sorted(range(len(sorted_idx)), key=lambda k: sorted_idx[k])
            memory_bank_numpy = memory_bank.data.cpu().numpy()
            sorted_memory_bank = []
            for item in memory_bank_numpy:
                sorted_memory_bank.append([item[idx] for idx in reverse_idx])
            if self.gpu:
                memory_bank = torch.cuda.FloatTensor(sorted_memory_bank)
            else:
                memory_bank = torch.FloatTensor(sorted_memory_bank)
            # enc_final
            enc_final_cat_layer = torch.cat(enc_final, 2)
            enc_final_cat_lstm = torch.cat([enc_final_cat_layer[0], enc_final_cat_layer[1]], 1)
            enc_final_cat_lstm_numpy = enc_final_cat_lstm.data.cpu().numpy()
            sorted_enc_final = [enc_final_cat_lstm_numpy[idx] for idx in reverse_idx]
            if self.gpu:
                enc_final = torch.cuda.FloatTensor(sorted_enc_final)
            else:
                enc_final = torch.FloatTensor(sorted_enc_final)
            enc_final_list.append(enc_final)
            memory_bank_list.append(memory_bank)
        return memory_bank_list, enc_final_list

    def forward(self, src, tgt, lengths, dec_state=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        # Hierarchical for context embedding
        memory_bank_list, enc_final_list = self.word_level(src)
        #memory_bank = torch.cat(memory_bank_list, 0)
        sen_src = torch.cat(enc_final_list, 0).reshape(len(enc_final_list), src.size()[1], -1)
        enc_final, _ = self.sentence_encoder(sen_src, None)

        # Term-rnn for attention and copy network
        _, memory_bank = self.encoder(src, lengths)

        # Decoding
        enc_state = self.decoder.init_decoder_state(memory_bank, enc_final)
        decoder_outputs, dec_state, attns = self.decoder(tgt, memory_bank,
                         enc_state if dec_state is None else dec_state, 
                         memory_lengths=lengths)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return decoder_outputs, attns, dec_state
