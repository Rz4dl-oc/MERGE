import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *


class TAMDUR_Layer(nn.Module):
    def __init__(self, total_input_dim, input_dims, hidden_dim, time_steps):
        super(TAMDUR_Layer, self).__init__()

        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.total_input_dim = total_input_dim
        self.time_steps = time_steps

        self.conv1ds = nn.ModuleList([])
        self.ebs = nn.ModuleList([])
        self.bilstms = nn.ModuleList([])
        self.tatts = nn.ModuleList([])

        for input_dim in input_dims:
            self.conv1ds.append(nn.Conv1d(in_channels=input_dim, out_channels=self.hidden_dim*2,
                                          kernel_size=self.time_steps).cuda())

            self.ebs.append(nn.Linear(input_dim, self.hidden_dim).cuda())
            self.bilstms.append(nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=1
                                    , batch_first=True, bidirectional=True).cuda())    # B*T*2D
            self.tatts.append(Location_Attention(hidden_dim*2).cuda())    # B*2D

        self.selfAtt = QKVSelfAttnWvv2(hidden_dim*2, hidden_dim, 128).cuda()

    def forward(self, input, input_ids, delays):
        batch_size = input.shape[0]
        time_steps = input.shape[1]
        totol_dim = input.shape[2]

        inputs = []
        for id in input_ids:
            inputs.append(input[:, :, id])

        all_aft_tl = None

        for i in range(len(inputs)):
            input_item = inputs[i]        # B*T*Di

            aft_cnn = self.conv1ds[i](input_item.permute(0, 2, 1)).squeeze()      # B*T*2D

            aft_eb = self.ebs[i](input_item)*delays        # B*T*D
            aft_rnn = self.bilstms[i](aft_eb, None)[0]      # B*T*2D
            aft_att = self.tatts[i](aft_rnn)      # B*T*2D

            aft_tl = (aft_att + aft_cnn).reshape(batch_size, 1, 2*self.hidden_dim)    # B*1*2D

            if all_aft_tl == None:
                all_aft_tl = aft_tl
            else:
                all_aft_tl = torch.cat((all_aft_tl, aft_tl), dim=1)

        opt = self.selfAtt(all_aft_tl).reshape(batch_size, len(inputs)*self.hidden_dim)    # B*ND

        return opt


class TAMDUR(nn.Module):
    def __init__(self, total_input_dim, input_dims, hidden_dim, time_steps, out_dim):
        super(TAMDUR, self).__init__()
        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.total_input_dim = total_input_dim
        self.time_steps = time_steps

        self.backbone = TAMDUR_Layer(total_input_dim, input_dims, hidden_dim, time_steps)
        self.out = nn.Linear(len(input_dims)*hidden_dim, out_dim)

    def forward(self, input, input_ids, delays):
        aft_back = self.backbone(input, input_ids, delays)
        opt = self.out(aft_back)

        return F.log_softmax(opt, dim=1)

