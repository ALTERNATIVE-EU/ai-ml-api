# 3 network.. 1 cddd, 1 nlp, 1 concat.. poi weighted          
# model 
import torch
from torch import nn

## Model
class multigate_multimodal(nn.Module):
    def __init__(self, input_parameters:dict):

        """Model architecture for multitask learning cardiotox:
        
        input parameters are contained in a dictionary: es


        input_parameters = {
                            'branch': {'NLPcustom': 0,
                                        'CDDD': 1,
                                        'morgan': 0,
                                        'MDs': 0,
                                        'ChemBERTa': 0},
                            
                            'gate_dimension': 64,

                            'input_dim_NN': {'MDs': 494,
                                            'CDDD': 512,
                                            'ChemBERTa': 768,
                                            'morgan': 1024},

                            'task_name': dataset_train.task_names,
                            
                            'nlp': {'emb_dim': 512,
                                    'vocab_dim': len(dataset_train.vocab)+1,
                                    'hidden_size_convs':[256, 128],
                                    'hidden_size_lstm': {'hidden_size':64, 'num_layers':3},
                                    'dropoutprob': 0.3},
                            
                            'NN': {'hidden_size': [256, 128, 64], # [256, 128] per i branch; single encoder invece [256, 128, 64],
                                'dropoutprob': 0.3}
                        }

        """

        super(multigate_multimodal, self).__init__()
        
        self.input_parameters = input_parameters
        self.task_n = len(input_parameters['task_name'])
        self.leakyrelu = nn.LeakyReLU()

        if input_parameters['branch']['NLPcustom'] == 1:
            # NLP part
            self.embedding = nn.Embedding(num_embeddings=input_parameters['nlp']['vocab_dim'], 
                                          embedding_dim=input_parameters['nlp']['emb_dim'])
    
            # conv [batch_size, sequence_length, embedding_dim].
            conv_module = []
            for n, conv_value in enumerate(input_parameters['nlp']['hidden_size_convs']):
                if n == 0: in_channels = input_parameters['nlp']['emb_dim']
                else: in_channels = input_parameters['nlp']['hidden_size_convs'][n-1]
                
                conv_module.append(nn.Conv1d(in_channels=in_channels,
                                             out_channels=conv_value,                        
                                             kernel_size=3,
                                             stride=1, 
                                             padding=0)) 
                
                # [batch_size, num_channels, seq_length]
                conv_module.append(nn.BatchNorm1d(conv_value))
                conv_module.append(nn.LeakyReLU())
                                             
            self.conv_module = nn.Sequential(*conv_module)
    
            self.lstm_layer = nn.LSTM(input_size=input_parameters['nlp']['hidden_size_convs'][-1],
                                      hidden_size=input_parameters['nlp']['hidden_size_lstm']['hidden_size'],
                                      num_layers=input_parameters['nlp']['hidden_size_lstm']['num_layers'],
                                      bias=True, 
                                      batch_first=True, # test era false e facevo permuattion prima
                                      dropout=0.1, 
                                      bidirectional=True, 
                                      proj_size=0, 
                                      device=None, 
                                      dtype=None)
            
            self.batch_norm_lstm = nn.BatchNorm1d(input_parameters['nlp']['hidden_size_lstm']['hidden_size']*2)
            
            self.linear_module_nlp = self.create_linear_module(in_features=input_parameters['nlp']['hidden_size_lstm']['hidden_size']*2, 
                                                 hidden_size_dense=input_parameters['NN']['hidden_size'],
                                                 dropoutprob=input_parameters['NN']['dropoutprob'])    
        
        if input_parameters['branch']['CDDD'] == 1:
            # cddd network
            self.linear_cddd = self.create_linear_module(in_features=input_parameters['input_dim_NN']['CDDD'],#input_parameters['NN']['input_size'], 
                                                 hidden_size_dense=input_parameters['NN']['hidden_size'],
                                                 dropoutprob=input_parameters['NN']['dropoutprob'])
        
        if input_parameters['branch']['MDs'] == 1:
            # MDs network
            self.linear_MDs = self.create_linear_module(in_features=input_parameters['input_dim_NN']['MDs'], #input_parameters['NN']['input_size'], 
                                                 hidden_size_dense=input_parameters['NN']['hidden_size'],
                                                 dropoutprob=input_parameters['NN']['dropoutprob'])

        if input_parameters['branch']['ChemBERTa'] == 1:
            # Chemberta network
            self.linear_chemberta = self.create_linear_module(in_features=input_parameters['input_dim_NN']['ChemBERTa'], #input_parameters['NN']['input_size'], 
                                                 hidden_size_dense=input_parameters['NN']['hidden_size'],
                                                 dropoutprob=input_parameters['NN']['dropoutprob'])
        
        if input_parameters['branch']['morgan'] == 1:
            # Morgan network
            self.linear_morgan = self.create_linear_module(in_features=input_parameters['input_dim_NN']['morgan'], #input_parameters['NN']['input_size'], 
                                                 hidden_size_dense=input_parameters['NN']['hidden_size'],
                                                 dropoutprob=input_parameters['NN']['dropoutprob'])
        
        
        # Define the learnable consensus weights GATING network
        self.n_net = sum(input_parameters['branch'].values())

        if self.n_net>1:
            self.consensus_weight = nn.Parameter(torch.ones(self.n_net, 
                                                             input_parameters['gate_dimension'], 
                                                             self.task_n, requires_grad=True)) #,
            
            torch.nn.init.orthogonal_(self.consensus_weight, gain=1)
    
            nets = []
            for _ in range(self.n_net):     
                # dense layers first part: feature extractor
                nets.append(self.create_linear_module(in_features=input_parameters['NN']['hidden_size'][-1], 
                                                 hidden_size_dense=[128, input_parameters['gate_dimension']],#, self.task_n], 
                                                 dropoutprob=input_parameters['NN']['dropoutprob']))
            
            self.nets = nn.ModuleList(nets)
    
            
        self.final_towers = nn.ModuleList([
            self.create_linear_module(in_features=input_parameters['gate_dimension'], 
                                      hidden_size_dense=[32, 16], dropoutprob=0.3) for _ in range(self.task_n)
        ])

        self.final_layers = nn.ModuleList([nn.Linear(in_features=16, out_features=1) for _ in range(self.task_n)])

    def forward(self, x):
        # NLP
        net_input = []
        if self.input_parameters['branch']['NLPcustom'] == 1:
            embedded = self.embedding(x['nlp'])
            # batch, seq_len, hidden_size
            x_conv = self.conv_module(embedded.permute(0, 2, 1))
            out, (h_n, c_n) = self.lstm_layer(x_conv.permute(0, 2, 1))
            # x = self.batch_norm_lstm(torch.cat((h_n[-2], h_n[-1]), dim=1)) # h_n[-1]) 
            # out, (h_n, c_n)  = self.lstm_layer(embedded)
            x_nlp = self.batch_norm_lstm(torch.cat((h_n[-2], h_n[-1]), dim=1)) # h_n[-1]) 
            x_nlp = self.leakyrelu(x_nlp)
    
            # NLP
            x_nlp = self.linear_module_nlp(x_nlp)
            net_input.append(x_nlp)

        if self.input_parameters['branch']['CDDD'] == 1:
            # CDDD
            x_cddd = self.linear_cddd(x['CDDD'])
            net_input.append(x_cddd)
            
        if self.input_parameters['branch']['MDs'] == 1:
            # MDs
            x_MDs = self.linear_MDs(x['MDs'])
            net_input.append(x_MDs)
            
        if self.input_parameters['branch']['ChemBERTa'] == 1:
            # ChemBerta
            x_ChemBerta = self.linear_chemberta(x['ChemBERTa'])
            net_input.append(x_ChemBerta)
            
        if self.input_parameters['branch']['morgan'] == 1:
            # morgan
            x_morgan = self.linear_morgan(x['morgan'])
            net_input.append(x_morgan)

        # net
        if self.n_net>1:
            net_output = []
            for x, net in zip(net_input, self.nets):
                x1 = net(x)
                net_output.append(x1)
            
            outputs = []
            for endpoint_n, (tower, out_layers) in enumerate(zip(self.final_towers, self.final_layers)):
                weighted_sum = 0
                normalized_weights = torch.softmax(self.consensus_weight, dim=0)
                for n, net_out in enumerate(net_output):
                    # GATING NETWORK
                    # normalized_weights = torch.softmax(self.consensus_weight[n, :, endpoint_n], dim=0)# self.consensus_weight[n, :, endpoint_n]/self.consensus_weight[:, n, endpoint_n].sum() #  
                    # print(net_out.shape, normalized_weights.shape)
                    # weighted_sum += torch.mul(net_out, normalized_weights)
                    weighted_sum += torch.mul(net_out, normalized_weights[n, :, endpoint_n])  
                outputs.append(out_layers(tower(weighted_sum)))
        else:
            outputs = []
            for endpoint_n, (tower, out_layers) in enumerate(zip(self.final_towers, self.final_layers)):
                outputs.append(out_layers(tower(net_input[0])))
                
        return torch.cat(outputs, dim=1)
        
    def create_linear_module(self, in_features, hidden_size_dense, dropoutprob=None):
        linear_module = []
        for n, hid in enumerate(hidden_size_dense):
            if n != 0: in_features = hidden_size_dense[n-1]
            linear_module.append(nn.Linear(in_features=in_features, out_features=hid))
            linear_module.append(nn.BatchNorm1d(hid))
            linear_module.append(nn.LeakyReLU())
            if dropoutprob: linear_module.append(nn.Dropout(dropoutprob))

        return nn.Sequential(*linear_module)
    
def initialize_weights(m):
    # Custom initialization function

    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)  # Xavier uniform initialization
        torch.nn.init.zeros_(m.bias)             # Initialize biases to zero

# Apply the initialization function to the model
