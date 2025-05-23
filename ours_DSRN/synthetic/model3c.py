import torch.nn as nn

class SpatialRegressor3(nn.Module):
    def __init__(self, hidden=32, features=4, prob=0.5):
        super(SpatialRegressor3, self).__init__()

        self.features = features
        self.hidden = hidden

        self.linear1_before_aggr = nn.Linear(self.features, hidden, bias=False)
        self.bn1_before_aggr = nn.BatchNorm1d(num_features=features)
        self.activation1_before_aggr = nn.Tanh() #nn.ReLU() #nn.Tanh() #nn.ReLU()
        self.dropout1_aggr = nn.Dropout(p=prob)

        self.linear12_before_aggr = nn.Linear(hidden, hidden, bias=False)
        self.bn12_before_aggr = nn.BatchNorm1d(num_features=hidden)
        self.activation12_before_aggr = nn.Tanh() #nn.ReLU() #nn.Tanh() #nn.PReLU() #LeakyReLU() #nn.Tanh() #nn.ReLU()
        self.dropout12_aggr = nn.Dropout(p=prob)

        #self.linear13_before_aggr = nn.Linear(hidden, hidden, bias=False)
        #self.bn13_before_aggr = nn.BatchNorm1d(num_features=hidden)
        #self.activation13_before_aggr = nn.Tanh() #nn.PReLU() #LeakyReLU() #nn.Tanh() #nn.ReLU()
        #self.dropout13_before_aggr = nn.Dropout(p=prob)
        
        self.linear2_before_aggr = nn.Linear(hidden, hidden, bias=False)
        self.bn2_before_aggr = nn.BatchNorm1d(num_features=hidden)
        self.activation2_before_aggr = nn.Tanh() #nn.ReLU() #nn.Tanh() #nn.PReLU() #LeakyReLU() #nn.Tanh()
        self.dropout2_before_aggr = nn.Dropout(p=prob)

        ##############

        self.linear1_attn = nn.Linear(self.features, hidden, bias=False)
        self.bn1_attn = nn.BatchNorm1d(num_features=features)
        self.activation1_attn = nn.Tanh() #nn.ReLU() #nn.Tanh() #nn.PReLU() #LeakyReLU() #nn.Tanh() #nn.ReLU()
        self.dropout1_attn = nn.Dropout(p=prob)

        self.linear12_attn = nn.Linear(hidden, hidden, bias=False)
        self.bn12_attn = nn.BatchNorm1d(num_features=hidden)
        self.activation12_attn = nn.Tanh() #nn.ReLU() #nn.Tanh() #nn.PReLU() #LeakyReLU() #nn.Tanh() #nn.ReLU()
        self.dropout12_attn = nn.Dropout(p=prob)

        #self.linear13_attn = nn.Linear(hidden, hidden, bias=False)
        #self.bn13_attn = nn.BatchNorm1d(num_features=hidden)
        #self.activation13_attn = nn.Tanh() #nn.PReLU() #LeakyReLU() #nn.Tanh() #nn.ReLU()
        #self.dropout13_attn = nn.Dropout(p=prob)
        
        self.dropout2_attn = nn.Dropout(p=prob)
        self.linear2_attn = nn.Linear(hidden, hidden, bias=False)
        self.bn2_attn = nn.BatchNorm1d(num_features=hidden)

        self.softmax = nn.Softmax(dim=1) # softmax in the neighbors dim
        ################

        self.dropout1_after_aggr = nn.Dropout(p=prob)
        self.linear1_after_aggr = nn.Linear(hidden, hidden, bias=False)
        self.bn1_after_aggr = nn.BatchNorm1d(num_features=hidden)
        self.activation1_after_aggr = nn.Tanh() #nn.PReLU() #LeakyReLU() #nn.Tanh() #nn.ReLU()

        self.dropout12_after_aggr = nn.Dropout(p=prob)
        self.linear12_after_aggr = nn.Linear(hidden, hidden, bias=False)
        self.bn12_after_aggr = nn.BatchNorm1d(num_features=hidden)
        self.activation12_after_aggr = nn.Tanh() #nn.ReLU() #nn.Tanh() #nn.PReLU() #LeakyReLU() #nn.Tanh() #nn.ReLU()

        #self.dropout13_after_aggr = nn.Dropout(p=prob)
        #self.linear13_after_aggr = nn.Linear(hidden, hidden, bias=False)
        #self.bn13_after_aggr = nn.BatchNorm1d(num_features=hidden)
        #self.activation13_after_aggr = nn.Tanh() #nn.PReLU() #LeakyReLU() #nn.Tanh() #nn.ReLU()
        
        self.dropout2_after_aggr = nn.Dropout(p=prob)
        self.linear2_after_aggr = nn.Linear(hidden, hidden, bias=False)
        self.bn2_after_aggr = nn.BatchNorm1d(num_features=hidden)
        self.activation2_after_aggr = nn.Tanh() #nn.ReLU() #nn.Tanh() #nn.PReLU() #LeakyReLU() #nn.Tanh()

        self.bn3_after_aggr = nn.BatchNorm1d(num_features=hidden)
        self.dropout3_after_aggr = nn.Dropout(p=prob)
        self.linear3_regression = nn.Linear(hidden, 1)

    def forward(self, u, mask):
        # u: (batch, neighbors, features)
        # example: (2, 3, 4)

        # mask: (batch, neighbors, features)
        # example: (2, 3, 4)

        # ------- pass the input features from the set of neighbots in the MLP-theta,
        # responsible for calculating the attention weights

        batch_size = u.shape[0]

        # (batch * neighbors, features)
        input = u.view(-1, self.features)
        
        # (batch * neighbors, hidden)

        x_main_0 = self.bn1_before_aggr(input)
        x_main_0 = self.linear1_before_aggr(x_main_0)
        self.x_main_0 = self.activation1_before_aggr(x_main_0)
        x_main_0 = self.dropout1_aggr(self.x_main_0)

        x_main_1 = self.bn12_before_aggr(x_main_0)
        x_main_1 = self.linear12_before_aggr(x_main_1)
        self.x_main_1 = self.activation12_before_aggr(x_main_1)
        x_main_1 = self.dropout12_aggr(self.x_main_1)
        x_main_1 = x_main_1 + x_main_0
        
        #x_main_1_1 = self.bn13_before_aggr(x_main_1)
        #x_main_1_1 = self.linear13_before_aggr(x_main_1_1)
        #self.x_main_1_1 = self.activation13_before_aggr(x_main_1_1)
        #x_main_1_1 = self.dropout13_before_aggr(self.x_main_1_1)
        #x_main_1_1 = x_main_1_1 + x_main_1

        x_main_2 = self.bn2_before_aggr(x_main_1)
        x_main_2 = self.linear2_before_aggr(x_main_2)
        self.x_main_2 = self.activation2_before_aggr(x_main_2)
        x_main_2 = self.dropout2_before_aggr(self.x_main_2)
        x_main_2 = x_main_2 + x_main_1

        # (batch, neighbors, hidden)
        x_main_2 = x_main_2.view(batch_size, -1, self.hidden)
   
        # attention leg

        # (batch * neighbors, features)
        x_attn_0 = self.bn1_attn(input)
        x_attn_0 = self.linear1_attn(x_attn_0)
        self.x_attn_0 = self.activation1_attn(x_attn_0)
        x_attn_0 = self.dropout1_attn(self.x_attn_0)

        x_attn_1 = self.bn12_attn(x_attn_0)
        x_attn_1 = self.linear12_attn(x_attn_1)
        self.x_attn_1 = self.activation12_attn(x_attn_1) 
        x_attn_1 = self.dropout12_attn(self.x_attn_1)
        x_attn_1 = x_attn_1 + x_attn_0

        #x_attn_1_1 = self.bn13_attn(x_attn_1)
        #x_attn_1_1 = self.linear13_attn(x_attn_1_1)
        #self.x_attn_1_1 = self.activation13_attn(x_attn_1_1)
        #x_attn_1_1 = self.dropout13_attn(self.x_attn_1_1)
        #x_attn_1_1 = x_attn_1_1 + x_attn_1

        # (batch * neighbors, hidden)
        x_attn_2 = self.bn2_attn(x_attn_1)
        x_attn_2 = self.linear2_attn(x_attn_2)
        x_attn_2 = self.dropout2_attn(x_attn_2)
        
        # (batch, neighbors, hidden)
        x_attn_2 = x_attn_2.view(batch_size, -1, self.hidden)

        # (batch, neighbors, hidden)
        x_attn_2 = x_attn_2.masked_fill(mask==0, -float("inf"))

        # (batch, neighbors, features)
        x_attn_2 = self.softmax(x_attn_2) # softmax in the neighbors dim

        x = x_main_2 * x_attn_2 # element wise multiplication by the weights

        # (batch, features)
        x = x.sum(dim=1)

        # ------- pass the aggregated node vetor in the MLP-phi, which is responsible for the regression
        #x = self.dropout1(x)

        # (batch, features)

        x_0 = self.bn1_after_aggr(x)
        x_0 = self.linear1_after_aggr(x_0)
        self.x_0 = self.activation1_after_aggr(x_0)
        x_0 = self.dropout1_after_aggr(self.x_0)
        x_0 = x_0 + x
        
        x_1 = self.bn12_after_aggr(x_0)
        x_1 = self.linear12_after_aggr(x_1)
        self.x_1 = self.activation12_after_aggr(x_1)
        x_1 = self.dropout12_after_aggr(self.x_1)
        x_1 = x_1 + x_0 

        #x_1_1 = self.bn13_after_aggr(x_1)
        #x_1_1 = self.linear13_after_aggr(x_1_1)
        #self.x_1_1 = self.activation13_after_aggr(x_1_1)
        #x_1_1 = self.dropout13_after_aggr(self.x_1_1)
        #x_1_1 = x_1_1 + x_1

        x_2 = self.bn2_after_aggr(x_1)
        x_2 = self.linear2_after_aggr(x_2)
        self.x_2 = self.activation2_after_aggr(x_2)
        x_2 = self.dropout2_after_aggr(self.x_2)
        x_2 = x_2 + x_1

        x_3 = self.bn3_after_aggr(x_2)
        x_3 = self.linear3_regression(x_3)

        return x_3
