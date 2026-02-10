import torch
import torch.nn as nn
from corecode.ealstm import EALSTM


class ModelBaseline(nn.Module):

    def __init__(self,
                 input_size_dyn: int,
                 input_size_stat: int,
                 hidden_size: int,
                 initial_forget_bias: int = 0,
                 dropout: float = 0.0):
        """Initialize model.

        Parameters
        ----------
        input_size_dyn: int
            Number of dynamic input features.
        input_size_stat: int
            Number of static input features (used in the EA-LSTM input gate).
        hidden_size: int
            Number of LSTM cells/hidden units.
        initial_forget_bias: int
            Value of the initial forget gate bias. (default: 5)
        dropout: float
            Dropout probability in range(0,1). (default: 0.0)
        """
        super(ModelBaseline, self).__init__()
        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat
        self.hidden_size = hidden_size
        self.initial_forget_bias = initial_forget_bias
        self.dropout_rate = dropout

        self.lstm_gpp = EALSTM(input_size_dyn=input_size_dyn,
                               input_size_stat=input_size_stat,
                               hidden_size=hidden_size,
                               initial_forget_bias=initial_forget_bias)

        self.lstm_reco = EALSTM(input_size_dyn=input_size_dyn,
                                input_size_stat=input_size_stat,
                                hidden_size=hidden_size,
                                initial_forget_bias=initial_forget_bias)

        self.fc_gpp = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_size, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        self.fc_reco = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_size, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor = None):
        """Run forward pass through the model.

        Parameters
        ----------
        x_d : torch.Tensor
            the dynamic input features of shape [batch, seq_length, n_features]
        x_s : torch.Tensor, optional
            Tensor containing the static catchment characteristics, by default None

        Returns
        -------
        out : torch.Tensor
            the network predictions
        h_n : torch.Tensor
            torch.Tensor containing the hidden states of each time step
        c_n : torch.Tensor
            torch.Tensor containing the cell states of each time step

        Args:
            x_a:
        """
        h_gpp, _ = self.lstm_gpp(x_d, x_s)
        h_reco, _ = self.lstm_reco(x_d, x_s)
        y_gpp = self.fc_gpp(h_gpp[:, -1, :])
        y_reco = self.fc_reco(h_reco[:, -1, :])
        # y_gpp = torch.exp(y_gpp)
        # y_reco = torch.exp(y_reco)

        return y_gpp, y_reco
