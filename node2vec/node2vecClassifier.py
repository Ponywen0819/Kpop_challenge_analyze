import torch.nn as nn


class Node2VecClassifier(nn.Module):
    def __init__(self, embedding_dim=512, hidden1=1024, hidden2=512, num_classes=2, dropout_p=0.3):
        super(Node2VecClassifier, self).__init__()
        
        self.input_pipe = nn.Sequential(
          nn.Linear(embedding_dim, hidden1),
          nn.LayerNorm(hidden1),
          nn.ReLU(inplace=True),
          nn.Dropout(dropout_p)
        )
        
        self.hidden_layers = nn.ModuleList()
        for i in range(25):
          pipe = nn.Sequential(
            nn.Linear(hidden1, hidden1),
            nn.LayerNorm(hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p)
          )
          self.hidden_layers.append(pipe)

        self.pipe_2 = nn.Sequential(
          nn.Linear(hidden1, hidden2),
          nn.LayerNorm(hidden2),
          nn.ReLU(inplace=True),
          nn.Dropout(dropout_p)
        )
        
        self.fc3 = nn.Linear(hidden2, num_classes)


    def forward(self, x_source, x_target):
        # 使用兩個點乘
        x = x_source * x_target
    
        x = self.input_pipe(x)
        
        for layer in self.hidden_layers:
          x = layer(x)

        x = self.pipe_2(x)

        logits = self.fc3(x)
        return logits