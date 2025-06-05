from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# Configurações do Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Definição do Modelo LSTM
class StockPriceLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(StockPriceLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])
        return out
    

# API FastAPI
app = FastAPI()


# Hyperparâmetros para LSTM
hidden_dim = 128
num_layers = 3
input_dim = 1
output_dim = 1
learning_rate = 0.001
epochs = 50


# Configuração do TensorBoard
writer = SummaryWriter()


# Data Model
class StockPrices(BaseModel):
    prices: List[float]


# Função para treinar o modelo
def train_model(data, model, criterion, optimizer):
    model.train()
    X, y, _ = preprocess_data(data)  # Agora desconsidera o scaler
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
        # Logar no TensorBoard
        writer.add_scalar("Loss/train", loss.item(), epoch)
    return model


# Preprocessamento dos Dados
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))
    X, y = [], []
    window_size = 30
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:i+window_size])
        y.append(scaled_data[i+window_size])
    return np.array(X), np.array(y), scaler  


# Função para prever os próximos 90 dias
def predict_next_90_days(model, data):
    model.eval()  # Coloca o modelo em modo de avaliação
    _, _, scaler = preprocess_data(data)  # Normaliza os dados com o mesmo scaler
    data_scaled = scaler.transform(np.array(data).reshape(-1, 1))  # Normaliza

    # Usa os últimos 30 dias como entrada inicial para a previsão
    input_seq = data_scaled[-30:]

    predictions = []
    with torch.no_grad():
        for _ in range(15):  # Prever os próximos 90 dias
            # Ajusta as dimensões para (1, sequence_length, input_dim)
            input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
            pred = model(input_tensor)  # Realiza a previsão
            predictions.append(pred.item())  # Converte para float
            # Adiciona a previsão ao input_seq (deslocando a janela)
            input_seq = np.append(input_seq[1:], [[pred.item()]], axis=0)  # Atualiza a sequência

    # Reescala as previsões para os valores originais
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predictions.tolist()  # Retorna uma lista


# Endpoint principal da API
@app.post("/predict")
async def predict_stock_prices(stock_prices: StockPrices):
    try:
        # Constrói o modelo
        model = StockPriceLSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # Treina o modelo
        train_model(stock_prices.prices, model, criterion, optimizer)
        # Realiza previsão para 90 dias
        predictions = predict_next_90_days(model, stock_prices.prices)

        return {"predictions": predictions}  # Garantir que seja uma lista
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# Fechar o TensorBoard Writer ao encerrar o app
@app.on_event("shutdown")
def shutdown_event():
    writer.close()