# Previsão de Preços de Ações com LSTM e FastAPI

## **Visão Geral do Projeto**

Este projeto implementa um sistema de previsão de preços de ações usando uma rede neural **Long Short-Term Memory (LSTM)**. Ele é composto por dois componentes principais:

1. Um **Notebook Jupyter** ou script Python (`ltsm_stock_price_prediction.py`) para treinar, avaliar e visualizar o desempenho do modelo LSTM em dados históricos de ações.
2. Uma aplicação **FastAPI** (`main.py`) que expõe um endpoint de API para receber preços históricos de ações e retornar previsões para os próximos 15 dias usando um modelo LSTM retreinado.

### Principais Tecnologias
- **Aquisição de Dados:** `yfinance`
- **Treinamento do Modelo:** `PyTorch`
- **Pré-processamento de Dados:** `scikit-learn`
- **Serviço Web:** `FastAPI`
- **Monitoramento de Treinamento:** `TensorBoard`

---

## **Funcionalidades**

- **Coleta de Dados Históricos**: Baixa dados históricos de ações (exemplo: Apple - AAPL) usando o `yfinance`.
- **Pré-processamento de Dados**:
  - Normaliza os preços das ações com `MinMaxScaler`.
  - Prepara os dados sequenciais para o modelo LSTM.
- **Modelo LSTM**: Rede neural personalizada para previsão de séries temporais.
- **Treinamento e Avaliação do Modelo**:
  - Treina o modelo com dados históricos.
  - Avalia desempenho usando métricas como `MAE`, `MSE`, `RMSE` e `MAPE`.
- **Registro com TensorBoard**: Monitora a perda de treinamento em tempo real.
- **Persistência do Modelo**: Salva e carrega o modelo LSTM treinado.
- **Serviço de API com FastAPI**:
  - Disponibiliza um endpoint RESTful para previsões.
- **Previsão Dinâmica**:
  - O modelo é retreinado dinamicamente com os dados de entrada para prever os próximos 15 dias baseados em preços históricos.

---

## **Primeiros Passos**

### **Pré-requisitos**
Certifique-se de ter o **Python 3.8+** instalado. Você pode baixá-lo em [python.org](https://www.python.org/).

---

### **Instalação**
1. Clone o repositório:
   ```bash
   git clone <https://github.com/Kbetti/LTSM_stock_price_prediction/tree/main>
   cd <LTSM_stock_price_prediction>

## Uso

### 1. Treinamento e Avaliação (Jupyter Notebook/Colab)

O script `ltsm_stock_price_prediction.py` (originalmente um notebook Colab) demonstra todo o pipeline de treinamento e avaliação.

Para executar esta parte:

1. **Abrir no Google Colab**:
Você pode abrir diretamente o arquivo `.ipynb` no Google Colab. Ele cuidará automaticamente da configuração da GPU, se disponível.

2. **Executar Localmente (como um script Python)**:
python "ltsm_stock_price_prediction.py"


Este script irá:

* Baixar dados históricos da AAPL de 2020-01-01 a 2025-04-30.

* Pré-processar e dividir os dados em conjuntos de treinamento e teste.

* Treinar um modelo LSTM com hiperparâmetros especificados (por exemplo, `hidden_dim=128`, `num_layers=3`).

* Registrar a perda de treinamento e teste no TensorBoard.

* Gerar um gráfico comparando os preços reais e previstos das ações.

* Imprimir métricas de avaliação: MAE, MSE, RMSE e MAPE.

* Salvar o modelo treinado como `lstm_stock_model.pth`.

### 2. Executando a Aplicação FastAPI

O arquivo `main.py` implementa o serviço FastAPI para fazer previsões.

1. **Inicie o servidor FastAPI:**
uvicorn main:app --reload

O sinalizador `--reload` é útil para desenvolvimento, pois reinicia o servidor quando há alterações no código.
A API estará acessível geralmente em `http://127.0.0.1:8000`.

2. **Acesse a Documentação da API:**
Com o servidor em execução, você pode acessar a documentação interativa da API (Swagger UI) em `http://127.0.0.1:8000/docs`.

3. **Faça uma Requisição de Previsão:**
Você pode usar `curl` ou qualquer cliente HTTP (como Postman, Insomnia ou um script Python `requests`) para enviar uma requisição POST para o endpoint `/predict`.

**Endpoint:** `POST /predict`
**Corpo da Requisição (JSON):**
# Previsão de Preços de Ações com LSTM e FastAPI

## Visão Geral do Projeto

Este projeto implementa um sistema de previsão de preços de ações usando uma rede neural Long Short-Term Memory (LSTM). Ele é composto por dois componentes principais:

1. Um Notebook Jupyter (`ltsm_stock_price_prediction.py`) para treinar, avaliar e visualizar o desempenho do modelo LSTM em dados históricos de ações.

2. Uma aplicação FastAPI (`main.py`) que expõe um endpoint de API para receber preços históricos de ações e retornar previsões para os próximos 15 dias usando um modelo LSTM retreinado.

O sistema utiliza `yfinance` para aquisição de dados, `PyTorch` para construir e treinar o modelo LSTM, `scikit-learn` para pré-processamento de dados e `FastAPI` para criar o serviço web. O `TensorBoard` é integrado para monitorar o processo de treinamento.

## Funcionalidades

* **Coleta de Dados Históricos**: Baixa dados históricos de ações (por exemplo, Apple - AAPL) usando `yfinance`.

* **Pré-processamento de Dados**: Normaliza os preços das ações usando `MinMaxScaler` e prepara dados sequenciais para LSTM.

* **Modelo LSTM**: Um modelo LSTM personalizável para previsão de séries temporais.

* **Treinamento e Avaliação do Modelo**: Treina o modelo LSTM em um conjunto de dados de treinamento e avalia seu desempenho usando métricas como MAE, MSE, RMSE e MAPE.

* **Registro no TensorBoard**: Integra o TensorBoard para visualização em tempo real da perda de treinamento e teste.

* **Persistência do Modelo**: Salva e carrega o modelo LSTM treinado.

* **Integração com FastAPI**: Fornece um endpoint de API RESTful para acionar previsões de preços de ações.

* **Previsão Dinâmica**: A aplicação FastAPI retreina o modelo com os dados de entrada e prevê os próximos 15 dias com base nos preços históricos fornecidos.

## Primeiros Passos

Estas instruções permitirão que você obtenha uma cópia do projeto em execução em sua máquina local para fins de desenvolvimento e teste.

### Pré-requisitos

Certifique-se de ter o **Python 3.8+** instalado. Você pode baixá-lo em [python.org](https://www.python.org/).

### Instalação

1. **Clone o repositório:**

git clone &lt;url_do_repositorio_aqui>
cd &lt;nome_do_repositorio>


*(Substitua `<url_do_repositorio_aqui>` e `<nome_do_repositorio>` pelos detalhes reais do seu repositório, caso este projeto estivesse hospedado no GitHub).*

2. **Instale as dependências:**
É altamente recomendável usar um ambiente virtual.

python -m venv venv
source venv/bin/activate  # No Windows, use venv\Scripts\activate
pip install -r requirements.txt


*Se você não tiver um arquivo `requirements.txt`, pode criar um listando os pacotes dos códigos fornecidos:*

pip install yfinance numpy pandas torch scikit-learn matplotlib fastapi uvicorn python-multipart tensorboard


## Uso

### 1. Treinamento e Avaliação (Jupyter Notebook/Colab)

O script `ltsm_stock_price_prediction.py` (originalmente um notebook Colab) demonstra todo o pipeline de treinamento e avaliação.

Para executar esta parte:

1. **Abrir no Google Colab**:
Você pode abrir diretamente o arquivo `.ipynb` no Google Colab. Ele cuidará automaticamente da configuração da GPU, se disponível.

2. **Executar Localmente (como um script Python)**:

python "ltsm_stock_price_prediction.py"


Este script irá:

* Baixar dados históricos da AAPL de 2020-01-01 a 2025-04-30.

* Pré-processar e dividir os dados em conjuntos de treinamento e teste.

* Treinar um modelo LSTM com hiperparâmetros especificados (por exemplo, `hidden_dim=128`, `num_layers=3`).

* Registrar a perda de treinamento e teste no TensorBoard.

* Gerar um gráfico comparando os preços reais e previstos das ações.

* Imprimir métricas de avaliação: MAE, MSE, RMSE e MAPE.

* Salvar o modelo treinado como `lstm_stock_model.pth`.

### 2. Executando a Aplicação FastAPI

O arquivo `main.py` implementa o serviço FastAPI para fazer previsões.

1. **Inicie o servidor FastAPI:**

uvicorn main:app --reload


O sinalizador `--reload` é útil para desenvolvimento, pois reinicia o servidor quando há alterações no código.
A API estará acessível geralmente em `http://127.0.0.1:8000`.

2. **Acesse a Documentação da API:**
Com o servidor em execução, você pode acessar a documentação interativa da API (Swagger UI) em `http://127.0.0.1:8000/docs`.

3. **Faça uma Requisição de Previsão:**
Você pode usar `curl` ou qualquer cliente HTTP (como Postman, Insomnia ou um script Python `requests`) para enviar uma requisição POST para o endpoint `/predict`.

**Endpoint:** `POST /predict`
**Corpo da Requisição (JSON):**

{
"prices": [
150.0,
151.2,
150.5,
152.1,
153.0,
... (pelo menos 30 preços de fechamento históricos)
]
}
**Exemplo usando `curl`:**
curl -X POST "http://127.0.0.1:8000/predict"

-H "Content-Type: application/json"

-d '{
"prices": [150.0, 151.2, 150.5, 152.1, 153.0, 153.5, 154.0, 153.8, 154.5, 155.0, 154.8, 155.5, 156.0, 155.9, 156.5, 157.0, 156.8, 157.5, 158.0, 157.7, 158.5, 159.0, 158.8, 159.5, 160.0, 159.7, 160.5, 161.0, 160.8, 161.5]
}'


*Nota: A lista `prices` deve conter pelo menos 30 preços de fechamento históricos para que a previsão funcione corretamente, pois o modelo usa uma `window_size` de 30.*

**Exemplo de Resposta (JSON):**

{
"predictions": [
162.0,
162.5,
162.3,
... (15 preços de ações previstos)
]
}


### 3. Visualizando os Logs do TensorBoard

Enquanto o script de treinamento (`ltsm_stock_price_prediction.py`) ou o aplicativo FastAPI (`main.py`) estiverem em execução (ou depois de terem sido executados), você pode visualizar o progresso do treinamento no TensorBoard.

1. **Navegue até o diretório do projeto no seu terminal.**

2. **Execute o TensorBoard:**

tensorboard --logdir logs


Ou, se estiver executando com o aplicativo FastAPI (que registra em um diretório `runs` padrão):

tensorboard --logdir runs


O TensorBoard geralmente estará disponível em `http://localhost:6006`.

## Estrutura do Projeto

* `ltsm_stock_price_prediction.py`: Notebook Jupyter (ou script Python) contendo o fluxo de trabalho completo de treinamento, avaliação e visualização para o modelo LSTM.

* `main.py`: Aplicação FastAPI que fornece uma API para previsão de preços de ações.

* `lstm_stock_model.pth`: (Gerado após a execução de `ltsm_stock_price_prediction.py`) O dicionário de estado do modelo PyTorch salvo.

* `logs/`: (Gerado por `ltsm_stock_price_prediction.py`) Diretório contendo arquivos de eventos do TensorBoard para logs de treinamento.

* `runs/`: (Gerado por `main.py`) Diretório padrão para logs do TensorBoard da aplicação FastAPI.

## Arquitetura do Modelo

O modelo LSTM (`StockPriceLSTM`) é definido da seguinte forma:

* **Dimensão de Entrada (`input_dim`)**: 1 (representando a única característica: preço de 'Fechamento').

* **Dimensão Oculta (`hidden_dim`)**: 128 (número de unidades LSTM em cada camada).

* **Dimensão de Saída (`output_dim`)**: 1 (prevendo o próximo preço de ação único).

* **Número de Camadas (`num_layers`)**: 3 (camadas LSTM empilhadas).

* **Dropout**: 0.2 (aplicado entre as camadas LSTM em `ltsm_stock_price_prediction.py`).

* **Ativação**: Camada de saída linear.

* **Função de Perda**: Erro Quadrático Médio (`nn.MSELoss`).

* **Otimizador**: Adam (`optim.Adam`) com uma taxa de aprendizado de 0.001.

## Lógica de Previsão

Na aplicação FastAPI `main.py`, a previsão para os próximos 15 dias funciona da seguinte forma:

1. Ao receber novos `prices` históricos, o modelo é reinicializado e retreinado por 50 épocas usando esses novos preços. Isso permite que o modelo se adapte aos dados mais recentes.

2. Os últimos `window_size` (30 dias) dos dados históricos *normalizados* são usados como a sequência de entrada inicial para a previsão.

3. O modelo faz uma previsão para o próximo dia.

4. Este valor previsto é então *adicionado* à sequência de entrada, e o valor *mais antigo* é removido, criando efetivamente uma janela deslizante.

5. Este processo é repetido 15 vezes para gerar previsões para os próximos 15 dias consecutivos.

6. Finalmente, os valores normalizados previstos são transformados inversamente de volta à sua escala de preço original.

## Tratamento de Erros

O endpoint FastAPI inclui tratamento básico de erros. Se ocorrer um erro inesperado durante o processamento, ele retornará um código de status HTTP 500 com uma mensagem detalhada.
