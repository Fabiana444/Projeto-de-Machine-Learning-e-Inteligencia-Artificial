![Capa do projeto Machine Learning e Inteligência Artificial](Assets/capa-machine-learning.png)

*Imagem de capa produzida com inteligência artificial (IA).*

# Projeto de Machine Learning e Inteligência Artificial<p>
# Previsão de Cancelamentos de Reservas Hoteleiras

## Sobre o projeto

O projeto de Machine Learning e Inteligência Artificial foi desenvolvido em Python com o objetivo de analisar dados de reservas hoteleiras e construir um modelo preditivo capaz de identificar a possibilidade de cancelamento de uma reserva.

O trabalho utiliza o dataset **Hotel Booking Demand**, que reúne informações sobre reservas de dois tipos de estabelecimentos: City Hotel e Resort Hotel.

O desenvolvimento contempla o carregamento, a inspeção, o tratamento e a preparação dos dados, a Análise Exploratória de Dados (EDA), a Engenharia de Features, a modelagem de Machine Learning e a avaliação dos resultados.

A variável `is_canceled` foi definida como variável-alvo (TARGET), sendo:

- `0`: reserva não cancelada.
- `1`: reserva cancelada.

O objetivo é identificar padrões relacionados aos cancelamentos e aplicar um modelo de classificação binária para prever a situação de novas reservas com base nas informações disponíveis.

## O que o projeto analisa

O projeto investiga diferentes características das reservas e sua relação com os cancelamentos.

**Análise exploratória de dados:**

- Distribuição de reservas por tipo de hotel.
- Comparação entre reservas canceladas e não canceladas.
- Taxa de cancelamento por tipo de hotel.
- Relação entre antecedência da reserva (`lead_time`) e cancelamento.
- Taxa de cancelamento por faixa de antecedência.
- Sazonalidade dos cancelamentos por mês.
- Evolução da taxa de cancelamento por ano.
- Relação entre tarifa diária média (`adr`) e cancelamento.
- Comparação da distribuição de tarifas entre reservas canceladas e não canceladas.
- Países de origem com maior número de reservas.
- Taxa de cancelamento entre os 15 países com maior volume de reservas.

**Visualizações em PNG:**

- Distribuição de reservas por tipo de hotel.
- Cancelamentos por tipo de hotel.
- Taxa de cancelamento por faixa de antecedência.
- Taxa de cancelamento por mês.
- Taxa de cancelamento por ano.
- Distribuição de `adr` por situação de cancelamento (boxplot).
- Taxa de cancelamento por país de origem.
- Matriz de confusão do modelo preditivo.

## Etapas desenvolvidas

### 1. Tratamento e preparação dos dados

O dataset original foi carregado utilizando Pandas. Inicialmente, foram verificadas as dimensões, as colunas disponíveis, os tipos de dados, os valores ausentes e os registros duplicados.

O conjunto original apresentou 119.390 registros e 32 colunas.

As principais atividades realizadas foram:

- Inspeção inicial dos dados com `df.head()` e `df.info()`.
- Identificação e quantificação de valores ausentes.
- Cálculo do percentual de valores ausentes por coluna.
- Identificação e investigação de registros duplicados.
- Remoção de duplicações exatas.
- Criação do DataFrame `df_tratado`, preservando o DataFrame original.
- Tratamento de valores ausentes nas colunas `children`, `country` e `agent`.
- Remoção da coluna `company`, devido à elevada quantidade de valores ausentes.
- Conversão de `reservation_status_date` para `datetime`.
- Análise estatística das variáveis numéricas.
- Identificação e investigação de possíveis outliers pelo método IQR.
- Verificação final dos valores ausentes, tipos de dados e dimensões.

Após os tratamentos, o DataFrame utilizado nas etapas seguintes apresentou 87.382 registros e 32 colunas, sem valores ausentes.

### 2. Análise Exploratória de Dados (EDA)

A análise exploratória foi realizada para compreender o comportamento das reservas e identificar características associadas aos cancelamentos.

Foram utilizadas estatísticas descritivas, filtros, agrupamentos e visualizações gráficas.

As principais análises envolveram:

- Distribuição da variável-alvo `is_canceled`.
- Comparação entre City Hotel e Resort Hotel.
- Taxa de cancelamento por tipo de hotel.
- Relação entre antecedência da reserva e cancelamento.
- Criação de faixas de antecedência.
- Sazonalidade mensal dos cancelamentos.
- Evolução dos cancelamentos por ano.
- Comparação das tarifas diárias médias (`adr`).
- Análise dos países de origem dos hóspedes.
- Identificação dos principais padrões e tendências.

### 3. Engenharia de Features

A Engenharia de Features foi utilizada para preparar as variáveis para a análise e a modelagem.

Foi criada a variável derivada `lead_time_faixa`, que agrupa a antecedência da reserva nas seguintes categorias:

- 0 a 30 dias.
- 31 a 90 dias.
- 91 a 180 dias.
- Mais de 180 dias.

Para a modelagem, foram definidas as variáveis preditoras (X) e a variável-alvo (y).

Também foram realizadas as seguintes ações:

- Definição de `is_canceled` como TARGET.
- Exclusão de `is_canceled` das variáveis de entrada.
- Exclusão de `reservation_status` e `reservation_status_date` para evitar vazamento de dados (*data leakage*).
- Exclusão de `lead_time_faixa` das entradas do modelo, pois `lead_time` foi utilizado diretamente.
- Identificação automática das variáveis numéricas e categóricas.
- Conversão de `agent` para texto.
- Aplicação de One-Hot Encoding às variáveis categóricas.
- Utilização de `handle_unknown='ignore'`.
- Ajuste do codificador somente com os dados de treinamento.
- Aplicação da mesma transformação aos dados de teste.

### 4. Modelagem de Machine Learning

O problema foi definido como uma classificação binária, com o objetivo de prever se uma reserva seria cancelada ou não.

A separação dos dados foi realizada da seguinte forma:

- 80% para treinamento.
- 20% para teste.
- Utilização de `stratify=y` para preservar a proporção entre as classes.
- Utilização de `random_state=42` para permitir a reprodução da divisão.

O algoritmo escolhido foi o **Random Forest Classifier**, configurado com:

- `n_estimators=100`
- `random_state=42`
- `n_jobs=-1`

O treinamento foi realizado exclusivamente com o conjunto de treinamento. Posteriormente, foram geradas previsões para o conjunto de teste, permitindo comparar os resultados previstos com os valores reais.

### 5. Avaliação do modelo

A avaliação foi realizada por meio de métricas de classificação e da Matriz de Confusão.

Foram calculadas e analisadas:

- Acurácia (Accuracy).
- Precision.
- Recall.
- F1-score.
- Verdadeiros Negativos (TN).
- Falsos Positivos (FP).
- Falsos Negativos (FN).
- Verdadeiros Positivos (TP).

As métricas foram avaliadas separadamente para as classes “Não cancelado” e “Cancelado”, permitindo identificar as diferenças de desempenho do modelo.

## Resultados obtidos

O modelo Random Forest apresentou **84,91% de acurácia** no conjunto de teste.

Os resultados para a classe “Cancelado” foram:

| Métrica | Resultado |
|---|---:|
| Acurácia geral | 84,91% |
| Precision — Cancelado | 78% |
| Recall — Cancelado | 63% |
| F1-score — Cancelado | 70% |

### Matriz de Confusão

A Matriz de Confusão apresentou os seguintes resultados:

| Resultado | Quantidade |
|---|---:|
| Verdadeiros Negativos (TN) | 11.813 |
| Falsos Positivos (FP) | 861 |
| Falsos Negativos (FN) | 1.777 |
| Verdadeiros Positivos (TP) | 3.026 |

Dos 4.803 cancelamentos reais existentes no conjunto de teste:

- 3.026 foram identificados corretamente.
- 1.777 não foram identificados pelo modelo.

### Desempenho por classe

**Classe 0 — Não cancelado**

- Precision: 87%.
- Recall: 93%.
- F1-score: 90%.

**Classe 1 — Cancelado**

- Precision: 78%.
- Recall: 63%.
- F1-score: 70%.

Os resultados demonstram que o modelo apresentou desempenho superior na identificação das reservas não canceladas. Para a classe de cancelamentos, a principal dificuldade foi identificar todos os cancelamentos reais, conforme demonstrado pelo Recall de 63%.

## Principais insights da análise exploratória

A análise dos dados permitiu identificar padrões relevantes no comportamento das reservas.

**Tipo de hotel**

O City Hotel apresentou maior volume de reservas e maior taxa proporcional de cancelamento:

- City Hotel: 30,04%.
- Resort Hotel: 23,45%.

**Antecedência da reserva**

Foi identificada uma associação entre maior antecedência e maior taxa de cancelamento:

- Até 30 dias: 16,42%.
- De 31 a 90 dias: 32,01%.
- De 91 a 180 dias: 34,98%.
- Mais de 180 dias: 39,68%.

**Sazonalidade**

A maior taxa de cancelamento ocorreu em agosto, com 32,18%. A menor foi observada em novembro, com 21,10%.

**Evolução anual**

A taxa de cancelamento apresentou crescimento no período analisado:

- 2015: 20,24%.
- 2016: 26,44%.
- 2017: 31,91%.

**Tarifa diária média**

As reservas canceladas apresentaram valores médios e medianos de `adr` superiores aos das reservas não canceladas:

- Não canceladas: média de 102,00.
- Canceladas: média de 117,61.

**País de origem**

Entre os 15 países com maior número de reservas, as maiores taxas de cancelamento foram identificadas em:

- Brasil: 36,44%.
- Portugal: 35,63%.
- Itália: 35,06%.

A menor taxa entre esses países foi observada na Áustria, com 17,95%.

Esses resultados representam associações identificadas no conjunto de dados e não estabelecem relações de causalidade.

## Conceitos aplicados

O desenvolvimento do projeto envolveu conceitos de programação, análise de dados e aprendizado de máquina:

- Python: variáveis, estruturas de dados, funções e manipulação de informações.
- Pandas: DataFrames, inspeção, filtros, tratamento e agrupamentos.
- NumPy: operações numéricas.
- Análise Exploratória de Dados (EDA): estatísticas descritivas e identificação de padrões.
- Tratamento de dados: valores ausentes, duplicados, tipos de dados e outliers.
- Engenharia de Features: criação e preparação de variáveis.
- One-Hot Encoding: transformação de variáveis categóricas.
- Separação de dados: treinamento e teste.
- Machine Learning: classificação binária.
- Random Forest Classifier: treinamento e previsão.
- Métricas de avaliação: Accuracy, Precision, Recall e F1-score.
- Matriz de Confusão: interpretação de acertos e erros do modelo.
- Matplotlib e Seaborn: criação de visualizações.

## Bibliotecas e ferramentas utilizadas

**Linguagem e bibliotecas Python**

- Python: linguagem utilizada no desenvolvimento.
- Pandas: manipulação, organização e análise dos dados.
- NumPy: operações numéricas.
- Matplotlib: criação e visualização de gráficos.
- Seaborn: visualizações estatísticas.
- Scikit-learn: preparação dos dados, codificação, treinamento e avaliação do modelo.

**Ambiente e ferramentas de desenvolvimento**

- Google Colab: ambiente utilizado para desenvolvimento e execução do notebook.
- Jupyter Notebook: formato interativo utilizado para organizar o código, as análises e os resultados.
- Git e GitHub: controle de versão e armazenamento do projeto.

## Como executar

### Pré-requisitos

Para executar o projeto, é necessário ter:

- Python 3.10 ou superior.
- Git, caso queira clonar o repositório.
- Jupyter Notebook ou Visual Studio Code com suporte a notebooks.
- Conexão com a internet para acessar o repositório e instalar as bibliotecas, quando necessário.

### Execução no Google Colab

O projeto pode ser executado diretamente no Google Colab.

1. Acesse o repositório no GitHub.
2. Abra o notebook `Desafio_Extra_IA_Hotel_Booking_Fabiana_Nonjah.ipynb`.
3. No Google Colab, abra o notebook por meio da opção de carregamento do GitHub ou faça o download e envie o arquivo `.ipynb`.
4. Disponibilize o arquivo `hotel_bookings.csv` no ambiente de execução.
5. Execute as células do notebook em sequência.

As bibliotecas utilizadas podem ser instaladas no ambiente, caso necessário:

```python
!pip install pandas numpy matplotlib seaborn scikit-learn
```

### Execução local

Clone o repositório:

```bash
git clone https://github.com/Fabiana444/Projeto-de-Machine-Learning-e-Inteligencia-Artificial.git
```

Acesse a pasta do projeto:

```bash
cd Projeto-de-Machine-Learning-e-Inteligencia-Artificial
```

Instale as dependências:

```bash
python -m pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Abra o notebook no Jupyter:

```bash
jupyter notebook Desafio_Extra_IA_Hotel_Booking_Fabiana_Nonjah.ipynb
```

Verifique se o arquivo `hotel_bookings.csv` está disponível no caminho esperado pelo notebook antes de executar as células.

## Estrutura do projeto

```text
Projeto-de-Machine-Learning-e-Inteligencia-Artificial/
│
├── Assets/
│   └── capa-machine-learning.png
│
├── Visualizações/
│   └── gráficos gerados durante as análises
│
├── Desafio_Extra_IA_Hotel_Booking_Fabiana_Nonjah.ipynb
│
├── Documentacao_Desafio_Extra_IA_Hotel_Booking_Fabiana_Nonjah.md
│
├── Documentacao_Desafio_Extra_IA_Hotel_Booking_Fabiana_Nonjah.pdf
│
├── hotel_bookings.csv
│
└── README.md
```

O notebook contém as etapas de carregamento, tratamento, análise exploratória, Engenharia de Features, treinamento e avaliação do modelo.

O arquivo `hotel_bookings.csv` contém o conjunto de dados utilizado no desenvolvimento.

A pasta `Visualizações` reúne os gráficos gerados durante as análises. A pasta `Assets` armazena a imagem de capa da documentação.

Os arquivos de documentação em Markdown e PDF apresentam o detalhamento das decisões técnicas, análises e resultados do desafio.

## Decisões técnicas

Durante o desenvolvimento, foram adotadas decisões para preservar a consistência dos dados e evitar distorções nas análises e na modelagem.

**Preservação do dataset original**

Foi criada uma cópia independente do DataFrame original por meio de `df_tratado = df.copy()`. Essa decisão permitiu realizar os tratamentos sem modificar diretamente os dados inicialmente carregados.

**Remoção de registros duplicados**

Foram identificados 31.994 registros duplicados exatos, aproximadamente 26,80% do dataset original. A remoção evitou que registros repetidos tivessem peso artificialmente maior nas análises e no treinamento.

**Tratamento de valores ausentes**

Cada coluna recebeu tratamento de acordo com sua natureza:

- `children`: preenchimento com a mediana.
- `country`: preenchimento com `Unknown`.
- `agent`: preenchimento com `Unknown`.
- `company`: remoção devido à elevada quantidade de valores ausentes.

**Investigação de outliers**

A identificação de possíveis valores extremos foi realizada pelo método IQR. Os registros não foram removidos automaticamente: os casos relevantes foram investigados individualmente.

Foram removidos registros considerados inconsistentes, incluindo um valor negativo de `adr`, 12 registros com mais de 10 adultos e um registro com `adr = 5400`.

**Prevenção de data leakage**

As colunas `reservation_status` e `reservation_status_date` foram excluídas das variáveis preditoras por conterem informações relacionadas ao resultado final da reserva. Essa decisão evita que o modelo utilize informações que não estariam disponíveis no momento em que a previsão deveria ser realizada.

**Separação entre treinamento e teste**

A divisão foi realizada antes do ajuste do codificador, utilizando 80% dos dados para treinamento e 20% para teste. O One-Hot Encoding foi ajustado somente com os dados de treinamento e posteriormente aplicado aos dados de teste.

Essa estratégia evita que informações do conjunto de teste influenciem a preparação dos dados.

**Escolha do modelo**

Foi utilizado o Random Forest Classifier, com 100 estimadores, `random_state=42` e processamento paralelo com `n_jobs=-1`.

## Conclusão

O projeto contemplou as etapas de preparação e tratamento de dados, Análise Exploratória, Engenharia de Features, modelagem de Machine Learning e avaliação de um modelo de classificação binária.

A análise exploratória permitiu identificar padrões relacionados ao tipo de hotel, antecedência da reserva, sazonalidade, tarifa diária e país de origem dos hóspedes.

O modelo Random Forest alcançou 84,91% de acurácia no conjunto de teste. Entretanto, apresentou desempenho diferente entre as classes, com maior dificuldade para identificar os cancelamentos reais.

O Recall de 63% para a classe “Cancelado” evidencia essa limitação e demonstra a importância de analisar diferentes métricas, em vez de considerar somente a acurácia.

Dessa forma, o projeto demonstra a aplicação prática de técnicas de análise de dados e aprendizado de máquina para investigar padrões e desenvolver um modelo introdutório de previsão de cancelamentos de reservas hoteleiras.

## Links úteis

| Recurso | Link |
|---|---|
| Repositório do projeto | [Acessar no GitHub](https://github.com/Fabiana444/Projeto-de-Machine-Learning-e-Inteligencia-Artificial) |
| Dataset Hotel Booking Demand | [Acessar no Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) |

## Limitações do projeto

O projeto foi desenvolvido com finalidade de estudo e aplicação prática de conceitos de Análise de Dados e Machine Learning.

Os resultados dependem das informações disponíveis no dataset Hotel Booking Demand e das decisões adotadas durante o tratamento e a preparação dos dados.

O modelo apresenta maior dificuldade na identificação dos cancelamentos reais, conforme demonstrado pelo Recall de 63% para a classe “Cancelado”.

Portanto, os resultados não devem ser interpretados como garantia de previsão de cancelamentos em operações hoteleiras reais. O modelo representa uma aplicação introdutória de classificação, com desempenho avaliado sobre o conjunto de teste utilizado no projeto.

---

Documento elaborado como parte do projeto de Machine Learning e Inteligência Artificial, com foco na análise de dados e previsão de cancelamentos de reservas hoteleiras.
