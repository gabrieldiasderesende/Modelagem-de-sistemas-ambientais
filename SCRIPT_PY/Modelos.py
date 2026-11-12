"""
Modelagem (classificação categórica chuva -> vazão) -- funções extraídas de
`Modelos.ipynb`, no mesmo padrão de `Datamining.py`/`Extracao_Pretratamento.py`:
o notebook final só importa e chama estas funções; nenhuma lógica de modelo
fica no notebook.

Pipeline (ver notebook de modelagem para as chamadas em ordem):
    1. pre_tratamento               -> calcula percentis e categoriza X (chuva) e y (vazão)
    2. pre_classificadores          -> cria lags, aplica LabelEncoder, faz split treino/teste
    3. rodar_<modelo>                -> treina um dos 18 classificadores disponíveis
    4. plotar_resultados            -> decodifica previsões, plota Real x Previsto, salva em disco
    5. funcao_diagnostico_overfitting / plotar_curva_validacao -> diagnóstico pós-treino

Observação importante (comportamento herdado do notebook original, não alterado
aqui): `plotar_resultados` assume que a variável dependente categorizada se
chama 'Q' (`label_encoders['Q']` está hardcoded). Por isso `pre_tratamento`
deve sempre ser chamado com `col_dependente='Q'`.

Os 18 `rodar_*` seguem todos a mesma estrutura (parâmetros default se
`parametros=None`, treina, prevê, imprime métricas, plota matriz de
confusão, retorna `{'clf':..., 'y_pred':...}`) -- foram mantidos como
funções separadas (uma por modelo), como estavam no notebook original.
"""

import os
import re
import warnings

import numpy as np
import pandas as pd
from tabulate import tabulate

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import validation_curve

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Perceptron
from sklearn.discriminant_analysis import (
    QuadraticDiscriminantAnalysis,
    LinearDiscriminantAnalysis,
)

from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier


# ============================================================================
# 1. PRÉ-TRATAMENTO PARA CLASSIFICAÇÃO (percentis + categorização)
# ============================================================================

def calcular_percentis(df, lista_percentis, printar=False):
    """
    Calcula os percentis das colunas numéricas do DataFrame.

    Parâmetros:
    - df: DataFrame de entrada
    - lista_percentis: lista ou array com os percentis a calcular (ex: [1, 2.5, 5, ...])
    - printar: bool, se True imprime os percentis calculados (default False)

    Retorna:
    - df_percentis: DataFrame com percentis por coluna
    """
    lista_percentis = [p for p in lista_percentis if 0 <= p <= 100]

    df_percentis = df.select_dtypes(include=['number']).quantile(q=[p / 100 for p in lista_percentis]).transpose()
    df_percentis.columns = [f'percentil_{str(p).zfill(2)}' for p in lista_percentis]

    if printar:
        print(tabulate(df_percentis, headers='keys', tablefmt='grid'))

    return df_percentis


def converter_variavel(X, percentis, variavel=None, printar=False):
    """
    Categoriza os valores numéricos de X em classes (strings), usando os
    percentis (índice = colunas de X, colunas = percentis) como limites.
    """
    X_categorizada = pd.DataFrame(index=X.index)

    for col in X.columns:
        pcts = percentis.loc[col].sort_values()
        pct_values = pcts.values
        pct_labels = pcts.index  # ex: ['percentil_01', 'percentil_05', ...]
        categorias = []

        for val in X[col]:
            if val < pct_values[0]:
                cat = f"Inferior_{variavel}_{pct_labels[0]}_{col}"
            elif val > pct_values[-1]:
                cat = f"Superior_{variavel}_{pct_labels[-1]}_{col}"
            else:
                cat = None
                for i in range(len(pct_values) - 1):
                    if pct_values[i] <= val <= pct_values[i + 1]:
                        cat = f"Meio_{variavel}_{pct_labels[i]}_{pct_labels[i + 1]}_{col}"
                        break
                if cat is None:
                    cat = f"{variavel}_Indefinido_{col}"

            categorias.append(cat)

        X_categorizada[col] = categorias

    if printar:
        print(tabulate(X_categorizada, headers='keys', tablefmt='grid'))

    return X_categorizada


def pre_tratamento(
    df,
    col_dependente,
    col_data,
    lista_percentis_chuva=None,
    lista_percentis_vazao=None,
    printar_percentis=False,
    retornar_percentis=False,
    retornar_categorizado=False,
    retornar_tudo=True,
    printar_categorizado=False
):
    """
    Realiza o pré-tratamento dos dados para classificação: separa X/y,
    calcula percentis e categoriza (via `calcular_percentis`/`converter_variavel`).

    Parâmetros:
    - df: DataFrame de entrada
    - col_dependente: nome da variável dependente (string) -- use 'Q', pois
      `plotar_resultados` espera esse nome.
    - col_data: nome da coluna de datas (string)
    - lista_percentis_chuva: lista de percentis para variáveis independentes (chuva)
    - lista_percentis_vazao: lista de percentis para variável dependente (vazão)
    - printar_percentis: bool, imprime percentis (default False)
    - retornar_percentis: bool, retorna os DataFrames dos percentis (default False)
    - retornar_categorizado: bool, retorna apenas o DataFrame categorizado (default False)
    - retornar_tudo: bool, retorna o dicionário completo (default True, tem prioridade)
    - printar_categorizado: bool, imprime o DataFrame categorizado (default False)

    Retorna (ordem de prioridade):
    - Se retornar_tudo=True (default):
        -> {'X_real', 'X_categorizada', 'y_real', 'y_categorizada', 'percentis_chuva', 'percentis_vazao'}
    - Se retornar_categorizado=True:
        -> {'X_categorizada': ..., 'y_categorizada': ...}
    - Se retornar_percentis=True:
        -> X, y, percentis_chuva, percentis_vazao
    - Senão:
        -> X, y
    """
    if col_data not in df.columns:
        raise Exception(f'Coluna de datas "{col_data}" não encontrada no DataFrame.')
    if col_dependente not in df.columns:
        raise Exception(f'Coluna dependente "{col_dependente}" não encontrada no DataFrame.')

    df[col_data] = pd.to_datetime(df[col_data])
    df = df.set_index(col_data)

    y = df[col_dependente].copy()
    X = df.drop(columns=[col_dependente])

    percentis_chuva = None
    percentis_vazao = None

    if lista_percentis_chuva is not None:
        percentis_chuva = calcular_percentis(X, lista_percentis_chuva, printar=printar_percentis)

    if lista_percentis_vazao is not None:
        df_y = y.to_frame()
        percentis_vazao = calcular_percentis(df_y, lista_percentis_vazao, printar=printar_percentis)

    if retornar_tudo:
        X_cat = converter_variavel(X, percentis_chuva, variavel='Chuva', printar=printar_categorizado)
        y_cat = converter_variavel(y.to_frame(), percentis_vazao, variavel='Vazao', printar=printar_categorizado)
        return {
            'X_real': X,
            'X_categorizada': X_cat,
            'y_real': y,
            'y_categorizada': y_cat,
            'percentis_chuva': percentis_chuva,
            'percentis_vazao': percentis_vazao
        }

    if retornar_categorizado:
        X_cat = converter_variavel(X, percentis_chuva, variavel='Chuva', printar=printar_categorizado)
        y_cat = converter_variavel(y.to_frame(), percentis_vazao, variavel='Vazao', printar=printar_categorizado)
        return {'X_categorizada': X_cat, 'y_categorizada': y_cat}

    if retornar_percentis:
        return X, y, percentis_chuva, percentis_vazao

    return X, y


# ============================================================================
# 2. PREPARAÇÃO PARA CLASSIFICADORES (lags + label encoding + split treino/teste)
# ============================================================================

def pre_classificadores(X_cat, y_cat, tamanho_teste=0.3, num_lags=1):
    """
    Prepara os DataFrames categorizados (strings) para uso em classificadores:
    cria lags de X e y, reformula o problema como Lags(Chuva) + Chuva(t) +
    Lags(Vazão) -> Vazão(t), faz split temporal treino/teste e aplica
    LabelEncoder em todas as colunas.

    Retorna um dict: {'X_treino', 'X_teste', 'y_treino', 'y_teste',
    'label_encoders', 'tamanho_teste'} -- todos já codificados numericamente.
    """

    def criar_lags(df, n_lags):
        df_lags = pd.DataFrame(index=df.index)
        for lag in range(1, n_lags + 1):
            lagged = df.shift(lag)
            lagged.columns = [f"{col}_lag{lag}" for col in df.columns]
            df_lags = pd.concat([df_lags, lagged], axis=1)
        df_com_lags = pd.concat([df_lags, df], axis=1)
        return df_com_lags

    def ordenar_colunas_por_lags(df, lags, variaveis):
        # var_lagN, var_lagN-1, ..., var_lag1, var_original
        colunas_ordenadas = []
        for var in variaveis:
            for lag in range(lags, 0, -1):
                col_lag = f"{var}_lag{lag}"
                if col_lag in df.columns:
                    colunas_ordenadas.append(col_lag)
            if var in df.columns:
                colunas_ordenadas.append(var)
        return colunas_ordenadas

    X_lag = criar_lags(X_cat, num_lags)
    y_lag = criar_lags(y_cat, num_lags)

    variaveis_X = X_cat.columns.tolist()
    variaveis_y = y_cat.columns.tolist()

    colunas_X_ordenadas = ordenar_colunas_por_lags(X_lag, num_lags, variaveis_X)
    colunas_y_ordenadas = ordenar_colunas_por_lags(y_lag, num_lags, variaveis_y)

    dados = pd.concat([X_lag[colunas_X_ordenadas], y_lag[colunas_y_ordenadas]], axis=1)
    dados = dados.dropna()

    print(f"Dados após criação de lags e remoção de NaNs (shape={dados.shape}):")
    print(tabulate(dados.head(), headers='keys', tablefmt='grid'))

    split_index = int(len(dados) * (1 - tamanho_teste))
    dados_treino = dados.iloc[:split_index]
    dados_teste = dados.iloc[split_index:]

    cols_chuva_lags = [col for col in dados.columns if col.startswith(tuple(variaveis_X)) and "_lag" in col]
    cols_chuva_originais = variaveis_X
    cols_vazao_lags = [col for col in dados.columns if col.startswith(tuple(variaveis_y)) and "_lag" in col]

    cols_X = cols_chuva_lags + cols_chuva_originais + cols_vazao_lags
    cols_y = variaveis_y

    X_treino = dados_treino[cols_X]
    X_teste = dados_teste[cols_X]
    y_treino = dados_treino[cols_y]
    y_teste = dados_teste[cols_y]

    print(f"X_treino shape: {X_treino.shape}")
    print(f"y_treino shape: {y_treino.shape}")
    print(f"X_teste shape: {X_teste.shape}")
    print(f"y_teste shape: {y_teste.shape}")

    le_dict = {}

    def label_encode_df(df, le_dict):
        df_encoded = pd.DataFrame(index=df.index)
        for col in df.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col])
            le_dict[col] = le

            valores_originais = le.classes_
            valores_labelados = list(range(len(le.classes_)))
            contagem = df[col].value_counts().reindex(valores_originais).fillna(0).astype(int)
            tabela = list(zip(valores_originais, valores_labelados, contagem))

            print(f"\nMapeamento LabelEncoder para a coluna '{col}':")
            print(tabulate(tabela, headers=["Valor Original", "Valor Labelado", "Número de dados"], tablefmt="grid"))
        return df_encoded

    X_treino_enc = label_encode_df(X_treino, le_dict)
    y_treino_enc = label_encode_df(y_treino, le_dict)
    X_teste_enc = pd.DataFrame(index=X_teste.index)
    y_teste_enc = pd.DataFrame(index=y_teste.index)

    for col in X_teste.columns:
        le = le_dict[col]
        X_teste_enc[col] = X_teste[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    for col in y_teste.columns:
        le = le_dict[col]
        y_teste_enc[col] = y_teste[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    return {
        'X_treino': X_treino_enc,
        'X_teste': X_teste_enc,
        'y_treino': y_treino_enc,
        'y_teste': y_teste_enc,
        'label_encoders': le_dict,
        'tamanho_teste': tamanho_teste
    }


# ============================================================================
# 3. PÓS-PROCESSAMENTO: decodificação + plot Real x Previsto + salvamento
# ============================================================================

def plotar_resultados(dados_cat, dados_entrada_modelo, dados_modelo_RF, plot_interativo=False, nome_modelo=None, salvar_resultados=True, salvar_figura=True):
    """
    Decodifica as previsões categóricas de um modelo (saída de um `rodar_*`)
    de volta para valores numéricos de vazão (usando os percentis originais),
    plota Real x Previsto e salva os resultados em
    `Resultados_Machine_Learning/<nome_modelo>/`.

    Requer que `dados_cat['y_categorizada']` tenha sido gerado com a variável
    dependente chamada 'Q' (ver observação no topo do módulo).
    """
    y_real = dados_cat['y_real']
    tamanho_teste = dados_entrada_modelo['tamanho_teste']
    label_encoders = dados_entrada_modelo['label_encoders']
    y_pred = dados_modelo_RF['y_pred']
    y_teste_enc = dados_entrada_modelo['y_teste']

    n_total = len(y_real)
    n_teste = int(n_total * tamanho_teste)
    y_real_teste = y_real.tail(n_teste)

    le_y = label_encoders['Q']
    y_pred_decoded = le_y.inverse_transform(y_pred)
    y_teste_decoded = le_y.inverse_transform(y_teste_enc.values.ravel())

    min_len = min(len(y_real_teste), len(y_pred_decoded), len(y_teste_decoded))
    y_real_teste = y_real_teste.tail(min_len)
    y_pred_decoded = y_pred_decoded[-min_len:]
    y_teste_decoded = y_teste_decoded[-min_len:]

    y_pred_decoded = pd.Series(y_pred_decoded, index=y_real_teste.index)
    y_teste_decoded = pd.Series(y_teste_decoded, index=y_real_teste.index)

    percentis_vazao = dados_cat['percentis_vazao']

    def extrair_numero_percentil(s):
        m = re.search(r'percentil_(\d+)', s)
        return int(m.group(1)) if m else -1

    percentis = [(col, percentis_vazao.loc['Q', col], extrair_numero_percentil(col)) for col in percentis_vazao.loc['Q'].index]
    percentis.sort(key=lambda x: x[2])

    pct_labels = [x[0] for x in percentis]
    pct_vals = [x[1] for x in percentis]

    y_pred_num = []

    for cat in y_pred_decoded:
        if cat.startswith('Inferior_Vazao_percentil'):
            partes = cat.split('_')
            menor_pct_label = partes[2] + '_' + partes[3]
            idx_menor = pct_labels.index(menor_pct_label)
            menor_val = pct_vals[idx_menor]
            segundo_menor_val = pct_vals[idx_menor + 1] if idx_menor + 1 < len(pct_vals) else menor_val
            valor = menor_val - (segundo_menor_val - menor_val)
            y_pred_num.append(valor)

        elif cat.startswith('Superior_Vazao_percentil'):
            partes = cat.split('_')
            maior_pct_label = partes[2] + '_' + partes[3]
            idx_maior = pct_labels.index(maior_pct_label)
            maior_val = pct_vals[idx_maior]
            segundo_maior_val = pct_vals[idx_maior - 1] if idx_maior - 1 >= 0 else maior_val
            valor = maior_val + (maior_val - segundo_maior_val)
            y_pred_num.append(valor)

        elif cat.startswith('Meio_Vazao_percentil'):
            partes = cat.split('_')
            pct1 = partes[2] + '_' + partes[3]
            pct2 = partes[4] + '_' + partes[5]
            if pct1 in pct_labels and pct2 in pct_labels:
                val1 = pct_vals[pct_labels.index(pct1)]
                val2 = pct_vals[pct_labels.index(pct2)]
                valor = (val1 + val2) / 2
            else:
                valor = np.nan
            y_pred_num.append(valor)

        else:
            y_pred_num.append(np.nan)

    y_pred_num = pd.Series(y_pred_num, index=y_pred_decoded.index)

    if salvar_resultados or salvar_figura:
        pasta_resultados = os.path.join(os.getcwd(), "Resultados_Machine_Learning", nome_modelo)
        os.makedirs(pasta_resultados, exist_ok=True)

    df_comparacao = pd.concat([
        y_real_teste.rename('Vazão Real'),
        y_pred_num.rename('Vazão Prevista'),
        y_teste_decoded.rename('Vazão Teste Decodificada'),
        y_pred_decoded.rename('Vazão Prevista Decodificada')
    ], axis=1)

    print(tabulate(df_comparacao[['Vazão Real', 'Vazão Prevista']].head(), headers='keys', tablefmt='grid'))

    if salvar_resultados:
        caminho_excel = os.path.join(pasta_resultados, f"{nome_modelo}.xlsx")
        df_comparacao.to_excel(caminho_excel, index=True)
        print(f"DataFrame salvo em: {caminho_excel}")

    if plot_interativo:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_comparacao.index, y=df_comparacao['Vazão Real'], mode='lines', name='Vazão Real', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df_comparacao.index, y=df_comparacao['Vazão Prevista'], mode='lines', name='Vazão Prevista', line=dict(color='red', dash='dash')))
        fig.update_layout(title='Comparação entre Vazão Real e Vazão Prevista',
                           xaxis_title='Data',
                           yaxis_title='Vazão')
        fig.show()
        if salvar_figura:
            caminho_figura = os.path.join(pasta_resultados, f"{nome_modelo}.html")
            fig.write_html(caminho_figura)
            print(f"Figura interativa salva em: {caminho_figura}")
    else:
        plt.figure(figsize=(12, 6))
        plt.plot(df_comparacao.index, df_comparacao['Vazão Real'], label='Vazão Real', color='blue')
        plt.plot(df_comparacao.index, df_comparacao['Vazão Prevista'], label='Vazão Prevista', color='red', linestyle='--')
        plt.xlabel('Data')
        plt.ylabel('Vazão')
        plt.title(f'Comparação entre Vazão Real e Vazão Prevista {nome_modelo}')
        plt.legend()
        plt.grid(True)
        if salvar_figura:
            caminho_figura = os.path.join(pasta_resultados, f"{nome_modelo}.png")
            plt.savefig(caminho_figura, bbox_inches='tight')
            print(f"Figura salva em: {caminho_figura}")
        plt.show()

    return df_comparacao


# ============================================================================
# 4. MODELOS DE CLASSIFICAÇÃO -- 18 funções `rodar_*`
#
#    Todas seguem a mesma estrutura: parâmetros default (se parametros=None),
#    treino, previsão, métricas (acurácia + classification_report), matriz de
#    confusão, e retorno de {'clf': modelo_treinado, 'y_pred': previsoes}.
# ============================================================================

def rodar_random_forest(dados_entrada_modelo, parametros=None, plotar_teste=True):
    X_train = dados_entrada_modelo['X_treino']
    X_test = dados_entrada_modelo['X_teste']
    y_train = dados_entrada_modelo['y_treino']
    y_test = dados_entrada_modelo['y_teste']

    if parametros is None:
        parametros = {
            'n_estimators': 100,
            'criterion': 'gini',
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'min_weight_fraction_leaf': 0.0,
            'max_features': 'sqrt',
            'max_leaf_nodes': None,
            'min_impurity_decrease': 0.0,
            'bootstrap': True,
            'oob_score': False,
            'n_jobs': None,
            'random_state': 42,
            'verbose': 0,
            'warm_start': False,
            'class_weight': None,
            'ccp_alpha': 0.0,
            'max_samples': None
        }

    clf = RandomForestClassifier(**parametros)
    clf.fit(X_train, y_train.values.ravel())

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acc:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    if plotar_teste:
        plt.show()

    return {'clf': clf, 'y_pred': y_pred}


def rodar_decision_tree(dados_entrada_modelo, parametros=None):
    if parametros is None:
        parametros = {
            'criterion': 'gini',
            'splitter': 'best',
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'min_weight_fraction_leaf': 0.0,
            'max_features': None,
            'random_state': 42,
            'max_leaf_nodes': None,
            'min_impurity_decrease': 0.0,
            'class_weight': None,
            'ccp_alpha': 0.0
        }
    X_train = dados_entrada_modelo['X_treino']
    X_test = dados_entrada_modelo['X_teste']
    y_train = dados_entrada_modelo['y_treino']
    y_test = dados_entrada_modelo['y_teste']

    clf = DecisionTreeClassifier(**parametros)
    clf.fit(X_train, y_train.values.ravel())

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acc:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title('Matriz de Confusão - Decision Tree')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.show()

    return {'clf': clf, 'y_pred': y_pred}


def rodar_gradient_boosting(dados_entrada_modelo, parametros=None, plotar_teste=True):
    if parametros is None:
        parametros = {
            'loss': 'log_loss',
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 1.0,
            'criterion': 'friedman_mse',
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'min_weight_fraction_leaf': 0.0,
            'max_depth': 3,
            'min_impurity_decrease': 0.0,
            'init': None,
            'random_state': 42,
            'max_features': None,
            'verbose': 0,
            'max_leaf_nodes': None,
            'warm_start': False,
            'validation_fraction': 0.1,
            'n_iter_no_change': None,
            'tol': 1e-4
        }
    X_train = dados_entrada_modelo['X_treino']
    X_test = dados_entrada_modelo['X_teste']
    y_train = dados_entrada_modelo['y_treino']
    y_test = dados_entrada_modelo['y_teste']

    clf = GradientBoostingClassifier(**parametros)
    clf.fit(X_train, y_train.values.ravel())

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acc:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão - Gradient Boosting')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    if plotar_teste:
        plt.show()

    return {'clf': clf, 'y_pred': y_pred}


def rodar_hist_gradient_boosting(dados_entrada_modelo, parametros=None, plotar_teste=True):
    if parametros is None:
        parametros = {
            'loss': 'log_loss',
            'learning_rate': 0.1,
            'max_iter': 100,
            'max_leaf_nodes': 31,
            'max_depth': None,
            'min_samples_leaf': 20,
            'max_bins': 255,
            'categorical_features': None,
            'warm_start': False,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 10,
            'tol': 1e-7,
            'scoring': None,
            'verbose': 0,
            'random_state': 42,
            'l2_regularization': 0.0,
        }
    X_train = dados_entrada_modelo['X_treino']
    X_test = dados_entrada_modelo['X_teste']
    y_train = dados_entrada_modelo['y_treino']
    y_test = dados_entrada_modelo['y_teste']

    clf = HistGradientBoostingClassifier(**parametros)
    clf.fit(X_train, y_train.values.ravel())

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acc:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão - HistGradientBoosting')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    if plotar_teste:
        plt.show()

    return {'clf': clf, 'y_pred': y_pred}


def rodar_mlp(dados_entrada_modelo, parametros=None, plotar_teste=True):
    if parametros is None:
        parametros = {
            'hidden_layer_sizes': (100,),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'batch_size': 'auto',
            'learning_rate': 'constant',
            'learning_rate_init': 0.001,
            'power_t': 0.5,
            'max_iter': 200,
            'shuffle': True,
            'random_state': 42,
            'tol': 1e-4,
            'verbose': False,
            'warm_start': False,
            'momentum': 0.9,
            'nesterovs_momentum': True,
            'early_stopping': False,
            'validation_fraction': 0.1,
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-8,
            'n_iter_no_change': 10,
            'max_fun': 15000
        }

    X_train = dados_entrada_modelo['X_treino']
    X_test = dados_entrada_modelo['X_teste']
    y_train = dados_entrada_modelo['y_treino']
    y_test = dados_entrada_modelo['y_teste']

    clf = MLPClassifier(**parametros)
    clf.fit(X_train, y_train.values.ravel())

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acc:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão - MLPClassifier')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    if plotar_teste:
        plt.show()

    return {'clf': clf, 'y_pred': y_pred}


def rodar_extra_trees(dados_entrada_modelo, parametros=None, plotar_teste=True):
    if parametros is None:
        parametros = {
            'n_estimators': 100,
            'criterion': 'gini',
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'min_weight_fraction_leaf': 0.0,
            'max_features': 'sqrt',
            'max_leaf_nodes': None,
            'min_impurity_decrease': 0.0,
            'bootstrap': False,
            'oob_score': False,
            'n_jobs': -1,
            'random_state': 42,
            'class_weight': None,
            'ccp_alpha': 0.0,
            'max_samples': None
        }
    X_train = dados_entrada_modelo['X_treino']
    X_test = dados_entrada_modelo['X_teste']
    y_train = dados_entrada_modelo['y_treino']
    y_test = dados_entrada_modelo['y_teste']

    clf = ExtraTreesClassifier(**parametros)
    clf.fit(X_train, y_train.values.ravel())

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acc:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão - ExtraTreesClassifier')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    if plotar_teste:
        plt.show()

    return {'clf': clf, 'y_pred': y_pred}


def rodar_svc(dados_entrada_modelo, parametros=None, plotar_teste=True):
    if parametros is None:
        parametros = {
            'C': 1.0,
            'kernel': 'rbf',
            'degree': 3,
            'gamma': 'scale',
            'coef0': 0.0,
            'shrinking': True,
            'probability': False,
            'tol': 1e-3,
            'cache_size': 200,
            'class_weight': None,
            'verbose': False,
            'max_iter': -1,
            'decision_function_shape': 'ovr',
            'break_ties': False,
            'random_state': 42
        }

    X_train = dados_entrada_modelo['X_treino']
    X_test = dados_entrada_modelo['X_teste']
    y_train = dados_entrada_modelo['y_treino']
    y_test = dados_entrada_modelo['y_teste']

    clf = SVC(**parametros)
    clf.fit(X_train, y_train.values.ravel())

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acc:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão - SVC')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    if plotar_teste:
        plt.show()

    return {'clf': clf, 'y_pred': y_pred}


def rodar_kneighbors(dados_entrada_modelo, parametros=None, plotar_teste=True):
    if parametros is None:
        parametros = {
            'n_neighbors': 5,
            'weights': 'uniform',
            'algorithm': 'auto',
            'leaf_size': 30,
            'p': 2,
            'metric': 'minkowski',
            'metric_params': None,
            'n_jobs': -1
        }

    X_train = dados_entrada_modelo['X_treino'].values
    X_test = dados_entrada_modelo['X_teste'].values
    y_train = dados_entrada_modelo['y_treino'].values.ravel()
    y_test = dados_entrada_modelo['y_teste'].values.ravel()

    clf = KNeighborsClassifier(**parametros)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acc:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão - KNeighborsClassifier')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    if plotar_teste:
        plt.show()

    return {'clf': clf, 'y_pred': y_pred}


def rodar_gaussian_nb(dados_entrada_modelo, parametros=None, plotar_teste=True):
    if parametros is None:
        parametros = {
            'priors': None,
            'var_smoothing': 1e-9
        }

    X_train = dados_entrada_modelo['X_treino']
    X_test = dados_entrada_modelo['X_teste']
    y_train = dados_entrada_modelo['y_treino']
    y_test = dados_entrada_modelo['y_teste']

    clf = GaussianNB(**parametros)
    clf.fit(X_train, y_train.values.ravel())

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acc:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão - GaussianNB')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    if plotar_teste:
        plt.show()

    return {'clf': clf, 'y_pred': y_pred}


def rodar_logistic_regression(dados_entrada_modelo, parametros=None, plotar_teste=True):
    """
    Executa Logistic Regression e retorna modelo + previsões
    """
    if parametros is None:
        parametros = {
            'solver': 'lbfgs',
            'max_iter': 1000,
            'random_state': 42,
            'C': 1.0,
            'penalty': 'l2',
            'n_jobs': None
        }

    X_train = dados_entrada_modelo['X_treino']
    X_test = dados_entrada_modelo['X_teste']
    y_train = dados_entrada_modelo['y_treino']
    y_test = dados_entrada_modelo['y_teste']

    clf = LogisticRegression(**parametros)
    clf.fit(X_train, y_train.values.ravel())

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acc:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão - LogisticRegression')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    if plotar_teste:
        plt.show()

    return {'clf': clf, 'y_pred': y_pred}


def rodar_linear_svc(dados_entrada_modelo, parametros=None, plotar_teste=True):
    """
    Executa LinearSVC e retorna modelo + previsões
    """
    if parametros is None:
        parametros = {
            'C': 1.0,
            'max_iter': 1000,
            'random_state': 42,
            'tol': 1e-4,
            'dual': True,
            'loss': 'squared_hinge'
        }
    X_train = dados_entrada_modelo['X_treino']
    X_test = dados_entrada_modelo['X_teste']
    y_train = dados_entrada_modelo['y_treino']
    y_test = dados_entrada_modelo['y_teste']

    clf = LinearSVC(**parametros)
    clf.fit(X_train, y_train.values.ravel())

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acc:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão - LinearSVC')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    if plotar_teste:
        plt.show()

    return {'clf': clf, 'y_pred': y_pred}


def rodar_ridge_classifier(dados_entrada_modelo, parametros=None, plotar_teste=True):
    """
    Executa RidgeClassifier e retorna modelo + previsões
    """
    if parametros is None:
        parametros = {
            'alpha': 1.0,
            'max_iter': 1000,
            'tol': 1e-3,
            'solver': 'auto',
            'random_state': 42
        }
    X_train = dados_entrada_modelo['X_treino']
    X_test = dados_entrada_modelo['X_teste']
    y_train = dados_entrada_modelo['y_treino']
    y_test = dados_entrada_modelo['y_teste']

    clf = RidgeClassifier(**parametros)
    clf.fit(X_train, y_train.values.ravel())

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acc:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão - RidgeClassifier')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    if plotar_teste:
        plt.show()

    return {'clf': clf, 'y_pred': y_pred}


def rodar_perceptron(dados_entrada_modelo, parametros=None, plotar_teste=True):
    """
    Executa Perceptron e retorna modelo + previsões
    """
    if parametros is None:
        parametros = {
            'max_iter': 1000,
            'eta0': 1.0,
            'tol': 1e-3,
            'random_state': 42,
            'penalty': None,
            'alpha': 0.0001,
            'fit_intercept': True,
            'shuffle': True
        }
    X_train = dados_entrada_modelo['X_treino']
    X_test = dados_entrada_modelo['X_teste']
    y_train = dados_entrada_modelo['y_treino']
    y_test = dados_entrada_modelo['y_teste']

    clf = Perceptron(**parametros)
    clf.fit(X_train, y_train.values.ravel())

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acc:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão - Perceptron')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    if plotar_teste:
        plt.show()

    return {'clf': clf, 'y_pred': y_pred}


def rodar_qda(dados_entrada_modelo, parametros=None, plotar_teste=True):
    """
    Executa QuadraticDiscriminantAnalysis e retorna modelo + previsões
    """
    if parametros is None:
        parametros = {
            'priors': None,
            'reg_param': 0.0,
            'store_covariance': False,
            'tol': 1e-4
        }

    X_train = dados_entrada_modelo['X_treino']
    X_test = dados_entrada_modelo['X_teste']
    y_train = dados_entrada_modelo['y_treino']
    y_test = dados_entrada_modelo['y_teste']

    clf = QuadraticDiscriminantAnalysis(**parametros)
    clf.fit(X_train, y_train.values.ravel())

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acc:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão - QDA')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    if plotar_teste:
        plt.show()

    return {'clf': clf, 'y_pred': y_pred}


def rodar_lda(dados_entrada_modelo, parametros=None, plotar_teste=True):
    """
    Executa LinearDiscriminantAnalysis e retorna modelo + previsões
    """
    if parametros is None:
        parametros = {
            'solver': 'svd',
            'shrinkage': None,
            'priors': None,
            'n_components': None,
            'store_covariance': False,
            'tol': 1e-4
        }

    X_train = dados_entrada_modelo['X_treino']
    X_test = dados_entrada_modelo['X_teste']
    y_train = dados_entrada_modelo['y_treino']
    y_test = dados_entrada_modelo['y_teste']

    clf = LinearDiscriminantAnalysis(**parametros)
    clf.fit(X_train, y_train.values.ravel())

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acc:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão - LDA')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    if plotar_teste:
        plt.show()

    return {'clf': clf, 'y_pred': y_pred}


def rodar_xgboost(dados_entrada_modelo, parametros=None, plotar_teste=True):
    """
    Treina e avalia um XGBoost Classifier no conjunto de dados fornecido.

    parametros: dict, parâmetros do XGBClassifier.
    """
    X_train = dados_entrada_modelo['X_treino']
    X_test = dados_entrada_modelo['X_teste']
    y_train = dados_entrada_modelo['y_treino']
    y_test = dados_entrada_modelo['y_teste']

    if parametros is None:
        parametros = {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': 42,
            'eval_metric': 'mlogloss'
        }

    clf = XGBClassifier(**parametros)
    clf.fit(X_train, y_train.values.ravel())

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acc:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão - XGBoost')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    if plotar_teste:
        plt.show()

    return {'clf': clf, 'y_pred': y_pred}


def rodar_lightgbm(dados_entrada_modelo, parametros=None, plotar_teste=True):
    """
    Treina e avalia um LightGBM Classifier no conjunto de dados fornecido.

    parametros: dict, parâmetros do LGBMClassifier.
    """
    X_train = dados_entrada_modelo['X_treino']
    X_test = dados_entrada_modelo['X_teste']
    y_train = dados_entrada_modelo['y_treino']
    y_test = dados_entrada_modelo['y_teste']

    warnings.filterwarnings('ignore')

    if parametros is None:
        parametros = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': -1,
            'num_leaves': 31,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'reg_alpha': 0.0,
            'reg_lambda': 0.0,
            'random_state': 42,
            'verbose': -1
        }

    clf = lgb.LGBMClassifier(**parametros)
    clf.fit(X_train, y_train.values.ravel())

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acc:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão - LightGBM')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    if plotar_teste:
        plt.show()

    return {'clf': clf, 'y_pred': y_pred}


def rodar_catboost(dados_entrada_modelo, parametros=None, plotar_teste=True):
    """
    Treina e avalia um CatBoost Classifier no conjunto de dados fornecido.

    parametros: dict, parâmetros do CatBoostClassifier.
    """
    X_train = dados_entrada_modelo['X_treino']
    X_test = dados_entrada_modelo['X_teste']
    y_train = dados_entrada_modelo['y_treino']
    y_test = dados_entrada_modelo['y_teste']

    if parametros is None:
        parametros = {
            'iterations': 1000,
            'learning_rate': 0.03,
            'depth': 6,
            'l2_leaf_reg': 3.0,
            'bootstrap_type': 'Bayesian',
            'bagging_temperature': 1.0,
            'random_strength': 1.0,
            'border_count': 254,
            'feature_border_type': 'GreedyLogSum',
            'verbose': 0,
            'random_seed': 42,
            'od_type': 'IncToDec',
            'od_wait': 20,
            'thread_count': -1,
            'class_weights': None,
            'allow_writing_files': False
        }

    clf = CatBoostClassifier(**parametros)
    clf.fit(X_train, y_train.values.ravel())

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia: {acc:.4f}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão - CatBoost')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    if plotar_teste:
        plt.show()

    return {'clf': clf, 'y_pred': y_pred}


# ============================================================================
# 5. DIAGNÓSTICO PÓS-TREINO
# ============================================================================

def funcao_diagnostico_overfitting(
    lista_modelos,
    dados_entrada,
    limiar_moderado=0.05,
    limiar_alto=0.20
):
    """
    Avalia o overfitting de múltiplos modelos comparando suas métricas de
    treino e teste.

    Parâmetros:
    - lista_modelos: list de dicts {'nome': str, 'dados_modelo': dict_retornado_por_rodar_*}
    - dados_entrada: dict 'dados_entrada_modelo' (saída de pre_classificadores)
    - limiar_moderado: float (default=0.05) -- diferença (treino-teste) para "Risco Moderado"
    - limiar_alto: float (default=0.20) -- diferença (treino-teste) para "ALTO RISCO"

    Retorna:
    - df_diagnostico: pd.DataFrame com Modelo, Acurácia Treino, Acurácia Teste, Diferença, Diagnóstico
    """
    try:
        X_treino = dados_entrada['X_treino']
        y_treino = dados_entrada['y_treino']
        X_teste = dados_entrada['X_teste']
        y_teste = dados_entrada['y_teste']
    except KeyError:
        print("Erro: 'dados_entrada' não parece ser o dicionário correto.")
        return None

    resultados = []

    print("Gerando Diagnóstico de Overfitting...")
    print("-" * 40)

    for item in lista_modelos:
        nome_modelo = item['nome']
        dados_modelo = item['dados_modelo']

        try:
            clf = dados_modelo['clf']
        except KeyError:
            print(f"AVISO: Modelo '{nome_modelo}' não possui a chave 'clf'. Pulando.")
            continue

        y_pred_treino = clf.predict(X_treino)
        acc_treino = accuracy_score(y_treino, y_pred_treino)

        y_pred_teste = clf.predict(X_teste)
        acc_teste = accuracy_score(y_teste, y_pred_teste)

        diferenca = acc_treino - acc_teste
        diagnostico = "Baixo Risco"

        if diferenca >= limiar_alto:
            diagnostico = "ALTO RISCO DE OVERFITTING"
        elif diferenca >= limiar_moderado:
            diagnostico = "Risco Moderado"

        resultados.append([
            nome_modelo,
            f"{acc_treino:.4f}",
            f"{acc_teste:.4f}",
            f"{diferenca:.4f}",
            diagnostico
        ])

    headers = ["Modelo", "Acurácia Treino", "Acurácia Teste", "Diferença", "Diagnóstico"]
    print(tabulate(resultados, headers=headers, tablefmt='grid'))

    df_diagnostico = pd.DataFrame(resultados, columns=headers)
    df_diagnostico_num = df_diagnostico.astype({
        'Acurácia Treino': 'float',
        'Acurácia Teste': 'float',
        'Diferença': 'float'
    })

    return df_diagnostico_num


def plotar_curva_validacao(
    estimator,
    param_name,
    param_range,
    X,
    y,
    cv=5,
    scoring='accuracy',
    print_table=True
):
    """
    Calcula e plota a curva de validação (treino x cross-validation) para um
    dado hiperparâmetro.

    Parâmetros:
    - estimator: modelo (classificador) do Scikit-learn.
    - param_name: nome do hiperparâmetro a variar (ex: 'max_depth').
    - param_range: lista de valores a testar para o hiperparâmetro.
    - X, y: features/alvo de treino (ex: X_treino, y_treino).
    - cv: número de folds da validação cruzada (default=5).
    - scoring: métrica a ser usada (default='accuracy').
    - print_table: se True, imprime uma tabela com os scores.
    """
    print(f"Calculando Curva de Validação para '{param_name}'...")

    train_scores, test_scores = validation_curve(
        estimator,
        X,
        y.values.ravel(),
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    if print_table:
        print("\nResultados da Curva de Validação:")
        tabela = []
        headers = [f"Valor {param_name}", "Média Treino", "Média Validação (CV)"]
        for i, val in enumerate(param_range):
            tabela.append([
                val,
                f"{train_mean[i]:.4f} (+/- {train_std[i]:.4f})",
                f"{test_mean[i]:.4f} (+/- {test_std[i]:.4f})"
            ])
        print(tabulate(tabela, headers=headers, tablefmt='grid'))

    plt.figure(figsize=(10, 6))

    x_axis = np.arange(len(param_range))

    plt.plot(x_axis, train_mean, 'o-', color='blue', label='Score de Treino', markersize=8)
    plt.plot(x_axis, test_mean, 'o-', color='red', label='Score de Validação (CV)', markersize=8)

    plt.fill_between(x_axis, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(x_axis, test_mean - test_std, test_mean + test_std, alpha=0.1, color='red')

    plt.title(f'Curva de Validação para: {estimator.__class__.__name__}')
    plt.xlabel(f'Hiperparâmetro: {param_name}')
    plt.ylabel(f'Score ({scoring})')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.xticks(x_axis, labels=[str(p) for p in param_range])

    plt.show()
