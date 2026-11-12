# %% [markdown]
# Aqui temos o caminho para o nosso arquivo base
# Também temos a pasta de saída das imagens, e caso ela não exista, o script irá criar-la

# %%
caminho_base = r'C:\Users\filipe.nogueira\OneDrive - Grupo Aguas do Brasil\Área de Trabalho\TrabalhoDataMining_python\df_datamining.xlsx'
caminho_figuras = r'C:\Users\filipe.nogueira\OneDrive - Grupo Aguas do Brasil\Área de Trabalho\TrabalhoDataMining_python\SAIDA\Figuras'
caminho_dataframes = r'C:\Users\filipe.nogueira\OneDrive - Grupo Aguas do Brasil\Área de Trabalho\TrabalhoDataMining_python\SAIDA\Dataframes'
colunas="A:AK"

# %%
import os
import pandas as pd
from tabulate import tabulate

def criar_df(caminho_base, caminho_figuras, caminho_dataframes, colunas):
    """
    Cria um DataFrame a partir de um arquivo Excel e garante a existência do diretório de saída.

    Parâmetros:
    - caminho_base (str): Caminho do arquivo Excel de origem.
    - caminho_saida (str): Caminho da pasta onde os resultados serão armazenados.
    - colunas (str): Intervalo de colunas a serem lidas no Excel (exemplo: "A:AK").

    Retorna:
    - df (DataFrame): DataFrame lido do Excel.
    """
    if not os.path.exists(caminho_figuras):
        os.makedirs(caminho_figuras)
    if not os.path.exists(caminho_dataframes):
        os.makedirs(caminho_dataframes)
    
    df = pd.read_excel(caminho_base, usecols=colunas, engine='openpyxl')
    df.set_index('Data', inplace=True)
    print(tabulate(df.head(), headers='keys', tablefmt='grid'))
    return df

if __name__ == "__main__":
    df = criar_df(caminho_base, caminho_figuras, caminho_dataframes, colunas)


# %%
import matplotlib.pyplot as plt

def plot_series_e_histograma(df, caminho_figuras):
    """
    Para cada coluna do DataFrame, plota a série temporal e o histograma lado a lado.
    O índice deve ser datetime (ex: 'Data').
    """
    
    #cria subpasta
    pasta_destino = os.path.join(caminho_figuras, 'Series_Temporais_e_Histogramas')
    os.makedirs(pasta_destino, exist_ok=True)
    
    colunas = df.columns

    for coluna in colunas:
        plt.figure(figsize=(14, 5))

        # Série temporal
        plt.subplot(1, 2, 1)
        plt.plot(df.index, df[coluna], color='blue')
        plt.title(f'Série Temporal - {coluna}')
        plt.xlabel('Data')
        plt.ylabel('Valor')
        plt.grid(True)

        # Histograma
        plt.subplot(1, 2, 2)
        plt.hist(df[coluna].dropna(), bins=30, color='green', edgecolor='black')
        plt.title(f'Histograma de Frequência - {coluna}')
        plt.xlabel('Valor')
        plt.ylabel('Frequência')
        plt.grid(True)
        plt.tight_layout()
        
        nome_arquivo = f'Serie_Temporal_e_Hist_{coluna}.png'
        caminho_series = os.path.join(pasta_destino, nome_arquivo)
        plt.savefig(caminho_series, dpi = 300)
        plt.show()
if __name__ == "__main__":
    # Chame a função:
    plot_series_e_histograma(df, caminho_figuras)

# %%
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

def visualizar_sazonalidades(df, caminho_base, periodo, m_movel_1, m_movel_2, se_diario, virar_mensal):
    """
    Para cada coluna em df, faz decomposição sazonal aditiva:
    - Se dados diários e virar_mensal=True, faz resample mensal somando os dados.
    - Se dados mensais, aplica decomposição com período 12 e médias móveis trimestral e anual.
    - Plota série original e médias móveis (m7 e m30) e suas decomposições.

    Parâmetros:
    - df: DataFrame com índice datetime.
    - caminho_base: pasta onde salvar as figuras.
    - se_diario: True se dados são diários, False se mensais.
    - virar_mensal: se True e se_diario=True, faz resample mensal somando.
    - periodo_diario: período sazonal para dados diários (default 365).
    - m_movel_1_diario: janela da 1ª média móvel para dados diários (default 90).
    - m_movel_2_diario: janela da 2ª média móvel para dados diários (default 365).
    """
    uni_tempo = 'Dias'
    if se_diario:
        if virar_mensal:
            print('Convertendo dados diarios em mensais!')
            # Define agregação personalizada
            agg_dict = {col: ('mean' if col == 'Q' else 'sum') for col in df.columns}

            # Resample aplicando regras
            df = df.resample("M").agg(agg_dict)
            periodo = 12
            m_movel_1 = 3
            m_movel_2 = 12
            uni_tempo = 'Meses'
        else:
            print('Dados diários mantidos')
    else:
        print('Dataframe mensal, fazendo médias móveis de 3 e 12 meses para um periodo de 12 meses')
        periodo = 12
        m_movel_1 = 3
        m_movel_2 = 12





    # Cria subpasta
    pasta_destino = os.path.join(caminho_base, "Series_Decompostas")
    os.makedirs(pasta_destino,exist_ok=True)
        
    colunas = df.columns
    
    for coluna in colunas:
        serie_original = df[coluna].dropna()
        m7 = serie_original.rolling(window=m_movel_1, center=True).mean().dropna()
        m30 = serie_original.rolling(window=m_movel_2, center=True).mean().dropna()
        
        # Decomposição sazonal para cada série
        try:
            decom_original = sm.tsa.seasonal_decompose(serie_original, model='additive', period=periodo)
            decom_m7 = sm.tsa.seasonal_decompose(m7, model='additive', period=periodo)
            decom_m30 = sm.tsa.seasonal_decompose(m30, model='additive', period=periodo)
        except Exception as e:
            print(f"Erro na decomposição da coluna {coluna}: {e}")
            continue
        
        fig, axs = plt.subplots(4, 3, figsize=(18, 12), sharex=False)
        fig.suptitle(f'Decomposição Sazonal - {coluna}', fontsize=16)
        
        # Lista para facilitar loop
        decom_list = [decom_original, decom_m7, decom_m30]
        titulos_colunas = ['Série Original', f'Média Móvel {m_movel_1} {uni_tempo}', f'Média Móvel {m_movel_2} {uni_tempo}']
        linhas = ['Observado', 'Tendência', 'Sazonalidade', 'Resíduo']
        
        for col_idx, decom in enumerate(decom_list):
            axs[0, col_idx].plot(decom.observed)
            axs[1, col_idx].plot(decom.trend)
            axs[2, col_idx].plot(decom.seasonal)
            axs[3, col_idx].plot(decom.resid)
            
            # Setando títulos colunas (apenas primeira linha)
            axs[0, col_idx].set_title(titulos_colunas[col_idx])
            
            for row_idx in range(4):
                axs[row_idx, col_idx].set_ylabel(linhas[row_idx])
                axs[row_idx, col_idx].grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        nome_arquivo = f'Series_Decompostas_de_{coluna}.png'
        caminho_arquivo = os.path.join(pasta_destino, nome_arquivo)
        plt.savefig(caminho_arquivo,dpi=300)
        plt.show()
if __name__ == "__main__":
    visualizar_sazonalidades(df, caminho_figuras, periodo=365, m_movel_1 = 90, m_movel_2 = 365, se_diario = True, virar_mensal = False)

# %%
# import pandas as pd
# import matplotlib.pyplot as plt
# import statsmodels.api as sm

# def visualizar_sazonalidade_mensal(df):
#     """
#     Faz decomposição sazonal mensal (periodicidade 12) para:
#     - Série original mensal
#     - Média móvel 3 meses
#     - Média móvel 6 meses

#     Parâmetros:
#     - df: DataFrame com índice datetime.
#     """
#     # Reamostrando para frequência mensal (soma das precipitações do mês)
#     df_mensal = df.resample('M').sum()

#     for coluna in df_mensal.columns:
#         serie = df_mensal[coluna].dropna()
#         mm3 = serie.rolling(window=3, center=True).mean().dropna()
#         mm6 = serie.rolling(window=12, center=True).mean().dropna()

#         # Verifica se a série tem dados suficientes
#         if len(serie) < 24:
#             print(f"Série muito curta para decomposição: {coluna}")
#             continue

#         try:
#             decom_original = sm.tsa.seasonal_decompose(serie, model='additive', period=12)
#             decom_mm3 = sm.tsa.seasonal_decompose(mm3, model='additive', period=12)
#             decom_mm6 = sm.tsa.seasonal_decompose(mm6, model='additive', period=12)
#         except Exception as e:
#             print(f"Erro na decomposição da coluna {coluna}: {e}")
#             continue

#         fig, axs = plt.subplots(4, 3, figsize=(18, 12), sharex=False)
#         fig.suptitle(f'Decomposição Sazonal Mensal - {coluna}', fontsize=16)

#         decomps = [decom_original, decom_mm3, decom_mm6]
#         titulos = ['Mensal Original', 'Média Móvel 3 meses', 'Média Móvel 6 meses']
#         componentes = ['observed', 'trend', 'seasonal', 'resid']
#         labels = ['Observado', 'Tendência', 'Sazonalidade', 'Resíduo']

#         for col_idx, decom in enumerate(decomps):
#             for row_idx, comp in enumerate(componentes):
#                 dados = getattr(decom, comp)
#                 axs[row_idx, col_idx].plot(dados)
#                 if col_idx == 0:
#                     axs[row_idx, col_idx].set_ylabel(labels[row_idx])
#                 if row_idx == 0:
#                     axs[row_idx, col_idx].set_title(titulos[col_idx])
#                 axs[row_idx, col_idx].grid(True)

#         plt.tight_layout(rect=[0, 0, 1, 0.95])
#         plt.show()

# visualizar_sazonalidade_mensal(df)

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def fazer_matriz_correlacao(df, caminho_figuras):
    """
    Plota duas matrizes de correlação (Pearson):
    1. Com todas as colunas
    2. Excluindo a coluna 'Q' (vazão)

    Cada uma como um heatmap com anotações.
    """

    # Cria a subpasta
    pasta_destino = os.path.join(caminho_figuras, 'matriz_correlacao')
    os.makedirs(pasta_destino, exist_ok=True)

    def plot_heatmap(corr, titulo, nome_arquivo):
        mask = np.triu(np.ones_like(corr, dtype=bool))  # Máscara para mostrar só triângulo inferior
        plt.figure(figsize=(16, 14))
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap='RdBu',
            center=0,
            square=True,
            linewidths=0.1,
            cbar_kws={"shrink": 0.75, "label": 'Correlação de Pearson' }
        )
        plt.title(titulo, fontsize=16)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        # Salva a figura
        caminho_salvar = os.path.join(pasta_destino, nome_arquivo)
        plt.savefig(caminho_salvar, dpi=300)
        plt.show()

    # --- 1. Todas as colunas ---
    corr_completa = df.corr(method='pearson')
    plot_heatmap(corr_completa, 'Matriz de Correlação - Todas as colunas', 'Matriz_correlacao_completa.png')

    # --- 2. Sem a coluna 'Q' ---
    if 'Q' in df.columns:
        df_sem_q = df.drop(columns='Q')
        corr_sem_q = df_sem_q.corr(method='pearson')
        plot_heatmap(corr_sem_q, 'Matriz de Correlação - Sem a coluna Q (Vazão)', 'Matriz_correlacao_sem_vazao.png')
    else:
        print("Coluna 'Q' não encontrada. Pulando segunda matriz.")

if __name__ == "__main__":
    fazer_matriz_correlacao(df, caminho_figuras)

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# def fazer_matriz_dissimilaridade(df, caminho_figuras):
#     """
#     Calcula e plota matrizes de dissimilaridade (distância Euclidiana) entre colunas do DataFrame.
#     Faz para todas colunas e para todas menos a coluna 'Q'.
    
#     Parâmetros:
#     - df: DataFrame com colunas numéricas e índice datetime.
#     """
#     # Cria a subpasta
#     pasta_destino = os.path.join(caminho_figuras, 'matriz_dissimilaridade')
#     os.makedirs(pasta_destino, exist_ok=True)

#     # Função auxiliar para calcular matriz de dissimilaridade
#     def matriz_dissim(df_):
#         # Transpõe para que pdist calcule distâncias entre colunas
#         dados = df_.T.values
#         dist = pdist(dados, metric='euclidean')
#         dist_mat = squareform(dist)
#         return pd.DataFrame(dist_mat, index=df_.columns, columns=df_.columns)
    
#     # Matriz com todas as colunas
#     dist_todas = matriz_dissim(df)
    
#     # Matriz sem coluna 'Q' (se existir)
#     if 'Q' in df.columns:
#         dist_sem_q = matriz_dissim(df.drop(columns=['Q']))
#     else:
#         dist_sem_q = None
    
#     # Plotagem
#     plt.figure(figsize=(18, 7))
    
#     # Heatmap - todas colunas
#     plt.subplot(1, 2, 1)
#     sns.heatmap(dist_todas, cmap='viridis', square=True, linewidths=0.3,
#                 cbar_kws={"shrink": 0.8, "label": "Distância Euclidiana"})
#     plt.title('Matriz de Dissimilaridade - Todas as Colunas')
#     plt.xticks(rotation=90, fontsize=10)
#     plt.yticks(rotation=0, fontsize=10)
    
#     # Heatmap - sem Q
#     if dist_sem_q is not None:
#         plt.subplot(1, 2, 2)
#         sns.heatmap(dist_sem_q, cmap='viridis', square=True, linewidths=0.3,
#                     cbar_kws={"shrink": 0.8, "label": "Distância Euclidiana"})
#         plt.title('Matriz de Dissimilaridade - Sem Coluna Q')
#         plt.xticks(rotation=90, fontsize=10)
#         plt.yticks(rotation=0, fontsize=10)
    
#     plt.tight_layout()
#     # Salva a figura
#     nome_arquivo = 'Matrizes_dissimilaridade'
#     caminho_salvar = os.path.join(pasta_destino, nome_arquivo)
#     plt.savefig(caminho_salvar, dpi=300)
#     plt.show()
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

def fazer_matriz_dissimilaridade(df, caminho_figuras, normalizar=True):
    """
    Calcula e plota matrizes de dissimilaridade (distância Euclidiana) entre colunas do DataFrame.
    Opção de normalizar os dados via Z-Score.

    Parâmetros:
    - df: DataFrame com colunas numéricas e índice datetime.
    - caminho_figuras: caminho para salvar as figuras
    - normalizar: bool, se True normaliza por Z-Score
    """

    # Normalização opcional (Z-score)
    if normalizar:
        df_proc = (df - df.mean()) / df.std(ddof=0)
    else:
        df_proc = df.copy()

    # Cria subpasta de destino
    pasta_destino = os.path.join(caminho_figuras, 'matriz_dissimilaridade')
    os.makedirs(pasta_destino, exist_ok=True)

    # Função interna para matriz de dissimilaridade
    def matriz_dissim(df_):
        dados = df_.T.values
        dist = pdist(dados, metric='euclidean')
        dist_mat = squareform(dist)
        return pd.DataFrame(dist_mat, index=df_.columns, columns=df_.columns)

    # Matriz com todas as colunas
    dist_todas = matriz_dissim(df_proc)

    # Matriz sem Q
    dist_sem_q = None
    if 'Q' in df_proc.columns:
        dist_sem_q = matriz_dissim(df_proc.drop(columns=['Q']))

    # Plotagem
    plt.figure(figsize=(18, 7))

    # Heatmap - todas
    plt.subplot(1, 2, 1)
    sns.heatmap(dist_todas, cmap='viridis', square=True, linewidths=0.3,
                cbar_kws={"shrink": 0.8, "label": "Distância Euclidiana"})
    titulo = 'Todas as Colunas'
    #if normalizar: titulo += ' (Normalizado)'
    plt.title(f'Matriz de Dissimilaridade - {titulo}')
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)

    # Heatmap - sem Q
    if dist_sem_q is not None:
        plt.subplot(1, 2, 2)
        sns.heatmap(dist_sem_q, cmap='viridis', square=True, linewidths=0.3,
                    cbar_kws={"shrink": 0.8, "label": "Distância Euclidiana"})
        titulo2 = 'Sem Q'
        #if normalizar: titulo2 += ' (Normalizado)'
        plt.title(f'Matriz de Dissimilaridade - {titulo2}')
        plt.xticks(rotation=90, fontsize=10)
        plt.yticks(rotation=0, fontsize=10)

    plt.tight_layout()

    # Nome do arquivo inclui info da normalização
    nome = 'Matrizes_dissimilaridade'
    caminho_salvar = os.path.join(pasta_destino, nome)
    plt.savefig(caminho_salvar, dpi=300)
    plt.show()

# Uso (exemplo)
if __name__ == "__main__":
    fazer_matriz_dissimilaridade(df, caminho_figuras)


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist

def cluster_hierarquico_com_media(df, k=4, caminho_figuras = None, caminho_dataframes = None, plotar = None):
    """
    Faz cluster hierárquico das colunas do df sem a coluna 'Q',
    plota dendrograma colorido conforme corte em k clusters,
    e retorna DataFrame com médias dos clusters.
    
    Parâmetros:
    - df: DataFrame com as séries, índice datetime.
    - k: número de clusters para cortar o dendrograma.
    - plotar: se True, plota dendrograma.
    
    Retorna:
    - df_clusters: DataFrame com índice igual ao df original e colunas C1, C2, ..., Ck (média dos clusters)
    """

    pasta_destino = os.path.join(caminho_figuras, 'clusters_hierarquicos')
    os.makedirs(pasta_destino, exist_ok=True)


    # Remove coluna Q se existir
    if 'Q' in df.columns:
        df_sem_q = df.drop(columns=['Q'])
    else:
        df_sem_q = df.copy()
    
    # Transpõe para ter colunas como observações
    dados = df_sem_q.T.values
    
    # Calcula a matriz de distâncias (vetor triangular)
    dist_vec = pdist(dados, metric='euclidean')
    
    # Faz linkage (método ward)
    Z = linkage(dist_vec, method='ward')
    
    # Define clusters pelo corte em k grupos
    clusters = fcluster(Z, k, criterion='maxclust')
    
    if plotar:
        # Para ajustar o color_threshold, pegamos o último merge que gera k clusters:
        # Os valores de distância (Z[:, 2]) em ordem crescente de merge.
        # Queremos o valor mínimo que gera exatamente k clusters.
        # Usamos uma abordagem prática:
        distances = Z[:, 2]
        
        # Distância de corte = menor distância para ter k clusters
        # Encontrar o índice que corresponde a n-k (porque são n-1 merges)
        # Total merges = len(distances), número de clusters iniciais = n = len(df_sem_q.columns)
        n = len(df_sem_q.columns)
        cutoff_idx = n - k - 1
        if cutoff_idx < 0:
            cutoff_idx = 0
        color_threshold = distances[cutoff_idx] if cutoff_idx < len(distances) else distances[-1]
        
        plt.figure(figsize=(14, 10))
        dendrogram(Z, labels=df_sem_q.columns, leaf_rotation=90, color_threshold=color_threshold)
        plt.title(f'Dendrograma - Cluster Hierárquico das Séries (sem Q), k={k}')
        plt.xlabel('Postos', fontsize = 20)
        plt.ylabel('Distância Euclidiana', fontsize = 20)
        plt.axhline(y=color_threshold, color='r', linestyle='--', lw=1)  # linha do corte
        plt.xticks(fontsize=14)
        plt.tight_layout()
        # Salva a figura
        nome_arquivo = 'Clusters_Hierarquicos.png'
        caminho_salvar = os.path.join(pasta_destino, nome_arquivo)
        plt.savefig(caminho_salvar, dpi=300)
        plt.show()
        plt.show()
    
    # Criar DataFrame para guardar médias dos clusters
    df_clusters = pd.DataFrame(index=df.index)
    
    # Para cada cluster, calcula média das colunas que pertencem a ele
    for cluster_num in range(1, k+1):
        colunas_do_cluster = df_sem_q.columns[clusters == cluster_num]
        df_clusters[f'C{cluster_num}'] = df_sem_q[colunas_do_cluster].mean(axis=1)
    # Exibir mapeamento
    print("\nColunas por cluster:")
    for cluster_num in range(1, k + 1):
        cols = df_sem_q.columns[clusters == cluster_num]
        print(f"Cluster {cluster_num}: {list(cols)}")

      # --- Salvar DataFrames em Excel ---
    if 'Q' in df.columns:
        df_clusters['Q'] = df['Q']
        df_clusters['Q'] = df['Q']
    caminho_pasta_salvar_df = caminho_dataframes
    pasta_salvar = os.path.join(caminho_pasta_salvar_df, 'BASES PARA MODELAGEM')
    os.makedirs(pasta_salvar, exist_ok=True)

    caminho_pca = os.path.join(pasta_salvar, 'df_clusters.xlsx')

    df_clusters.to_excel(caminho_pca)
    
    return df_clusters

if __name__ == "__main__":
    # Exemplo de uso:
    df_clusters = cluster_hierarquico_com_media(df, k=10, caminho_figuras = caminho_figuras, caminho_dataframes = caminho_dataframes,  plotar=True )
    df_clusters.head()


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# def fazer_pca(df, caminho_figuras = None, caminho_dataframes = None, n_components = None):
#     """
#     Aplica PCA ao DataFrame df (excluindo a coluna 'Q', se existir),
#     plota variância explicada, gráficos 2D e 3D dos principais componentes,
#     e retorna um novo DataFrame df_pca com os componentes.

#     Parâmetros:
#     - df: DataFrame original com séries temporais (índice datetime).
#     - n_components: número de componentes a manter (default: todas).

#     Retorna:
#     - df_pca: DataFrame com os componentes principais e mesmo índice do df original.
#     """

#     pasta_destino = os.path.join(caminho_figuras, 'PCA')
#     os.makedirs(pasta_destino, exist_ok=True)

#     # Remove 'Q' se existir
#     if 'Q' in df.columns:
#         df_pca_base = df.drop(columns=['Q'])
#     else:
#         df_pca_base = df.copy()

#     # Padronização (média 0, desvio padrão 1)
#     scaler = StandardScaler()
#     dados_padronizados = scaler.fit_transform(df_pca_base)

#     # PCA
#     pca = PCA(n_components=n_components)
#     componentes = pca.fit_transform(dados_padronizados)

#     # Criação do DataFrame dos componentes
#     nomes_componentes = [f'PC{i+1}' for i in range(componentes.shape[1])]
#     df_pca = pd.DataFrame(componentes, columns=nomes_componentes, index=df.index)

    
#     var_exp = pca.explained_variance_ratio_ * 100
#     componentes_validos = np.where(var_exp > 3.5)[0]

#     var_exp_filtrada = var_exp[componentes_validos]
#     componentes_ids = [f'PC{i+1}' for i in componentes_validos]
#     qnt_componentes = len(componentes_ids)
#     nomes_variaveis = df_pca_base.columns
#     df_pca_filtrado = df_pca[componentes_ids].copy()

#     # 1. Plotar variância explicada
#     plt.figure(figsize=(10, 4))
#     bars = plt.bar(componentes_ids, var_exp_filtrada, color='skyblue')

#     # Adicionar texto dentro das barras
#     for i, bar in enumerate(bars):
#         altura = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width() / 2, altura * 0.5, f'{altura:.1f}%', 
#                 ha='center', va='center', fontsize=9, color='black')

#     # Linha da soma acumulada
#     plt.plot(componentes_ids, np.cumsum(var_exp_filtrada), marker='o', color='black', linestyle='--', label='Acumulada')

#     plt.xlabel('Componente Principal')
#     plt.ylabel('Variância Explicada (%)')
#     plt.title('Variância Explicada por Componente (> 2.5%)')
#     plt.grid(True, axis='y', linestyle='--', alpha=0.6)
#     plt.legend()
#     plt.tight_layout()
#     nome_arquivo = 'PCA_VARIANCIA.png'
#     caminho_salvar = os.path.join(pasta_destino, nome_arquivo)
#     plt.savefig(caminho_salvar, dpi = 300)
#     plt.show()

#     # 2. Plotar PC1 vs PC2
#     if componentes.shape[1] >= 2:
#         plt.figure(figsize=(7, 6))
#         plt.scatter(df_pca['PC1'], df_pca['PC2'], alpha=0.6, c='darkblue')
#         plt.xlabel('PC1')
#         plt.ylabel('PC2')
#         plt.title('PCA: PC1 x PC2')
#         plt.grid(True)
#         plt.tight_layout()
#         nome_arquivo = 'PC1_X_PC2.png'
#         caminho_salvar = os.path.join(pasta_destino, nome_arquivo)
#         plt.savefig(caminho_salvar, dpi = 300)
#         plt.show()

#     # 3. Plotar PC1 vs PC2 vs PC3 (3D)
#     if componentes.shape[1] >= 3:
#         fig = plt.figure(figsize=(8, 6))
#         ax = fig.add_subplot(111, projection='3d')
#         ax.scatter(df_pca['PC1'], df_pca['PC2'], df_pca['PC3'], alpha=0.5, c='green')
#         ax.set_xlabel('PC1')
#         ax.set_ylabel('PC2')
#         ax.set_zlabel('PC3')
#         ax.set_title('PCA: PC1 x PC2 x PC3')
#         plt.tight_layout()
#         plt.show()

#     # 4. Plotar gráfico Pn x Pn (loadings) para PC1 e PC2 com cores distintas
#     # 4. Plotar gráfico Pn x Pn (loadings) para PC1 e PC2 com legenda
#     if componentes.shape[1] >= 2:
#         loadings = pca.components_.T
#         #nomes_variaveis = df_pca_base.columns

#         # Criar mapa de cores
#         cmap = plt.get_cmap('tab20')
#         cores = [cmap(i % 20) for i in range(len(nomes_variaveis))]

#         plt.figure(figsize=(8, 6))
#         plt.axhline(0, color='grey', lw=1)
#         plt.axvline(0, color='grey', lw=1)

#         for i, nome in enumerate(nomes_variaveis):
#             plt.scatter(loadings[i, 0], loadings[i, 1], color=cores[i], label=nome)

#         plt.xlabel('PC1')
#         plt.ylabel('PC2')
#         plt.title('Contribuição dos Postos (Pn) - Loadings PC1 x PC2')
#         plt.grid(True)

#         # Agrupar a legenda fora do gráfico, se tiver muitos pontos
#         plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Postos', fontsize=9)
#         plt.tight_layout()
#         nome_arquivo = 'Postos_PC1_X_PC2'
#         caminho_salvar = os.path.join(pasta_destino, nome_arquivo)
#         plt.savefig(caminho_salvar, dpi = 300)
#         plt.show()

#     # 5. Plotar contribuição das variáveis para cada componente significativo
#     if len(componentes_validos) > 0:
#         loadings = pca.components_.T  # shape: (n_variáveis, n_componentes)
#         #nomes_variaveis = df_pca_base.columns

#         for i in componentes_validos:
#             loading_pc = loadings[:, i]
#             pc_label = f'PC{i+1}'
            
#             plt.figure(figsize=(10, 6))
#             bars = plt.barh(nomes_variaveis, loading_pc, color='steelblue')
#             plt.title(f'Contribuição das Variáveis para {pc_label}')
#             plt.xlabel('Peso (loading)')
#             plt.grid(True, axis='x', linestyle='--', alpha=0.6)

#             # Adicionar valores nos extremos das barras
#             for bar in bars:
#                 largura = bar.get_width()
#                 plt.text(largura + 0.01*np.sign(largura), bar.get_y() + bar.get_height()/2,
#                          f'{largura:.2f}', va='center', ha='left' if largura > 0 else 'right', fontsize=8)

#             plt.tight_layout()
#             nome_arquivo = f'Contribuicao_Postos_Para_PC{i+1}.png'
#             caminho_salvar = os.path.join(pasta_destino, nome_arquivo)
#             plt.savefig(caminho_salvar, dpi = 300)
#             plt.show()
    
#     # Printar os principais postos para cada PC
#         print("\n===== Principais postos por Componente Principal =====")
#         for i in componentes_validos:
#             pc_label = f'PC{i+1}'
#             loading_pc = loadings[:, i]
#             abs_loading_pc = np.abs(loading_pc)
#             idx_sorted = np.argsort(abs_loading_pc)[::-1]

#             print(f"\n{pc_label} - Top 5 postos:")
#             for rank in range(min(5, len(nomes_variaveis))):
#                 idx_var = idx_sorted[rank]
#                 sinal = '+' if loading_pc[idx_var] > 0 else '' #tirei "-" para não ficar "--", já que o float já retorna um negativo nesses casos
#                 print(f"  {rank+1}. {nomes_variaveis[idx_var]}: {sinal}{loading_pc[idx_var]:.3f}")
    
#       # --- Salvar DataFrames em Excel ---
#     if 'Q' in df.columns:
#         df_pca['Q'] = df['Q']
#         df_pca_filtrado['Q'] = df['Q']
#     caminho_salvar_df = caminho_dataframes
#     pasta_salvar = os.path.join(caminho_salvar_df,'BASES PARA MODELAGEM')
#     os.makedirs(pasta_salvar, exist_ok=True)

#     caminho_pca = os.path.join(pasta_salvar, 'df_pca.xlsx')
#     caminho_pca_filtrado = os.path.join(pasta_salvar, 'df_pca_filtrado.xlsx')

#     df_pca.to_excel(caminho_pca)
#     df_pca_filtrado.to_excel(caminho_pca_filtrado)

#     return df_pca, df_pca_filtrado

# if __name__ == "__main__":
#     df_pca, df_pca_filtrado = fazer_pca(df, caminho_figuras, caminho_dataframes)

#     print(df_pca_filtrado)

def _executar_analise_pca(df_para_analise, df_original_index, q_para_anexar, 
                          caminho_figuras, caminho_dataframes, n_components, sufixo_analise):
    """
    Função auxiliar interna que executa a lógica principal do PCA.
    Não deve ser chamada diretamente.
    """
    
    print(f"\n===== INICIANDO ANÁLISE PCA ({sufixo_analise.upper().replace('_', '')}) =====")

    # 1. Preparar caminhos de saída
    pasta_destino_fig = os.path.join(caminho_figuras, f'PCA{sufixo_analise}')
    os.makedirs(pasta_destino_fig, exist_ok=True)
    
    pasta_destino_df = os.path.join(caminho_dataframes, 'BASES PARA MODELAGEM')
    os.makedirs(pasta_destino_df, exist_ok=True)

    df_pca_base = df_para_analise.copy()

    # 2. Padronização (média 0, desvio padrão 1)
    scaler = StandardScaler()
    dados_padronizados = scaler.fit_transform(df_pca_base)

    # 3. PCA
    pca = PCA(n_components=n_components)
    componentes = pca.fit_transform(dados_padronizados)

    # 4. Criação do DataFrame dos componentes
    nomes_componentes = [f'PC{i+1}' for i in range(componentes.shape[1])]
    df_pca = pd.DataFrame(componentes, columns=nomes_componentes, index=df_original_index)

    # 5. Filtragem de componentes
    var_exp = pca.explained_variance_ratio_ * 100
    # NOTA: O limite de 3.5% pode ser muito alto quando 'Q' está incluído.
    # Considere ajustar ou remover se nenhum componente passar.
    componentes_validos = np.where(var_exp > 3.5)[0] 
    
    if len(componentes_validos) == 0:
        print(f"AVISO: Nenhum componente explicou mais de 3.5% da variância na análise {sufixo_analise}.")
        # Se nenhum passar, pegar pelo menos os 2 primeiros para os gráficos
        componentes_validos = np.arange(min(2, len(var_exp)))
        if len(componentes_validos) == 0:
             print("ERRO: PCA não produziu componentes.")
             return pd.DataFrame(index=df_original_index), pd.DataFrame(index=df_original_index)

    var_exp_filtrada = var_exp[componentes_validos]
    componentes_ids = [f'PC{i+1}' for i in componentes_validos]
    nomes_variaveis = df_pca_base.columns
    df_pca_filtrado = df_pca[componentes_ids].copy()

    # --- Início dos Plots ---

    # 1. Plotar variância explicada
    plt.figure(figsize=(10, 4))
    bars = plt.bar(componentes_ids, var_exp_filtrada, color='skyblue')
    for i, bar in enumerate(bars):
        altura = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, altura * 0.5, f'{altura:.1f}%', 
                 ha='center', va='center', fontsize=9, color='black')
    plt.plot(componentes_ids, np.cumsum(var_exp_filtrada), marker='o', color='black', linestyle='--', label='Acumulada')
    plt.xlabel('Componente Principal')
    plt.ylabel('Variância Explicada (%)')
    plt.title(f'Variância Explicada ({sufixo_analise}) - Componentes > 3.5%')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    nome_arquivo = 'PCA_VARIANCIA.png'
    caminho_salvar = os.path.join(pasta_destino_fig, nome_arquivo)
    plt.savefig(caminho_salvar, dpi=300)
    plt.show()

    # 2. Plotar PC1 vs PC2
    if componentes.shape[1] >= 2:
        plt.figure(figsize=(7, 6))
        plt.scatter(df_pca['PC1'], df_pca['PC2'], alpha=0.6, c='darkblue')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(f'PCA: PC1 x PC2 ({sufixo_analise})')
        plt.grid(True)
        plt.tight_layout()
        nome_arquivo = 'PC1_X_PC2.png'
        caminho_salvar = os.path.join(pasta_destino_fig, nome_arquivo)
        plt.savefig(caminho_salvar, dpi=300)
        plt.show()

    # 3. Plotar PC1 vs PC2 vs PC3 (3D)
    if componentes.shape[1] >= 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df_pca['PC1'], df_pca['PC2'], df_pca['PC3'], alpha=0.5, c='green')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title(f'PCA: PC1 x PC2 x PC3 ({sufixo_analise})')
        plt.tight_layout()
        nome_arquivo = 'PC1_X_PC2_X_PC3.png' # Nome de arquivo adicionado
        caminho_salvar = os.path.join(pasta_destino_fig, nome_arquivo)
        plt.savefig(caminho_salvar, dpi=300)
        plt.show()

    # 4. Plotar gráfico Pn x Pn (loadings) para PC1 e PC2
    if componentes.shape[1] >= 2:
        loadings = pca.components_.T
        cmap = plt.get_cmap('tab20')
        cores = [cmap(i % 20) for i in range(len(nomes_variaveis))]

        plt.figure(figsize=(8, 6))
        plt.axhline(0, color='grey', lw=1)
        plt.axvline(0, color='grey', lw=1)

        for i, nome in enumerate(nomes_variaveis):
            plt.scatter(loadings[i, 0], loadings[i, 1], color=cores[i], label=nome)

        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title(f'Contribuição das Variáveis (Loadings) PC1 x PC2 ({sufixo_analise})')
        plt.grid(True)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Variáveis', fontsize=9)
        plt.tight_layout()
        nome_arquivo = 'Variaveis_Loadings_PC1_X_PC2.png'
        caminho_salvar = os.path.join(pasta_destino_fig, nome_arquivo)
        plt.savefig(caminho_salvar, dpi=300)
        plt.show()

    # 5. Plotar contribuição das variáveis para cada componente significativo
    if len(componentes_validos) > 0:
        loadings = pca.components_.T 
        for i in componentes_validos:
            loading_pc = loadings[:, i]
            pc_label = f'PC{i+1}'
            
            # Ordenar para melhor visualização
            idx_sorted = np.argsort(loading_pc)
            nomes_ordenados = nomes_variaveis[idx_sorted]
            loading_ordenado = loading_pc[idx_sorted]
            
            plt.figure(figsize=(10, max(6, len(nomes_ordenados) * 0.4))) # Ajuste dinâmico da altura
            bars = plt.barh(nomes_ordenados, loading_ordenado, color='steelblue')
            plt.title(f'Contribuição das Variáveis para {pc_label} ({sufixo_analise})')
            plt.xlabel('Peso (loading)')
            plt.grid(True, axis='x', linestyle='--', alpha=0.6)

            for bar in bars:
                largura = bar.get_width()
                ha_val = 'left' if largura > 0 else 'right'
                x_pos = largura + 0.01 * np.sign(largura) if largura != 0 else 0.01
                plt.text(x_pos, bar.get_y() + bar.get_height()/2,
                         f'{largura:.2f}', va='center', ha=ha_val, fontsize=8)

            plt.tight_layout()
            nome_arquivo = f'Contribuicao_Variaveis_Para_PC{i+1}.png'
            caminho_salvar = os.path.join(pasta_destino_fig, nome_arquivo)
            plt.savefig(caminho_salvar, dpi=300)
            plt.show()
    
    # --- Fim dos Plots ---

    # 6. Printar os principais postos para cada PC
    print(f"\n===== Principais Variáveis por Componente ({sufixo_analise}) =====")
    for i in componentes_validos:
        pc_label = f'PC{i+1}'
        loading_pc = loadings[:, i]
        abs_loading_pc = np.abs(loading_pc)
        idx_sorted = np.argsort(abs_loading_pc)[::-1]

        print(f"\n{pc_label} - Top 5 Variáveis (Absoluto):")
        for rank in range(min(5, len(nomes_variaveis))):
            idx_var = idx_sorted[rank]
            sinal = '+' if loading_pc[idx_var] > 0 else ''
            print(f"  {rank+1}. {nomes_variaveis[idx_var]}: {sinal}{loading_pc[idx_var]:.3f}")
    
    # 7. Salvar DataFrames em Excel
    
    # Anexa 'Q' de volta APENAS se esta for a análise "sem Q"
    if q_para_anexar is not None:
        df_pca['Q'] = q_para_anexar
        df_pca_filtrado['Q'] = q_para_anexar

    caminho_pca = os.path.join(pasta_destino_df, f'df_pca{sufixo_analise}.xlsx')
    caminho_pca_filtrado = os.path.join(pasta_destino_df, f'df_pca_filtrado{sufixo_analise}.xlsx')

    df_pca.to_excel(caminho_pca)
    df_pca_filtrado.to_excel(caminho_pca_filtrado)

    print(f"===== ANÁLISE PCA ({sufixo_analise.upper().replace('_', '')}) CONCLUÍDA =====")

    return df_pca, df_pca_filtrado

def fazer_pca(df, caminho_figuras = None, caminho_dataframes = None, n_components = None):
    """
    Aplica PCA ao DataFrame df de duas maneiras:
    1. Excluindo a coluna 'Q' (para análise de preditores).
    2. Incluindo a coluna 'Q' (para análise do sistema completo).
    
    Salva todos os gráficos e DataFrames para ambas as análises com
    sufixos '_sem_Q' e '_com_Q'.

    Parâmetros:
    - df: DataFrame original com séries temporais (índice datetime).
    - caminho_figuras: Pasta raiz para salvar as subpastas de figuras.
    - caminho_dataframes: Pasta raiz para salvar os arquivos .xlsx.
    - n_components: número de componentes a manter (default: todas).

    Retorna:
    - (df_pca_sem_q, df_pca_filtrado_sem_q): Tupla contendo os DataFrames
      da análise SEM 'Q', mantendo a compatibilidade com o código anterior.
    """

    # --- Análise 1: SEM 'Q' ---
    # Esta é a análise original, boa para Regressão de Componentes Principais
    
    q_original = df.get('Q') # Pega 'Q' se existir, senão retorna None
    
    if q_original is not None:
        print("--- INICIANDO ANÁLISE 1: PCA sobre os postos (SEM 'Q') ---")
        df_sem_q = df.drop(columns=['Q'])
    else:
        print("--- INICIANDO ANÁLISE 1: PCA (coluna 'Q' não encontrada) ---")
        df_sem_q = df.copy()

    df_pca_sem_q, df_pca_filtrado_sem_q = _executar_analise_pca(
        df_para_analise=df_sem_q,
        df_original_index=df.index,
        q_para_anexar=q_original, # Passa 'Q' para ser anexado no final
        caminho_figuras=caminho_figuras,
        caminho_dataframes=caminho_dataframes,
        n_components=n_components,
        sufixo_analise='_sem_Q'
    )

    # --- Análise 2: COM 'Q' ---
    # Esta análise explora a variabilidade do sistema completo
    
    if q_original is not None:
        print("\n\n--- INICIANDO ANÁLISE 2: PCA sobre o sistema completo (COM 'Q') ---")
        df_com_q = df.copy() # Usa o DataFrame inteiro
        
        # Chamamos a análise, mas não precisamos capturar o retorno
        # (a menos que você queira usá-lo)
        _executar_analise_pca(
            df_para_analise=df_com_q,
            df_original_index=df.index,
            q_para_anexar=None, # 'Q' já está na análise, não anexar no final
            caminho_figuras=caminho_figuras,
            caminho_dataframes=caminho_dataframes,
            n_components=n_components,
            sufixo_analise='_com_Q'
        )
    else:
        print("\n--- ANÁLISE 2 (COM 'Q') PULADA: Coluna 'Q' não encontrada. ---")

    # Retorna os resultados da análise "sem Q" para manter a 
    # compatibilidade com seu código principal
    return df_pca_sem_q, df_pca_filtrado_sem_q

