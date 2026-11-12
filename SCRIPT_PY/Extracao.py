# %%
def extrair_ANA(
    codEstDE=None,
    codEstATE=None,
    tpEst=None,
    nmEst=None,
    nmRio=None,
    codSubBacia=None,
    codBacia=None,
    nmMunicipio=None,
    nmEstado=None,
    sgResp=None,
    sgOper=None,
    telemetrica=None
):
    from zeep import Client
    from zeep.helpers import serialize_object
    wsdl_url = "http://telemetriaws1.ana.gov.br/ServiceANA.asmx?wsdl"
    client = Client(wsdl=wsdl_url)

    parametros = {
        'codEstDE': codEstDE or '',
        'codEstATE': codEstATE or '',
        'tpEst': tpEst or '',
        'nmEst': nmEst or '',
        'nmRio': nmRio or '',
        'codSubBacia': codSubBacia or '',
        'codBacia': codBacia or '',
        'nmMunicipio': nmMunicipio or '',
        'nmEstado': nmEstado or '',
        'sgResp': sgResp or '',
        'sgOper': sgOper or '',
        'telemetrica': telemetrica or ''
    }

    try:
        resposta = client.service.HidroInventario(**parametros)
        estacoes = serialize_object(resposta)

        # Garante que o retorno seja uma lista
        resultado = estacoes.get('_value_1', [])
        if isinstance(resultado, dict):
            return [resultado]
        return resultado

    except Exception as e:
        print("Erro ao acessar o serviço:", e)
        return []


# %%
# dados = extrair_ANA(tpEst='1', nmEstado='Rio de Janeiro')

# print(f"Total estações: {len(dados)}")

# from pprint import pprint
# pprint(dados[0])


# %%


def extrair_bacia(
    codBacia=None,
    codSubBacia=None
):
    from zeep import Client
    from zeep.helpers import serialize_object
    
    wsdl_url = "http://telemetriaws1.ana.gov.br/ServiceANA.asmx?wsdl"
    client = Client(wsdl=wsdl_url)

    parametros = {
        'codBacia': codBacia or '',
        'codSubBacia': codSubBacia or ''
    }

    try:
        resposta = client.service.HidroBaciaSubBacia(**parametros)
        dados = serialize_object(resposta)

        resultado = dados.get('_value_1', [])
        if isinstance(resultado, dict):
            return [resultado]
        return resultado

    except Exception as e:
        print("Erro ao acessar o serviço HidroBaciaSubBacia:", e)
        return []


# %%
# # Exemplo de uso:
# if __name__ == "__main__":
#     resultado = extrair_bacia()
#     pprint(resultado)

# %%


def gerar_df_hidro(
    # parâmetros para extrair_ANA
    codEstDE=None,
    codEstATE=None,
    tpEst=None,
    nmEst=None,
    nmRio=None,
    codSubBacia=None,
    codBacia=None,
    nmMunicipio=None,
    nmEstado=None,
    sgResp=None,
    sgOper=None,
    telemetrica=None,
   
):
    import pandas as pd
    # Chama extrair_ANA
    dados_ana_raw = extrair_ANA(
        codEstDE=codEstDE,
        codEstATE=codEstATE,
        tpEst=tpEst,
        nmEst=nmEst,
        nmRio=nmRio,
        codSubBacia=codSubBacia,
        codBacia=codBacia,
        nmMunicipio=nmMunicipio,
        nmEstado=nmEstado,
        sgResp=sgResp,
        sgOper=sgOper,
        telemetrica=telemetrica
    )
    
    lista_ana = []
    for entry in dados_ana_raw:
        blocos = entry.get('_value_1', [])
        for bloco in blocos:
            table = bloco.get('Table')
            if table:
                lista_ana.append(table)
    
    df_ana = pd.DataFrame(lista_ana)
    
    # Chama extrair_bacia
    dados_bacia_raw = extrair_bacia(
        codBacia=codBacia,
        codSubBacia=codSubBacia
    )
    
    lista_bacia = []
    if dados_bacia_raw and isinstance(dados_bacia_raw, list):
        primeiro_item = dados_bacia_raw[0]
        sublistas = primeiro_item.get('_value_1', [])
        for subitem in sublistas:
            table = subitem.get('Table')
            if table:
                lista_bacia.append(table)
    
    df_bacia = pd.DataFrame(lista_bacia)
    
    # Padroniza nome da coluna para merge
    if 'SubBaciaCodigo' in df_ana.columns:
        df_ana_renamed = df_ana.rename(columns={'SubBaciaCodigo': 'SubBaciaCodigo'})
    else:
        df_ana_renamed = df_ana.copy()

    if 'codSubBacia' in df_bacia.columns:
        df_bacia_renamed = df_bacia.rename(columns={'codSubBacia': 'SubBaciaCodigo'})
    else:
        df_bacia_renamed = df_bacia.copy()

    # Corrige tipos para merge (converte para string)
    df_ana_renamed['SubBaciaCodigo'] = df_ana_renamed['SubBaciaCodigo'].astype(str)
    df_bacia_renamed['SubBaciaCodigo'] = df_bacia_renamed['SubBaciaCodigo'].astype(str)

    # Colunas para adicionar do df_bacia (excluindo as que já tem no df_ana)
    cols_para_adicionar = [col for col in df_bacia_renamed.columns if col not in df_ana_renamed.columns and col != 'SubBaciaCodigo']

    # Faz a mesclagem (merge) só trazendo as colunas novas do df_bacia
    df_merge = df_ana_renamed.merge(
        df_bacia_renamed[['SubBaciaCodigo'] + cols_para_adicionar],
        on='SubBaciaCodigo',
        how='left'
    )
    
    return df_merge


# %%
if __name__ == "__main__":

    df_merge = gerar_df_hidro(codSubBacia = 58,  telemetrica = 0)

    from tabulate import tabulate

    # print("\n=== DataFrame extrair_ANA completo ===")
    # print(tabulate(df_ana, headers='keys', tablefmt='fancy_grid', showindex=False))

    # print("\n=== DataFrame extrair_bacia completo ===")
    # print(tabulate(df_bacia, headers='keys', tablefmt='fancy_grid', showindex=False))

    print("\n=== DataFrame mesclado completo ===")
    print(tabulate(df_merge, headers='keys', tablefmt='fancy_grid', showindex=False))


# %% [markdown]
# # Vazão

# %%
def extrair_vazao_ana(cod_estacao, data_inicio="", data_fim="", nivel_consistencia=[1, 2]):
    import requests
    import xml.etree.ElementTree as ET
    import pandas as pd

    def fazer_requisicao(nivel):
        url = "http://telemetriaws1.ana.gov.br/ServiceANA.asmx"

        soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
        <soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
                       xmlns:xsd="http://www.w3.org/2001/XMLSchema" 
                       xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
          <soap:Body>
            <HidroSerieHistorica xmlns="http://MRCS/">
              <codEstacao>{cod_estacao}</codEstacao>
              <dataInicio>{data_inicio}</dataInicio>
              <dataFim>{data_fim}</dataFim>
              <tipoDados>3</tipoDados>
              <nivelConsistencia>{nivel}</nivelConsistencia>
            </HidroSerieHistorica>
          </soap:Body>
        </soap:Envelope>"""

        headers = {
            "Content-Type": "text/xml; charset=utf-8",
            "SOAPAction": "http://MRCS/HidroSerieHistorica"
        }

        try:
            response = requests.post(url, data=soap_body, headers=headers, timeout=30)
            if response.status_code == 200:
                return response.text
            else:
                print(f"Erro HTTP {response.status_code} na requisição (nível {nivel}).")
                return None
        except Exception as e:
            print(f"Erro ao requisitar dados para nível {nivel}: {e}")
            return None

    def xml_para_dataframe(xml_string):
        ns = {
            'soap': 'http://schemas.xmlsoap.org/soap/envelope/',
            'diffgr': 'urn:schemas-microsoft-com:xml-diffgram-v1'
        }

        try:
            root = ET.fromstring(xml_string)
            diffgram = root.find('.//diffgr:diffgram', ns)
            if diffgram is None:
                return pd.DataFrame()

            document_element = diffgram.find('DocumentElement')
            if document_element is None:
                return pd.DataFrame()

            records = []
            for record in document_element:
                rec_dict = {}
                for field in record:
                    rec_dict[field.tag] = field.text
                records.append(rec_dict)

            return pd.DataFrame(records)
        except Exception as e:
            print(f"Erro ao processar XML: {e}")
            return pd.DataFrame()

    if isinstance(nivel_consistencia, int):
        nivel_consistencia = [nivel_consistencia]

    todos_dfs = []
    for nivel in nivel_consistencia:
        xml = fazer_requisicao(nivel)
        if xml:
            df = xml_para_dataframe(xml)
            if not df.empty:
                # Verifica se a coluna existe antes de filtrar
                if "NivelConsistencia" in df.columns:
                    df_filtrado = df[df["NivelConsistencia"] == str(nivel)]
                    if not df_filtrado.empty:
                        todos_dfs.append(df_filtrado)
                else:
                    print(f"Nenhuma coluna 'NivelConsistencia' nos dados retornados da estação {cod_estacao} (nível {nivel}).")
            else:
                print(f"Sem dados retornados para estação {cod_estacao} (nível {nivel}).")
        else:
            print(f"Requisição falhou para estação {cod_estacao} (nível {nivel}).")

    if todos_dfs:
        return pd.concat(todos_dfs, ignore_index=True)
    else:
        #print(f"❌ Não existem dados para a estação {cod_estacao}.")
        return pd.DataFrame()


# %%
# def extrair_vazao_ana_multiplas(df_estacoes, data_inicio="", data_fim="", nivel_consistencia=[1, 2]):
#     """
#     Recebe DataFrame com colunas 'Codigo', 'TipoEstacao' e 'TipoEstacaoTelemetrica'.
#     Filtra por TipoEstacao == 1 e TipoEstacaoTelemetrica == 0.
#     Itera pelos códigos filtrados, chama extrair_vazao_ana para cada código e retorna dict {codigo: df}.
#     """
#     resultados = {}

#     # Filtrar as estações válidas
#     df_filtrado = df_merge[(df_merge['TipoEstacao'] == '1') & (df_merge['TipoEstacaoTelemetrica'] == '0')]

#     codigos = df_filtrado['Codigo'].unique()

#     for codigo in codigos:
#         print(f"Extraindo dados para a estação {codigo}...")
#         df = extrair_vazao_ana(codigo, nivel_consistencia=nivel_consistencia)
#         print(tabulate(df.head(), headers='keys', tablefmt='grid'))
#         resultados[codigo] = df

#     return resultados
# resultados = extrair_vazao_ana_multiplas(df_merge,  nivel_consistencia=[2])

# %%
def extrair_vazao_ana_multiplas(df_estacoes, data_inicio="", data_fim="", nivel_consistencia=[1, 2]):
    """
    Recebe DataFrame com colunas 'Codigo', 'TipoEstacao' e 'TipoEstacaoTelemetrica'.
    Filtra por TipoEstacao == 1 e TipoEstacaoTelemetrica == 0.
    Itera pelos códigos filtrados, chama extrair_vazao_ana para cada código,
    transforma para formato diário e retorna dict {codigo: df}.
    """
    import pandas as pd
    from tabulate import tabulate

    def transformar_vazoes(df):
        # Garante que a coluna DataHora está em datetime
        df['DataHora'] = pd.to_datetime(df['DataHora'])

        # Extrai ano e mês para montagem da data depois
        df['AnoMes'] = df['DataHora'].dt.strftime('%m/%Y')

        # Colunas de vazão e status
        col_vazao = [f"Vazao{str(i).zfill(2)}" for i in range(1, 32)]
        #col_status = [f"Vazao{str(i).zfill(2)}Status" for i in range(1, 32)]

        # Melt para transformar colunas em linhas
        df_vazao = df.melt(
            id_vars=['EstacaoCodigo', 'NivelConsistencia','AnoMes'],
            value_vars=col_vazao,
            var_name='Dia',
            value_name='Vazao'
        )
        # df_status = df.melt(
        #     id_vars=['EstacaoCodigo', 'AnoMes'],
        #     value_vars=col_status,
        #     var_name='Dia',
        #     value_name='StatusVazao'
        # )

        # Extrai o número do dia das colunas tipo 'Vazao01'
        df_vazao['Dia'] = df_vazao['Dia'].str.extract(r'(\d+)').astype(int)
        #df_status['Dia'] = df_status['Dia'].str.extract(r'(\d+)').astype(int)

        # # Junta as duas tabelas
        # df_mergeado = pd.merge(df_vazao, df_status,
        #                        on=['EstacaoCodigo', 'AnoMes', 'Dia'],
        #                        how='left')

        # Monta a data completa
        df_vazao['Data'] = df_vazao['Dia'].astype(str).str.zfill(2) + '/' + df_vazao['AnoMes']
        df_vazao['Data'] = pd.to_datetime(df_vazao['Data'], format='%d/%m/%Y', errors='coerce')

        # Remove datas inválidas
        df_vazao = df_vazao.dropna(subset=['Data'])

        # Organiza e retorna
        df_final = df_vazao[['EstacaoCodigo', 'NivelConsistencia','Data', 'Vazao']].sort_values(by=['EstacaoCodigo', 'Data']).reset_index(drop=True)

        return df_final

    resultados = {}

    # Filtrar estações válidas
    df_filtrado = df_estacoes[(df_estacoes['TipoEstacao'] == '1') & (df_estacoes['TipoEstacaoTelemetrica'] == '0')]
    codigos = df_filtrado['Codigo'].unique()

    for codigo in codigos:
        print(f"Extraindo dados para a estação {codigo}...")
        df = extrair_vazao_ana(codigo, nivel_consistencia=nivel_consistencia)

        if not df.empty:
            df = transformar_vazoes(df)
            print(f'✅ Dados extraidos com sucesso para {codigo}')
            #print(tabulate(df, headers='keys', tablefmt='grid'))
            resultados[codigo] = df
        else:
            print(f"❌ Sem dados para {codigo}. Nada será adicionado.")

    return resultados
if __name__ == "__main__":
    # Chamada
    resultados_vazao = extrair_vazao_ana_multiplas(df_merge, nivel_consistencia=[1,2])


# %%
if __name__ == "__main__":
    resultados_vazao

# %%


def plotar_quantidade_dados_por_dia(resultados):
    """
    Plota um gráfico interativo com:
    - Quantidade de dados por dia com NivelConsistencia = 1 e = 2
    - Quantidade de valores vazios por dia (linha vermelha)
    - Eixos Y duplos e título com número de estações
    """
    
    import pandas as pd
    import plotly.graph_objects as go

    # Combina todos os DataFrames do dicionário
    df_tudo = pd.concat(resultados.values(), ignore_index=True)

    # # Exemplo da lista de estações que você quer manter:
    # estacoes_selecionadas = ['58033000', '58030000', '58035000']

    # # Filtra o DataFrame mantendo só as linhas cujo EstacaoCodigo está na lista
    # df_tudo = df_tudo[df_tudo['EstacaoCodigo'].astype(str).isin(estacoes_selecionadas)]

    # Garante que a coluna 'Data' esteja no tipo datetime
    df_tudo['Data'] = pd.to_datetime(df_tudo['Data'])

    # Cria coluna para marcar vazios
    df_tudo['Vazio'] = df_tudo['Vazao'].isna()

    # Agrupamento: contagem por Data e NivelConsistencia
    contagem = (
        df_tudo.groupby(['Data', 'NivelConsistencia'])
        .size()
        .unstack(fill_value=0)
    )
    contagem.columns = [f'Nível {col}' for col in contagem.columns]

    # Agrupamento: contagem de vazios por data
    vazios_por_data = df_tudo[df_tudo['Vazio']].groupby('Data').size()

    # Número total de estações
    total_postos = len(resultados)
    # total_postos = len(estacoes_selecionadas)

    # Criar figura
    fig = go.Figure()

    # Linha para Nível 1
    fig.add_trace(go.Scatter(
        x=contagem.index,
        y=contagem.get('Nível 1', [0]*len(contagem)),
        mode='lines',
        name='Nível 1',
        line=dict(color='blue', dash='dot'),
        yaxis='y1'
    ))

    # Linha para Nível 2
    fig.add_trace(go.Scatter(
        x=contagem.index,
        y=contagem.get('Nível 2', [0]*len(contagem)),
        mode='lines',
        name='Nível 2',
        line=dict(color='green', dash='dot'),
        yaxis='y2'
    ))

    # Linha para vazios
    fig.add_trace(go.Scatter(
        x=vazios_por_data.index,
        y=vazios_por_data.values,
        mode='lines',
        name='Vazios',
        line=dict(color='red', dash='dot'),
        yaxis='y1'  # Mesmo eixo Y da contagem de dados
    ))

    # Layout com dois eixos Y e subtítulo
    fig.update_layout(
        title=dict(
            text='Quantidade de dados por dia - Níveis de Consistência<br>'
                 f'<sup>Total de Postos Observados: {total_postos}</sup>',
            x=0.5
        ),
        xaxis=dict(title='Data'),
        yaxis=dict(
            title='Quantidade (Nível 1 + Vazios)',
            titlefont=dict(color='blue'),
            tickfont=dict(color='blue'),
        ),
        yaxis2=dict(
            title='Quantidade (Nível 2)',
            titlefont=dict(color='green'),
            tickfont=dict(color='green'),
            overlaying='y',
            side='right',
        ),
        legend=dict(x=0.01, y=0.99),
        height=500,
        width=1000,
    )

    fig.show()
    
if __name__ == "__main__":  
    plotar_quantidade_dados_por_dia(resultados_vazao)


# %%

def plotar_colunas_dados_por_mes(resultados):
    """
    Plota um gráfico de colunas empilhadas por mês:
    - Quantidade de dados Nível 1 (azul)
    - Quantidade de dados Nível 2 (verde)
    - Quantidade de vazios (vermelho)
    """

    import pandas as pd
    import plotly.graph_objects as go

    # Junta todos os DataFrames
    df_tudo = pd.concat(resultados.values(), ignore_index=True)
    df_tudo['Data'] = pd.to_datetime(df_tudo['Data'])
    df_tudo['Vazio'] = df_tudo['Vazao'].isna()
   

    # Agrupamento mensal
    df_tudo['Mes'] = df_tudo['Data'].dt.to_period('M').dt.to_timestamp()

    # Contagem por nível de consistência e mês
    contagem = (
        df_tudo.groupby(['Mes', 'NivelConsistencia'])
        .size()
        .unstack(fill_value=0)
    )
    contagem.columns = [f'Nível {col}' for col in contagem.columns]

    # Contagem de vazios por mês
    vazios_por_mes = df_tudo[df_tudo['Vazio']].groupby('Mes').size()

    # Juntar com DataFrame principal
    contagem['Vazios'] = vazios_por_mes
    contagem['Vazios'] = contagem['Vazios'].fillna(0).astype(int)
    contagem = contagem.sort_index()

    # Preencher 0s para níveis ausentes
    if 'Nível 1' not in contagem.columns:
        contagem['Nível 1'] = 0
    if 'Nível 2' not in contagem.columns:
        contagem['Nível 2'] = 0

    total_postos = len(resultados)

    # Gráfico de colunas empilhadas por mês
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=contagem.index,
        y=contagem['Nível 1'],
        name='Nível 1',
        marker_color='blue'
    ))

    fig.add_trace(go.Bar(
        x=contagem.index,
        y=contagem['Nível 2'],
        name='Nível 2',
        marker_color='green'
    ))

    fig.add_trace(go.Bar(
        x=contagem.index,
        y=contagem['Vazios'],
        name='Vazios',
        marker_color='red'
    ))

    fig.update_layout(
        barmode='stack',
        title=dict(
            text='Composição de Dados por Mês<br>'
                 f'<sup>Total de Postos Observados: {total_postos}</sup>',
            x=0.5
        ),
        xaxis_title='Mês',
        yaxis_title='Quantidade de Registros',
        legend=dict(x=0.01, y=0.99),
        height=500,
        width=1200,
    )

    fig.show()
    
if __name__ == "__main__":
    plotar_colunas_dados_por_mes(resultados_vazao)


# %% [markdown]
# # Chuva

# %%
def extrair_chuva_ana(cod_estacao, data_inicio="", data_fim="", nivel_consistencia=[1, 2]):
    import requests
    import xml.etree.ElementTree as ET
    import pandas as pd

    def fazer_requisicao(nivel):
        url = "http://telemetriaws1.ana.gov.br/ServiceANA.asmx"

        soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
        <soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
                       xmlns:xsd="http://www.w3.org/2001/XMLSchema" 
                       xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
          <soap:Body>
            <HidroSerieHistorica xmlns="http://MRCS/">
              <codEstacao>{cod_estacao}</codEstacao>
              <dataInicio>{data_inicio}</dataInicio>
              <dataFim>{data_fim}</dataFim>
              <tipoDados>2</tipoDados>
              <nivelConsistencia>{nivel}</nivelConsistencia>
            </HidroSerieHistorica>
          </soap:Body>
        </soap:Envelope>"""

        headers = {
            "Content-Type": "text/xml; charset=utf-8",
            "SOAPAction": "http://MRCS/HidroSerieHistorica"
        }

        try:
            response = requests.post(url, data=soap_body, headers=headers, timeout=30)
            if response.status_code == 200:
                return response.text
            else:
                print(f"Erro HTTP {response.status_code} na requisição (nível {nivel}).")
                return None
        except Exception as e:
            print(f"Erro ao requisitar dados para nível {nivel}: {e}")
            return None

    def xml_para_dataframe(xml_string):
        ns = {
            'soap': 'http://schemas.xmlsoap.org/soap/envelope/',
            'diffgr': 'urn:schemas-microsoft-com:xml-diffgram-v1'
        }

        try:
            root = ET.fromstring(xml_string)
            diffgram = root.find('.//diffgr:diffgram', ns)
            if diffgram is None:
                return pd.DataFrame()

            document_element = diffgram.find('DocumentElement')
            if document_element is None:
                return pd.DataFrame()

            records = []
            for record in document_element:
                rec_dict = {}
                for field in record:
                    rec_dict[field.tag] = field.text
                records.append(rec_dict)

            return pd.DataFrame(records)
        except Exception as e:
            print(f"Erro ao processar XML: {e}")
            return pd.DataFrame()

    if isinstance(nivel_consistencia, int):
        nivel_consistencia = [nivel_consistencia]

    todos_dfs = []
    for nivel in nivel_consistencia:
        xml = fazer_requisicao(nivel)
        if xml:
            df = xml_para_dataframe(xml)
            if not df.empty:
                # Verifica se a coluna existe antes de filtrar
                if "NivelConsistencia" in df.columns:
                    df_filtrado = df[df["NivelConsistencia"] == str(nivel)]
                    if not df_filtrado.empty:
                        todos_dfs.append(df_filtrado)
                else:
                    print(f"Nenhuma coluna 'NivelConsistencia' nos dados retornados da estação {cod_estacao} (nível {nivel}).")
            else:
                print(f"Sem dados retornados para estação {cod_estacao} (nível {nivel}).")
        else:
            print(f"Requisição falhou para estação {cod_estacao} (nível {nivel}).")

    if todos_dfs:
        return pd.concat(todos_dfs, ignore_index=True)
    else:
        #print(f"❌ Não existem dados para a estação {cod_estacao}.")
        return pd.DataFrame()


# %%
def extrair_chuva_ana_multiplas(df_estacoes, data_inicio="", data_fim="", nivel_consistencia=[1, 2]):
    """
    Recebe DataFrame com colunas 'Codigo', 'TipoEstacao' e 'TipoEstacaoTelemetrica'.
    Filtra por TipoEstacao == 1 e TipoEstacaoTelemetrica == 0.
    Itera pelos códigos filtrados, chama extrair_vazao_ana para cada código,
    transforma para formato diário e retorna dict {codigo: df}.
    """
    import pandas as pd
    from tabulate import tabulate

    def transformar_chuvas(df):
        # Garante que a coluna DataHora está em datetime
        df['DataHora'] = pd.to_datetime(df['DataHora'])

        # Extrai ano e mês para montagem da data depois
        df['AnoMes'] = df['DataHora'].dt.strftime('%m/%Y')

        # Colunas de vazão e status
        col_chuva = [f"Chuva{str(i).zfill(2)}" for i in range(1, 32)]
        #col_status = [f"Vazao{str(i).zfill(2)}Status" for i in range(1, 32)]

        # Melt para transformar colunas em linhas
        df_chuva = df.melt(
            id_vars=['EstacaoCodigo', 'NivelConsistencia','AnoMes'],
            value_vars=col_chuva,
            var_name='Dia',
            value_name='Chuva'
        )
        # df_status = df.melt(
        #     id_vars=['EstacaoCodigo', 'AnoMes'],
        #     value_vars=col_status,
        #     var_name='Dia',
        #     value_name='StatusVazao'
        # )

        # Extrai o número do dia das colunas tipo 'Vazao01'
        df_chuva['Dia'] = df_chuva['Dia'].str.extract(r'(\d+)').astype(int)
        #df_status['Dia'] = df_status['Dia'].str.extract(r'(\d+)').astype(int)

        # # Junta as duas tabelas
        # df_mergeado = pd.merge(df_vazao, df_status,
        #                        on=['EstacaoCodigo', 'AnoMes', 'Dia'],
        #                        how='left')

        # Monta a data completa
        df_chuva['Data'] = df_chuva['Dia'].astype(str).str.zfill(2) + '/' + df_chuva['AnoMes']
        df_chuva['Data'] = pd.to_datetime(df_chuva['Data'], format='%d/%m/%Y', errors='coerce')

        # Remove datas inválidas
        df_chuva = df_chuva.dropna(subset=['Data'])

        # Organiza e retorna
        df_final = df_chuva[['EstacaoCodigo', 'NivelConsistencia','Data', 'Chuva']].sort_values(by=['EstacaoCodigo', 'Data']).reset_index(drop=True)

        return df_final

    resultados = {}

    # Filtrar estações válidas
    df_filtrado = df_estacoes[(df_estacoes['TipoEstacao'] == '2') & (df_estacoes['TipoEstacaoTelemetrica'] == '0')]
    codigos = df_filtrado['Codigo'].unique()

    for codigo in codigos:
        print(f"Extraindo dados para a estação {codigo}...")
        df = extrair_chuva_ana(codigo, nivel_consistencia=nivel_consistencia)

        if not df.empty:
            df = transformar_chuvas(df)
            print(f'✅ Dados extraidos com sucesso para {codigo}')
            #print(tabulate(df, headers='keys', tablefmt='grid'))
            resultados[codigo] = df
        else:
            print(f"❌ Sem dados para {codigo}. Nada será adicionado.")

    return resultados

if __name__ == "__main__":
    # Chamada
    resultados_chuva = extrair_chuva_ana_multiplas(df_merge, nivel_consistencia=[1,2])


# %%
if __name__ == "main":
    resultados_chuva

# %%


def plotar_quantidade_dados_por_dia(resultados):
    """
    Plota um gráfico interativo com:
    - Quantidade de dados por dia com NivelConsistencia = 1 e = 2
    - Quantidade de valores vazios por dia (linha vermelha)
    - Eixos Y duplos e título com número de estações
    """

    import pandas as pd
    import plotly.graph_objects as go

    # Combina todos os DataFrames do dicionário
    df_tudo = pd.concat(resultados.values(), ignore_index=True)

    # # Exemplo da lista de estações que você quer manter:
    # estacoes_selecionadas = ['58033000', '58030000', '58035000']

    # # Filtra o DataFrame mantendo só as linhas cujo EstacaoCodigo está na lista
    # df_tudo = df_tudo[df_tudo['EstacaoCodigo'].astype(str).isin(estacoes_selecionadas)]

    # Garante que a coluna 'Data' esteja no tipo datetime
    df_tudo['Data'] = pd.to_datetime(df_tudo['Data'])

    # Cria coluna para marcar vazios
    df_tudo['Vazio'] = df_tudo['Chuva'].isna()

    # Agrupamento: contagem por Data e NivelConsistencia
    contagem = (
        df_tudo.groupby(['Data', 'NivelConsistencia'])
        .size()
        .unstack(fill_value=0)
    )
    contagem.columns = [f'Nível {col}' for col in contagem.columns]

    # Agrupamento: contagem de vazios por data
    vazios_por_data = df_tudo[df_tudo['Vazio']].groupby('Data').size()

    # Número total de estações
    total_postos = len(resultados)
    # total_postos = len(estacoes_selecionadas)

    # Criar figura
    fig = go.Figure()

    # Linha para Nível 1
    fig.add_trace(go.Scatter(
        x=contagem.index,
        y=contagem.get('Nível 1', [0]*len(contagem)),
        mode='lines',
        name='Nível 1',
        line=dict(color='blue', dash='dot'),
        yaxis='y1'
    ))

    # Linha para Nível 2
    fig.add_trace(go.Scatter(
        x=contagem.index,
        y=contagem.get('Nível 2', [0]*len(contagem)),
        mode='lines',
        name='Nível 2',
        line=dict(color='green', dash='dot'),
        yaxis='y2'
    ))

    # Linha para vazios
    fig.add_trace(go.Scatter(
        x=vazios_por_data.index,
        y=vazios_por_data.values,
        mode='lines',
        name='Vazios',
        line=dict(color='red', dash='dot'),
        yaxis='y1'  # Mesmo eixo Y da contagem de dados
    ))

    # Layout com dois eixos Y e subtítulo
    fig.update_layout(
        title=dict(
            text='Quantidade de dados por dia - Níveis de Consistência<br>'
                 f'<sup>Total de Postos Observados: {total_postos}</sup>',
            x=0.5
        ),
        xaxis=dict(title='Data'),
        yaxis=dict(
            title='Quantidade (Nível 1 + Vazios)',
            titlefont=dict(color='blue'),
            tickfont=dict(color='blue'),
        ),
        yaxis2=dict(
            title='Quantidade (Nível 2)',
            titlefont=dict(color='green'),
            tickfont=dict(color='green'),
            overlaying='y',
            side='right',
        ),
        legend=dict(x=0.01, y=0.99),
        height=500,
        width=1000,
    )

    fig.show()
    
if __name__ == "__main__":
    plotar_quantidade_dados_por_dia(resultados_chuva)


# %% [markdown]
# # Pre tratamento

# %%


def plotar_estacao_falha(dicionario, nivel_consistencia, variavel, intervalo_tempo):
    """
    Plota a curva do número de estações válidas para diferentes níveis de falha.
    """

    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    datas_inicio = []
    datas_fim = []
    total_dias = None
    
    # Primeiro, extrair datas mínimas e máximas para considerar o intervalo
    for df in dicionario.values():
        if variavel not in df.columns:
            continue

        df = df[df['NivelConsistencia'] == str(nivel_consistencia)]
        if not df.empty:
            datas_inicio.append(pd.to_datetime(df['Data'].min()))
            datas_fim.append(pd.to_datetime(df['Data'].max()))
    
    if intervalo_tempo is None:
        data_inicio = min(datas_inicio)
        data_fim = max(datas_fim)
    else:
        data_inicio = pd.to_datetime(intervalo_tempo[0])
        data_fim = pd.to_datetime(intervalo_tempo[1])
    
    total_dias = (data_fim - data_inicio).days + 1
    
    niveis_falha = np.linspace(0, 1, 101)  # de 0.0 a 1.0 em 101 passos
    estacoes_validas = []

    for nivel in niveis_falha:
        cont_validas = 0
        for df in dicionario.values():
            if variavel not in df.columns:
                continue

            df_filtro = df[df['NivelConsistencia'] == str(nivel_consistencia)].copy()
            df_filtro['Data'] = pd.to_datetime(df_filtro['Data'], errors='coerce')
            df_filtro = df_filtro[(df_filtro['Data'] >= data_inicio) & (df_filtro['Data'] <= data_fim)]
            if df_filtro.empty:
                continue

            df_filtro[variavel] = pd.to_numeric(df_filtro[variavel], errors='coerce')
            dados_validos = df_filtro[variavel].notna().sum()
            proporcao_falhas = 1 - (dados_validos / total_dias)

            if proporcao_falhas <= nivel:
                cont_validas += 1
        estacoes_validas.append(cont_validas)

    plt.figure(figsize=(10,6))
    plt.plot(niveis_falha, estacoes_validas)
    
    # Define ticks no eixo X a cada 0.05 (5%)
    ticks = np.arange(0, 1.05, 0.05)  # inclui 1.0
    plt.xticks(ticks, [f"{int(t*100)}%" for t in ticks])
    
    plt.xlabel('Nível máximo permitido de falhas')
    plt.ylabel('Número de estações válidas')
    plt.title(f'Estações válidas por nível de falha para {variavel}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %%
def preencher_falhas(dicionario, df_depara, quantidade_estacoes=3, raio_maximo=100.0, variavel=None):
    """
    Preenche valores ausentes nos DataFrames do dicionário usando a média das N estações
    mais próximas com dado válido na mesma data (usando Haversine em km).
    """
    from math import radians, sin, cos, sqrt, atan2
    import pandas as pd
    import numpy as np

    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    # Prepara localizações
    localizacao = df_depara.set_index('Codigo')[['Latitude', 'Longitude']].copy()
    localizacao['Latitude'] = pd.to_numeric(localizacao['Latitude'], errors='coerce')
    localizacao['Longitude'] = pd.to_numeric(localizacao['Longitude'], errors='coerce')

    novo_dicionario = {}

    for codigo, df in dicionario.items():
        df = df.copy()
        if variavel not in df.columns or codigo not in localizacao.index:
            novo_dicionario[codigo] = df
            continue

        lat_ref, lon_ref = localizacao.loc[codigo]
        df['Data'] = pd.to_datetime(df['Data'])
        df[variavel] = pd.to_numeric(df[variavel], errors='coerce')

        for i, row in df[df[variavel].isna()].iterrows():
            data_falha = row['Data']

            # Candidatas com dado válido na data
            candidatas = []
            for cod_vizinho, (lat2, lon2) in localizacao.iterrows():
                if cod_vizinho == codigo or cod_vizinho not in dicionario:
                    continue
                df_vizinho = dicionario[cod_vizinho].copy()
                df_vizinho['Data'] = pd.to_datetime(df_vizinho['Data'])
                df_vizinho[variavel] = pd.to_numeric(df_vizinho[variavel], errors='coerce')

                valor_vizinho = df_vizinho.loc[df_vizinho['Data'] == data_falha, variavel]
                if not valor_vizinho.empty and pd.notna(valor_vizinho.values[0]):
                    dist = haversine_km(lat_ref, lon_ref, lat2, lon2)
                    if dist <= raio_maximo:
                        candidatas.append((cod_vizinho, dist, valor_vizinho.values[0]))

            # Ordenar pelas mais próximas com dado
            candidatas = sorted(candidatas, key=lambda x: x[1])[:quantidade_estacoes]

            if candidatas:
                media = np.mean([val for _, _, val in candidatas])
                df.at[i, variavel] = media
            else:
                print(f"[SEM DADO] Estação {codigo} | Data: {data_falha.date()} | Nenhuma vizinha com valor válido no raio")

        novo_dicionario[codigo] = df

    return novo_dicionario


# %%


def plotar_serie(dicionario, variavel=None):
    """
    Plota um gráfico individual para cada estação no dicionário.

    Parâmetros:
    - dicionario: dict com DataFrames contendo colunas ['Data', variavel]
    - variavel: nome da variável ('Chuva' ou 'Vazao')
    """

    import matplotlib.pyplot as plt

    for codigo, df in dicionario.items():
        if variavel not in df.columns or 'Data' not in df.columns:
            continue
        df_plot = df[['Data', variavel]].dropna().sort_values('Data')
        if df_plot.empty:
            continue

        plt.figure(figsize=(12, 5))
        plt.plot(df_plot['Data'], df_plot[variavel], label=str(codigo))
        plt.title(f'Série Temporal da Estação {codigo} - {variavel}')
        plt.xlabel('Data')
        plt.ylabel(variavel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# %%


def gerar_df_datamining(dicionario, intervalo, variavel):
    """
    Gera um DataFrame para mineração com dados de várias estações.

    Parâmetros:
    - dicionario: dict de DataFrames contendo colunas ['Data', variavel]
    - intervalo: tupla/lista com (data_inicio, data_fim), strings ou datetime
    - variavel: nome da variável a extrair (ex: 'Chuva' ou 'Vazao')

    Retorna:
    - DataFrame com índice como intervalo de datas e colunas como códigos das estações
    """

    import pandas as pd
    from tabulate import tabulate
    
    data_inicio = pd.to_datetime(intervalo[0])
    data_fim = pd.to_datetime(intervalo[1])

    # Cria índice de datas completo
    idx_datas = pd.date_range(start=data_inicio, end=data_fim, freq='D')

    dfs = []

    for codigo, df in dicionario.items():
        if variavel not in df.columns or 'Data' not in df.columns:
            continue
        df_tmp = df[['Data', variavel]].copy()
        df_tmp['Data'] = pd.to_datetime(df_tmp['Data'])
        df_tmp = df_tmp[(df_tmp['Data'] >= data_inicio) & (df_tmp['Data'] <= data_fim)]
        df_tmp = df_tmp.set_index('Data').sort_index()
        # Renomeia a coluna para o código da estação
        df_tmp = df_tmp.rename(columns={variavel: str(codigo)})
        dfs.append(df_tmp[str(codigo)])

    if not dfs:
        return pd.DataFrame(index=idx_datas)  # vazio com índice de datas

    # Junta todas as séries em um DataFrame, alinhando pelo índice
    df_final = pd.concat(dfs, axis=1)

    # Reindexa para o índice completo (preenchendo com NaN onde não há dados)
    df_final = df_final.reindex(idx_datas)
    print(df_final.info())
    print('\n')
    print(tabulate(df_final, headers='keys', tablefmt='grid'))

    return df_final


# %%

def padronizacao_datamining(
    dicionario,
    nivel_consistencia,
    variavel,
    intervalo_tempo=None,
    nivel_falhas=0.1,
    plotar_grafico_falhas=True,
    preencher_dados_faltantes=False,
    df_depara=None,
    quantidade_estacoes=3,
    raio_maximo=1.0,
    plotar_series=False,
    retornar_df_mining=False
):
    """
    Padroniza os dados de chuva ou Vazao para mineração de dados.
    
    Parâmetros:
    - dicionario: dict de DataFrames (ex: resultados_chuva ou resultados_vazao)
    - nivel_consistencia: int (1 ou 2)
    - variavel: str ("Chuva" ou "Vazao")
    - intervalo_tempo: tupla ou lista com duas datas no formato 'YYYY-MM-DD'
    - nivel_falhas: float entre 0 e 1 (proporção máxima de dados ausentes permitida)
    
    Retorna:
    - Um dicionário filtrado apenas com os DataFrames aceitáveis (falhas abaixo do limite)
    """

    import pandas as pd
    
    # Validações
    if nivel_consistencia not in [1, 2]:
        raise ValueError("O nível de consistência deve ser 1 ou 2.")
        
    if variavel not in ["Chuva", "Vazao"]:
        raise ValueError("A variável deve ser 'Chuva' ou 'Vazao'.")
        
    if not 0 <= nivel_falhas <= 1:
        raise ValueError("O nível de falhas deve estar entre 0 e 1.")
    
    # Determinar intervalo de datas se não fornecido
    datas_min = []
    datas_max = []
    
    for df in dicionario.values():
        if variavel not in df.columns:
            continue

        df = df[df['NivelConsistencia'] == str(nivel_consistencia)]
        if not df.empty:
            datas_min.append(pd.to_datetime(df['Data'].min()))
            datas_max.append(pd.to_datetime(df['Data'].max()))

    if not datas_min or not datas_max:
        raise ValueError("Nenhum dado disponível com o nível de consistência e variável especificados.")

    if intervalo_tempo is None:
        data_inicio = min(datas_min)
        data_fim = max(datas_max)
    else:
        data_inicio = pd.to_datetime(intervalo_tempo[0])
        data_fim = pd.to_datetime(intervalo_tempo[1])
    
    print(f"Intervalo considerado: {data_inicio.date()} a {data_fim.date()}")
   
    # Criação do dicionário filtrado
    dicionario_filtrado = {}
    total_dias = (data_fim - data_inicio).days + 1
    
    for codigo, df in dicionario.items():
        if variavel not in df.columns:
            continue

        df = df[df['NivelConsistencia'] == str(nivel_consistencia)].copy()
        df['Data'] = pd.to_datetime(df['Data'], errors='coerce')
        df = df[(df['Data'] >= data_inicio) & (df['Data'] <= data_fim)]

        if df.empty:
            continue

        df[variavel] = pd.to_numeric(df[variavel], errors='coerce')
        dados_validos = df[variavel].notna().sum()
        proporcao_falhas = 1 - (dados_validos / total_dias)

        if proporcao_falhas <= nivel_falhas:
            dicionario_filtrado[codigo] = df
    print(f"Estações restantes após filtragem por falhas (≤ {nivel_falhas:.1%}): {len(dicionario_filtrado)}")
   
    if plotar_grafico_falhas:
        plotar_estacao_falha(dicionario, nivel_consistencia, variavel, intervalo_tempo)

    # Preenchimento opcional
    if preencher_dados_faltantes:
        if df_depara is None:
            raise ValueError("Você deve fornecer df_depara para preencher os dados faltantes.")
        print("Preenchendo valores ausentes com estações vizinhas...")
        dicionario_filtrado = preencher_falhas(
            dicionario=dicionario_filtrado,
            df_depara=df_depara,
            quantidade_estacoes=quantidade_estacoes,
            raio_maximo=raio_maximo,
            variavel=variavel
        )
    if plotar_series:
        plotar_serie(dicionario_filtrado, variavel=variavel)

    df_mining = None
    if retornar_df_mining:
        if intervalo_tempo is None:
            # usa intervalo do próprio filtro
            intervalo = (data_inicio, data_fim)
        else:
            intervalo = intervalo_tempo
        df_mining = gerar_df_datamining(dicionario_filtrado, intervalo, variavel)

    if retornar_df_mining:
        return dicionario_filtrado, df_mining
    else:
        return dicionario_filtrado
    
if __name__ == '__main__':
    dicionario_filtrado, df_mining = padronizacao_datamining(
        dicionario=resultados_vazao,
        nivel_consistencia=2,
        variavel='Vazao',
        intervalo_tempo=('1982-01-01', '1995-12-31'),
        nivel_falhas=0.005,
        plotar_grafico_falhas=True,
        preencher_dados_faltantes=True,
        df_depara=df_merge,
        quantidade_estacoes=3,
        raio_maximo=200.0,
        plotar_series=True,
        retornar_df_mining=True
    )


