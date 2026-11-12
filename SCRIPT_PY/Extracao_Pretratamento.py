"""
Extração e pré-tratamento de dados ANA (chuva) + MDT, integrando o que havia
em `Extracao_ANA.ipynb` e `Extracao_ANA_resumido.ipynb` em um único módulo.

Organização (nessa ordem de uso típico):
    1. NÚCLEO ANA (SOAP)          -> extrair_ANA, extrair_bacia, gerar_df_hidro
    2. SHAPES / BACIAS            -> bln_to_shp, listar_bacias_em_shapes, gerar_regiao, recortar_limites_bacias
    3. SELEÇÃO DE ESTAÇÕES        -> filtro_estacoes, criar_mapa_selecao_interativo, filtrar_df_por_selecao_json
    4. EXTRAÇÃO DE SÉRIES (CHUVA) -> extrair_chuva_ana, extrair_chuva_ana_multiplas (paralelo)
    5. PRÉ-TRATAMENTO             -> processar_dicionario_chuva, montar_df_datamining,
                                      plotar_falhas, preencher_falhas_vetorizado
    6. MDT (memory-safe)          -> nome_tile, gerar_mosaico_bacia_streaming
    7. MAPAS                      -> plotar_mapa, plotar_mapa_raster_com_shape,
                                      plotar_mapa_shapes_light, plot_mapa_bacia_zoom

As seções 1-5 mantêm exatamente a lógica dos notebooks originais (só foi
paralelizada a extração de séries, seção 4). A seção 6 foi reescrita: em vez
de baixar todos os tiles e fazer `rasterio.merge` com todos os arrays na
RAM de uma vez (o que estourava memória no Colab e em bacias grandes), o
mosaico é construído tile a tile, escrevendo direto num raster de saída em
disco -- nunca mais que um tile inteiro fica carregado na memória por vez.
Quando possível, o tile é lido remotamente via /vsicurl/ (sem baixar o
arquivo inteiro); se isso não for suportado pelo ambiente, cai para
download + cache local.
"""

import os
import re
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from tabulate import tabulate


# ============================================================================
# 1. NÚCLEO ANA (SOAP) -- idêntico aos dois notebooks originais
# ============================================================================

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

        resultado = estacoes.get('_value_1', [])
        if isinstance(resultado, dict):
            return [resultado]
        return resultado

    except Exception as e:
        print("Erro ao acessar o serviço:", e)
        return []


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


def gerar_df_hidro(
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

    if 'SubBaciaCodigo' in df_ana.columns:
        df_ana_renamed = df_ana.rename(columns={'SubBaciaCodigo': 'SubBaciaCodigo'})
    else:
        df_ana_renamed = df_ana.copy()

    if 'codSubBacia' in df_bacia.columns:
        df_bacia_renamed = df_bacia.rename(columns={'codSubBacia': 'SubBaciaCodigo'})
    else:
        df_bacia_renamed = df_bacia.copy()

    df_ana_renamed['SubBaciaCodigo'] = df_ana_renamed['SubBaciaCodigo'].astype(str)
    df_bacia_renamed['SubBaciaCodigo'] = df_bacia_renamed['SubBaciaCodigo'].astype(str)

    cols_para_adicionar = [col for col in df_bacia_renamed.columns if col not in df_ana_renamed.columns and col != 'SubBaciaCodigo']

    df_merge = df_ana_renamed.merge(
        df_bacia_renamed[['SubBaciaCodigo'] + cols_para_adicionar],
        on='SubBaciaCodigo',
        how='left'
    )

    return df_merge


# ============================================================================
# 2. SHAPES / BACIAS -- idêntico ao Extracao_ANA.ipynb (completo)
# ============================================================================

def _parse_bln_polygon(input_bln_path):
    """
    Lê um arquivo .bln (Golden Software Blanking File -- lista de pares
    x,y de um polígono, sem CRS embutido) e retorna (polygon, basin_name).

    Levanta FileNotFoundError se o arquivo não existir e ValueError se
    nenhuma coordenada válida for encontrada.
    """
    from shapely.geometry import Polygon

    coords = []
    basin_name = "default_basin"

    with open(input_bln_path, 'r') as f:
        try:
            header = f.readline().strip()
            name_match = re.search(r"['\"](.*?)['\"]", header)
            if name_match:
                basin_name = name_match.group(1)
            else:
                basin_name = header.split(',')[-1].strip()
        except Exception:
            f.seek(0)

        for line in f:
            parts = line.replace(',', ' ').strip().split()
            if len(parts) >= 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    coords.append((x, y))
                except ValueError:
                    continue

    if not coords:
        raise ValueError(f"Nenhuma coordenada válida encontrada em '{input_bln_path}'.")

    return Polygon(coords), basin_name


def ler_geodataframe_bacia(caminho, input_crs_bln="EPSG:4326"):
    """
    Lê a geometria de uma bacia a partir de um shapefile (.shp) OU de um
    arquivo .bln (Golden Software Blanking File), e retorna sempre um
    GeoDataFrame em EPSG:4326.

    Isso permite que todas as funções que recebem um "shapefile de bacia"
    (`filtro_estacoes`, `plotar_mapa_raster_com_shape`,
    `plotar_mapa_shapes_light`, `criar_mapa_selecao_interativo`,
    `plot_mapa_bacia_zoom`) funcionem tanto com bacias que só têm .shp
    quanto com bacias que só têm .bln -- basta passar o caminho do arquivo,
    a extensão é detectada automaticamente.

    `input_crs_bln` é o CRS assumido para as coordenadas do .bln (que não
    tem CRS embutido, ao contrário do .shp que tem .prj). Ajuste se o seu
    .bln estiver em UTM ou outra projeção em vez de lat/lon.
    """
    caminho = Path(caminho)

    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo de bacia não encontrado: {caminho}")

    sufixo = caminho.suffix.lower()

    if sufixo == '.shp':
        gdf = gpd.read_file(caminho)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326", allow_override=True)
        return gdf.to_crs("EPSG:4326")

    if sufixo == '.bln':
        polygon, basin_name = _parse_bln_polygon(caminho)
        gdf = gpd.GeoDataFrame(data=[{'name': basin_name}], geometry=[polygon], crs=input_crs_bln)
        return gdf.to_crs("EPSG:4326")

    raise ValueError(f"Formato de bacia não suportado: '{sufixo}' (use .shp ou .bln).")


def bln_to_shp(input_bln_path, output_shp_path, input_crs="EPSG:4326"):
    """
    Converte um arquivo de polígono .bln para um shapefile (.shp),
    garantindo que a saída esteja no CRS EPSG:4326. Utilitário para quem
    quer persistir a conversão em disco -- as demais funções do módulo já
    leem .bln direto via `ler_geodataframe_bacia`, sem precisar converter.
    """
    output_crs = "EPSG:4326"

    print(f"Iniciando conversão de: {input_bln_path}")

    try:
        polygon, basin_name = _parse_bln_polygon(input_bln_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em {input_bln_path}")
        return
    except ValueError as e:
        print(f"Erro: {e}")
        return
    except Exception as e:
        print(f"Erro ao ler o arquivo {input_bln_path}: {e}")
        return

    print(f"Encontrados {len(polygon.exterior.coords)} pontos para a bacia: '{basin_name}'")

    gdf = gpd.GeoDataFrame(
        data=[{'name': basin_name}],
        geometry=[polygon],
        crs=input_crs
    )

    if gdf.crs != output_crs:
        print(f"Reprojetando de {input_crs} para {output_crs}...")
        gdf = gdf.to_crs(output_crs)
    else:
        print(f"CRS de entrada já é {output_crs}. Nenhuma reprojeção necessária.")

    try:
        output_dir = os.path.dirname(output_shp_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Diretório criado: {output_dir}")

        gdf.to_file(output_shp_path, driver='ESRI Shapefile')
        print(f"\nSucesso! Shapefile salvo em: {output_shp_path}")

    except Exception as e:
        print(f"Erro ao salvar o shapefile: {e}")


def listar_bacias_em_shapes(pasta_shapes: str = "shapes_ibge"):
    pasta = Path(pasta_shapes)

    if not pasta.exists():
        raise FileNotFoundError(f"Pasta '{pasta_shapes}' não encontrada.")

    for subpasta in sorted(pasta.iterdir()):
        if subpasta.is_dir() and subpasta.name.startswith("bacias_nivel_"):
            arquivos_shp = list(subpasta.glob("*.shp"))

            for shp_path in arquivos_shp:
                print(f"\nLendo opções localizadas em: {shp_path.relative_to(pasta)}")
                try:
                    gdf = gpd.read_file(shp_path)

                    if 'nome_bacia' in gdf.columns:
                        nomes = [[i + 1, nome] for i, nome in enumerate(gdf['nome_bacia'])]
                        print(tabulate(nomes, headers=["#", "nome_bacia"], tablefmt="grid"))
                    else:
                        print(" [!] Coluna 'nome_bacia' não encontrada.")
                except Exception as e:
                    print(f" [Erro ao ler {shp_path.name}]: {e}")


def gerar_regiao(nomes_bacias, shapes_dir="shapes_ibge", plotar=True):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if isinstance(nomes_bacias, str):
        nomes_bacias = [nomes_bacias]

    todos_shapes = []
    for nivel in range(2, 7):
        pasta = Path(shapes_dir) / f"bacias_nivel_{nivel}"
        for shp in pasta.glob("*.shp"):
            gdf = gpd.read_file(shp)
            gdf["nivel"] = nivel
            todos_shapes.append(gdf)

    if not todos_shapes:
        raise ValueError("Nenhum shapefile encontrado nas pastas de níveis.")

    gdf_total = gpd.GeoDataFrame(pd.concat(todos_shapes, ignore_index=True), crs=todos_shapes[0].crs)

    resultado_final = gpd.GeoDataFrame(columns=gdf_total.columns)

    for nome in nomes_bacias:
        encontrados = gdf_total[gdf_total["nome_bacia"] == nome]
        if encontrados.empty:
            print(f"Bacia '{nome}' não encontrada em nenhum nível.")
            continue

        if len(encontrados) == 1:
            resultado_final = pd.concat([resultado_final, encontrados])
        else:
            print(f"Bacia '{nome}' encontrada em múltiplos níveis:")
            tabela = [
                {"index": idx, "nome_bacia": row["nome_bacia"], "nivel": row["nivel"]}
                for idx, row in encontrados.iterrows()
            ]
            print(tabulate(tabela, headers="keys", tablefmt="fancy_grid"))

            entrada = input("Digite os índices separados por vírgula para selecionar (ou pressione ENTER para pular): ")
            if entrada.strip() == "":
                continue

            try:
                indices = [int(i.strip()) for i in entrada.split(",")]
                selecionadas = encontrados.loc[indices]
                resultado_final = pd.concat([resultado_final, selecionadas])
            except Exception as e:
                print(f"Erro ao interpretar entrada: {e}. Pulando...")

    if resultado_final.empty:
        print("Nenhuma bacia válida selecionada.")
        return None

    if plotar:
        fig, ax = plt.subplots(figsize=(10, 8))
        cores = plt.cm.tab20.colors
        legend_patches = []

        for i, (nome_bacia, grupo) in enumerate(resultado_final.groupby("nome_bacia")):
            cor = cores[i % len(cores)]
            grupo.plot(ax=ax, edgecolor='black', facecolor=cor)
            legend_patches.append(Patch(facecolor=cor, edgecolor='black', label=nome_bacia))

        ax.legend(handles=legend_patches, title="Bacias", loc='lower right', fontsize=8, title_fontsize=9)
        ax.set_title("Bacia(s) Selecionada(s)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True)
        plt.tight_layout()
        plt.show()

    return resultado_final


def recortar_limites_bacias(geodf):
    resultados = {}
    for idx, row in geodf.iterrows():
        bacia_id = row['id']
        nome = row['nome_bacia']
        classe = row['nivel']
        bounds = row['geometry'].bounds  # (xmin, ymin, xmax, ymax)
        resultados[bacia_id] = {
            "Nome": nome,
            "Classe": classe,
            "xmin": bounds[0],
            "ymin": bounds[1],
            "xmax": bounds[2],
            "ymax": bounds[3]
        }
        print(f"Bacia Nome: {nome} - ID: {bacia_id}")
        print(resultados[bacia_id])
        print('-------------------------')
    return resultados


# ============================================================================
# 3. SELEÇÃO DE ESTAÇÕES -- idêntico ao Extracao_ANA_resumido.ipynb
# ============================================================================

def filtro_estacoes(shapefile_bacia, df_postos=None, input_crs_bln="EPSG:4326"):
    """
    Extrai os postos contidos dentro do shape da bacia desejada.
    Aceita tanto .shp quanto .bln (ver `ler_geodataframe_bacia`).
    Verifica se o shape é Polígono; se for Linha, tenta convertê-lo.
    """
    from shapely.ops import polygonize

    target_crs = "EPSG:4326"

    try:
        shape_gdf = ler_geodataframe_bacia(shapefile_bacia, input_crs_bln=input_crs_bln)
    except (FileNotFoundError, ValueError) as e:
        print(f"Erro: {e}")
        return None
    except Exception as e:
        print(f"Erro ao ler a bacia: {e}")
        return None

    geom_types = shape_gdf.geom_type.unique()
    print(f"Tipos de geometria no arquivo: {geom_types}")

    if shape_gdf.geom_type.isin(['Polygon', 'MultiPolygon']).all():
        print("Geometria da bacia é Polígono. Prosseguindo...")

    elif shape_gdf.geom_type.isin(['LineString', 'MultiLineString']).any():
        print("Geometria da bacia é Linha. Tentando converter para Polígono...")
        try:
            linhas_unidas = shape_gdf.unary_union
            poligonos_gerados = list(polygonize(linhas_unidas))

            if not poligonos_gerados:
                print("ERRO: As linhas da bacia não formam um loop fechado. Não é possível converter.")
                return None

            print(f"Sucesso: {len(poligonos_gerados)} polígono(s) criado(s).")
            shape_gdf = gpd.GeoDataFrame(geometry=poligonos_gerados, crs=shape_gdf.crs)
            shape_gdf = shape_gdf.dissolve()

        except Exception as e:
            print(f"Falha na conversão de Linha para Polígono: {e}")
            return None

    else:
        print(f"Tipo de geometria não suportado ({geom_types}).")
        return None

    if not isinstance(df_postos, gpd.GeoDataFrame):
        if 'Longitude' not in df_postos.columns or 'Latitude' not in df_postos.columns:
            print("Erro: df_postos não tem colunas 'Longitude' e 'Latitude'.")
            return None

        geometry = gpd.points_from_xy(df_postos['Longitude'], df_postos['Latitude'])
        gdf_postos = gpd.GeoDataFrame(df_postos, geometry=geometry, crs=target_crs)

    else:
        gdf_postos = df_postos.copy()
        if gdf_postos.crs is None:
            print("Aviso: O GeoDataFrame 'df_postos' de entrada está sem CRS.")
            print(f"Assumindo CRS de origem como {target_crs} (WGS84).")
            gdf_postos = gdf_postos.set_crs(target_crs, allow_override=True)
        else:
            gdf_postos = gdf_postos.to_crs(target_crs)

    print("Realizando o spatial join...")
    postos_dentro = gpd.sjoin(gdf_postos, shape_gdf, how="inner", predicate='within')

    print(f"Total de estações encontradas dentro da bacia: {len(postos_dentro)}")

    return postos_dentro


def criar_mapa_selecao_interativo(shapefile_path, df_postos, usina_coords, output_filename="mapa_selecao_estacoes.html", input_crs_bln="EPSG:4326"):
    """
    Cria um mapa HTML interativo com o Folium para seleção manual de estações.
    Salva um mapa com botão de exportação para `estacoes_selecionadas.json`,
    que pode ser lido depois por `filtrar_df_por_selecao_json`.

    `shapefile_path` aceita tanto .shp quanto .bln (ver `ler_geodataframe_bacia`).
    """
    import folium
    import branca.element

    print(f"--- Iniciando a criação do mapa: {output_filename} ---")

    try:
        print(f"Carregando shapefile da bacia de: {shapefile_path}")
        shape_gdf_folium = ler_geodataframe_bacia(shapefile_path, input_crs_bln=input_crs_bln)

        df_geo_folium = df_postos.copy()
        if df_geo_folium.crs.to_string() != "EPSG:4326":
            df_geo_folium = df_geo_folium.to_crs("EPSG:4326")
        print("Dados geográficos prontos.")

        center_y, center_x = usina_coords
        m = folium.Map(location=[center_y, center_x], zoom_start=10, tiles="CartoDB positron")

        folium.GeoJson(
            shape_gdf_folium,
            name='Bacia Hidrográfica',
            style_function=lambda x: {'color': 'blue', 'weight': 2, 'fillColor': 'blue', 'fillOpacity': 0.1}
        ).add_to(m)

        print(f"Adicionando {len(df_geo_folium)} estações selecionáveis ao mapa...")

        point_to_layer_js = """
        function(feature, latlng) {
            return L.circleMarker(latlng, {
                radius: 5, color: 'red', fill: true, fillColor: 'red', fillOpacity: 0.7, weight: 1
            });
        }
        """
        on_each_feature_js = """
        function(feature, layer) {
            if (typeof parent.stationLayerMap === 'undefined') { parent.stationLayerMap = {}; }
            const props = feature.properties;
            const code = props.Codigo ? props.Codigo.toString() : 'N/A';
            const name = props.Nome ? props.Nome.toString() : code;
            const js_code = code.replace(/'/g, "\\'");
            layer.bindTooltip(`<b>${name}</b><br>Código: ${code}`);
            const popup_html = `
                <b>Estação:</b> ${name}<br><b>Código:</b> ${code}<br><br>
                <button onclick="parent.addStation('${js_code}')"
                    style="color: white; background-color: #007bff; border: none; padding: 5px 10px; border-radius: 5px; cursor: pointer;">
                    Selecionar Estação
                </button>
            `;
            layer.bindPopup(popup_html, {maxWidth: 300});
            parent.stationLayerMap[code] = layer;
            if (parent.selectedStations && parent.selectedStations.has(code)) {
                layer.setStyle({color: 'green', fillColor: 'green'});
            }
        }
        """
        folium.GeoJson(
            df_geo_folium,
            name="Estações",
            point_to_layer=folium.JsCode(point_to_layer_js),
            on_each_feature=folium.JsCode(on_each_feature_js)
        ).add_to(m)

        print("Adicionando marcador da usina.")
        folium.Marker(
            location=usina_coords,
            tooltip="Usina (Ponto de Interesse)",
            icon=folium.Icon(color='orange', icon='bolt')
        ).add_to(m)

        script_definitions = """
        <script>
            if (typeof parent.selectedStations === 'undefined') { parent.selectedStations = new Set(); }
            if (typeof parent.stationLayerMap === 'undefined') { parent.stationLayerMap = {}; }

            parent.addStation = function(code) {
                parent.selectedStations.add(code);
                parent.updateSelectedList();
                const layer = parent.stationLayerMap[code];
                if (layer) {
                    layer.setStyle({ color: 'green', fillColor: 'green', fillOpacity: 0.9 });
                }
            }

            parent.updateSelectedList = function() {
                const listDiv = document.getElementById('selected-list');
                if (!listDiv) { return; }
                if (parent.selectedStations.size === 0) {
                    listDiv.innerHTML = 'Nenhuma estação selecionada.';
                    return;
                }
                let html = '<ul>';
                parent.selectedStations.forEach(code => { html += `<li>${code}</li>`; });
                html += '</ul>';
                listDiv.innerHTML = html;
            }

            parent.exportJson = function() {
                if (parent.selectedStations.size === 0) {
                    alert("Nenhuma estação foi selecionada!");
                    return;
                }
                const selectedArray = Array.from(parent.selectedStations);
                const finalJsonData = selectedArray.map(code => ({ "Codigo": code }));
                const jsonString = JSON.stringify(finalJsonData, null, 2);
                const blob = new Blob([jsonString], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'estacoes_selecionadas.json';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }
        </script>
        """
        m.get_root().header.add_child(branca.element.Element(script_definitions))

        ui_html_and_script = """
        {% macro html(this, kwargs) %}
            <div id="selection-panel" style="position: fixed;
                        top: 10px; right: 10px; width: 250px;
                        background-color: white; border: 2px solid grey;
                        padding: 10px; z-index: 9999; border-radius: 5px;
                        font-family: Arial, sans-serif;">

                <h4>Estações Selecionadas</h4>

                <div id="selected-list" style="max-height: 150px; overflow-y: auto; background-color: #f9f9f9; border: 1px solid #ddd; padding: 5px; margin-bottom: 10px; min-height: 50px;">
                    Carregando...
                </div>

                <button
                    onclick="parent.exportJson()"
                    style="color: white; background-color: #28a745; border: none; padding: 10px; width: 100%; border-radius: 5px; cursor: pointer; font-weight: bold;">
                    Finalizar e Exportar JSON
                </button>
            </div>

            <script>
                setTimeout(function() {
                    parent.updateSelectedList();
                }, 500);
            </script>
        {% endmacro %}
        """

        class CustomUI(branca.element.MacroElement):
            _template = branca.element.Template(ui_html_and_script)

            def __init__(self):
                super().__init__()
                self._name = 'CustomUI'

        m.get_root().add_child(CustomUI())

        folium.LayerControl().add_to(m)

        m.save(output_filename)

        print(f"\nSucesso! Mapa interativo salvo como: {output_filename}")
        print("Abra este arquivo HTML em seu navegador para usar a seleção interativa.")

    except FileNotFoundError as e:
        print("Erro: Arquivo não encontrado. Verifique o caminho do shapefile.")
        print(f"Detalhe: {e}")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")


def filtrar_df_por_selecao_json(df_geo_completo, json_filename="estacoes_selecionadas.json"):
    """
    Filtra um GeoDataFrame com base em uma lista de códigos de estação
    carregada de um arquivo JSON (exportado do mapa interativo).
    """
    print(f"--- Iniciando filtro com base no arquivo: {json_filename} ---")

    try:
        df_selecao = pd.read_json(json_filename)

        if 'Codigo' not in df_selecao.columns:
            print(f"Erro: O arquivo JSON '{json_filename}' não contém a coluna 'Codigo'.")
            return None

        codigos_selecionados = df_selecao['Codigo'].astype(str).tolist()

        if not codigos_selecionados:
            print("Aviso: O arquivo JSON não contém nenhum código. Retornando DataFrame vazio.")
            return df_geo_completo.iloc[0:0]

        print(f"Encontrados {len(codigos_selecionados)} códigos selecionados no JSON.")

        coluna_codigo_str = df_geo_completo['Codigo'].astype(str)
        df_geo_filtrado = df_geo_completo[coluna_codigo_str.isin(codigos_selecionados)].copy()

        print("\n--- DataFrame filtrado com sucesso! ---")
        print(f"Total de estações no DataFrame original: {len(df_geo_completo)}")
        print(f"Total de estações selecionadas (filtradas): {len(df_geo_filtrado)}")

        return df_geo_filtrado

    except FileNotFoundError:
        print("--- ERRO ---")
        print(f"Arquivo '{json_filename}' não encontrado!")
        return None
    except Exception as e:
        print(f"Ocorreu um erro inesperado ao ler o arquivo JSON: {e}")
        return None


# ============================================================================
# 4. EXTRAÇÃO DE SÉRIES (CHUVA) -- extrair_chuva_ana idêntica; a versão
#    "_multiplas" foi paralelizada com ThreadPoolExecutor (I/O-bound: cada
#    chamada é uma requisição SOAP via `requests`, thread-safe pois não usa
#    nenhum estado global compartilhado).
# ============================================================================

def extrair_chuva_ana(cod_estacao, data_inicio="", data_fim="", nivel_consistencia=[1, 2]):
    import xml.etree.ElementTree as ET

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
        return pd.DataFrame()


def _transformar_chuvas(df):
    df['DataHora'] = pd.to_datetime(df['DataHora'])
    df['AnoMes'] = df['DataHora'].dt.strftime('%m/%Y')

    col_chuva = [f"Chuva{str(i).zfill(2)}" for i in range(1, 32)]

    df_chuva = df.melt(
        id_vars=['EstacaoCodigo', 'NivelConsistencia', 'AnoMes'],
        value_vars=col_chuva,
        var_name='Dia',
        value_name='Chuva'
    )

    df_chuva['Dia'] = df_chuva['Dia'].str.extract(r'(\d+)').astype(int)
    df_chuva['Data'] = df_chuva['Dia'].astype(str).str.zfill(2) + '/' + df_chuva['AnoMes']
    df_chuva['Data'] = pd.to_datetime(df_chuva['Data'], format='%d/%m/%Y', errors='coerce')

    df_chuva = df_chuva.dropna(subset=['Data'])

    df_final = df_chuva[['EstacaoCodigo', 'NivelConsistencia', 'Data', 'Chuva']].sort_values(by=['EstacaoCodigo', 'Data']).reset_index(drop=True)

    return df_final


def extrair_chuva_ana_multiplas(
    df_estacoes,
    data_inicio="",
    data_fim="",
    nivel_consistencia=[1, 2],
    max_workers=5,
):
    """
    Recebe DataFrame com colunas 'Codigo' e 'TipoEstacao'. Filtra por
    TipoEstacao == 2 (pluviométrica) e extrai a série histórica de cada
    estação EM PARALELO (thread pool), já que cada chamada é uma requisição
    SOAP independente (I/O-bound) via `requests` -- sem estado compartilhado,
    portanto thread-safe.

    `max_workers` controla quantas requisições simultâneas são feitas ao
    webservice da ANA. Comece conservador (ex: 5) e ajuste conforme a
    estabilidade observada; valores muito altos podem gerar timeouts/erros
    HTTP se o servidor limitar conexões simultâneas.

    Retorna o mesmo formato de sempre: dict {codigo: DataFrame}.
    """
    df_filtrado = df_estacoes[(df_estacoes['TipoEstacao'] == '2')]
    codigos = df_filtrado['Codigo'].unique()
    print(f"{len(codigos)} estações pluviométricas a extrair (max_workers={max_workers}).")

    def _tarefa(codigo):
        df = extrair_chuva_ana(codigo, data_inicio=data_inicio, data_fim=data_fim, nivel_consistencia=nivel_consistencia)
        if df.empty:
            return codigo, None
        return codigo, _transformar_chuvas(df)

    resultados = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futuros = {executor.submit(_tarefa, codigo): codigo for codigo in codigos}

        for futuro in tqdm(as_completed(futuros), total=len(futuros), desc="Extraindo chuva ANA"):
            codigo_original = futuros[futuro]
            try:
                codigo, df = futuro.result()
            except Exception as e:
                print(f"❌ Erro inesperado na estação {codigo_original}: {e}")
                continue

            if df is not None:
                resultados[codigo] = df
                print(f"✅ Dados extraídos com sucesso para {codigo}")
            else:
                print(f"❌ Sem dados para {codigo}. Nada será adicionado.")

    return resultados


def plotar_quantidade_dados_por_dia(resultados):
    import plotly.graph_objects as go

    df_tudo = pd.concat(resultados.values(), ignore_index=True)
    df_tudo['Data'] = pd.to_datetime(df_tudo['Data'])
    df_tudo['Vazio'] = df_tudo['Chuva'].isna()

    contagem = (
        df_tudo.groupby(['Data', 'NivelConsistencia'])
        .size()
        .unstack(fill_value=0)
    )
    contagem.columns = [f'Nível {col}' for col in contagem.columns]

    vazios_por_data = df_tudo[df_tudo['Vazio']].groupby('Data').size()

    total_postos = len(resultados)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=contagem.index,
        y=contagem.get('Nível 1', [0] * len(contagem)),
        mode='lines',
        name='Nível 1',
        line=dict(color='blue', dash='dot'),
        yaxis='y1'
    ))

    fig.add_trace(go.Scatter(
        x=contagem.index,
        y=contagem.get('Nível 2', [0] * len(contagem)),
        mode='lines',
        name='Nível 2',
        line=dict(color='green', dash='dot'),
        yaxis='y2'
    ))

    fig.add_trace(go.Scatter(
        x=vazios_por_data.index,
        y=vazios_por_data.values,
        mode='lines',
        name='Vazios',
        line=dict(color='red', dash='dot'),
        yaxis='y1'
    ))

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


# ============================================================================
# 5. PRÉ-TRATAMENTO -- idêntico aos dois notebooks originais
# ============================================================================

def _tratar_nivel_consistencia_df(df):
    """
    (Função Auxiliar) Trata o nível de consistência de um único DataFrame.
    Resolve duplicatas de data priorizando o NivelConsistencia 2 sobre o 1.
    """
    df_sorted = df.sort_values(by=['Data', 'NivelConsistencia'], ascending=[True, False])
    df_tratado = df_sorted.drop_duplicates(subset=['Data'], keep='first')
    return df_tratado


def processar_dicionario_chuva(dicionario_original):
    """
    Aplica `_tratar_nivel_consistencia_df` em cada estação do dicionário.
    """
    print("Iniciando o processamento do dicionário de chuvas...")
    dicionario_tratado = {}

    for codigo_estacao, df_original in dicionario_original.items():
        print(f"  - Processando estação: {codigo_estacao} ({len(df_original)} linhas)")
        df_processado = _tratar_nivel_consistencia_df(df_original)
        dicionario_tratado[codigo_estacao] = df_processado
        print(f"  - Concluído. Estação com {len(df_processado)} linhas após tratamento.")

    print("\nProcessamento finalizado com sucesso!")
    return dicionario_tratado


def montar_df_datamining(dicionario_tratado):
    """
    Monta um DataFrame para mineração a partir de um dicionário de DataFrames
    de chuva. Converte Chuva para float e evita fragmentação usando pd.concat.
    """
    print("Iniciando montagem do df_datamining...")

    series_list = []

    for codigo_estacao, df in dicionario_tratado.items():
        df['Chuva'] = pd.to_numeric(df['Chuva'], errors='coerce')
        serie_chuva = df.set_index('Data')['Chuva']
        serie_chuva.name = codigo_estacao
        series_list.append(serie_chuva)

    df_datamining = pd.concat(series_list, axis=1)
    df_datamining.sort_index(inplace=True)

    print(df_datamining.head())
    print("\nInformações do df_datamining:")
    print(df_datamining.info())
    print("\nResumo estatístico:")
    print(df_datamining.describe().T)

    return df_datamining


def plotar_falhas(df, data_inicio, data_fim, pct_desejada=None, plotar_mapa=False, gerar_df=False):
    """
    Plota o número de estações em função do percentual máximo de falhas
    permitido. Opcionalmente retorna um DataFrame filtrado pelas estações que
    atendem a `pct_desejada`.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import plotly.graph_objects as go

    df_intervalo = df.loc[data_inicio:data_fim]
    pct_falhas = df_intervalo.isna().mean() * 100

    limites = np.arange(0, 105, 5)
    num_estacoes = [(pct_falhas <= lim).sum() for lim in limites]

    if not plotar_mapa:
        plt.figure(figsize=(10, 6))
        plt.plot(limites, num_estacoes, marker='o')
        plt.xlabel("Percentual máximo de falhas (%)")
        plt.ylabel("Número de estações")
        plt.title(f"Distribuição de estações por percentual de falhas\n({data_inicio} até {data_fim})")
        plt.grid(True)
        plt.xticks(limites)
        ax = plt.gca()
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=15))
        plt.show()
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=limites, y=num_estacoes, mode='lines+markers'))

        fig.update_layout(
            title=f"Distribuição de estações por percentual de falhas\n({data_inicio} até {data_fim})",
            xaxis=dict(title="Percentual máximo de falhas (%)", dtick=5, autorange=True, fixedrange=False),
            yaxis=dict(title="Número de estações", autorange=True, tickmode='auto', fixedrange=False),
            template='plotly_white'
        )
        fig.show()

    df_filtrado = None
    if pct_desejada is not None:
        pct_percent = pct_desejada * 100
        ate_pct = (pct_falhas <= pct_percent).sum()
        acima_pct = (pct_falhas > pct_percent).sum()
        print(f"\nPara até {pct_percent:.1f}% de falhas:")
        print(f"  - Número de estações com até {pct_percent:.1f}% de falhas: {ate_pct}")
        print(f"  - Número de estações com mais de {pct_percent:.1f}% de falhas: {acima_pct}")

        if gerar_df:
            colunas_validas = pct_falhas[pct_falhas <= pct_percent].index
            df_filtrado = df_intervalo[colunas_validas]
            print(f"\nDataFrame filtrado gerado com {len(colunas_validas)} estações.")

    return df_filtrado


def preencher_falhas_vetorizado(df, df_geo, n_vizinhos=3):
    """
    Preenche valores ausentes em df usando a média das n_vizinhos estações
    mais próximas que têm valor no mesmo dia (KD-Tree, vetorizado).
    """
    from scipy.spatial import cKDTree

    df_result = df.copy()

    df_geo_sorted = df_geo.set_index('Codigo').loc[df.columns]
    coords = df_geo_sorted[['Latitude', 'Longitude']].astype(float).to_numpy()

    tree = cKDTree(coords)
    _, vizinhos_proximos = tree.query(coords, k=len(coords))

    n_linhas, n_colunas = df.shape

    print("Iniciando preenchimento vetorizado dos valores ausentes...")

    data_np = df.to_numpy()

    for i in tqdm(range(n_linhas), desc="Preenchendo linhas"):
        linha = data_np[i, :]
        nan_idx = np.where(np.isnan(linha))[0]

        if len(nan_idx) == 0:
            continue

        for idx in nan_idx:
            vizinhos = vizinhos_proximos[idx][vizinhos_proximos[idx] != idx]
            mask_validos = ~np.isnan(linha[vizinhos])
            vizinhos_validos = vizinhos[mask_validos]
            if len(vizinhos_validos) > 0:
                selecionados = vizinhos_validos[:n_vizinhos]
                linha[idx] = np.mean(linha[selecionados])

        data_np[i, :] = linha

    df_result.iloc[:, :] = data_np
    print("Preenchimento vetorizado concluído!")

    return df_result


# ============================================================================
# 6. MDT (memory-safe) -- reescrito. Copernicus DEM 30m (COGs públicos no S3).
#
#    Em vez de baixar TODOS os tiles e chamar rasterio.merge com todos os
#    arrays completos na RAM (comportamento antigo, que os próprios notebooks
#    já alertavam "provavelmente não funciona no Google Colab"), o mosaico é
#    construído tile a tile:
#      - cada tile é aberto individualmente (remoto via /vsicurl/, sem
#        baixar o arquivo inteiro quando o ambiente suporta; senão cai para
#        download com cache local em disco);
#      - só a JANELA do tile que sobrepõe a bacia é lida;
#      - essa janela é escrita direto no raster de saída em disco;
#      - o tile é fechado antes de processar o próximo.
#    Nunca mais que um tile inteiro fica carregado na memória, e o raster de
#    saída é escrito incrementalmente pelo GDAL (blocos), não como um array
#    numpy gigante em memória.
# ============================================================================

COPERNICUS_DEM_BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"


def nome_tile(lat, lon):
    ns = 'N' if lat >= 0 else 'S'
    ew = 'E' if lon >= 0 else 'W'
    lat_str = f"{abs(int(lat)):02d}"
    lon_str = f"{abs(int(lon)):03d}"
    return f"Copernicus_DSM_COG_10_{ns}{lat_str}_00_{ew}{lon_str}_00_DEM"


def _construir_lista_tiles(xmin, ymin, xmax, ymax):
    tiles = []
    lat_range = range(math.floor(ymin), math.floor(ymax) + 1)
    lon_range = range(math.floor(xmin), math.floor(xmax) + 1)
    for lat in lat_range:
        for lon in lon_range:
            tiles.append(nome_tile(lat, lon))
    return tiles


def _abrir_tile_dem(tile, cache_dir=None, tentar_remoto=True, timeout=30):
    """
    Abre um tile Copernicus DEM 30m e retorna um dataset rasterio aberto
    (chame `.close()` depois de usar, ou use dentro de um `with`).

    Estratégia (conforme decisão do usuário -- "tenta vsicurl, cai para
    download se falhar"):
      1. Tenta abrir remotamente via GDAL /vsicurl/, que só transfere os
         blocos de bytes necessários por leitura via HTTP range requests
         (nunca baixa o .tif inteiro).
      2. Se isso falhar (ambiente sem suporte a curl no GDAL, firewall,
         etc.), baixa o .tif inteiro (usando `cache_dir` se fornecido, para
         não baixar de novo em execuções futuras) e abre localmente.

    Um único tile é pequeno (~25-30 MB, grid 1x1 grau a 30m), então mesmo no
    fallback de download isso não compromete a RAM -- o que estourava
    memória era abrir e mesclar MUITOS tiles simultaneamente, não um só.

    Retorna None se o tile não existir (ex: tiles oceânicos sem cobertura).
    """
    import rasterio

    url = f"{COPERNICUS_DEM_BASE_URL}/{tile}/{tile}.tif"

    if tentar_remoto:
        try:
            ds = rasterio.open(f"/vsicurl/{url}")
            _ = ds.profile  # força uma leitura de metadados para validar o acesso
            return ds
        except Exception as e:
            print(f"[MDT] Leitura remota (vsicurl) falhou para {tile}: {e}. Tentando download...")

    destino = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        destino = cache_dir / f"{tile}.tif"
        if destino.exists():
            try:
                return rasterio.open(destino)
            except Exception:
                destino.unlink(missing_ok=True)  # cache corrompido, refaz download

    try:
        resp = requests.get(url, timeout=timeout)
    except Exception as e:
        print(f"[MDT] Falha ao baixar {tile}: {e}")
        return None

    if resp.status_code != 200:
        print(f"[MDT] Tile indisponível (HTTP {resp.status_code}): {tile}")
        return None

    if destino is not None:
        with open(destino, "wb") as f:
            f.write(resp.content)
        return rasterio.open(destino)
    else:
        from rasterio.io import MemoryFile
        memfile = MemoryFile(resp.content)
        return memfile.open()


def gerar_mosaico_bacia_streaming(
    bacia_geometry,
    id_bacia,
    nome_bacia,
    base_dir="raster_analise",
    cache_tiles_dir="RASTERS_BACIA_CACHE",
    tentar_remoto=True,
    max_workers_download=4,
    salvar_saida=True,
):
    """
    Gera o raster de elevação recortado para uma bacia, tile a tile, sem
    nunca carregar mais de um tile inteiro (ou o mosaico completo) na RAM.

    Parâmetros
    ----------
    bacia_geometry : shapely geometry (Polygon/MultiPolygon) em EPSG:4326
        Geometria da bacia (ex: `linha.geometry` de uma linha de um
        GeoDataFrame retornado por `gerar_regiao`).
    id_bacia, nome_bacia : identificação (usados só para nomear a saída).
    base_dir : pasta onde o raster final recortado é salvo.
    cache_tiles_dir : pasta de cache para tiles baixados (fallback de
        download). Passe None para não cachear em disco (usa MemoryFile).
    tentar_remoto : se True, tenta ler os tiles via /vsicurl/ antes de
        baixar (recomendado -- evita download quando possível).
    max_workers_download : threads usadas para PRÉ-BAIXAR os tiles em
        paralelo quando a leitura remota falha (I/O-bound). A escrita no
        raster de destino permanece sempre sequencial/single-thread, então
        isso só acelera o download, não muda o comportamento de memória.
    salvar_saida : se True, salva o raster final recortado em disco.

    Retorna um dict compatível com o antigo `gerar_mapa_bacia`:
        {"array", "transform", "crs", "meta", "saida", "boundary"}
    """
    import rasterio
    from rasterio.windows import from_bounds as window_from_bounds
    from rasterio.mask import mask as rasterio_mask

    xmin, ymin, xmax, ymax = bacia_geometry.bounds
    tiles = _construir_lista_tiles(xmin, ymin, xmax, ymax)
    print(f"[MDT] Bacia '{nome_bacia}' (ID {id_bacia}): {len(tiles)} tile(s) necessário(s).")

    # --- 1. Abre o primeiro tile válido só para pegar resolução/dtype/CRS ---
    primeiro_tile_ds = None
    for tile in tiles:
        ds = _abrir_tile_dem(tile, cache_dir=cache_tiles_dir, tentar_remoto=tentar_remoto)
        if ds is not None:
            primeiro_tile_ds = ds
            break

    if primeiro_tile_ds is None:
        raise RuntimeError(f"Nenhum tile Copernicus DEM disponível para a bacia '{nome_bacia}'.")

    res_x = primeiro_tile_ds.transform.a
    res_y = primeiro_tile_ds.transform.e  # negativo (norte->sul)
    dtype = primeiro_tile_ds.dtypes[0]
    nodata_origem = primeiro_tile_ds.nodata if primeiro_tile_ds.nodata is not None else -32768.0
    crs = primeiro_tile_ds.crs
    primeiro_tile_ds.close()

    # --- 2. Monta transform/tamanho do raster de destino, alinhado à bacia ---
    from rasterio.transform import from_origin
    largura = max(1, math.ceil((xmax - xmin) / abs(res_x)))
    altura = max(1, math.ceil((ymax - ymin) / abs(res_y)))
    dst_transform = from_origin(xmin, ymax, abs(res_x), abs(res_y))

    nome_bacia_fmt = str(nome_bacia).replace(" ", "_")
    pasta_saida = Path(base_dir) / nome_bacia_fmt / "raster_bacia"
    pasta_saida.mkdir(parents=True, exist_ok=True)
    caminho_bruto = pasta_saida / f"bacia_{id_bacia}_mosaico_bruto.tif"

    perfil_destino = {
        "driver": "GTiff",
        "height": altura,
        "width": largura,
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": dst_transform,
        "nodata": nodata_origem,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": "deflate",
    }

    # --- 3. Pré-baixa (em paralelo, se necessário) só para acelerar I/O; a
    #        gravação no destino permanece sequencial e tile-a-tile. ---
    def _obter_tile(tile):
        return tile, _abrir_tile_dem(tile, cache_dir=cache_tiles_dir, tentar_remoto=tentar_remoto)

    with rasterio.open(caminho_bruto, "w", **perfil_destino) as dst:
        with ThreadPoolExecutor(max_workers=max_workers_download) as executor:
            futuros = [executor.submit(_obter_tile, tile) for tile in tiles]

            for futuro in tqdm(as_completed(futuros), total=len(futuros), desc=f"Mosaico {nome_bacia_fmt}"):
                tile, src = futuro.result()
                if src is None:
                    continue

                try:
                    with src:
                        # Overlap entre o tile e a bacia, em coordenadas geográficas
                        t_xmin, t_ymin, t_xmax, t_ymax = src.bounds
                        ov_xmin = max(t_xmin, xmin)
                        ov_ymin = max(t_ymin, ymin)
                        ov_xmax = min(t_xmax, xmax)
                        ov_ymax = min(t_ymax, ymax)

                        if ov_xmin >= ov_xmax or ov_ymin >= ov_ymax:
                            continue  # tile não sobrepõe de fato a bacia (canto do bbox)

                        src_window = window_from_bounds(ov_xmin, ov_ymin, ov_xmax, ov_ymax, src.transform)
                        dst_window = window_from_bounds(ov_xmin, ov_ymin, ov_xmax, ov_ymax, dst_transform)

                        # Lê SÓ a janela de overlap (pequena) desse tile -- nunca o tile
                        # inteiro nem o mosaico inteiro.
                        dado = src.read(1, window=src_window, out_shape=(
                            max(1, round(dst_window.height)),
                            max(1, round(dst_window.width)),
                        ))

                        dst.write(dado, 1, window=dst_window)
                except Exception as e:
                    print(f"[MDT] Erro processando tile {tile}: {e}")

    # --- 4. Recorta pela geometria exata da bacia. O raster de entrada aqui já
    #        está limitado ao bbox da bacia (não à união de todos os tiles),
    #        então essa leitura é pequena e segura em memória. ---
    with rasterio.open(caminho_bruto) as dst_bruto:
        geom = [bacia_geometry.__geo_interface__]
        out_image, out_transform = rasterio_mask(dst_bruto, geom, crop=True)
        out_meta = dst_bruto.meta.copy()

    out_image = np.where(out_image == nodata_origem, np.nan, out_image).astype("float32")
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
        "dtype": "float32",
        "nodata": np.nan,
    })

    saida = None
    if salvar_saida:
        saida = pasta_saida / f"bacia_{id_bacia}_recorte.tif"
        with rasterio.open(saida, "w", **out_meta) as dest:
            dest.write(out_image)
        print(f"[MDT] Raster recortado salvo em: {saida}")

    # Remove o mosaico bruto intermediário (não é mais necessário)
    try:
        caminho_bruto.unlink(missing_ok=True)
    except Exception:
        pass

    return {
        "array": out_image,
        "transform": out_transform,
        "crs": out_meta["crs"],
        "meta": out_meta,
        "saida": str(saida) if saida else None,
    }


# ============================================================================
# 7. MAPAS -- leem o raster já recortado (pequeno) salvo em disco, portanto
#    já eram memory-safe; mantidos como no notebook completo original.
# ============================================================================

def plotar_mapa(pasta_raster, df_shape=None, id_bacia=None, cmap="magma", salvar=True, titulo=None, Nome_Bacia=None):
    import rasterio
    from rasterio.plot import show
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable

    pasta_raster = Path(pasta_raster)

    arquivos_tif = list(pasta_raster.glob("*.tif"))
    if not arquivos_tif:
        print(f"Nenhum arquivo .tif encontrado em {pasta_raster}")
        return

    raster_path = arquivos_tif[0]

    with rasterio.open(raster_path) as src:
        array = src.read(1)
        transform = src.transform
        nodata = src.nodata

        array = np.where(array == nodata, np.nan, array)

        fig, ax = plt.subplots(figsize=(8, 6))
        show(array, transform=transform, ax=ax, cmap=cmap)

        if df_shape is not None and id_bacia is not None:
            bacia_gdf = df_shape[df_shape["id"] == id_bacia]
            if not bacia_gdf.empty:
                bacia_gdf.boundary.plot(ax=ax, edgecolor="black", linewidth=1)

        ax.set_title(f"Mapa: {titulo}", fontsize=12)

        sm = ScalarMappable(cmap=cmap)
        sm.set_array(array)
        cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
        cbar.set_label("Altura", rotation=270, labelpad=15)

        caminho_saida = None
        if salvar:
            pasta_saida = Path("SAIDA_ETAPA_MAPAS")
            pasta_saida.mkdir(parents=True, exist_ok=True)
            nome_arquivo = f"{Nome_Bacia}.png"
            caminho_saida = pasta_saida / nome_arquivo
            plt.savefig(caminho_saida, dpi=300, bbox_inches="tight")
            print(f"Mapa salvo em: {caminho_saida}")

        plt.show()

    return caminho_saida


def plotar_mapa_raster_com_shape(
    pasta_raster,
    shapefile_bacia,
    shapefile_rio=None,
    cmap="magma",
    salvar=True,
    titulo=None,
    nome_usina=None,
    rio_principal=None,
    plotar_postos=False,
    df_postos=None,
    postos_destaque=None,
    localizacao_usina=None,
    apenas_postos_selecionados=False,
    input_crs_bln="EPSG:4326"
):
    """
    Plota um raster recortado pelo shapefile da bacia (raster já pequeno,
    gerado por `gerar_mosaico_bacia_streaming`). Permite sobrepor rios,
    postos e destacar postos e a usina. Versão consolidada (mais completa
    das 3 iterações do notebook original).

    `shapefile_bacia` aceita tanto .shp quanto .bln (ver `ler_geodataframe_bacia`).
    """
    import rasterio
    from rasterio.plot import show
    from rasterio.mask import mask
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable

    pasta_raster = Path(pasta_raster)

    try:
        shape_gdf = ler_geodataframe_bacia(shapefile_bacia, input_crs_bln=input_crs_bln)
    except (FileNotFoundError, ValueError) as e:
        print(f"Erro: {e}")
        return
    except Exception as e:
        print(f"Erro ao ler a bacia: {e}")
        return

    rios_gdf = None
    if shapefile_rio is not None and Path(shapefile_rio).exists():
        rios_gdf = gpd.read_file(shapefile_rio)
        rios_gdf = rios_gdf.to_crs(shape_gdf.crs)

    arquivos_tif = list(pasta_raster.glob("*.tif"))
    if not arquivos_tif:
        print(f"Nenhum arquivo .tif encontrado em {pasta_raster}")
        return
    raster_path = arquivos_tif[0]

    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        geom = [shape_gdf.geometry.union_all().__geo_interface__]
        out_image, out_transform = mask(src, geom, crop=True)
        out_image = np.where(out_image == nodata, np.nan, out_image)

    fig, ax = plt.subplots(figsize=(16, 12))
    show(out_image, transform=out_transform, ax=ax, cmap=cmap)
    shape_gdf.boundary.plot(ax=ax, edgecolor="black", linewidth=1)

    if rios_gdf is not None:
        rios_clip = gpd.clip(rios_gdf, shape_gdf).to_crs(shape_gdf.crs)
        if rio_principal is None:
            rios_clip.plot(ax=ax, edgecolor="blue", linewidth=0.5)
        else:
            rio_main = rios_clip[rios_clip["NORIOCOMP"] == rio_principal]
            rios_rest = rios_clip[rios_clip["NORIOCOMP"] != rio_principal]
            if not rios_rest.empty:
                rios_rest.plot(ax=ax, edgecolor="lightblue", linewidth=0.5)
            if not rio_main.empty:
                rio_main.plot(ax=ax, edgecolor="blue", linewidth=1.2)

    postos_dentro = None
    if plotar_postos and df_postos is not None:
        gdf_postos = df_postos.copy() if isinstance(df_postos, gpd.GeoDataFrame) else \
            gpd.GeoDataFrame(df_postos, geometry=gpd.points_from_xy(df_postos['Longitude'], df_postos['Latitude']), crs="EPSG:4326")
        gdf_postos = gdf_postos.to_crs(shape_gdf.crs)

        postos_dentro = gpd.sjoin(gdf_postos, shape_gdf, how="inner", predicate='within')

        if apenas_postos_selecionados and postos_destaque is not None:
            postos_dentro = postos_dentro[postos_dentro["Codigo"].isin(postos_destaque)]

        if not apenas_postos_selecionados and not postos_dentro.empty:
            ax.scatter(
                postos_dentro.geometry.x,
                postos_dentro.geometry.y,
                color='red',
                s=30,
                label='Postos dentro da bacia'
            )

        if postos_destaque is not None:
            destaque = postos_dentro[postos_dentro["Codigo"].isin(postos_destaque)]
            if not destaque.empty:
                destaque.plot(ax=ax, marker='*', color='gold', markersize=200, label='Postos em destaque')
                print("\nPostos destacados:")
                print(destaque[["Codigo", "Nome"]])

    if localizacao_usina is not None and nome_usina is not None:
        gdf_usina = gpd.GeoDataFrame(
            {"Nome": [nome_usina]},
            geometry=gpd.points_from_xy([localizacao_usina[0]], [localizacao_usina[1]]),
            crs="EPSG:4326"
        ).to_crs(shape_gdf.crs)
        gdf_usina.plot(ax=ax, marker='P', color='green', markersize=100, label=f'Usina {nome_usina}')
        for x, y, label in zip(gdf_usina.geometry.x, gdf_usina.geometry.y, gdf_usina["Nome"]):
            ax.text(x, y, f" {label}", fontsize=10, color="green", weight="bold")

    if titulo:
        ax.set_title(titulo, fontsize=12)
    sm = ScalarMappable(cmap=cmap)
    sm.set_array(out_image)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label("Altura", rotation=270, labelpad=15)

    if salvar and nome_usina:
        pasta_saida = Path("SAIDA_ETAPA_MAPAS")
        pasta_saida.mkdir(parents=True, exist_ok=True)
        nome_arquivo = f"{nome_usina}_com_postos_escolhidos.png"
        plt.savefig(pasta_saida / nome_arquivo, dpi=500, bbox_inches="tight")
        print(f"Mapa salvo em: {pasta_saida / nome_arquivo}")

    ax.legend(loc='upper left')
    plt.show()

    return postos_dentro


def plotar_mapa_shapes_light(
    shapefile_bacia,
    shapefile_rio=None,
    salvar=True,
    titulo=None,
    nome_usina=None,
    rio_principal=None,
    plotar_postos=False,
    df_postos=None,
    input_crs_bln="EPSG:4326"
):
    """
    Mapa "leve" (sem raster) com bacia, rios e postos -- útil no Colab ou
    quando não se quer/precisa gerar o MDT. Filtra e retorna os postos
    dentro da bacia.

    `shapefile_bacia` aceita tanto .shp quanto .bln (ver `ler_geodataframe_bacia`).
    """
    import matplotlib.pyplot as plt

    try:
        shape_gdf = ler_geodataframe_bacia(shapefile_bacia, input_crs_bln=input_crs_bln)
    except (FileNotFoundError, ValueError) as e:
        print(f"Erro: {e}")
        return None
    except Exception as e:
        print(f"Erro ao ler a bacia: {e}")
        return None
    print("CRS da bacia:", shape_gdf.crs)

    rios_gdf = None
    if shapefile_rio is not None:
        shapefile_rio = Path(shapefile_rio)
        if shapefile_rio.exists():
            rios_gdf = gpd.read_file(shapefile_rio)
            rios_gdf = rios_gdf.to_crs(shape_gdf.crs)
        else:
            print(f"Shapefile de rios não encontrado: {shapefile_rio}")

    fig, ax = plt.subplots(figsize=(16, 12))
    shape_gdf.plot(ax=ax, facecolor='#f0f0f0', edgecolor="black", linewidth=1, label='Bacia')

    if rios_gdf is not None:
        rios_clip = gpd.clip(rios_gdf, shape_gdf)
        if not rios_clip.empty:
            if rio_principal is None:
                rios_clip.plot(ax=ax, edgecolor="blue", linewidth=0.5, label='Rios')
            else:
                if "NORIOCOMP" in rios_clip.columns:
                    rio_main = rios_clip[rios_clip["NORIOCOMP"] == rio_principal]
                    rios_rest = rios_clip[rios_clip["NORIOCOMP"] != rio_principal]

                    if not rios_rest.empty:
                        rios_rest.plot(ax=ax, edgecolor="lightblue", linewidth=0.5, label='Afluentes')
                    if not rio_main.empty:
                        rio_main.plot(ax=ax, edgecolor="blue", linewidth=1.2, label=rio_principal)
                    elif rios_rest.empty:
                        print(f"Aviso: Rio principal '{rio_principal}' não encontrado. Plotando todos os rios.")
                        rios_clip.plot(ax=ax, edgecolor="blue", linewidth=0.5, label='Rios')
                else:
                    print("Aviso: Coluna 'NORIOCOMP' não encontrada no shape de rios. Plotando todos os rios.")
                    rios_clip.plot(ax=ax, edgecolor="blue", linewidth=0.5, label='Rios')

    postos_dentro = None
    if plotar_postos and df_postos is not None:
        if not isinstance(df_postos, gpd.GeoDataFrame):
            if 'Longitude' not in df_postos.columns or 'Latitude' not in df_postos.columns:
                print("Erro: df_postos não tem colunas 'Longitude' e 'Latitude'.")
                return None
            geometry = gpd.points_from_xy(df_postos['Longitude'], df_postos['Latitude'])
            gdf_postos = gpd.GeoDataFrame(df_postos, geometry=geometry, crs="EPSG:4326")
        else:
            gdf_postos = df_postos.copy()

        gdf_postos = gdf_postos.to_crs(shape_gdf.crs)

        postos_dentro = gpd.sjoin(gdf_postos, shape_gdf, how="inner", predicate='within')

        if not postos_dentro.empty:
            ax.scatter(
                postos_dentro.geometry.x,
                postos_dentro.geometry.y,
                color='red',
                s=30,
                label='Postos dentro da bacia',
                zorder=5
            )
            if "Codigo" in postos_dentro.columns and "Nome" in postos_dentro.columns:
                print("\nPostos encontrados dentro da bacia:")
                print(postos_dentro[["Codigo", "Nome"]].to_string(index=False))
        else:
            print("Nenhum posto encontrado dentro da bacia.")

    if titulo is not None:
        ax.set_title(titulo, fontsize=12)

    caminho_saida = None
    if salvar and nome_usina is not None:
        pasta_saida = Path("SAIDA_ETAPA_MAPAS")
        pasta_saida.mkdir(parents=True, exist_ok=True)
        nome_arquivo = f"{nome_usina}_todos_os_postos_light.png"
        caminho_saida = pasta_saida / nome_arquivo
        plt.savefig(caminho_saida, dpi=300, bbox_inches="tight")
        print(f"Mapa salvo em: {caminho_saida}")

    ax.legend()
    plt.show()

    return postos_dentro


def plot_mapa_bacia_zoom(
    raster_mosaico_path,
    shapefile_grande,
    shapefile_sub,
    cmap="terrain",
    titulo="Mapa da Bacia com Subbacia em Destaque",
    nome_usina=None,
    localizacao_usina=None,
    plotar_postos=False,
    df_postos=None,
    postos_destaque=None,
    input_crs_bln="EPSG:4326"
):
    """
    Plota o raster (mosaico JÁ PRONTO, gerado por
    `gerar_mosaico_bacia_streaming`) da bacia completa com shapefile da bacia
    maior e subbacia destacada. Lê só a banda necessária -- não refaz mosaico.

    `shapefile_grande` (se não for já um GeoDataFrame) e `shapefile_sub`
    aceitam tanto .shp quanto .bln (ver `ler_geodataframe_bacia`).
    """
    import matplotlib.pyplot as plt
    from rasterio.plot import show
    import rasterio
    import rasterio.errors

    if isinstance(shapefile_grande, gpd.GeoDataFrame):
        shape_grande = shapefile_grande
    else:
        shape_grande = ler_geodataframe_bacia(shapefile_grande, input_crs_bln=input_crs_bln)

    shape_sub = ler_geodataframe_bacia(shapefile_sub, input_crs_bln=input_crs_bln)
    shape_sub = shape_sub.to_crs(shape_grande.crs)

    try:
        with rasterio.open(raster_mosaico_path) as src:
            mosaic = src.read(1)
            out_transform = src.transform
            nodata = src.nodata

        if nodata is not None:
            mosaic = np.where(mosaic == nodata, np.nan, mosaic)

    except rasterio.errors.RasterioIOError as e:
        print(f"Erro ao ler o arquivo raster: {e}")
        return
    except Exception as e:
        print(f"Ocorreu um erro ao processar o raster: {e}")
        return

    fig, ax = plt.subplots(figsize=(12, 12))

    show(mosaic, transform=out_transform, ax=ax, cmap=cmap)
    shape_grande.boundary.plot(ax=ax, edgecolor="black", linewidth=1)

    minx, miny, maxx, maxy = shape_sub.total_bounds
    rect = plt.Rectangle(
        (minx, miny), maxx - minx, maxy - miny,
        linewidth=2, edgecolor='red', facecolor='none', linestyle='--'
    )
    ax.add_patch(rect)

    shape_sub.boundary.plot(ax=ax, edgecolor='blue', linewidth=2)

    if localizacao_usina is not None and nome_usina is not None:
        gdf_usina = gpd.GeoDataFrame(
            {"Nome": [nome_usina]},
            geometry=gpd.points_from_xy([localizacao_usina[0]], [localizacao_usina[1]]),
            crs="EPSG:4674"
        ).to_crs(shape_grande.crs)
        gdf_usina.plot(ax=ax, marker='P', color='black', markersize=100, label=f'Usina {nome_usina}')
        for x, y, label in zip(gdf_usina.geometry.x, gdf_usina.geometry.y, gdf_usina["Nome"]):
            ax.text(x, y, f" {label}", fontsize=10, color="black", weight="bold")

    if plotar_postos and df_postos is not None:
        if not isinstance(df_postos, gpd.GeoDataFrame):
            geometry = gpd.points_from_xy(df_postos['Longitude'], df_postos['Latitude'])
            gdf_postos = gpd.GeoDataFrame(df_postos, geometry=geometry, crs="EPSG:4674")
        else:
            gdf_postos = df_postos.copy()
        gdf_postos = gdf_postos.to_crs(shape_grande.crs)

        postos_dentro = gpd.sjoin(gdf_postos, shape_sub, how="inner", predicate='within')
        if not postos_dentro.empty:
            ax.scatter(
                postos_dentro.geometry.x,
                postos_dentro.geometry.y,
                color='red', s=10, label='Postos dentro da subbacia'
            )
            if postos_destaque is not None:
                destaque = postos_dentro[postos_dentro["Codigo"].isin(postos_destaque)]
                if not destaque.empty:
                    destaque.plot(ax=ax, marker='*', color='gold', markersize=50, label='Postos em destaque')

    ax.set_title(titulo, fontsize=14)
    plt.legend()
    plt.show()


# ============================================================================
# EXEMPLO DE USO (fluxo completo de referência -- não executa nada sozinho)
# ============================================================================

if __name__ == "__main__":
    # 1) Inventário de estações + bacias
    # df_geo_completo = gerar_df_hidro(codSubBacia=58, telemetrica=0)

    # 2) Seleção (opcional) via mapa interativo, ou por shapefile de bacia
    # df_geo = filtro_estacoes("shapes_bacia_trabalho/shape_bacia/minha_bacia.shp", df_postos=df_geo_completo)
    # criar_mapa_selecao_interativo("shapes_bacia_trabalho/shape_bacia/minha_bacia.shp", df_geo, usina_coords=(-22.48, -43.84))
    # df_geo = filtrar_df_por_selecao_json(df_geo, "estacoes_selecionadas.json")

    # 3) Extração paralela das séries de chuva
    # resultados_chuva = extrair_chuva_ana_multiplas(df_geo, nivel_consistencia=[1, 2], max_workers=5)

    # 4) Pré-tratamento
    # resultados_chuva_tratados = processar_dicionario_chuva(resultados_chuva)
    # df_datamining = montar_df_datamining(resultados_chuva_tratados)
    # df_datamining_filtrado = plotar_falhas(df_datamining, "1981-01-01", "1995-12-31", pct_desejada=0.05, gerar_df=True)
    # df_datamining_no_NA = preencher_falhas_vetorizado(df_datamining_filtrado, df_geo, n_vizinhos=3)

    # 5) MDT memory-safe (rode só se precisar dos mapas de elevação)
    # bacias = gerar_regiao("Bacia do Rio Paraíba do Sul", shapes_dir="shapes_IBGE", plotar=False)
    # linha = bacias.iloc[0]
    # resultado_mdt = gerar_mosaico_bacia_streaming(linha.geometry, linha["id"], linha["nome_bacia"])
    pass
