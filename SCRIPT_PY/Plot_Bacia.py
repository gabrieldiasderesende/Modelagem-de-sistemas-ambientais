# %% [markdown]
# 

# %%
import geopandas as gpd
from pathlib import Path
from tabulate import tabulate

def listar_bacias_em_shapes(pasta_shapes: str = "shapes"):
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
                    
if __name__ == '__main__':
# Executar
    listar_bacias_em_shapes()



# %%
from tabulate import tabulate

def gerar_regiao(nomes_bacias, shapes_dir="shapes", plotar=True):
    import geopandas as gpd
    from pathlib import Path
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import pandas as pd

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

if __name__ == '__main__':
    resultado_final = gerar_regiao(
        [#"Bacia do Rio Doce",
        "Bacia do Rio Paraíba do Sul",
        # "Bacia do Rio Uruguai"
        ], plotar=True)
    print(resultado_final)


# %%
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
if __name__ == '__main__':
    limites = recortar_limites_bacias(resultado_final)


# %%
import os
import math
import requests
from tqdm import tqdm
from pathlib import Path
from rasterio.io import MemoryFile

def nome_tile(lat, lon):
    ns = 'N' if lat >= 0 else 'S'
    ew = 'E' if lon >= 0 else 'W'
    lat_str = f"{abs(int(lat)):02d}"
    lon_str = f"{abs(int(lon)):03d}"
    return f"Copernicus_DSM_COG_10_{ns}{lat_str}_00_{ew}{lon_str}_00_DEM"

def baixar_tiles_multiplas_bacias(bacias_dict, base_dir=None, save_to_disk=True):
    resultados = {}

    for id_bacia, info in bacias_dict.items():
        nome = info["Nome"]
        limites = {
            'xmin': info["xmin"],
            'ymin': info["ymin"],
            'xmax': info["xmax"],
            'ymax': info["ymax"]
        }

        tiles = []
        xmin, ymin, xmax, ymax = limites["xmin"], limites["ymin"], limites["xmax"], limites["ymax"]
        lat_range = range(math.floor(ymin), math.floor(ymax) + 1)
        lon_range = range(math.floor(xmin), math.floor(xmax) + 1)
        for lat in lat_range:
            for lon in lon_range:
                tiles.append(nome_tile(lat, lon))

        if save_to_disk:
            pasta_bacia = Path(base_dir) / f"{nome.replace(' ', '_')}_ID_{id_bacia}"
            os.makedirs(pasta_bacia, exist_ok=True)
        else:
            pasta_bacia = {}  # Será um dicionário com nome do tile e objeto em memória

        print(f"\nBaixando tiles para a bacia '{nome}' (ID: {id_bacia})")
        for tile in tqdm(tiles, desc=f"Baixando {nome}"):
            url = f"https://copernicus-dem-30m.s3.amazonaws.com/{tile}/{tile}.tif"
            response = requests.get(url)
            if response.status_code == 200:
                if save_to_disk:
                    path_tile = pasta_bacia / f"{tile}.tif"
                    with open(path_tile, "wb") as f:
                        f.write(response.content)
                else:
                    # Armazena o MemoryFile com o dataset aberto
                    memfile = MemoryFile(response.content)
                    dataset = memfile.open()
                    pasta_bacia[tile] = {
                        "dataset": dataset,
                        "memfile": memfile  # manter referência para evitar fechamento
                    }
            else:
                print(f"Erro ao baixar {tile}.tif")

        resultados[id_bacia] = pasta_bacia

    return resultados

# %%
if __name__ == '__main__':
    resultados_em_memoria = baixar_tiles_multiplas_bacias(limites, save_to_disk=False)


# %%
# import matplotlib.pyplot as plt

# print(plt.colormaps())  # lista todos os nomes

# %%
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Point
import operator


def plotar_mapa(
    gdf_bacias,
    salvar_imagem=False,
    rasters_memoria=None,
    filtros_desejados=None,
):
    """
    Plota mapas de elevação para múltiplas bacias com dados em memória.

    Parâmetros:
    - gdf_bacias: GeoDataFrame com as bacias selecionadas
    - salvar_imagem: se True, salva a figura como PNG
    - rasters_memoria: dicionário {id: {tile_id: {'dataset': ..., 'memfile': ...}}}
    - filtro_altura: dicionário com estrutura:
        {
            id: {
                "altura": float,
                "comparacao": str (ex: '>', '<=', '=='),
                "altura_desejada": bool,
                "pontos": [(x1, y1), (x2, y2), ...]  # opcional
            }
        }
    """
    def plotar_cota(ax2, linha, out_meta, out_image, bacia_nome, espacamento_m=100):
        import numpy as np
        import matplotlib.pyplot as plt
        import rasterio
        from pyproj import Geod
        from shapely.geometry import LineString
        from shapely.ops import substring

        # --- cria uma linha mais "densa" interpolando pontos ao longo dela ---
        # calcula comprimento da linha geodésica
        geod = Geod(ellps="WGS84")
        coords = list(linha.coords)

        total_dist_m = 0
        for i in range(1, len(coords)):
            lon1, lat1 = coords[i-1]
            lon2, lat2 = coords[i]
            _, _, dist = geod.inv(lon1, lat1, lon2, lat2)
            total_dist_m += dist

        # calcula número de pontos a cada 'espacamento_m'
        n_pontos = int(total_dist_m // espacamento_m) + 1

        # cria lista com distâncias acumuladas a cada espacamento_m
        dist_acumulada = np.linspace(0, total_dist_m, n_pontos)

        # Função para interpolar pontos ao longo da linha baseada na distância fracionária
        def ponto_em_frac(frac):
            # frac entre 0 e 1
            comprimento_total = linha.length
            return linha.interpolate(frac * comprimento_total)

        # calcula pontos interpolados ao longo da linha (usando fração do comprimento geométrico)
        # Obs: isso é a interpolação linear no sistema de coordenadas (graus), mas para perfil isso é aceitável
        pontos_interp = [ponto_em_frac(d / total_dist_m) for d in dist_acumulada]

        # pega as coordenadas desses pontos (lon, lat)
        coords_interp = [(p.x, p.y) for p in pontos_interp]

        # abre raster em memória para amostragem
        with rasterio.MemoryFile() as memfile:
            with memfile.open(**out_meta) as dataset:
                dataset.write(out_image)
                samples = list(dataset.sample(coords_interp))

        elevations = np.array([s[0] for s in samples])

        # dist_acumulada já está em metros, converte para km para plotar
        dist_km = dist_acumulada / 1000

        # plota perfil mais contínuo
        ax2.plot(dist_km, elevations, color='red')
        ax2.set_xlabel('Distância ao longo da linha (km)')
        ax2.set_ylabel('Elevação (m)')
        ax2.set_title(f'Perfil de Elevação - {bacia_nome}')
        ax2.grid(True)



    OPERADORES = {
        '==': operator.eq,
        '>': operator.gt,
        '<': operator.lt,
        '>=': operator.ge,
        '<=': operator.le
    }
    resultados = {}
    for _, linha in gdf_bacias.iterrows():
        bacia_id = linha['id']
        bacia_nome = linha['nome_bacia']
        print(f"\nProcessando: {bacia_nome} (ID: {bacia_id})")

        bacia = gpd.GeoDataFrame(geometry=[linha.geometry], crs=gdf_bacias.crs)

        if not rasters_memoria or bacia_id not in rasters_memoria:
            print(f"Raster não encontrado na memória para bacia ID {bacia_id}.")
            continue

        raster_dict = rasters_memoria[bacia_id]
        rasters = [r['dataset'] for r in raster_dict.values()]

        mosaic, out_trans = merge(rasters)
        out_meta = rasters[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "count": mosaic.shape[0]
        })

        # # Verifica e exibe o CRS do shapefile e do raster
        # print(f"CRS do shapefile (gdf_bacias): {gdf_bacias.crs}")
        # print(f"CRS do primeiro raster da bacia {bacia_id}: {rasters[0].crs}")

        bacia = bacia.to_crs(rasters[0].crs)
        geoms = [feature.__geo_interface__ for feature in bacia.geometry]

        with rasterio.MemoryFile() as memfile:
            with memfile.open(**out_meta) as dataset:
                dataset.write(mosaic)
                out_image, out_transform = mask(dataset=dataset, shapes=geoms, crop=True)

        img = out_image[0].astype(float)
        img[img == 0] = np.nan

        if filtros_desejados and bacia_id in filtros_desejados:
            dados_filtro = filtros_desejados[bacia_id]
            if dados_filtro.get("altura_desejada"):
                altura = dados_filtro.get("altura")
                comp = dados_filtro.get("comparacao")
                operador = OPERADORES.get(comp)
                if operador:
                    mask_altura = operador(img, altura)
                    img[~mask_altura] = np.nan
                else:
                    print(f"Operador '{comp}' inválido para bacia {bacia_id}")

        alt_min = np.nanmin(img)
        alt_max = np.nanmax(img)
        print(f"Altitude mínima: {alt_min:.2f} m")
        print(f"Altitude máxima: {alt_max:.2f} m")
        
        bounds = rasterio.transform.array_bounds(img.shape[0], img.shape[1], out_transform)
        extent = [bounds[0], bounds[2], bounds[1], bounds[3]]

        cor_desejada = filtros_desejados.get(bacia_id, {}).get("cor_desejada", "gist_earth")
        usar_cor_personalizada = filtros_desejados.get(bacia_id, {}).get("aplicar_cor_desejada", False)

        cmap = cor_desejada if usar_cor_personalizada else "gist_earth"

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(img, cmap=cmap, extent=extent)
        bacia.boundary.plot(ax=ax, edgecolor='black', linewidth=1.5)

        
        # Verifica se deve traçar os pontos para a bacia atual
        linha = None
        if filtros_desejados.get(bacia_id, {}).get("tracar_pontos", False):
            pontos_info = filtros_desejados.get(bacia_id, {}).get("pontos", None)
            if pontos_info:
                # Filtra os pontos que estão dentro da bacia
                pontos_validos = [
                    pt for pt in pontos_info
                    if bacia.geometry.iloc[0].contains(Point(pt))
                ]

                if len(pontos_validos) >= 2:
                    linha = LineString(pontos_validos)
                    linha_gdf = gpd.GeoDataFrame(geometry=[linha], crs=bacia.crs)
                    linha_gdf.plot(ax=ax, color='red', linewidth=2)
                elif 0 < len(pontos_validos) < 2:
                    print(f"Bacia ID {bacia_id}: menos de dois pontos válidos, linha não traçada.")
                else:
                    print(f"Bacia ID {bacia_id}: todos os pontos estão fora da bacia, linha ignorada.")
        
        plt.colorbar(im, ax=ax, label='Elevação (m)')
        ax.set_title(f"{bacia_nome}: Cotas")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.tick_params(axis='both', labelsize=10)

        if dados_filtro.get("plotar_cota", False):
            if linha:
                fig2, ax2 = plt.subplots(figsize=(10, 5))
                plotar_cota(ax2, linha, out_meta, out_image, bacia_nome)
                plt.show()
            else:
                print(f"Bacia ID {bacia_id}: não há linha válida para traçar o perfil de elevação.")

        if salvar_imagem:
            nome_arquivo = f"{bacia_nome.replace(' ', '_')}_mapa.png"
            plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
            print(f"Imagem salva: {nome_arquivo}")

        plt.show()

        resultados[bacia_id] = {
            "id": bacia_id,
            "nome": bacia_nome,
            "shape": bacia,
            "mosaic": np.squeeze(mosaic),
            "contorno": bacia.geometry.iloc[0].boundary,
            "meta": {
                "transform": out_trans,
                "crs": out_meta["crs"],
                "width": mosaic.shape[2],
                "height": mosaic.shape[1]
            }
        }

    return resultados

# Exemplo de uso:
# filtro_altura agora pode conter também os pontos para cada bacia
if __name__ == '__main__':
    filtros = {
        5: {
            "altura": 1200,
            "comparacao": ">=",
            "altura_desejada": False,
            "pontos": 
            [
                (-43.5, -21.00), 
                (-42.50, -20.00), 
                (-42.00, -19.00),
                (-41.5, -19.25),
                (-41.0, -19.5),
                (-40.5, -19.5),
                (-40.0, -19.25)
            ],
            "tracar_pontos": True,
            "plotar_cota": True,
            "cor_desejada": 'magma',
            "aplicar_cor_desejada": False,
        },
        97: {
            "altura": 2000,
            "comparacao": ">=",
            "altura_desejada": False,
            "pontos": 
            [
                (-46.0, -23.25),
                (-45.0, -22.75),
                (-44.0, -22.75),
                (-43.0, -22.25),
                (-42.0, -21.75),
                (-42.3, -21.25)
            ],
            "tracar_pontos": True,
            "plotar_cota": True,
            "cor_desejada": 'CMRmap',
            "aplicar_cor_desejada": False,
        }
    }

    dados_bacias = plotar_mapa(
        gdf_bacias=resultado_final,
        salvar_imagem=False,
        rasters_memoria=resultados_em_memoria,
        filtros_desejados=filtros
    )



