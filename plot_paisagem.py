import pylandstats as pls
import matplotlib.pyplot as plt

# Carrega o raster oficial
ls = pls.Landscape("ls100_06.tif")

# Gera a visualização da paisagem
ls.plot_landscape()

# Salva a imagem
plt.savefig("paisagem_oficial.png", dpi=300)

print("Imagem criada com sucesso!")