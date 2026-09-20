# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 12:26:31 2026

@author: thela
"""

import os
import sys
import csv
import warnings
import glob
import shutil
import string
import subprocess
from os import listdir
from os.path import isfile, join
from datetime import datetime, timedelta
from datetime import time as dtime
from itertools import combinations
from tqdm import tqdm
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker, colors, dates
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from brokenaxes import brokenaxes
import seaborn as sns
from scipy.stats import linregress, gaussian_kde
from scipy import stats
import qualRpy.qualR as qr

import torch
import torchvision
from torchvision import transforms

# =============================================================================
# 0. CONFIGURATIONS & PATHS
# =============================================================================
BASE_SMPS_PATH = 'E:/totais/txts_smps/10.2 - 429.4'
MASK_NPF_PATH = r'C:\Users\thela\maskNPF'
OUTPUT_CONTOURS_PATH = 'E:/totais/contornos_diarios'

sys.path.append(MASK_NPF_PATH)

from utils import get_instance_segmentation_model
from utils import num2time, time2num, mkdirs
from utils import psd2im, draw_subplots, reshape_mask
from utils import get_SE, get_GR, get_GR_old, convert_matlab_time

# Configurações gerais
pd.options.display.max_columns = None
font = {'family': 'Arial'}
matplotlib.rc('font', **font)
warnings.filterwarnings("ignore")

os.chdir(BASE_SMPS_PATH)
path = BASE_SMPS_PATH
save_path = BASE_SMPS_PATH
pd.options.display.max_columns = None

font = {'family': 'Arial'}
matplotlib.rc('font', **font)

def calculate_penetration_diffusion(dp_nm, L=6.35, Q_lpm=1.0, T=298.15):
    """
    Calcula a eficiência de penetração de aerossóis em uma linha cilíndrica
    devido à difusão, baseado na equação de Gormley & Kennedy (1948).

    Parâmetros:
    dp_nm : Diâmetro da partícula em nm
    L     : Comprimento do tubo em metros (padrão: 6.35 m)
    Q_lpm : Fluxo de aerossol em L/min (padrão: 1.0 L/min)
    T     : Temperatura em Kelvin (padrão 298.15 K)
    """
    # Constantes físicas
    k_B = 1.380649e-23      # Constante de Boltzmann (J/K)
    mu = 1.81e-5            # Viscosidade dinâmica do ar (kg/(m*s))
    lambda_air = 6.65e-8    # Caminho livre médio do ar (m)

    # Conversões
    dp_m = dp_nm * 1e-9           # nm para metros
    Q_m3s = Q_lpm * 1e-3 / 60.0   # L/min para m³/s

    # Número de Knudsen e Fator de Correção de Cunningham (Cc)
    Kn = 2 * lambda_air / dp_m
    Cc = 1 + Kn * (1.142 + 0.558 * np.exp(-0.999 / Kn))

    # Coeficiente de Difusão (D) [m²/s]
    D = (k_B * T * Cc) / (3 * np.pi * mu * dp_m)

    # Parâmetro adimensional de deposição (mi)
    # Nota: A fórmula de Gormley & Kennedy tradicional usa mi = (pi * D * L) / Q
    mi = (np.pi * D * L) / Q_m3s

    # Eficiência de penetração (eta)
    if mi < 0.02:
        eta = 1 - 2.56 * (mi ** (2/3)) + 1.2 * mi + 0.177 * (mi ** (4/3))
    else:
        eta = 0.819 * np.exp(-3.657 * mi) + 0.097 * np.exp(-22.3 * mi) + 0.032 * np.exp(-57 * mi)

    return max(eta, 0.01) # Limita a no mínimo 1% de penetração para evitar divisão por zero

warnings.filterwarnings("ignore")
# =============================================================================
# 2. FUNÇÕES DO SMPS
# =============================================================================

def get_header_value(file_path, key):
    # Alterado de "utf-8-sig" para "unicode_escape" ou "latin1" para suportar caracteres especiais do SMPS
    with open(file_path, encoding="unicode_escape", newline="") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) >= 2 and row[0].strip() == key:
                return row[1].strip()

    return None

def filter_smps(df, file_path):
    """ Filtra condições de erro passo a passo e imprime amostras dos dados rejeitados """

    total_inicial = len(df)
    if total_inicial == 0:
        return df

    print(f"\n--- Filtrando arquivo SMPS ({total_inicial} linhas iniciais) ---")

    # 1. Filtro de Concentração Total
    # Busca dinamicamente qual é o nome da coluna de concentração total neste arquivo
    col_conc = None
    for col in df.columns:
        if 'Total Conc' in col or 'Total Concentration' in col:
            col_conc = col
            break

    if col_conc is not None:
        mask_conc = (df[col_conc] >= 2.0) & (df[col_conc] < 10e6)
        rejeitados = df[~mask_conc]
        if len(rejeitados) > 0:
            amostra = rejeitados[col_conc].dropna().head(5).tolist()
            print(f"Após Concentração: Corte de {len(rejeitados)} linhas.")
            print(f"   -> Exemplo de valores rejeitados: {amostra}")
        df = df[mask_conc]
    else:
        print("⚠️ Coluna de Concentração Total não encontrada! Pulando este filtro...")

    # 2. Filtro de Sheath Flow (agora dinâmico)
    col_sheath = next((col for col in df.columns if 'Sheath Flow' in col), None)
    if col_sheath is not None:
        mask_sheath = (df[col_sheath] > 3) & (df[col_sheath] < 12)
        rejeitados = df[~mask_sheath]
        if len(rejeitados) > 0:
            amostra = rejeitados[col_sheath].dropna().head(5).tolist()
            print(f"Após Sheath Flow: Corte de {len(rejeitados)} linhas.")
            print(f"   -> Exemplo de valores rejeitados: {amostra}")
        df = df[mask_sheath]
    else:
        print("⚠️ Coluna de Sheath Flow não encontrada! Pulando este filtro...")

    # 3. Filtro de Impactor / Aerosol / Sample Flow
    col_impactor = next((col for col in df.columns if 'Impactor Flow' in col or 'Aerosol Flow' in col or 'Sample Flow' in col), None)
    if col_impactor is not None:
        mask_impactor = (df[col_impactor] > 0.3) & (df[col_impactor] < 2)
        rejeitados = df[~mask_impactor]
        if len(rejeitados) > 0:
            amostra = rejeitados[col_impactor].dropna().head(5).tolist()
            print(f"Após Fluxo de Aerosol/Impactor: Corte de {len(rejeitados)} linhas.")
            print(f"   -> Exemplo de valores rejeitados: {amostra}")
        df = df[mask_impactor]
    else:
        print("⚠️ Coluna de Impactor/Aerosol Flow não encontrada! Pulando este filtro...")

    # 4. Filtro de Detector Status
    col_detector = next((col for col in df.columns if 'Detector Status' in col), None)
    if col_detector is not None:
        mask_detector = (df[col_detector] == 'Normal Scan')
        rejeitados = df[~mask_detector]
        if len(rejeitados) > 0:
            amostra = rejeitados[col_detector].dropna().head(5).tolist()
            print(f"Após Detector Status: Corte de {len(rejeitados)} linhas.")
            print(f"   -> Exemplo de status rejeitados: {amostra}")
        df = df[mask_detector]

    # 5. Filtro de Classifier Errors (Mantido comentado como no original)
    # col_classifier = next((col for col in df.columns if 'Classifier Errors' in col), None)
    # if col_classifier is not None:
    #     mask_classifier = (df[col_classifier] == 'Normal Scan')
    #     df = df[mask_classifier]

    # 6. Filtro de Neutralizer Status
    col_neutralizer = next((col for col in df.columns if 'Neutralizer Status' in col), None)
    if col_neutralizer is not None:
        mask_neutralizer = (df[col_neutralizer] == 'ON')
        rejeitados = df[~mask_neutralizer]
        if len(rejeitados) > 0:
            amostra = rejeitados[col_neutralizer].dropna().head(5).tolist()
            print(f"Após Neutralizer Status: Corte de {len(rejeitados)} linhas.")
            print(f"   -> Exemplo de status rejeitados: {amostra}")
        df = df[mask_neutralizer]

    # --- Variáveis de Cabeçalho Restauradas ---
    # (Elas usam a variável global 'file_path' declarada no seu loop principal)
    mcc = get_header_value(file_path, "Multiple Charge Correction")
    diffusion = get_header_value(file_path, "Diffusion Loss Correction")
    sampling = get_header_value(file_path, "Sampling System Particle Loss Correction")

    if mcc is not None:
        if mcc.lower() == "true":
            print("✓ Multiple Charge Correction: Ativo")
        else:
            print("⚠ Multiple Charge Correction: DESATIVADO")

    if diffusion is not None:
        if diffusion.lower() == "true":
            print("✓ Diffusion Loss Correction: Ativo")
        else:
            print("⚠ Diffusion Loss Correction: DESATIVADO")

    if sampling is not None:
        if sampling.lower() == "on":
            print("✓ Sampling System Particle Loss Correction: Ativo")
        else:
            print("⚠ Sampling System Particle Loss Correction: DESATIVADO")

    print(f"*** Linhas restantes válidas: {len(df)} ***")
    print("-" * 50)

    return df

def apply_stp(df):
    """ Aplica correção STP baseada em Temperatura e Pressão do dia """

    # 1. Busca dinâmica da coluna de Temperatura (aceitando 'Sample Temp')
    col_temp = next((c for c in df.columns if 'Temp' in c and ('Sheath' in c or 'Aerosol' in c or 'C' in c or 'Sample' in c)), None)

    if col_temp is not None:
        temp = pd.to_numeric(df[col_temp], errors='coerce')
    else:
        print("⚠️ Coluna de Temperatura não encontrada! Assumindo 20°C para o cálculo STP.")
        temp = pd.Series(20.0, index=df.index)

    # 2. Busca dinâmica da coluna de Pressão (aceitando 'Sample Pressure')
    col_press = next((c for c in df.columns if 'Press' in c or 'kPa' in c), None)

    if col_press is not None:
        pressure = pd.to_numeric(df[col_press], errors='coerce')
    else:
        print("⚠️ Coluna de Pressão não encontrada! Assumindo 101.325 kPa para o cálculo STP.")
        pressure = pd.Series(101.325, index=df.index)

    # 3. Calcula o fator STP
    df['STP'] = (1013.25 * (temp + 273.15)) / (293.15 * (pressure * 10))

    # 4. Encontra as colunas de partículas (Garante que é número puro, ex: '14.1', ignorando '_14.1')
    psd_cols = []
    for c in df.columns:
        # Se a coluna começar com underline (ex: _10.09 do csv), nós ignoramos
        if str(c).startswith('_'):
            continue
        try:
            float(c) # Só aceita se for convertível para número puro
            psd_cols.append(c)
        except ValueError:
            pass

    # 5. Aplica a correção STP em todas as colunas de partículas numéricas encontradas
    for col in psd_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce') * df['STP']

    df.drop(columns=['STP'], inplace=True)
    return df, psd_cols

# =============================================================================
# 3. PROCESSAMENTO DOS DADOS DO SMPS
# =============================================================================
print("--- Processando SMPS ---")
smps_path = BASE_SMPS_PATH
os.chdir(smps_path)

file_list_smps = [f for f in os.listdir(smps_path) if f.lower().endswith(('.csv', '.txt'))]
dfs_smps = []

for file_name in file_list_smps:
    print(file_name)
    file_path = os.path.join(smps_path, file_name)
    # 1. Escolhe o delimitador certo baseado na extensão do arquivo
# 1. Define o delimitador com base na extensão
    delimitador = ',' if file_name.lower().endswith('.csv') else '\t'

    # 2. Abre o arquivo rapidamente apenas para descobrir onde está o cabeçalho
    header_line = 0
    with open(file_path, 'r', encoding='unicode_escape', errors='ignore') as f:
        for i, line in enumerate(f):
            # Procura pela palavra 'Sample' que indica o início das colunas
            if line.startswith('Sample #') or line.startswith('"Sample #"') or 'Date' in line:
                header_line = i
                break

    # 3. Agora lê o arquivo no Pandas pulando o número EXATO de linhas de texto inútil
    df_s = pd.read_csv(file_path, sep=delimitador, encoding='unicode_escape', on_bad_lines='skip', skiprows=header_line)

    # 4. Limpa espaços extras nos nomes das colunas
    df_s.columns = df_s.columns.str.strip()

    # 5. Lógica da Data e Hora
    if 'DateTime Sample Start' in df_s.columns:
        df_s['datetime'] = pd.to_datetime(df_s['DateTime Sample Start'], errors='coerce')
    elif 'Date' in df_s.columns and 'Start Time' in df_s.columns:
        df_s['datetime'] = pd.to_datetime(df_s['Date'] + ' ' + df_s['Start Time'], errors='coerce')
    else:
        print(f"⚠️ Atenção: Pulando o arquivo {file_name}. Colunas de data não encontradas.")
        print(f"Colunas lidas: {df_s.columns.tolist()[:10]}...")
        continue

    df_s = df_s.dropna(subset=['datetime']).set_index('datetime')

    df_s = filter_smps(df_s, file_path)
    df_s, psd_cols = apply_stp(df_s)
    dfs_smps.append(df_s[psd_cols].copy())

merged_smps = pd.concat(dfs_smps).sort_index()

# =============================================================================
# --- APLICAÇÃO DA CORREÇÃO DE PERDAS POR DIFUSÃO ---
# =============================================================================
print("--- Aplicando correção teórica de perdas por difusão nas linhas ---")
colunas_corrigidas = 0
for col in merged_smps.columns:
    try:
        # Tenta converter o nome da coluna para float (diâmetro em nm)
        dp_nm = float(col)

        # Calcula a penetração (L=6.35 m, Q=1.0 L/min como descrito na dissertação)
        eta = calculate_penetration_diffusion(dp_nm, L=6.35, Q_lpm=1.0)

        # A concentração "real" é a medida dividida pela taxa de penetração
        merged_smps[col] = merged_smps[col] / eta
        colunas_corrigidas += 1
    except ValueError:
        # Se a coluna não for um número (ex: nome de outra variável), ele ignora
        pass

print(f"--- Correção aplicada em {colunas_corrigidas} colunas de diâmetro ---")
# =============================================================================

stats = pd.DataFrame({
    "median": merged_smps.median(),
    "p99": merged_smps.quantile(0.99),
    "max": merged_smps.max()
})

stats["ratio"] = stats["max"] / stats["median"]
stats["ratio99"] = stats["p99"] / stats["median"]

# Aumentamos o limite para 5000 para preservar os picos reais de NPF
bad_cols = stats.loc[
    (stats["ratio"] > 5000) |
    (stats["ratio99"] > 500)
].index.tolist()

print(f"Colunas com erro de leitura (apagadas): {bad_cols}")
stats["ratio99"] = stats["p99"] / stats["median"]

bad_cols = stats.loc[
    (stats["ratio"] > 100) |
    (stats["ratio99"] > 50)
].index.tolist()

merged_smps[bad_cols] = np.nan
merged_smps = merged_smps.interpolate(axis=1)

# Remover linhas duplicadas (caso existam)
merged_smps = merged_smps.drop_duplicates()

merged_smps.replace('nan', np.nan, inplace=True)
merged_smps = merged_smps.dropna(axis=0, how='all')
merged_smps = merged_smps.dropna(axis=1, how='all')

# 2. Em vez de deletar a coluna toda por causa de uma mudança de configuração,
# preencha os espaços vazios (NaN) com 0. Assim as partículas < 100 nm do
# Dia 1 não são perdidas só porque o Dia 2 não as mediu!
merged_smps = merged_smps.fillna(0)

# 3. Agora a conversão para int não vai dar erro, pois não há mais NaNs
try:
    merged_smps = merged_smps.astype(int)
except ValueError as e:
    print(f"Erro ao converter para inteiros: {e}")
path_part = path.split('/')[-1].split(' - ')[-1]

# Construir o nome do arquivo final usando o 'save_path' e o 'path_part'
final_path = os.path.join(save_path, 'columns_merged' + path_part + '.csv')
index_path = os.path.join(save_path, 'index_merged' + path_part + '.csv')
# Criar a coluna com o nome '0' e valores de 1 até o comprimento do DataFrame
merded_df_2=merged_smps.copy()
merded_df_2.insert(0, '0', range(1, len(merged_smps) + 1))

# Reordenar as colunas do DataFrame para mover a coluna '0' para a primeira posição
merded_df_2 = merded_df_2[["0"] + [col for col in merded_df_2.columns if col != "0"]]

# Salvar o DataFrame com a coluna '0' como a primeira coluna
merded_df_2.to_csv(final_path, index=False)
merded_df_2.index.to_series().to_csv(index_path, index=False)


path = BASE_SMPS_PATH

pd.options.display.max_columns = None

font = {'family': 'Arial'}
matplotlib.rc('font', **font)

warnings.filterwarnings("ignore")

model = get_instance_segmentation_model()
modelfp = os.path.join(MASK_NPF_PATH, 'checkpoints/maskrcnnfull.pth')

# Verificar se CUDA está disponível
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Carregar o modelo para o dispositivo apropriado
model.load_state_dict(torch.load(modelfp, map_location=device))
model.to(device)
model.eval()
plt.rcParams.update({'font.size': 20})

def psd2im(df,
           ax=None,
           fig=None,
           mask=None,
           savefp=None,
           dpi=600,
           n_xticks=5,
           figsize=(1.6, 1.2),
           show_figure=True,
           use_title=False,
           fit_data=None,
           line_data=None,
           index=None,
           vmax=None,
           lcolor='white',
           lwidth=3,
           use_xaxis=True,
           use_xlabel=False,
           use_yaxis=True,
           use_cbar=False,
           ftsize=22
           ):
    r"""Draw single or multiple surface plots.

    Parameters:
        df (dataframe)     --  particle size distribution data for one day or multiple days
        ax (ax)            --  specify the ax to visualize the psd. If not specified, a new one will be created
        fig (fig)          --  the whole figure
        mask (array)       --  numpy array with the same shape as the input psds
        savefp (str)       --  path for storing the figures
        dpi (int)          --  default is 600
        n_xticks (int)     --  how many ticklabels shown on the x-axis
        figsize (tuple)    --  used only if a new ax is created
        show_figure (bool) --  clear all the figures if drawing many surface plots
        use_title (bool)   --  use the date as the title for the psd
        fit_data (list)    --  the fitted time points and related Dps
        line_data (list)   --  the lines to show the determined GRs
        index (int)        --  there many be more than one masks detected for one day's psd
        vmax (float|none)  --  color scale for visualization, default is None.
        lcolor (str)       --  color for visualizing the GR
        lwidth (int)       --  linewidth
        use_xaxis (bool)   --  whether to draw the x-axis
        use_yaxis (bool)   --  whether to draw the y-axis
        use_cbar (bool)    --  whether to use the colorbar
        ftsize (int)       --  fontsize for plotting

    """

    # get the psd data
    dfc = df.copy(deep=True)    # get a copy version
    df_min = np.nanmin(dfc.replace(0, np.nan))    # find the minimul value
    # use the minimul value to replace the na values
    dfc.fillna(df_min, inplace=True)
    dfc[dfc == 0] = df_min               # use the minimul value to replace 0
    dfc = dfc.replace(0, df_min)

    # use the dynamic vmax
    if vmax is None:
        max_val = np.nanmax(dfc.values)
        # check the number of digits
        n_digits = len(str(int(max_val)))
        vmax = np.power(10, n_digits)

    values = dfc.values.T if mask is None else (
        dfc.values*mask).T   # values for visualization
    dps = [float(dp) for dp in list(dfc.columns)]    # Dps
    tm = dfc.index.values    # time points

    # check how many days of data to be shown
    whole_dates = np.unique([item.date() for item in df.index])
    num_days = (whole_dates[-1] - whole_dates[0]).days + 1

    # once the ax is none, create a new one
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    im = ax.pcolormesh(tm,    # time points
                       dps,    # particle sizes
                       values,  # distribution
                       norm=colors.LogNorm(vmin=1e1, vmax=vmax),
                       cmap='nipy_spectral',
                       shading='auto')

    # add the fitted line for determinating the GRs
    if fit_data is not None:
        ax.scatter(fit_data[0], fit_data[1], c='k', s=15, marker='o')

    if line_data is not None:
        ax.plot(line_data[0], line_data[1], c=lcolor, linewidth=lwidth)

    # use the log scale for y-axis
    ax.set_yscale('log')

    # Ensure specific y-axis labels are shown
    y_ticks = [10, 20, 50, 100, 200, 300, 400, 500, 600,700,800]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{tick}' for tick in y_ticks])
    ax.tick_params(axis='y', labelsize=20)

    # get title
    title = str(whole_dates[0]) if num_days <= 1 else str(
        whole_dates[0])+'_'+str(whole_dates[-1])

    # add index
    if index is not None:
        title = title + f' {index}'

    # add the title on the figure
    if use_title:
        ax.set_title(title, fontsize=ftsize+6, fontweight='bold')

    # add y-axis
    if use_yaxis:
        ax.set_ylabel(r'$\mathrm{D_p}$ (nm)', fontsize=ftsize+2)
    else:
        ax.get_yaxis().set_visible(False)

    # add x-axis
    if use_xaxis:
        xtick = [datetime(sdate.year, sdate.month, sdate.day) + timedelta(i/(n_xticks-1)) for sdate in whole_dates
                 for i in range(n_xticks-1)] + [datetime(whole_dates[-1].year, whole_dates[-1].month, whole_dates[-1].day)+timedelta(1)]
        xtick_labels = ['00', '02','04', '06','08','10','12','14','16','18','20','22'] * num_days + ['00']
        ax.set_xticks(xtick)
        ax.set_xticklabels(xtick_labels)
        ax.tick_params(axis='x', labelsize=22)
    else:
        ax.get_xaxis().set_visible(False)

    if use_xlabel:
        ax.set_xlabel('Local time (h)', fontsize=22)

    # add colorbar
    if use_cbar:
        # here fig is the default input for subplots
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(
            r'dN/dlog$\mathrm{D_p} (\mathrm{cm}^{-3})$', fontsize=ftsize+2)

    # to avoid the black edges
    if (not use_xaxis) and (not use_yaxis):
        ax.set_axis_off()

    # save the currect figure
    if (savefp is not None) and (not use_xaxis) and (not use_yaxis):
        fig.savefig(os.path.join(savefp, title + '.png'),
                    bbox_inches='tight', pad_inches=0, dpi=dpi)
    elif (savefp is not None) and (use_xaxis or use_yaxis):
        fig.savefig(os.path.join(savefp, title + '.png'),
                    bbox_inches='tight', pad_inches=0.1, dpi=dpi)

    if not show_figure:
        plt.cla()
        plt.clf()
        plt.close('all')
    return im

df_ext = merged_smps.dropna()

logDp_values = pd.to_numeric(df_ext.columns, errors='coerce')

# Supondo que df_ext e logDp_values já estão definidos
total_conc = []
gmd = []
gstd = []
mode = []
median = []

# Inicializar a lista para armazenar concentrações totais
total_conc = []

# Calcular a concentração total
for index, row in df_ext.iterrows():
    mean_concentration = row
    dlogDp = []  # Inicializa a lista de delta_log para cada linha

    # Calcula os deltas dos logs dos diâmetros (usando log natural)
    for i in range(len(logDp_values) - 1):
        delta_log = np.log10(logDp_values[i + 1]) - np.log10(logDp_values[i])
        dlogDp.append(delta_log)

    dlogDp = np.array(dlogDp)

    # Calcular a concentração total e adicionar à lista
    integral_N = np.sum(mean_concentration.values[:-1] * dlogDp)
    total_conc.append(integral_N)
    # Verificar se integral_N não é zero para evitar divisão por zero
    if integral_N == 0:
        gmd.append(np.nan)
        gstd.append(np.nan)
        mode.append(np.nan)
        median.append(np.nan)
        continue

    # Calcular o diâmetro médio geométrico (usando log natural)
    geometric_mean_diameter = np.exp(np.sum(np.log(logDp_values[:-1]) * mean_concentration.values[:-1] * dlogDp) / integral_N)
    gmd.append(geometric_mean_diameter)

    # Calcular o diâmetro médio ponderado (usando log natural)
    Dp_weighted_mean = np.exp(np.sum(np.log(logDp_values[:-1]) * mean_concentration.values[:-1] * dlogDp) / integral_N)

    # Calcular o termo dentro da raiz quadrada da fórmula do GSD (usando log natural)
    squared_diff = (np.log(logDp_values[:-1]) - np.log(Dp_weighted_mean)) ** 2

    # Calcular a média do termo dentro da raiz quadrada
    squared_diff_mean = np.sum(squared_diff * mean_concentration.values[:-1] * dlogDp) / integral_N

    # Calcular o GSD
    geometric_standard_deviation = np.exp(np.sqrt(squared_diff_mean))
    gstd.append(geometric_standard_deviation)

    # Calcular a concentração média para cada intervalo de diâmetro de partícula
    delta_logDp = np.diff(np.log(logDp_values)).mean()
    concentration_mean = mean_concentration / delta_logDp

    # Identificar o intervalo com a maior concentração média (moda)
    moda = concentration_mean.idxmax()
    mode.append(moda)

    # Calcular a mediana
    sorted_diametros = np.sort(logDp_values[:-1])
    cdf = np.cumsum(mean_concentration.values[:-1]) / np.sum(mean_concentration.values[:-1])
    mediana_index = np.argmax(cdf >= 0.5)
    mediana = sorted_diametros[mediana_index]
    median.append(mediana)

# Criar um novo DataFrame com as variáveis calculadas
df_ext_stat = pd.DataFrame({
    'Total Conc. (#/cm³)': total_conc,
    'Geo. Mean (nm)': gmd,
    'Geo. Std. Dev.': gstd,
    'Mode (nm)': mode,
    'Median (nm)': median
}, index=df_ext.index)
df_ext_stat = df_ext_stat.apply(pd.to_numeric, errors='coerce')

# Verificar se houve erros na conversão
if df_ext_stat.isnull().values.any():
    print("Erro: Não foi possível converter todos os dados para tipo numérico float.")
else:
    print("Todos os dados foram convertidos com sucesso para tipo numérico float.")

def extrair_valor(coluna):
    try:
        # Tente converter o nome da coluna em um valor float
        return float(coluna)
    except ValueError:
        # Se não for possível converter, retorne 0.0
        return 0.0

# Ordene as colunas com base nos valores extraídos dos nomes das colunas
df_ext = df_ext[sorted(df_ext.columns, key=extrair_valor)]

dfs_por_dia = []

# Iterar sobre os dias e gerar uma DataFrame para cada dia
for day, df_day in df_ext.groupby(df_ext.index.date):
    dfs_por_dia.append(df_day)

# Função para remover linhas com 4 ou mais zeros consecutivos
def remove_linhas_com_zeros(df, limite=4):
    mask = []
    arr = df.to_numpy()
    for row in arr:
        # Converte a linha em um array booleano: True onde o valor é zero
        zeros = (row == 0)
        # Verifica se há sequência de "limite" ou mais zeros consecutivos
        if np.any(np.convolve(zeros, np.ones(limite, dtype=int), mode='valid') == limite):
            mask.append(False)  # descartar linha
        else:
            mask.append(True)   # manter linha
    return df[mask]

# Aplicar a função em todos os DataFrames da lista
#dfs_por_dia = [remove_linhas_com_zeros(df) for df in dfs_por_dia]

df_ext_stat=df_ext_stat.resample("30T").mean()
# Supondo que df_ext_stat tenha 5 colunas e datetime no index
num_columns = len(df_ext_stat.columns)  # Conta o número de colunas no DataFrame

# Cria uma figura e um array de eixos com base no número de colunas
fig, axs = plt.subplots(num_columns, 1, figsize=(10, 15), sharex=True)

# Caso haja apenas um subplot, axs não será uma lista, então garantimos que seja uma lista
if num_columns == 1:
    axs = [axs]

# Percorre as colunas do DataFrame e plota cada uma em um subplot
for i, column in enumerate(df_ext_stat.columns):
    axs[i].plot(df_ext_stat.index, df_ext_stat[column])
    axs[i].set_ylabel(column)
    axs[i].grid(True)

# Configurações finais
axs[-1].set_xlabel('Data e Hora')  # Seta o rótulo do eixo x no último subplot
plt.tight_layout()  # Ajusta o layout para evitar sobreposição de elementos
plt.show()

# Certifique-se de que 'Total Conc. (#/cm³)' está no DataFrame
if 'Total Conc. (#/cm³)' in df_ext_stat.columns:

    df_boxplot_seaborn = df_ext_stat[['Total Conc. (#/cm³)']].copy()
    df_boxplot_seaborn['day'] = df_ext_stat.index.date
    df_boxplot_seaborn = df_boxplot_seaborn.replace(
    [np.inf, -np.inf],
    np.nan
    )

    df_boxplot_seaborn = df_boxplot_seaborn.dropna(
    subset=['Total Conc. (#/cm³)']
    )
    # Configurar estilo do Seaborn
    sns.set(style="whitegrid")

    # Criar a figura
    plt.figure(figsize=(12, 6))

    # Criar boxplot com seaborn
    sns.boxplot(
        x='day',
        y='Total Conc. (#/cm³)',
        data=df_boxplot_seaborn,
        showfliers=False
    )

    # Ajustar rótulos e título
    plt.title('Daily Boxplot of Total Concentration (#/cm³)', fontsize=16, fontweight='bold')
    plt.xlabel('Day', fontsize=14)
    plt.ylabel('Total Conc. (#/cm³)', fontsize=14)

    plt.xticks(rotation=45)  # Rotaciona os rótulos dos dias no eixo x
    plt.tight_layout()

    plt.show()

else:
    print("Erro: A variável 'Total Conc. (#/cm³)' não está no DataFrame.")

for df in dfs_por_dia:
    # Percorre a lista de dataframes
    psd2im(df,dpi=300, figsize=(12, 8),vmax=2e4, use_cbar=True, n_xticks=13, ftsize=10, use_title=True,savefp=OUTPUT_CONTOURS_PATH)
