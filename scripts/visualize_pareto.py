import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ---------------------------------------------------------
# 1. Data Processing
# ---------------------------------------------------------

data = [
    # TSP-50
    {"Method": "Concorde", "Type": "Exact", "N": 50, "Gap": 0.00, "Time": "3m"},
    {"Method": "LKH3", "Type": "Exact", "N": 50, "Gap": 0.00, "Time": "3m"},
    {"Method": "2Opt", "Type": "Exact", "N": 50, "Gap": 2.95, "Time": None},
    {"Method": "AM", "Type": "RL", "N": 50, "Gap": 1.76, "Time": "2s"},
    {"Method": "GCN", "Type": "RL", "N": 50, "Gap": 3.10, "Time": "55s"},
    {"Method": "Transformer", "Type": "RL", "N": 50, "Gap": 0.31, "Time": "14s"},
    {"Method": "POMO", "Type": "RL", "N": 50, "Gap": 0.64, "Time": "1s"},
    {"Method": "Image Diffusion", "Type": "Diffusion", "N": 50, "Gap": 1.23, "Time": None},
    {"Method": "DIFUSCO (Speed)", "Type": "Diffusion", "N": 50, "Gap": 12.84, "Time": "16s"},
    {"Method": "DIFUSCO (Accuracy)", "Type": "Diffusion", "N": 50, "Gap": 0.41, "Time": "18m"},
    {"Method": "T2T (Speed)", "Type": "T2T", "N": 50, "Gap": 8.15, "Time": "55s"},
    {"Method": "T2T (Accuracy)", "Type": "T2T", "N": 50, "Gap": 0.03, "Time": "26m"},
    {"Method": "Fast T2T (Speed)", "Type": "T2T", "N": 50, "Gap": 0.31, "Time": "11s"},
    {"Method": "Fast T2T (Accuracy)", "Type": "T2T", "N": 50, "Gap": 0.01, "Time": "3m"},
    {"Method": "CycFlow", "Type": "Ours", "N": 50, "Gap": 0.08, "Time": "0.004s"},

    # TSP-100
    {"Method": "Concorde", "Type": "Exact", "N": 100, "Gap": 0.00, "Time": "12m"},
    {"Method": "LKH3", "Type": "Exact", "N": 100, "Gap": 0.00, "Time": "33m"},
    {"Method": "2Opt", "Type": "Exact", "N": 100, "Gap": 3.54, "Time": None},
    {"Method": "AM", "Type": "RL", "N": 100, "Gap": 4.53, "Time": "6s"},
    {"Method": "GCN", "Type": "RL", "N": 100, "Gap": 8.38, "Time": "6m"},
    {"Method": "Transformer", "Type": "RL", "N": 100, "Gap": 1.42, "Time": "5s"},
    {"Method": "POMO", "Type": "RL", "N": 100, "Gap": 1.07, "Time": "2s"},
    {"Method": "Sym-NCO", "Type": "RL", "N": 100, "Gap": 0.94, "Time": "2s"},
    {"Method": "Image Diffusion", "Type": "Diffusion", "N": 100, "Gap": 2.11, "Time": None},
    {"Method": "DIFUSCO (Speed)", "Type": "Diffusion", "N": 100, "Gap": 20.20, "Time": "20s"},
    {"Method": "DIFUSCO (Accuracy)", "Type": "Diffusion", "N": 100, "Gap": 1.16, "Time": "18m"},
    {"Method": "T2T (Speed)", "Type": "T2T", "N": 100, "Gap": 16.09, "Time": "1m"},
    {"Method": "T2T (Accuracy)", "Type": "T2T", "N": 100, "Gap": 0.11, "Time": "42m"},
    {"Method": "Fast T2T (Speed)", "Type": "T2T", "N": 100, "Gap": 1.31, "Time": "16s"},
    {"Method": "Fast T2T (Accuracy)", "Type": "T2T", "N": 100, "Gap": 0.03, "Time": "3m"},
    {"Method": "CycFlow", "Type": "Ours", "N": 100, "Gap": 0.42, "Time": "0.01s"},

    # TSP-500
    {"Method": "Concorde", "Type": "Exact", "N": 500, "Gap": 0.00, "Time": "37.7m"},
    {"Method": "LKH3", "Type": "Exact", "N": 500, "Gap": 0.00, "Time": "46.3m"},
    {"Method": "AM", "Type": "RL", "N": 500, "Gap": 20.99, "Time": "1.5m"},
    {"Method": "GCN", "Type": "RL", "N": 500, "Gap": 79.61, "Time": "6.7m"},
    {"Method": "POMO", "Type": "RL", "N": 500, "Gap": 48.22, "Time": "11.6h"},
    {"Method": "DIMES", "Type": "RL", "N": 500, "Gap": 14.38, "Time": "1.0m"},
    {"Method": "DIFUSCO", "Type": "Diffusion", "N": 500, "Gap": 1.5, "Time": "4.5m"},
    {"Method": "T2T (Speed)", "Type": "T2T", "N": 500, "Gap": 4.28, "Time": "36s"},
    {"Method": "T2T (Accuracy)", "Type": "T2T", "N": 500, "Gap": 5.61, "Time": "6.4m"},
    {"Method": "Fast T2T (Speed)", "Type": "T2T", "N": 500, "Gap": 1.23, "Time": "15s"},
    {"Method": "Fast T2T (Accuracy)", "Type": "T2T", "N": 500, "Gap": 0.39, "Time": "2.2m"},
    {"Method": "CycFlow", "Type": "Ours", "N": 500, "Gap": 7.60, "Time": "0.06s"},

    # TSP-1000
    {"Method": "Concorde", "Type": "Exact", "N": 1000, "Gap": 0.00, "Time": "6.65h"},
    {"Method": "LKH3", "Type": "Exact", "N": 1000, "Gap": 0.00, "Time": "2.57h"},
    {"Method": "AM", "Type": "RL", "N": 1000, "Gap": 34.75, "Time": "3.2m"},
    {"Method": "GCN", "Type": "RL", "N": 1000, "Gap": 110.29, "Time": "28.5m"},
    {"Method": "POMO", "Type": "RL", "N": 1000, "Gap": 114.36, "Time": "63.5h"},
    {"Method": "DIMES", "Type": "RL", "N": 1000, "Gap": 14.97, "Time": "2.1m"},
    {"Method": "DIFUSCO", "Type": "Diffusion", "N": 1000, "Gap": 1.89, "Time": "14.4m"},
    {"Method": "T2T (Accuracy)", "Type": "T2T", "N": 1000, "Gap": 9.04, "Time": "19.4m"},
    {"Method": "Fast T2T (Speed)", "Type": "T2T", "N": 1000, "Gap": 1.42, "Time": "57s"},
    {"Method": "Fast T2T (Accuracy)", "Type": "T2T", "N": 1000, "Gap": 0.58, "Time": "8.6m"},
    {"Method": "CycFlow", "Type": "Ours", "N": 1000, "Gap": 9.89, "Time": "0.22s"},
]

df = pd.DataFrame(data)


def parse_time(t_str):
    if t_str is None: return None
    if 'h' in t_str: return float(t_str.replace('h', '')) * 3600
    if 'm' in t_str: return float(t_str.replace('m', '')) * 60
    if 's' in t_str: return float(t_str.replace('s', ''))
    return None


df['Time_Seconds'] = df['Time'].apply(parse_time)
df = df.dropna(subset=['Time_Seconds', 'Gap']).copy()

# Fix 0.0 Gap for Log Plot (Set floor to 0.01% for visualization)
MIN_GAP_DISPLAY = 0.005
df['Gap_Plot'] = df['Gap'].apply(lambda x: max(x, MIN_GAP_DISPLAY))

# ---------------------------------------------------------
# 2. Style & Palette Definition
# ---------------------------------------------------------

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

# Base Colors per Type
type_colors = {
    'Exact': '#34495e',  # Dark Slate (Black/Gray)
    'RL': '#27ae60',  # Green
    'Diffusion': '#d35400',  # Burnt Orange
    'T2T': '#2980b9',  # Strong Blue
    'Ours': '#c0392b'  # Red
}

# Available Styles to Cycle Through
# We ensure there are enough unique combinations within a category
available_markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '<', '>']
available_linestyles = ['--', '-.', ':', (0, (3, 1, 1, 1))]  # Dashed, DashDot, Dotted, Dense DashDot

# ---------------------------------------------------------
# 3. Dynamic Style Mapping
# ---------------------------------------------------------

method_styles = {}

# We group by Type, then assign unique styles to each Method in that Type
for m_type in df['Type'].unique():
    methods_in_type = df[df['Type'] == m_type]['Method'].unique()
    methods_in_type = sorted(methods_in_type)  # Sort to ensure stability

    for i, method in enumerate(methods_in_type):
        is_ours = (m_type == 'Ours')

        # Base config
        color = type_colors.get(m_type, 'gray')

        if is_ours:
            # Highlight Ours
            marker = '*'
            linestyle = '-'
            linewidth = 3.5
            alpha = 1.0
            zorder = 100
            markersize_mult = 2.5
        else:
            # Cycle through styles for baselines
            marker = available_markers[i % len(available_markers)]

            # Use solid line for the "best" or first in list, others get fancy dashes
            # But user asked for differentiation, so we cycle linestyles too
            if "Concorde" in method:
                linestyle = '-'  # Force solid for Gold Standard
            elif "LKH3" in method:
                linestyle = '--'
            else:
                linestyle = available_linestyles[i % len(available_linestyles)]

            linewidth = 1.5
            alpha = 0.75
            zorder = 10
            markersize_mult = 1.0

        method_styles[method] = {
            'color': color,
            'marker': marker,
            'ls': linestyle,
            'lw': linewidth,
            'alpha': alpha,
            'zorder': zorder,
            'ms_mult': markersize_mult
        }

# ---------------------------------------------------------
# 4. Plotting
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 10))

# Iterate through every method to plot its trajectory
for method in df['Method'].unique():
    subset = df[df['Method'] == method].sort_values(by='N')
    if subset.empty: continue

    style = method_styles[method]

    # 1. Draw Line (Trajectory)
    ax.plot(subset['Time_Seconds'], subset['Gap_Plot'],
            color=style['color'], linestyle=style['ls'], linewidth=style['lw'],
            alpha=style['alpha'], zorder=style['zorder'], label=method)

    # 2. Draw Markers (Variable Size based on N)
    # Base sizes: 50->80, 100->120, 500->180, 1000->240
    sizes = [80 + (n / 1000) * 160 for n in subset['N']]
    sizes = [s * style['ms_mult'] for s in sizes]  # Scale up if Ours

    ax.scatter(subset['Time_Seconds'], subset['Gap_Plot'],
               s=sizes, marker=style['marker'], color=style['color'],
               edgecolors='white', linewidth=1.0,
               alpha=1.0 if style['zorder'] == 100 else 0.9,
               zorder=style['zorder'] + 1)  # Markers on top of lines

# ---------------------------------------------------------
# 5. Formatting & Annotations
# ---------------------------------------------------------

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_xlabel("Inference Time (Seconds) - Log Scale", fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel("Optimality Gap (%) - Log Scale", fontsize=14, fontweight='bold', labelpad=10)

# Custom Y-Axis formatting
y_major = ticker.LogLocator(base=10.0, numticks=10)
ax.yaxis.set_major_locator(y_major)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:.2g}%'.format(y) if y >= 0.01 else 'Exact (0%)'))

ax.set_title("TSP Solver Efficiency Frontier: Size N=50 to N=1000", fontsize=16, fontweight='bold', pad=15)
ax.grid(True, which="major", ls="-", alpha=0.5)
ax.grid(True, which="minor", ls=":", alpha=0.2)

# Annotate specific points (Ours + Concorde) to guide the eye
ours_data = df[df['Method'] == 'CycFlow'].sort_values(by='N').iloc[-1]
ax.annotate(f"Ours (N={int(ours_data['N'])})",
            (ours_data['Time_Seconds'], ours_data['Gap_Plot']),
            xytext=(0, 15), textcoords='offset points', ha='center',
            color=type_colors['Ours'], fontweight='bold', fontsize=12,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))

# ---------------------------------------------------------
# 6. Custom Legend Construction
# ---------------------------------------------------------

# We create a legend that groups by Type, but lists specific markers/lines
legend_handles = []

for m_type, color in type_colors.items():
    # Header for the category
    legend_handles.append(Line2D([0], [0], color='white', label=f"$\\bf{{--- {m_type} ---}}$"))

    # Get methods for this type
    methods = sorted(df[df['Type'] == m_type]['Method'].unique())
    for m in methods:
        s = method_styles[m]
        # Create a proxy artist for the legend
        h = Line2D([0], [0], color=s['color'], marker=s['marker'], linestyle=s['ls'],
                   lw=2, markersize=8, label=m)
        legend_handles.append(h)

# Place legend outside or to the side depending on layout
ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.01, 1),
          fontsize=9, frameon=True, title="Method Details", title_fontsize=11)

plt.tight_layout()
plt.savefig('tsp_detailed_results.png', dpi=300, bbox_inches='tight')
plt.show()