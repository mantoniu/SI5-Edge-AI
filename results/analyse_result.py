import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.lines import Line2D

# 1. CONFIGURATION DES FICHIERS
OUTPUT_DIR = './results/graph_results'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

PROFILING_CONFIG = {
    './results/power_monitoring/rasp_prepared.csv': (1.0, "Raspberry Pi 4"),
    './results/power_monitoring/oakd_prepared.csv': (1.0, "OAK-D Lite"), 
    ('./results/power_monitoring/nvidia_gpu_prepared.csv', './results/power_monitoring/cpu_with_gpu_prepared.csv'): (0.5, "CPU + GPU"),
    './results/power_monitoring/cpu_prepared.csv': (0.5, "CPU Seul"),
}

GLOBAL_BENCH_CONFIG = [
    {'device': 'Raspberry Pi', 'bench': './results/benchmark_rasp.csv', 'power_files': ['./results/power_monitoring/rasp_prepared.csv']},
    {'device': 'OAK-D', 'bench': './results/benchmark_oakd.csv', 'power_files': ['./results/power_monitoring/oakd_prepared.csv']},
    {'device': 'CPU', 'bench': './results/benchmark_cpu.csv', 'power_files': ['./results/power_monitoring/cpu_prepared.csv']},
    {'device': 'CPU + GPU', 'bench': './results/benchmark_gpu.csv', 'power_files': ['./results/power_monitoring/nvidia_gpu_prepared.csv', './results/power_monitoring/cpu_with_gpu_prepared.csv']}
]

# 2. FONCTIONS DE NETTOYAGE ET CALCUL
def clean_model_name(name):
    """Nettoie le nom du modèle pour la légende."""
    name = str(name).replace('.onnx', '').replace('.blob', '')
    if '_openvino' in name:
        name = name.split('_openvino')[0]
    return name

def get_avg_power(df, phase_name):
    """Calcule la puissance moyenne pour une phase spécifique."""
    col = 'Power' if 'Power' in df.columns else 'Watt (W)'
    if 'Phase' not in df.columns: return 0
    subset = df[df['Phase'].astype(str).str.strip().str.lower() == str(phase_name).strip().lower()]
    return subset[col].mean() if not subset.empty else 0

def get_global_max_power(config_dict):
    """Scanne pour trouver la puissance max (Totale)."""
    max_p = 1.0 
    for file_key in config_dict.keys():
        files = [file_key] if isinstance(file_key, str) else file_key
        combined_series = pd.Series(dtype=float)
        for f in files:
            if os.path.exists(f):
                temp_df = pd.read_csv(f)
                col = 'Power' if 'Power' in temp_df.columns else 'Watt (W)'
                combined_series = combined_series.add(temp_df[col], fill_value=0)
        if not combined_series.empty:
            max_p = max(max_p, combined_series.max())
    return max_p

# 3. GÉNÉRATION DES PROFILS INDIVIDUELS
def generate_individual_profiles(config_dict, global_max):
    print(f"--- 1. Génération des profils individuels (Max global: {global_max:.2f}W) ---")
    
    for file_key, (time_step, custom_label) in config_dict.items():
        files = [file_key] if isinstance(file_key, str) else list(file_key)
        
        for f in files:
            if not os.path.exists(f): continue
            temp_df = pd.read_csv(f)
            c = 'Power' if 'Power' in temp_df.columns else 'Watt (W)'
            if df_combined is None:
                df_combined = temp_df.copy()
                col_conso = c
            else:
                df_combined[col_conso] = df_combined[col_conso].add(temp_df[c], fill_value=0)
        
        if not valid_dfs:
            continue

        # Détection automatique des colonnes
        col_conso = 'Power' if 'Power' in valid_dfs[0].columns else 'Watt (W)'
        col_phase = 'Phase' if 'Phase' in valid_dfs[0].columns else 'Model'
        
        # Somme pour le graphique des courbes (on s'aligne sur la longueur minimale)
        min_len = min(len(d) for d in valid_dfs)
        df_total = valid_dfs[0].iloc[:min_len].copy()
        if len(valid_dfs) > 1:
            for other_df in valid_dfs[1:]:
                df_total[col_conso] += other_df[col_conso].iloc[:min_len].values
        
        models = [p for p in df_total[col_phase].unique() if p != 'Idle']
        
        # --- A. GRAPHE DES COURBES (PUISSANCE TOTALE) ---
        plt.figure(figsize=(12, 6))
        for model in models:
            indices = df_total.index[df_total[col_phase] == model].tolist()
            if not indices: continue
            
            s, e = max(0, indices[0]-5), min(len(df_total)-1, indices[-1]+5)
            subset = df_total.iloc[s:e+1].copy()
            subset['Relative Time'] = np.arange(len(subset)) * time_step
            
            # Soustraction de l'idle pour la courbe temporelle
            dynamic_power_curve = (subset[col_conso] - idle_power).clip(lower=0.01)
            
            label_c = clean_model_name(model)
            plt.plot(subset['Relative Time'], subset[col_conso], label=label_c)

        plt.title(f"Profil de Puissance Totale : {custom_label}")
        plt.xlabel("Secondes")
        plt.ylabel("Puissance (Watt)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        file_name_clean = custom_label.replace(' ', '_').replace('+', 'plus')
        plt.savefig(os.path.join(OUTPUT_DIR, f"courbes_{file_name_clean}.png"))
        plt.close()

        # --- B. GRAPHE DES MOYENNES (EMPILÉES SI MULTI-FICHIERS) ---
        phases = ['Idle'] + models
        avg_data = {}
        
        for p in phases:
            p_clean = clean_model_name(p) if p != 'Idle' else 'Idle'
            vals = []
            for d in valid_dfs:
                subset = d[d[col_phase] == p]
                vals.append(subset[col_conso].mean() if not subset.empty else 0)
            avg_data[p_clean] = vals

        # Tri par puissance totale décroissante
        sorted_keys = sorted(avg_data.keys(), key=lambda k: sum(avg_data[k]), reverse=True)
        
        plt.figure(figsize=(12, 6))
        bottoms = np.zeros(len(sorted_keys))
        
        # Couleurs et labels pour les composants
        if len(valid_dfs) == 2:
            comp_names = ["GPU", "CPU"] # Basé sur l'ordre de votre tuple
            comp_colors = ['#3498db', '#e74c3c'] # Bleu pour GPU, Rouge pour CPU
        else:
            comp_names = [f"Fichier {i+1}" for i in range(len(valid_dfs))]
            comp_colors = sns.color_palette("muted", len(valid_dfs))

        for i in range(len(valid_dfs)):
            heights = [avg_data[k][i] for k in sorted_keys]
            plt.bar(sorted_keys, heights, bottom=bottoms, color=comp_colors[i], 
                    edgecolor='black', label=comp_names[i])
            bottoms += np.array(heights)

        # Ajout du texte de la puissance totale au-dessus de chaque barre
        for idx, k in enumerate(sorted_keys):
            total = sum(avg_data[k])
            plt.text(idx, total + 0.1, f"{total:.2f}W", ha='center', fontweight='bold')

        plt.title(f"Consommation Moyenne Détaillée : {custom_label}")
        plt.ylabel("Watts")
        plt.xticks(rotation=45, ha='right')
        if len(valid_dfs) > 1:
            plt.legend(title="Composants")
        
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"moyennes_{file_name_clean}.png"))
        plt.close()
        
        print(f"Terminé pour {custom_label}")

# 4. GÉNÉRATION DES COMPARAISONS GLOBALES
def plot_efficiency(df, map_col, map_label, filename):
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    models_list = sorted(df['Model'].unique())
    devices_list = df['Device'].unique()
    colors = dict(zip(models_list, sns.color_palette("husl", len(models_list))))
    markers = {dev: m for dev, m in zip(devices_list, ['o', 's', 'D', '^', 'v'])}

    for mod in models_list:
        mod_df = df[df['Model'] == mod].sort_values(map_col)
        plt.plot(mod_df[map_col], mod_df['Score'], color=colors[mod], alpha=0.4, linewidth=1.5)
        for dev in devices_list:
            pt = mod_df[mod_df['Device'] == dev]
            if not pt.empty:
                plt.scatter(pt[map_col], pt['Score'], color=colors[mod], marker=markers[dev], 
                            s=150, edgecolors='black', alpha=0.9, zorder=5)

    plt.title(f"Efficacité Énergétique Dynamique : {map_label} vs Joules", fontsize=15)
    plt.xlabel(f"Précision ({map_label})", fontsize=12)
    plt.ylabel("Énergie Dynamique Totale (Joules)")
    plt.yscale('log')

    # Légendes
    m_leg = [Line2D([0], [0], color=colors[m], lw=2, label=m) for m in models_list]
    leg1 = plt.legend(handles=m_leg, title="Modèles", loc='lower right', frameon=True, shadow=True)
    plt.gca().add_artist(leg1)
    d_leg = [Line2D([0], [0], marker=markers[d], color='w', label=d, markerfacecolor='gray', markersize=9) for d in devices_list]
    plt.legend(handles=d_leg, title="Périphériques", loc='upper left', frameon=True, shadow=True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close()

def generate_global_comparisons(bench_config):
    print("\n--- 2. Génération des comparaisons mAP / Efficacité ---")
    results = []
    for cfg in bench_config:
        if not os.path.exists(cfg['bench']): continue
        dfs_power = [pd.read_csv(pf) for pf in cfg['power_files'] if os.path.exists(pf)]
        if not dfs_power: continue

        idle_p = sum(get_avg_power(df, 'Idle') for df in dfs_power)
        df_bench = pd.read_csv(cfg['bench'])

        for _, row in df_bench.iterrows():
            total_model_p = sum(get_avg_power(df, row['Model']) for df in dfs_power)
            if total_model_p > 0:
                dynamic_p = max(0.01, total_model_p - idle_p)
                score = (row['time_avg'] / 1000.0) * dynamic_p
                results.append({
                    'Device': cfg['device'],
                    'Model': clean_model_name(row['Model']),
                    'mAP_box': row['mAP_box'],
                    'mAP_mask': row['mAP_mask'],
                    'Score': score
                })

    if not results: return
    df_final = pd.DataFrame(results)
    plot_efficiency(df_final, 'mAP_box', 'mAP Box', 'comparaison_efficacite_box.png')
    plot_efficiency(df_final, 'mAP_mask', 'mAP Mask', 'comparaison_efficacite_mask.png')

# 5. EXÉCUTION
if __name__ == "__main__":
    # Max global pour harmoniser tous les axes Y
    global_max_found = get_global_max_power(PROFILING_CONFIG)
    
    generate_individual_profiles(PROFILING_CONFIG, global_max_found)
    generate_global_comparisons(GLOBAL_BENCH_CONFIG)
    
    print(f"\nTraitement terminé. Les graphes temporels sont DYNAMIQUES, les moyennes sont TOTALES (avec Idle).")