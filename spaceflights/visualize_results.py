import json
import matplotlib
matplotlib.use('Agg')  # Para guardar sin mostrar
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

print("🎨 Iniciando generación de visualizaciones...")

# Configuración
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Cargar datos
with open('data/08_reporting/tuning_summary.json') as f:
    tuning_data = json.load(f)

# ============================================
# GRÁFICO 1: R² Score - Regresión
# ============================================
print("\n📊 Generando gráfico 1/3: R² Score (Regresión)...")

regression_data = []
for target, models in tuning_data['regression'].items():
    for model, metrics in models.items():
        regression_data.append({
            'Target': target.replace('target_', '').replace('_', '\n'),
            'Model': model.replace('_', ' ').title(),
            'R2': metrics['tuned_cv_mean'],
            'Std': metrics['tuned_cv_std']
        })

df_reg = pd.DataFrame(regression_data)

fig, ax = plt.subplots(figsize=(16, 10))
targets = df_reg['Target'].unique()
x = np.arange(len(targets))
width = 0.25

models = df_reg['Model'].unique()
colors = ['#3498db', '#e74c3c', '#2ecc71']

for i, model in enumerate(models):
    model_data = df_reg[df_reg['Model'] == model]
    values = []
    errors = []
    for t in targets:
        data = model_data[model_data['Target'] == t]
        if len(data) > 0:
            values.append(data['R2'].values[0])
            errors.append(data['Std'].values[0])
        else:
            values.append(0)
            errors.append(0)
    
    ax.bar(x + i*width, values, width, yerr=errors, label=model, 
           color=colors[i % len(colors)], alpha=0.8, capsize=5)

ax.set_xlabel('Target Variable', fontsize=14, fontweight='bold')
ax.set_ylabel('R² Score', fontsize=14, fontweight='bold')
ax.set_title('🎯 Comparación de Modelos de Regresión', fontsize=18, fontweight='bold', pad=20)
ax.set_xticks(x + width)
ax.set_xticklabels(targets, rotation=45, ha='right', fontsize=10)
ax.legend(fontsize=12, loc='upper right')
ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.3, linewidth=2)
ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.3, linewidth=2)
ax.text(len(targets)-0.5, 0.92, 'Excelente', color='green', fontsize=10)
ax.text(len(targets)-0.5, 0.72, 'Bueno', color='orange', fontsize=10)
ax.set_ylim(-0.1, 1.1)
plt.tight_layout()
plt.savefig('data/08_reporting/regression_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Guardado: regression_comparison.png")

# ============================================
# GRÁFICO 2: F1-Score - Clasificación
# ============================================
print("\n📊 Generando gráfico 2/3: F1-Score (Clasificación)...")

classification_data = []
for target, models in tuning_data['classification'].items():
    for model, metrics in models.items():
        classification_data.append({
            'Target': target.replace('target_', '').replace('_', '\n'),
            'Model': model.replace('_', ' ').title(),
            'F1': metrics['tuned_cv_mean'],
            'Std': metrics['tuned_cv_std']
        })

df_cls = pd.DataFrame(classification_data)

fig, ax = plt.subplots(figsize=(16, 10))
targets_cls = df_cls['Target'].unique()
x = np.arange(len(targets_cls))

models_cls = df_cls['Model'].unique()
colors_cls = ['#9b59b6', '#f39c12']
width_cls = 0.35

for i, model in enumerate(models_cls):
    model_data = df_cls[df_cls['Model'] == model]
    values = []
    errors = []
    for t in targets_cls:
        data = model_data[model_data['Target'] == t]
        if len(data) > 0:
            values.append(data['F1'].values[0])
            errors.append(data['Std'].values[0])
        else:
            values.append(0)
            errors.append(0)
    
    ax.bar(x + i*width_cls, values, width_cls, yerr=errors, label=model,
           color=colors_cls[i % len(colors_cls)], alpha=0.8, capsize=5)

ax.set_xlabel('Target Variable', fontsize=14, fontweight='bold')
ax.set_ylabel('F1 Score', fontsize=14, fontweight='bold')
ax.set_title('🎯 Comparación de Modelos de Clasificación', fontsize=18, fontweight='bold', pad=20)
ax.set_xticks(x + width_cls/2)
ax.set_xticklabels(targets_cls, rotation=45, ha='right', fontsize=10)
ax.legend(fontsize=12, loc='upper right')
ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.3, linewidth=2)
ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.3, linewidth=2)
ax.text(len(targets_cls)-0.5, 0.92, 'Excelente', color='green', fontsize=10)
ax.text(len(targets_cls)-0.5, 0.72, 'Bueno', color='orange', fontsize=10)
ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig('data/08_reporting/classification_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Guardado: classification_comparison.png")

# ============================================
# GRÁFICO 3: Resumen de Mejoras
# ============================================
print("\n📊 Generando gráfico 3/3: Resumen General...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

# Top 5 modelos de regresión
top_reg = df_reg.nlargest(8, 'R2')[['Target', 'Model', 'R2']]
colors_top = ['green' if x > 0.8 else 'orange' if x > 0.5 else 'red' for x in top_reg['R2']]
ax1.barh(range(len(top_reg)), top_reg['R2'], color=colors_top, alpha=0.7)
ax1.set_yticks(range(len(top_reg)))
ax1.set_yticklabels([f"{r['Target'][:15]}\n{r['Model']}" for _, r in top_reg.iterrows()], fontsize=9)
ax1.set_xlabel('R² Score', fontsize=12, fontweight='bold')
ax1.set_title('🏆 Top 8 Modelos - Regresión', fontsize=14, fontweight='bold')
ax1.set_xlim(0, 1.1)
ax1.grid(axis='x', alpha=0.3)

# Top 5 modelos de clasificación
top_cls = df_cls.nlargest(8, 'F1')[['Target', 'Model', 'F1']]
colors_top_cls = ['green' if x > 0.8 else 'orange' if x > 0.5 else 'red' for x in top_cls['F1']]
ax2.barh(range(len(top_cls)), top_cls['F1'], color=colors_top_cls, alpha=0.7)
ax2.set_yticks(range(len(top_cls)))
ax2.set_yticklabels([f"{r['Target'][:15]}\n{r['Model']}" for _, r in top_cls.iterrows()], fontsize=9)
ax2.set_xlabel('F1 Score', fontsize=12, fontweight='bold')
ax2.set_title('🏆 Top 8 Modelos - Clasificación', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 1.1)
ax2.grid(axis='x', alpha=0.3)

# Distribución de R² Score
ax3.hist(df_reg['R2'], bins=15, color='#3498db', alpha=0.7, edgecolor='black')
ax3.axvline(df_reg['R2'].mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {df_reg["R2"].mean():.3f}')
ax3.set_xlabel('R² Score', fontsize=12, fontweight='bold')
ax3.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
ax3.set_title('📊 Distribución de R² - Regresión', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# Distribución de F1 Score
ax4.hist(df_cls['F1'], bins=15, color='#9b59b6', alpha=0.7, edgecolor='black')
ax4.axvline(df_cls['F1'].mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {df_cls["F1"].mean():.3f}')
ax4.set_xlabel('F1 Score', fontsize=12, fontweight='bold')
ax4.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
ax4.set_title('📊 Distribución de F1 - Clasificación', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('data/08_reporting/summary_dashboard.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Guardado: summary_dashboard.png")

# ============================================
# RESUMEN
# ============================================
print("\n" + "="*70)
print("🎉 VISUALIZACIONES COMPLETADAS EXITOSAMENTE")
print("="*70)
print("\n📁 Archivos generados:")
print("   1. regression_comparison.png    - Comparación de modelos de regresión")
print("   2. classification_comparison.png - Comparación de modelos de clasificación")
print("   3. summary_dashboard.png         - Dashboard general de resultados")
print("\n📍 Ubicación: /opt/airflow/project/data/08_reporting/")
print("\n💡 Para copiarlos a tu PC, ejecuta desde PowerShell:")
print("   docker cp spaceflights-airflow-scheduler-1:/opt/airflow/project/data/08_reporting/*.png C:\\Users\\cmuru\\OneDrive\\Escritorio\\graficos\\")
print("="*70)
