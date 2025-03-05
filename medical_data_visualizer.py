import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
#read the data from medical_examination.csv
df = pd.read_csv('medical_examination.csv')

# 2
#calculate the BMI in order to determine if a person is overweight 
df['bmi'] = df['weight'] / ((df['height'] / 100)**2)

df['overweight'] = (df['bmi'] > 25).astype(int)

# 3
#Normalize data by making 0 always good and 1 always bad
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4
#Categorical Plot
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'],
                     var_name='variable', value_name='value')


    # 6
    df_cat = df_cat.groupby(['cardio','variable','value']).size().reset_index(name='total')
    

    # 7
    g = sns.catplot(data = df_cat, x='variable', hue='value', col='cardio', kind='count', palette='Set1', height=4, aspect=1.2)


    # 8
    
    fig = g.fig


    # 9
    fig.savefig('catplot.png')
    return fig


def clean_data():
    # Charger les données
    df = pd.read_csv('medical_examination.csv')

    # 1. Pression diastolique <= pression systolique
    df = df[df['ap_lo'] <=df['ap_hi']]

    # 2. Taille entre le 2.5e et le 97.5e percentile
    height_lower = df['height'].quantile(0.025)
    height_upper = df['height'].quantile(0.975)
    df = df[(df['height'] >= height_lower) & (df['height'] >= height_upper)]

    # 3. Poids entre le 2.5e et le 97.5e percentile
    weight_lower = df['weight'].quantile(0.025)
    weight_upper = df['weight'].quantile(0.975)
    df = df[(df['weight'] >= weight_lower) & (df['weight'] >= weight_upper)]

    return df
# 10
def draw_heat_map():
    # 11
    df_heat = clean_data()
    print(df_heat.head())

    # 12
    corr = df_heat.corr()

    # Créer une heatmap de la matrice de corrélation
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Matrice de corrélation')
    plt.show()

    # 13
    # Générer le masque pour le triangle supérieur
    mask = np.triu(np.ones_like(corr, dtype=bool))
    print("Masque pour le triangle supérieur : ")
    print(mask)

    # 14 
    # Configure the figure and axes
    fig, ax = plt.subplots(figsize=(8,6), dpi=100) # Taille de la figure : 8x6 pouces, résolution : 100 DPI

    # 15 
    # Plot the heatmap with the mask
    sns.heatmap(corr, 
                mask=mask, # Mask for the upper triangle
                annot=True, 
                fmt=".2f", 
                cmap="coolwarm", 
                vmin=-1, 
                vmax=1, 
                square=True,
                cbar_kws={"shrink":0.8}, 
                ax=ax) # Axes on which to draw the heatmap
    # 16
    fig.savefig('heatmap.png')
    return fig
