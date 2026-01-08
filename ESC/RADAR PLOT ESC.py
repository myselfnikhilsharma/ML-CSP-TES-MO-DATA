
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set global font properties
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 22

def create_radar_chart(data, title, cmap, label_color_map, show_legend=False):
    """Creates and shows a radar chart for the given data."""
    labels = data.columns[1:]
    stats = data.iloc[:, 1:].values
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for i in range(len(data)):
        model_name = data.iloc[i, 0]
        values = stats[i].tolist()
        values += values[:1]
        color = label_color_map.get(model_name, cmap(i % 10))
        ax.plot(angles, values, label=model_name, color=color)
        ax.fill(angles, values, color=color, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    # Place the subplot label (A)/(B) at the top left, bold, blue, Times New Roman, size 22
    ax.text(
        0, 1.08, title,
        fontdict={'family': 'Times New Roman', 'weight': 'bold', 'size': 22, 'color': 'black'},
        horizontalalignment='left', verticalalignment='top',
        transform=ax.transAxes
    )
    if show_legend:
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=14, frameon=False)
    plt.tight_layout()
    plt.show()

def plot_color_code(label_color_map):
    """Creates a separate plot showing the color code for each model in a legend-like style."""
    import matplotlib.lines as mlines

    fig, ax = plt.subplots(figsize=(12, 1))
    ax.axis('off')

    # Prepare legend handles with model names and colors
    handles = []
    for model, color in label_color_map.items():
        handles.append(
            mlines.Line2D([], [], color=color, linewidth=3, label=model)
        )

    # Create a legend with large font and bold style, matching your example
    legend = ax.legend(
        handles=handles,
        loc='center',
        ncol=len(label_color_map),
        frameon=False,
        fontsize=18,
        handlelength=2.5,
        handletextpad=1,
        columnspacing=2,
        borderaxespad=0
    )

    # Set legend text to bold
    for text in legend.get_texts():
        text.set_fontweight('bold')
    plt.tight_layout()
    plt.show()


def main():
    
    file_path = 'RADAR DATA ESC.xlsx'
    


    # Read data from the first and second sheets
    testing_data = pd.read_excel(file_path, sheet_name=1)
    training_data = pd.read_excel(file_path, sheet_name=0)
    
    cmap = plt.colormaps["tab10"]  # Colormap for consistent colors
    
    # Combine all unique model names from both sheets
    all_model_names = pd.concat([
        training_data.iloc[:, 0],
        testing_data.iloc[:, 0]
    ]).unique()
    
    # Assign a color to each model name
    label_color_map = {name: cmap(i % 10) for i, name in enumerate(all_model_names)}
    
    # Plot radar charts WITHOUT legends
    create_radar_chart(training_data, '(A)', cmap, label_color_map, show_legend=False)
    create_radar_chart(testing_data, '(B)', cmap, label_color_map, show_legend=False)
    
    # Plot color code chart
    plot_color_code(label_color_map)

if __name__ == "__main__":
    main()


