import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as mpatches
from matplotlib import pyplot as pl
import numpy as np

def plot_plate_positions(df:pd.DataFrame, plate:str, dot_size:int, save_path:Path, save_filename:str, grids=(8,12), pos_color="teal", neg_color="gray"):
    """Function plots samples recorded on each position in a plate. This helps to visually track if there are any missing samples in a given dataset.

    Args:
        df (pd.DataFrame): Pandas dataframe contaning information. Dataset must at least contain two columns labeled as "Plate" and "Position"
        plate (str): Plate identifier. This plate identifier must correspond to an id in the "Plate" column.
        dot_size (int): dot size to plot each position on the plate.
        save_path (Path): Path to file figure.
        save_filename (str): file name to save figure.
        pos_color (str, optional): Positive color. All samples present. Defaults to "teal".
        neg_color (str, optional): Negative color. All samples missing. Defaults to "gray".
        grids (tuple, optional): Figure size. Defaults to (8,12).
    """
    grid_size=grids
    fig, ax = plt.subplots(figsize=(grids[1], grids[0]))
    positions=df[df["Plate"]==plate]["Position"].unique()
    for pos in positions:
        row=ord(pos[0])-ord("A")
        col=int(pos[1:])-1
    
        # Plot the position
        ax.plot(col,row, "bo", color=pos_color, markersize=dot_size)
    
    # Plot missing positions 
    all_positions = set([f'{chr(row + ord("A"))}{col + 1:02d}' for row in range(grid_size[0]) for col in range(grid_size[1])])
    missing_positions = all_positions - set(positions)
    for pos in missing_positions:
        row = ord(pos[0]) - ord('A')  # Convert the letter to a row index
        col = int(pos[1:]) - 1  # Convert the number to a column index
        ax.plot(col, row, 'o', color=neg_color, markersize=dot_size)  # Plot missing positions in gray circles
    
    # Set labels and title
    ax.set_xticks(range(grid_size[1]))  # Set x-ticks to match the number of columns
    ax.set_yticks(range(grid_size[0]))  # Set y-ticks to match the number of rows
    ax.set_xticklabels(range(1, grid_size[1] + 1))  # Set x-tick labels
    ax.set_yticklabels([chr(row + ord('A')) for row in range(grid_size[0])])  # Set y-tick labels
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title(f'Sample distribution plate {plate}')

    # Show the plot
    plt.grid(True)
    plt.gca().invert_yaxis()  # Invert y-axis to match the grid orientation
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, save_filename))
    plt.close()
    
def make_barplot_sample_dist(df:pd.DataFrame, group_col:str, plot_title:str, y_axis_title:str,
        x_axis_title:str, fig_size:tuple=(8,6), cmap_name:str="tab10", keep_xlabs:bool=True) -> object:
    """Make a barplot with sample distributions, how many patients and how many controls we have in the dataframe

    Args:
        df (pd.DataFrame): Dataframe containing patients and a column with information on patient/controls classification.
        group_col (str): Column name where group is indicated.
        plot_title (str): Plot main title.
        y_axis_title (str): Plot y axis title.
        x_axis_title (str): Plot x axis title.
        fig_size (tuple, optional): Figure size. Defaults to (8,6).
        cmap_name (str): cmap name to color bars.

    Returns:
        plot: Returns plot.
    """
    values=df[group_col]
    # Get counts
    value_counts=values.value_counts()
    
    # Make plot
    fig, ax= plt.subplots(figsize=fig_size)
    cmap=plt.get_cmap(cmap_name)
    colors=[cmap(i) for i in np.linspace(0,1, len(value_counts))]
    barplot=ax.bar(value_counts.index, value_counts.values, color=colors)
    ax.set_xlabel(x_axis_title)
    ax.set_ylabel(y_axis_title)
    ax.set_title(plot_title)
    
    if keep_xlabs:
        plt.xticks(rotation=90)
    else:
        ax.set_xticklabels([]) 
    
    # Add text labels on top of each bar
    for bar in barplot:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, str(height), ha='center', va='bottom')

    # Add legends
    legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, value_counts.index)]
    ax.legend(handles=legend_patches, title=group_col)
    
    # Add title

    
    # Add total number of counts
    total_count = sum(value_counts.values)
    ax.text(1, 1, f'Total Count: {total_count}', transform=ax.transAxes,
            horizontalalignment='right', verticalalignment='bottom')
    
    plt.tight_layout()
    
    return ax

def make_barplot_sample_dist_twogroups(df:pd.DataFrame, group1:str, group2:str, plot_title:str, y_axis_title:str,
                    x_axis_title:str, fig_size:tuple=(8,6), cmap_name:str="tab10") -> object:
    """Make a barplot with sample distributions when we want to present information about two different groups.

    Args:
        df (pd.DataFrame): Dataframe containing information on the two categorical variables to present.
        group1 (str): Column name where group1 is indicated.
        group2 (str): Column name where group2 is indeicated.
        plot_title (str): Plot main title.
        y_axis_title (str): Plot y axis title.
        x_axis_title (str): Plot x axis title.
        fig_size (tuple, optional): Figure size. Defaults to (8,6).
        cmap_name (str): cmap name for bar colors.

    Returns:
        plot: Returns plot.
    """
    # Get grouped counts
    grouped_counts=df.groupby([group1, group2]).size().unstack(fill_value=0)
    # Make plot
    fig, ax= plt.subplots(figsize=fig_size)
    cmap=plt.get_cmap(cmap_name)
    colors=[cmap(i) for i in np.linspace(0,1, len(grouped_counts.columns))]
    
    # Plot stacked bars
    bars=grouped_counts.plot(kind="bar", stacked=True, color=colors, ax=ax)

    # Modify figure axes
    ax.set_xlabel(x_axis_title)
    ax.set_ylabel(y_axis_title)
    ax.set_title(plot_title)
    plt.xticks(rotation=90)
    
        # Add text labels on top of each bar for group2 counts in the first stack
    for i, (index,row) in enumerate(grouped_counts.iterrows()):
        height=sum(row)
        text=", ".join(map(str, row.values))
        ax.text(i, height+5,text, fontsize=9, ha="center", va="center")

    # Add legends
    legend_patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, grouped_counts.columns)]
    ax.legend(handles=legend_patches, title=group2)
    
    
    
    # Add total number of counts
    total_count = sum(grouped_counts.values)
    ax.text(1, 1, f'total counts: {", ".join(map(str, total_count))}', transform=ax.transAxes,
            horizontalalignment='right', verticalalignment='bottom')
    
    plt.tight_layout()
    
    return ax
