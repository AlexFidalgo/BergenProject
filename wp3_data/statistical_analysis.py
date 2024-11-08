import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import scipy.cluster.hierarchy as sch


def calculate_rcm_correlations_for_fixed_gcms_average_then_pivot(cons):

    gcm_list = cons['id_GCM'].unique()
    correlation_results = {}

    for gcm_id in gcm_list:

        gcm_data = cons[cons['id_GCM'] == gcm_id]

        # Pivot the data to have RCMs' errors as columns and gridpoints as rows
        pivot_data = gcm_data.pivot_table(index='gp_region', columns='id_RCM', values='mat_vector')

        # Calculate pairwise correlations among RCMs for this GCM
        correlation_matrix = pivot_data.corr(method='pearson')

        correlation_results[gcm_id] = correlation_matrix

        print(f"Partial correlations for GCM {gcm_id} calculated.")
    
    return correlation_results

def calculate_rcm_correlations_for_fixed_gcms_granular_correlations(cons):

    gcm_list = cons['id_GCM'].unique()
    correlation_results = {}

    for gcm_id in gcm_list:
        gcm_data = cons[cons['id_GCM'] == gcm_id]

        # Dictionary to hold correlations for each combination of physical_variable and metric
        temp_correlations = []

        # Loop over each combination of physical_variable and metric
        for (phys_var, metric), subset in gcm_data.groupby(['physical_variable', 'Metric']):
            # Pivot for the current subset
            pivot_data = subset.pivot_table(index='gp_region', columns='id_RCM', values='mat_vector')
            
            # Calculate pairwise correlations among RCMs for this subset
            correlation_matrix = pivot_data.corr(method='pearson')
            temp_correlations.append(correlation_matrix)

        # Average the correlation matrices across all combinations
        # Ensure all matrices have the same index and columns for this to work
        if temp_correlations:
            averaged_correlation_matrix = sum(temp_correlations) / len(temp_correlations)
            correlation_results[gcm_id] = averaged_correlation_matrix

        print(f"Partial correlations for GCM {gcm_id} calculated.")
    
    return correlation_results

def calculate_rcm_correlations_for_fixed_gcms_specific_metric_variable(cons):
    # Filter for the specific metric and physical variable
    filtered_data = cons[(cons['Metric'] == 1) & (cons['physical_variable'] == 'ppt')]

    gcm_list = filtered_data['id_GCM'].unique()
    correlation_results = {}

    for gcm_id in gcm_list:
        gcm_data = filtered_data[filtered_data['id_GCM'] == gcm_id]

        # Pivot the data to have RCMs' errors as columns and gridpoints as rows
        pivot_data = gcm_data.pivot_table(index='gp_region', columns='id_RCM', values='mat_vector')

        # Calculate pairwise correlations among RCMs for this GCM
        correlation_matrix = pivot_data.corr(method='pearson')

        correlation_results[gcm_id] = correlation_matrix

        print(f"Partial correlations for GCM {gcm_id} with Metric 1 and Variable 'ppt' calculated.")
    
    return correlation_results


def visualize_rcms_correlations_for_fixed_gcms(correlation_results, folder):

    # Set up the main folder
    main_folder = 'rcms_correlations_for_fixed_gcms'
    os.makedirs(main_folder, exist_ok=True)

    # Loop over each GCM and save the heatmap in the corresponding folder
    for gcm, correlation_matrix in correlation_results.items():
        # Create a folder for each GCM inside the main folder
        gcm_folder = os.path.join(main_folder, folder)
        os.makedirs(gcm_folder, exist_ok=True)
        
        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                    cbar=True, square=True, linewidths=0.5, linecolor='black')
        
        plt.title(f"Partial Correlations Among RCMs for GCM {gcm}")
        plt.xlabel("RCM")
        plt.ylabel("RCM")
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        
        # Save the plot to the GCM-specific folder
        filename = os.path.join(gcm_folder, f'rcm_correlations_gcm_{gcm}.png')
        plt.tight_layout()
        plt.savefig(filename, format='png', dpi=300)
        plt.close()  # Close the figure to avoid display in the notebook/environment

        print(f"Saved correlation heatmap for GCM {gcm} to {filename}")


def create_average_error_and_std_maps_for_gcms(cons):
    # Define region boundaries
    regions = {
        'BI': {'name': 'Ilhas Britânicas', 'west': -10, 'east': 2, 'south': 50, 'north': 59},
        'IP': {'name': 'Península Ibérica', 'west': -10, 'east': 3, 'south': 36, 'north': 44},
        'FR': {'name': 'França', 'west': -5, 'east': 5, 'south': 44, 'north': 50},
        'ME': {'name': 'Europa Central', 'west': 2, 'east': 16, 'south': 48, 'north': 55},
        'SC': {'name': 'Escandinávia', 'west': 5, 'east': 30, 'south': 55, 'north': 70},
        'AL': {'name': 'Alpes', 'west': 5, 'east': 15, 'south': 44, 'north': 48},
        'MD': {'name': 'Mediterrâneo', 'west': 3, 'east': 25, 'south': 36, 'north': 44},
        'EA': {'name': 'Europa Oriental', 'west': 16, 'east': 30, 'south': 44, 'north': 55}
    }

    # Main folder for maps
    main_folder = 'heatmap'
    os.makedirs(main_folder, exist_ok=True)

    # Get unique GCMs
    gcm_list = cons['id_GCM'].unique()

    # Loop over each GCM
    for gcm_id in gcm_list:
        # Filter data for the specific GCM
        gcm_data = cons[cons['id_GCM'] == gcm_id]

        # Calculate average error and std deviation for each region
        region_stats = {}
        for region_code, bounds in regions.items():
            # Filter data for the current region
            region_data = gcm_data[gcm_data['gp_region'].str.contains(region_code)]
            mean_error = region_data['mat_vector'].mean()
            std_dev_error = region_data['mat_vector'].std()
            region_stats[region_code] = {'mean': mean_error, 'std_dev': std_dev_error}

        # Set up map for Average Errors
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent([-12, 35, 35, 72], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.COASTLINE)
        ax.set_title(f"Average Errors Across RCMs for GCM {gcm_id}")

        # Plot each region's average error
        for region_code, bounds in regions.items():
            mean_error = region_stats[region_code]['mean']
            ax.add_patch(plt.Rectangle(
                (bounds['west'], bounds['south']),
                bounds['east'] - bounds['west'],
                bounds['north'] - bounds['south'],
                transform=ccrs.PlateCarree(),
                color=plt.cm.viridis(mean_error / max([v['mean'] for v in region_stats.values()])),
                alpha=0.6,
                label=f"{regions[region_code]['name']} (Mean Error: {mean_error:.2f})"
            ))

            # Annotate region with mean error value
            ax.text(
                (bounds['west'] + bounds['east']) / 2,
                (bounds['south'] + bounds['north']) / 2,
                f"{region_code}\n{mean_error:.2f}",
                transform=ccrs.PlateCarree(),
                ha='center', va='center', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='black')
            )

        # Save the average error map
        gcm_folder = os.path.join(main_folder, f'gcm_id_{gcm_id}')
        os.makedirs(gcm_folder, exist_ok=True)
        mean_filename = os.path.join(gcm_folder, f'average_error_map_gcm_{gcm_id}.png')
        plt.savefig(mean_filename, format='png', dpi=300)
        plt.close()

        # Set up map for Standard Deviation of Errors
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.set_extent([-12, 35, 35, 72], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.COASTLINE)
        ax.set_title(f"Standard Deviation of Errors Across RCMs for GCM {gcm_id}")

        # Plot each region's standard deviation of error
        for region_code, bounds in regions.items():
            std_dev_error = region_stats[region_code]['std_dev']
            ax.add_patch(plt.Rectangle(
                (bounds['west'], bounds['south']),
                bounds['east'] - bounds['west'],
                bounds['north'] - bounds['south'],
                transform=ccrs.PlateCarree(),
                color=plt.cm.plasma(std_dev_error / max([v['std_dev'] for v in region_stats.values()])),
                alpha=0.6,
                label=f"{regions[region_code]['name']} (Std Dev: {std_dev_error:.2f})"
            ))

            # Annotate region with std deviation value
            ax.text(
                (bounds['west'] + bounds['east']) / 2,
                (bounds['south'] + bounds['north']) / 2,
                f"{region_code}\n{std_dev_error:.2f}",
                transform=ccrs.PlateCarree(),
                ha='center', va='center', fontsize=8,
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='black')
            )

        # Save the standard deviation map
        std_filename = os.path.join(gcm_folder, f'std_dev_error_map_gcm_{gcm_id}.png')
        plt.savefig(std_filename, format='png', dpi=300)
        plt.close()

        print(f"Saved average error and std deviation maps for GCM {gcm_id} in folder {gcm_folder}")

def correlation_analysis_by_region_and_gcm_granular(cons, region_code):
    # Create a folder for saving the correlation heatmaps
    main_folder = 'rcms_correlations_for_fixed_gcms_regions'
    os.makedirs(main_folder, exist_ok=True)

    # Filter data for the specified region
    region_data = cons[cons['gp_region'].str.contains(region_code)]

    # Extract unique GCMs within this region
    gcm_list = region_data['id_GCM'].unique()

    # Iterate over each GCM
    for gcm_id in gcm_list:
        # Filter data for the specific GCM
        gcm_data = region_data[region_data['id_GCM'] == gcm_id]

        # Dictionary to store correlations for each (metric, physical_variable) combination
        correlation_matrices = []

        # Group by Metric and physical_variable to calculate granular correlations
        for (metric, physical_variable), subset in gcm_data.groupby(['Metric', 'physical_variable']):
            # Pivot data to have RCMs as columns and gridpoints as rows
            pivot_data = subset.pivot_table(index='gp_region', columns='id_RCM', values='mat_vector')

            # Calculate pairwise correlations among RCMs for this specific metric and physical variable
            correlation_matrix = pivot_data.corr(method='pearson')
            correlation_matrices.append(correlation_matrix)

        # Average the correlation matrices across all (metric, physical_variable) combinations
        if correlation_matrices:
            averaged_correlation_matrix = sum(correlation_matrices) / len(correlation_matrices)

            # Determine the color scale based on the data in the correlation matrix
            min_corr = averaged_correlation_matrix.min().min()
            max_corr = averaged_correlation_matrix.max().max()

            # Plot the averaged correlation matrix as a heatmap with adjusted color scale
            plt.figure(figsize=(10, 8))
            sns.heatmap(averaged_correlation_matrix, annot=True, cmap="coolwarm", vmin=min_corr, vmax=max_corr, 
                        square=True, linewidths=0.5, linecolor='black', cbar=True)
            plt.title(f"Average Correlation of RCM Errors for GCM {gcm_id} in Region {region_code}")
            plt.xlabel("RCM")
            plt.ylabel("RCM")
            plt.xticks(rotation=45)
            plt.yticks(rotation=45)
            plt.tight_layout()

            # Save the heatmap for each GCM
            region_folder = os.path.join(main_folder, f'gcm_id_{gcm_id}')
            os.makedirs(region_folder, exist_ok=True)
            filename = os.path.join(region_folder, f'correlation_heatmap_gcm_{gcm_id}_region_{region_code}.png')
            plt.savefig(filename, format='png', dpi=300)
            plt.close()
            
            print(f"Saved averaged correlation heatmap for GCM {gcm_id} in region {region_code}.")

def hierarchical_clustering(cons, error_metric, region):

    # Create the folder for saving dendrograms
    output_folder = 'hierarchical_clustering'
    os.makedirs(output_folder, exist_ok=True)

    # Filter for the region and specified error metric
    region_data = cons[(cons['gp_region'].str.contains(region)) & (cons['Metric'] == error_metric)]

    # Pivot data to have RCMs as columns and gridpoints as rows
    pivot_data = region_data.pivot_table(index='gp_region', columns='id_RCM', values='mat_vector')

    # Drop any RCM columns that are entirely NaN (if some RCMs have no data for the British Isles)
    pivot_data = pivot_data.dropna(axis=1, how='all')

    # Fill any remaining NaN values with the column mean (optional, depending on data sparsity)
    pivot_data = pivot_data.fillna(pivot_data.mean())

    # Transpose the data so that RCMs are rows, ready for clustering
    rcm_data = pivot_data.T

    # Perform hierarchical clustering
    plt.figure(figsize=(12, 8))
    dendrogram = sch.dendrogram(sch.linkage(rcm_data, method='ward'))
    plt.title(f"Hierarchical Clustering of RCMs for {region} with Error Metric {error_metric}")
    plt.xlabel("RCM")
    plt.ylabel("Euclidean Distance")
    plt.tight_layout()

    # Save the dendrogram to the specified folder
    filename = os.path.join(output_folder, f'hierarchical_clustering_{region}_metric_{error_metric}.png')
    plt.savefig(filename, format='png', dpi=300)
    plt.close()

    print(f"Dendrogram saved as {filename}")

def hierarchical_clustering_fixed_gcm(cons, error_metric, region, gcm_id):
    # Create the folder for saving dendrograms
    output_folder = 'hierarchical_clustering_fixed_gcm'
    os.makedirs(output_folder, exist_ok=True)

    # Filter for the British Isles region ('BI'), specified error metric, and GCM 2
    data = cons[(cons['gp_region'].str.contains(region)) & 
                              (cons['Metric'] == error_metric) &
                              (cons['id_GCM'] == gcm_id)]

    # Pivot data to have RCMs as columns and gridpoints as rows
    pivot_data = data.pivot_table(index='gp_region', columns='id_RCM', values='mat_vector')

    # Check if we have sufficient data
    if pivot_data.empty:
        print("No data available for the British Isles with the specified error metric and GCM 2.")
        return

    # Drop any RCM columns that are entirely NaN (if some RCMs have no data for the region)
    pivot_data = pivot_data.dropna(axis=1, how='all')

    # Fill any remaining NaN values with the column mean (optional, depending on data sparsity)
    pivot_data = pivot_data.fillna(pivot_data.mean())

    # Transpose the data so that RCMs are rows, ready for clustering
    rcm_data = pivot_data.T

    # Perform hierarchical clustering
    plt.figure(figsize=(12, 8))
    dendrogram = sch.dendrogram(sch.linkage(rcm_data, method='ward'))
    plt.title(f"Hierarchical Clustering of RCMs for BI with Error Metric {error_metric} (GCM 2)")
    plt.xlabel("RCM")
    plt.ylabel("Euclidean Distance")
    plt.tight_layout()

    # Save the dendrogram to the specified folder
    filename = os.path.join(output_folder, f'hierarchical_clustering_{region}_metric_{error_metric}_gcm_{gcm_id}.png')
    plt.savefig(filename, format='png', dpi=300)
    plt.close()

    print(f"Dendrogram saved as {filename}")

def calculate_general_rcm_correlations(cons):
    # Dictionary to hold correlations and counts for each combination of physical_variable and metric
    temp_correlations = []
    temp_counts = []

    # Loop over each combination of physical_variable and metric
    for (phys_var, metric), subset in cons.groupby(['physical_variable', 'Metric']):
        # Pivot for the current subset
        pivot_data = subset.pivot_table(index='gp_region', columns='id_RCM', values='mat_vector')
        
        # Calculate pairwise correlations among RCMs for this subset
        correlation_matrix = pivot_data.corr(method='pearson')
        temp_correlations.append(correlation_matrix)

        # Calculate the pair counts (non-NaN pairs for each RCM pair)
        pair_counts_matrix = pivot_data.notna().T.dot(pivot_data.notna())
        temp_counts.append(pair_counts_matrix)

    # Average the correlation matrices across all combinations
    if temp_correlations:
        averaged_correlation_matrix = sum(temp_correlations) / len(temp_correlations)
    
    # Sum the counts matrices across all combinations
    if temp_counts:
        averaged_counts_matrix = sum(temp_counts)

    print("General correlation between RCMs calculated.")
    print("Pair counts for each RCM pair calculated.")
    return averaged_correlation_matrix, averaged_counts_matrix

def visualize_general_rcm_correlations(correlation_matrix, folder_name='general_rcm_correlations'):

    os.makedirs(folder_name, exist_ok=True)

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                cbar=True, square=True, linewidths=0.5, linecolor='black')
    
    plt.title("General Correlations Among RCMs")
    plt.xlabel("RCM")
    plt.ylabel("RCM")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    
    # Save the plot to the specified folder
    filename = os.path.join(folder_name, 'general_rcm_correlation_heatmap.png')
    plt.tight_layout()
    plt.savefig(filename, format='png', dpi=300)
    plt.close()  # Close the figure to avoid display in the notebook/environment

    print(f"Saved general correlation heatmap to {filename}")

def visualize_pair_counts(pair_counts_matrix, folder_name='general_rcm_correlations'):

    os.makedirs(folder_name, exist_ok=True)

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(pair_counts_matrix, annot=True, fmt=".0f", cmap="YlGnBu", 
                cbar=True, square=True, linewidths=0.5, linecolor='black')
    
    plt.title("Pair Counts for RCM Correlations")
    plt.xlabel("RCM")
    plt.ylabel("RCM")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    
    # Save the plot to the specified folder
    filename = os.path.join(folder_name, 'general_rcm_pair_counts_heatmap.png')
    plt.tight_layout()
    plt.savefig(filename, format='png', dpi=300)
    plt.close()

    print(f"Saved pair counts heatmap to {filename}")