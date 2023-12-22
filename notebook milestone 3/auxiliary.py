import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter
import seaborn as sns
from typing import Callable, List
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

############################################## Part 1 : auxiliary functions ##############################################
def create_link_count_hist(links_count, bins, incoming=False):
    """
    This function creates and displays a histogram for the distribution of the number of links per article.

    Parameters:
    links_count (iterable): An iterable (like a list or array) containing the count of links for each article.
    bins : the bin edges for the histogram.
    incoming (bool, optional): A flag to determine the type of links to be displayed in the histogram.
                               If True, the histogram will show incoming links, otherwise outgoing links.
                               Defaults to False, which means outgoing links are displayed by default.

    Returns:
    None: This function does not return anything. It displays a histogram plot.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(links_count, bins= bins, edgecolor='black')
    plt.title(f'Distribution of Number of {"Incoming" if incoming else "Outgoing"} Links per Article')
    plt.xlabel(f'Number of {"Incoming" if incoming else "Outgoing"} Links')
    plt.ylabel('Number of Articles')
    plt.xticks(bins)
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    
def get_top_articles_by_page_rank(page_rank, top_n=20):
    """
    This function takes the result of the page rank algorithm and returns the top N articles
    with the highest page rank.

    Args:
    page_rank (dict): A dictionary with page ranks where keys are article identifiers
                      and values are the page ranks.
    top_n (int): The number of top articles to return.

    Returns:
    list: A list of tuples containing the top N articles and their page ranks.
    """
    keys = list(page_rank.keys())
    vals = list(page_rank.values())
    
    # Sorting the page_rank dictionary based on values and getting the sorted keys
    sorted_indices = np.argsort(vals)[::-1]
    sorted_page_rank = {keys[i]: vals[i] for i in sorted_indices}
    
    # Extracting the sorted keys and values
    sorted_keys = list(sorted_page_rank.keys())
    sorted_vals = list(sorted_page_rank.values())

    # Collecting the top N articles
    top_articles = []
    
    # Print the top N articles
    print('The top {} articles in terms of page rank are:'.format(top_n))
    for i in range(top_n):
        top_articles.append(sorted_keys[i])
        print(sorted_keys[i], ' : ', sorted_vals[i])
        
    return top_articles


def remove_unvisited_pages(path):
    """
    Removes markers for unvisited pages from the path.
    
    This function iterates over the path list and removes the back clicks
    
    Args:
    path (list): the article path potentially containing back clicks
    
    Returns:
    list: the article path filteres from back clicks
    """

    # Check if the path contains any '<' characters; if not, return the path as is.
    if path.count('<') == 0:
        return path

    # Start iterating over the path list.
    i = 0
    while i < len(path):
        # If the current character is '<', indicating the start of a sequence of unvisited pages.
        if path[i] == '<':
            counter = 0  # Initialize a counter for consecutive '<' characters.
            tmp_i = i  # Temporary index to track the position in the path.
            # Count the number of consecutive '<' characters.
            while tmp_i < len(path) and path[tmp_i] == '<':
                tmp_i += 1
                counter += 1
            # Calculate the starting index for removal.
            # It ensures that the number of characters removed is equal to the number of '<' found.
            m = max(0, i - counter)

            # Replace visited page markers with '<' before the current sequence of unvisited pages.
            for j in range(m, i):
                path[j] = '<'
            # Set the index to the end of the current sequence of unvisited pages.
            i = tmp_i
        else:
            # If the current character is not '<', move to the next character.
            i += 1

    # Find all indices of '<' characters in the path.
    indx = np.where(np.array(path) == '<')[0]
    # Remove all '<' characters from the path using numpy's delete function.
    path = np.delete(np.array(path), indx)
    # Convert the numpy array back to a list and return it.
    return list(path)


def parse_and_clean_path(row):
    """
    Parses and cleans a path 
    
    Args:
    row (str): the article path that needs to be parsed
    
    Returns:
    list: the parsed path in list format
    """
    path_split = row.split(';')
    path_split = remove_unvisited_pages(path_split)
    return path_split


def pad_path(path_split,max_path_length = 407):
    """
    We pad the paths with zeros to allow comparison between aths with different lengths
    """
    padding_length = max_path_length - len(path_split)
    return path_split + [0] * padding_length if padding_length > 0 else path_split


def calculate_path_lengths(paths_df):
    """
    calculating the length of a path   
    """
    paths_length = paths_df.apply(lambda x: (x != 0).sum(), axis=1)
    paths_df['length'] = paths_length
    return paths_length.value_counts()

def plot_path_length_frequencies(frequencies):
    """
    computing the frequency of path lengths
    """
    plt.bar(frequencies.index, frequencies.values)
    plt.xticks(rotation=90)
    plt.title('Paths having the same number of visited pages')
    plt.ylabel('# Paths')
    plt.xlabel('Path Length')
    plt.xlim(0, 30)
    plt.show()

############################################## End Part 1 : auxiliary functions ##############################################

############################################## Part 2 : auxiliary functions ##############################################


def plot_avg_page_rank_for_path_length(paths_df, path_length, ax):
    """
    Plots the average page rank for paths of a specific length on the given axes.

    Args:
    paths_df (DataFrame): A pandas DataFrame containing paths data where each row is a path and 
                          each column represents a step in the path, with values being page ranks.
    path_length (int): The length of the paths to filter and plot.
    ax (matplotlib.axes.Axes): The matplotlib Axes object where the plot will be drawn.

    Returns:
    None: The function directly plots on the provided Axes object.
    """
    # Filter paths of the specified length
    paths_fixed_len = paths_df.copy()[paths_df['length'] == path_length]

    # Drop columns beyond the specified path length
    columns_to_drop = paths_fixed_len.columns[path_length:]
    new_paths_fixed_len = paths_fixed_len.drop(columns_to_drop, axis=1)

    # Calculate the mean page rank for each step in the path
    x = new_paths_fixed_len.mean(axis=0)
    x_vals = np.arange(len(x))

    # Plotting
    ax.plot(x.values)
    ax.set_xticks(x_vals)
    ax.set_xlabel('Index of Intermediate Article', fontsize=8)
    ax.set_ylabel('Average Page Rank', fontsize=8)
    ax.set_title(f'Average Page Rank for Paths of Length {path_length}')


def plot_categories_frequencies(df, column_name):
    """
    Plots the frequency of each unique value in the specified column of the DataFrame.

    Args:
    df (DataFrame): The pandas DataFrame containing the data.
    column_name (str): The name of the val in the DataFrame to analyze.

    Returns:
    None: The function directly plots the frequency of categories.
    """
    # Calculate unique categories and their frequencies
    unique_categories = df[column_name].drop_duplicates()
    print(len(unique_categories), 'different categories')
    freq = df[column_name].value_counts()

    # Plotting
    plt.figure(figsize=(20, 8))
    sns.barplot(x=freq.index, y=freq.values, width=0.8)
    plt.xticks(rotation=90, fontsize=7)
    plt.title('Number of articles per category')
    plt.ylabel('# of articles')
    plt.show()
    

def count_periods(s):
    """
    Count the number of periods in a string 
    
    Args:
    s (str): the string 
    
    Returns:
    count : This function plots a pie chart and does not return anything.
    """
    return s.count('.')    

def plot_top_categories_pie_chart(categories, values, colors, top_n=10):
    """
    Plots a pie chart of the top N categories based on provided values.

    Args:
    categories (list): A list of category names.
    values (list): A list of values corresponding to each category.
    top_n (int, optional): The number of top categories to include in the pie chart. Defaults to 10.

    Returns:
    None: This function plots a pie chart and does not return anything.
    """
    # Sort the categories and values
    idx_sorted = np.argsort(values)
    categories_sorted = np.array(categories)[idx_sorted]
    values_sorted = np.array(values)[idx_sorted]

    # Plot pie chart
    plt.pie(values_sorted[-top_n:], autopct='%1.1f%%', startangle=-33, colors=colors[-top_n:])
    plt.title(f'Top {top_n} Categories of Top Articles in Page Rank')
    plt.legend(labels=categories_sorted[-top_n:], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()



def process_paths_on_rank(paths, page_rank ,upPath = True) :
    """
    Processes and filters paths based on their page ranks and a directional flag.

    For each path in the provided DataFrame, the function splits the path into components,
    removes any unvisited pages, and then calculates the rank for each page component using the 
    provided page_rank mapping. It then selects a portion of the path based on the highest rank 
    and the value of the upPath flag: if upPath is True, it takes the path up to the highest rank; 
    otherwise, it takes the path from the highest rank onwards. The function also tracks pages 
    without a rank.

    Args:
    paths (DataFrame): A pandas DataFrame containing paths in a column named 'path'.
    page_rank (dict): A dictionary mapping page names to their respective ranks.
    upPath (bool, optional): Determines the portion of the path to be processed. If True, 
                             processes the segment up to the highest rank. Otherwise, processes 
                             the segment from the highest rank onwards. Defaults to True.

    Returns:
    tuple: A tuple containing two lists. The first list contains the processed paths, and the 
           second list contains the names of pages that were found to have no rank.
    """
    processed_paths = []
    pages_with_no_rank = []
    for i, s in paths.iterrows():
        path_split=(paths['path'].iloc[i]).split(';')
        path_split = remove_unvisited_pages(path_split)
        ranks = []
        for elem in path_split:
            r = page_rank.get(elem, -1)
            if r==-1:
                pages_with_no_rank.append(elem)
            ranks.append(r)
        if upPath : 
            processed_paths.append(path_split[:np.argmax(ranks)])
        else : 
            processed_paths.append(path_split[np.argmax(ranks) :])
            
    return processed_paths, pages_with_no_rank


def extract_downpath(path,page_rank):
    path_split = path.split(';')
    path_split = remove_unvisited_pages(path_split)
    ranks = []
    for elem in path_split:
        r = page_rank.get(elem, -1)
        ranks.append(r)
    return path_split[np.argmax(ranks) :]


def build_category_connections(category_paths) :
    """
    Builds a matrix of connections between categories based on provided paths.

    This function creates a pairwise counter for consecutive categories in each path and then 
    uses this information to populate a DataFrame representing connections between categories.
    The resulting DataFrame is a matrix where the index and columns are unique categories and 
    the cell values are the counts of direct transitions from one category to another.

    Args:
    category_paths (list of lists): A list where each element is a path represented as a list of categories.

    Returns:
    DataFrame: A pandas DataFrame representing the connection strengths between categories.
    """
    category_pairs_counter = defaultdict(int)

    # process each path
    for path in category_paths:
        for i in range(len(path) - 1):
            # Increment the counter for each found pair
            pair = (path[i], path[i + 1])
            category_pairs_counter[pair] += 1

    # unique category list that appear in our paths, 
    # no need for all categories of articles in  

    unique_categories = list(set(cat for path in category_paths for cat in path))

    # initialize an the connections
    category_connections = pd.DataFrame(index=unique_categories, columns=unique_categories).fillna(0)

    # filling up the values with the counts
    for (cat1, cat2), count in category_pairs_counter.items():
        category_connections.loc[cat1, cat2] = count
    
    return category_connections




def heatmap_general_categories(category_connections) :
    """
    Creates and displays a heatmap for the general category connections.

    This function processes the category connections DataFrame by removing a specific category 
    (e.g., 'Unknown_Category'), adjusting the category labels, and aggregating the connections 
    under general categories. It then normalizes the data and plots a heatmap to visualize 
    the strength of connections between these general categories.

    Args:
    category_connections (DataFrame): A DataFrame representing connections between categories.

    Returns:
    DataFrame: A transformed DataFrame used for plotting the heatmap.
    """
    df = category_connections.copy() 
    df = df.drop(index = "Unknown_Category")
    df = df.drop(columns = "Unknown_Category")
    df = df.fillna(0)
    #Remove 'subject.' prefix since it is common to all categories
    prev_list = [ li  for li in df.columns]
    
    unique_vals = list(set(stri.split('.')[1] for stri in df.index))
    print(unique_vals)
    final_df = pd.DataFrame(index = unique_vals, columns = unique_vals)
    final_df = final_df.fillna(0)
    

    for ind1,row1 in df.iterrows() :
        for value in prev_list :
            final_df.loc[ind1.split('.')[1],value.split('.')[1]] =  final_df.loc[ind1.split('.')[1],value.split('.')[1]] + row1[value]
      
    # Replacing diagonal elements with 0
    
    rows, cols = final_df.shape
    for i in range(rows):
        final_df.iloc[i, i] = 0  
    
    
        
    for ind, row in final_df.iterrows():
        max = row.max()
        final_df.loc[ind]=(final_df.loc[ind])*100./max
            
      
    

    #final_df = final_df.div(final_df.sum(axis=1), axis=0)    
    #plotting the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(final_df, annot=False, fmt='.2f', cmap="YlGnBu", cbar=True)
    plt.title('Heatmap of Weights between Categories')
    plt.show()   
    return final_df


def heatMap_Special_Category(category,category_connections) : 
    """
    Creates and displays a heatmap for a specific category within the category connections.

    This function focuses on a specific category and its subcategories, creating a filtered heatmap 
    to visualize the internal connections within this category. It processes the category connections 
    DataFrame by removing prefixes, filtering based on the specified category, normalizing the data, 
    and then plotting a heatmap to show the connections between the subcategories.

    Args:
    category (str): The specific category to focus on.
    category_connections (DataFrame): A DataFrame representing connections between categories.

    Returns:
    None: The function plots a heatmap and does not return anything.
    """
    df = category_connections.copy() 
    
    #Remove 'subject.' prefix again
    df.index = df.index.str.replace('subject.', '')
    df.columns = df.columns.str.replace('subject.', '')
    #filtering   all the categories to keep only those that belong to the general one
    df = df.filter(like=category, axis=1).filter(like=category, axis=0)
    
    #Again dropping all the diagonal values for the same reasons as before
    np.fill_diagonal(df.values, 0)
    df = df.div(df.sum(axis=1), axis=0)
    #Plotting the heatMap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, fmt='.2f', cmap="YlGnBu", cbar=True)
    plt.title('Heatmap of Weights between Categories of %s' %category)
    plt.show()   



def plot_degree_distribution(finished_paths, unfinished_paths):
    """
    Plots the degree distribution for finished and unfinished paths.

    Parameters:
        finished_paths (list): List of in-degrees for finished paths.
        unfinished_paths (list): List of in-degrees for unfinished paths.
    """
    sns.histplot(finished_paths, kde=True, stat='density', color='orange', label='finished_paths', log_scale=True)
    sns.histplot(unfinished_paths, kde=True, stat='density', color='blue', label='unfinished_paths', log_scale=True)
    plt.legend()
    plt.ylabel("Distribution")
    plt.xlabel("In-degrees")
    plt.show()


############################################## End Part 2 : auxiliary functions ##############################################


############################### Part 3 : classifier auxiliary functions ##################################################### 
  
def compute_games_played(finished_paths, unfinished_paths):
    unique_ips = pd.concat([finished_paths["hashedIpAddress"], unfinished_paths["hashedIpAddress"]]).unique()
    dicti = {key: 0 for key in unique_ips}
    for indx, row in finished_paths.iterrows():
        dicti[row["hashedIpAddress"]] += 1
    for indx, row in unfinished_paths.iterrows():
        dicti[row["hashedIpAddress"]] += 1
    return dicti

# Define a function to label IPs based on counts
def user_games_played(finished_paths, unfinished_paths, dicti):
    p1 = finished_paths["hashedIpAddress"].apply(lambda t: dicti.get(t, 0))
    p2 = unfinished_paths["hashedIpAddress"].apply(lambda t: dicti.get(t, 0))
    return p1, p2


# Define a function to get hub ranks
def get_hub_ranks(processed_finished_paths_downPath, processed_unfinished_paths_downPath, page_rank):
    l_finished = [page_rank.get(elem[0], -1) for elem in processed_finished_paths_downPath]
    l_unfinished = [page_rank.get(elem[0], -1) for elem in processed_unfinished_paths_downPath]
    return l_finished, l_unfinished




# Define a function to standardize the data
def standardize_data(X, p1, p2):
    X = (X - X.mean()) / X.std()
    X["num_of_games"] = pd.concat([p1, p2])
    return X

def calculate_and_update_indegrees(X, finished_paths, unfinished_paths, graph):
    unfinished_indegrees = unfinished_paths["target"].apply(lambda t: graph.in_degree(t)).fillna(0).apply(lambda t: t if isinstance(t, (int)) else 0) + 1
    finished_indegrees = finished_paths["path"].apply(lambda t: graph.in_degree(t.split(";")[-1])).apply(lambda t: t if isinstance(t, (int)) else 0) + 1
    X["indegrees"] = pd.concat([finished_indegrees, unfinished_indegrees])
    return X


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model



def evaluate_model(model, X_test, y_test):
    predictions = model.predict_proba(X_test)
    predictions = np.where(predictions[:, 1] < 0.5, 0, 1)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report
# Define a function to create a DataFrame with feature coefficients
def feature_coefficients(model, X_train):
    tmp = []
    for feature, coef in zip(X_train.columns, model.coef_[0]):
        tmp.append({"feature": feature, "coefficient": coef})
    features_coef = pd.DataFrame(tmp).sort_values("coefficient")
    return features_coef

# Define a function to plot feature coefficients
def plot_feature_coefficients(features_coef):
    plt.subplots(figsize=(5, 7))
    plt.barh(features_coef.feature, features_coef.coefficient, alpha=0.6)
    plt.title('Importance of features')
    plt.show()

# Function to get targets and starting points
def get_targets_and_starting_points(finished_paths, unfinished_paths):
    finished_targets = finished_paths["path"].apply(lambda path: path.split(';')[-1])
    finished_starting = finished_paths["path"].apply(lambda path: path.split(';')[0])
    unfinished_targets = unfinished_paths["target"]
    unfinished_starting = unfinished_paths["path"].apply(lambda path: path.split(';')[0])
    return finished_targets, finished_starting, unfinished_targets, unfinished_starting


# Function to associate targets and starting points with page ranks
def associate_targets_and_starting_points_with_ranks(targets, starting_points, page_rank):
    targets_rank = targets.apply(lambda target: page_rank.get(target, 0))
    starting_rank = starting_points.apply(lambda target: page_rank.get(target, 0))
    return targets_rank, starting_rank


############################### End Part 3 auxiliary functions ######################################## 

############################### Part 4 :  auxiliary functions ######################################### 
from matplotlib.colors import Normalize
from sklearn.manifold import MDS

def category_selection_from_mult(df, specific_categories, cat):
    df_mult = df.copy()
    
    # Filter the dataframe to keep only the paths that aim at the specified category
    df_mult["specific_categ"] = [specific_categories.loc[t[-1], 'category'] if t[-1] in specific_categories.index else 'Unknown_Category'  for t in df_mult["path"]]    
    df_mult = df_mult[df_mult["specific_categ"] == cat]
    print("There are ",df_mult.shape[0]," targets of this specific category")
    
    # Sort the dataframe according to the timestamp
    df_mult = df_mult.sort_values(["timestamp"])
    
    # Store the number of clicks in each path
    df_mult['number_clicks'] = df_mult['path'].apply(lambda x: len(x)-1)
    
    return df_mult

def dividing_df_mult(df, specific_categories, cat):
    df_mult = df.copy()

    # Divide the dataframe in two parts: 
    #      * The first quarter of the attempts of each player is considered as the training part
    #      * The second half of the attempts of each player is taken as the paths where the player has learned 

    fraction = 1/4
    df_mult_start = df_mult.groupby('hashedIpAddress').apply(lambda x: x.head(int(len(x)*fraction))).reset_index(drop=True)
    df_mult_end = df_mult.groupby('hashedIpAddress').apply(lambda x: x.tail(int(len(x)*fraction*2))).reset_index(drop=True)
    
    # Filter the dataframes --> take only the specific category that corresponds to cat
    df_start = category_selection_from_mult(df_mult_start, specific_categories, cat)
    df_end = category_selection_from_mult(df_mult_end, specific_categories, cat)

    # Balance the data
    min_size = min(df_start.shape[0],df_end.shape[0])
    df_start = df_start.head(min_size)
    df_end = df_end.head(min_size)

    print("After reshaping, there are", df_start.shape[0]," targets of the specific category", cat)
    print("After reshaping, there are", df_end.shape[0]," targets of the specific category", cat)

    return df_start,df_end

def dividing_df_compare_fin_unfin(df_fin, df_un, specific_categories, cat):
    df_mult_start = df_fin.copy()
    df_mult_end = df_un.copy()
    
    # Filter the dataframes --> take only the genral category that corresponds to cat
    df_start = category_selection_from_mult(df_mult_start, specific_categories, cat)
    df_end = category_selection_from_mult(df_mult_end, specific_categories, cat)

    # Balance the data
    min_size = min(df_start.shape[0],df_end.shape[0])
    df_start = df_start.head(min_size)
    df_end = df_end.tail(min_size)

    print("After reshaping, there are", df_start.shape[0]," targets of the specific category", cat)
    print("After reshaping, there are", df_end.shape[0]," targets of the specific category", cat)
    
    return df_start,df_end
    

def weighting_links(df, specific_categories):
    df_mult_graph = df.copy()
    df_mult_graph['time_per_click'] = df_mult_graph['durationInSec']/df_mult_graph['number_clicks']
    df_mult_graph['path'] =  df_mult_graph['path'].apply(lambda x: [specific_categories.loc[elem, 'category'].split('.')[-1] if elem in specific_categories.index else 'Unknown_Category' for elem in x])
    dict_edges={}
    for _,elem in df_mult_graph.iterrows():
        p = elem['path']
        count = elem['number_clicks']
        t = elem['time_per_click'] 
        for i in range(1,len(p)):
            # Check if node1 exists in list_edges, if not, create it
            node1 = p[i-1]
            if node1 not in dict_edges:
                dict_edges[node1] = {}
            node2 = p[i]
            # Check if node2 exists as a key in the dictionary corresponding to node1, if not, create it
            if node2 not in dict_edges[node1]:
                dict_edges[node1][node2] = 0  # Initialize edge weight
            dict_edges[node1][node2] += ((i/count)*1/t)

    all_weights = [weight for edge in dict_edges.values() for weight in edge.values()]

    threshold = pd.Series(all_weights).quantile(0.5) # Define your threshold here

    # Filter nodes based on threshold
    filtered_edges = {k: {k1: v1 for k1, v1 in v.items() if v1 >= threshold} for k, v in dict_edges.items() if any(val >= threshold for val in v.values())}

    # Conversion of weights to distances
    for source, targets in filtered_edges.items():
        for target, weight in targets.items():
            filtered_edges[source][target] = abs(np.log(1/(2*weight)))
    return filtered_edges

def study_weights(dict):
    all_weights = [weight for edge in dict.values() for weight in edge.values()]
    print("mean  ",np.mean(all_weights))
    print("std  ",np.std(all_weights))
    # Plotting the histogram
    plt.hist(all_weights, bins=30, color='skyblue', edgecolor='black')  # Adjust the number of bins as needed
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Histogram')
    plt.grid(True)
    plt.show()

def nodes_positioning(df, specific_categories):
    filtered_edges = weighting_links(df,specific_categories)
    # Extract nodes from distances_dict
    nodes = set(filtered_edges.keys())

    # Create an empty graph
    G = nx.Graph()

    # Add nodes to the graph
    G.add_nodes_from(nodes)

    # Initialize default distance for non-connected nodes
    default_distance = 100.0  # You can adjust this default distance value

    # Create a matrix of pairwise distances with default distance for non-connected nodes
    num_nodes = len(nodes)
    distance_matrix = np.full((num_nodes, num_nodes), default_distance)
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if j > i:
                d = np.min([filtered_edges.get(node1, {}).get(node2, filtered_edges.get(node2, {}).get(node1, default_distance)), filtered_edges.get(node1, {}).get(node2, filtered_edges.get(node1, {}).get(node2, default_distance))])
                distance_matrix[i][j] = d
                distance_matrix[j][i] = d

    # Initialize and fit the MDS model
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    node_positions = mds.fit_transform(distance_matrix)

    # Create a dictionary of positions for each node
    pos = {node: (node_positions[i, 0], node_positions[i, 1]) for i, node in enumerate(nodes)}
    return pos

def distance_from_target(df, specific_categories, central_point):
    pos = nodes_positioning(df, specific_categories)

    # Calculate distances from central_node to other nodes
    distances_to_C = {node: np.linalg.norm(np.array(pos[node]) - np.array(pos[central_point])) for node in pos if node != central_point}

    # Sort nodes by their distance to central_node
    sorted_nodes_by_distance = sorted(distances_to_C.keys(), key=lambda x: distances_to_C[x])

    # Get the list of distances
    distances_list = [distances_to_C[node] for node in sorted_nodes_by_distance]

    # Create scatter plot data
    x_values = [pos[node][0] for node in sorted_nodes_by_distance]
    y_values = [pos[node][1] for node in sorted_nodes_by_distance]
    
    central_position = pos[central_point]

    return [distances_list, x_values, y_values, central_position]


def plot_distance_from_target(df, specific_categories, cat, compare_training = True, df_mult_unfinished=None):
    if not compare_training:
        df_start, df_end = dividing_df_compare_fin_unfin(df, df_mult_unfinished, specific_categories, cat)
    else:
        df_start, df_end = dividing_df_mult(df, specific_categories, cat)

    central_point = cat.split('.')[-1]
    [d_start, x_start, y_start,central_start] = distance_from_target(df_start, specific_categories, central_point)
    [d_end, x_end, y_end, central_end] = distance_from_target(df_end, specific_categories, central_point)
    
    # Determine the overall min and max values for normalization
    min_val = min(min(d_start), min(d_end))
    max_val = max(max(d_start), max(d_end))

    # Normalize colors based on the overall min and max values
    norm = Normalize(vmin=min_val, vmax=max_val)

    # Invert the colormap 'coolwarm' to have blue for close and red for far
    plt_cm = plt.cm.get_cmap('coolwarm')
    new_cmap = plt_cm.reversed()

    # Create a figure and subplots with a horizontal layout (1 row, 2 columns)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [0.1, 0.1]})

    # Plotting data in the first panel (left)
    ax[0].scatter(x_start, y_start, c=d_start, cmap=new_cmap, s=500, edgecolor='black',norm = norm)
        # Add labels to nodes

    # Plot the central point separately with a distinct color
    ax[0].scatter(central_start[0], central_start[1], color='yellow', s=500, edgecolor='black', marker='*')
    ax[0].text(central_start[0], central_start[1], central_point, ha='center', va='center', color='black', fontsize=8)
    
    if compare_training:
        ax[0].set_title('Before training')
    else: 
        ax[0].set_title('Finished Paths')
    ax[0].set_xlabel('X-axis')
    ax[0].set_ylabel('Y-axis')

    # Plotting data in the second panel (right)
    scatter_end = ax[1].scatter(x_end, y_end, c=d_end, cmap=new_cmap, s=500, edgecolor='black', norm = norm)

    # Plot the central point separately with a distinct color
    ax[1].scatter(central_end[0], central_end[1], color='yellow', s=500, edgecolor='black', marker='*')
    

    ax[1].text(central_end[0], central_end[1], central_point, ha='center', va='center', color='black', fontsize=8)
    if compare_training:
        ax[0].set_title('After training')
    else: 
        ax[0].set_title('Unfinished Paths')
    ax[1].set_xlabel('X-axis')
    ax[1].set_ylabel('Y-axis')

    # Create an axis for the colorbar and position it between the subplots
    cbar_ax = fig.add_axes([0.25, -0.05, 0.5, 0.05])  # Adjust these values for desired positioning
    plt.colorbar(scatter_end, cax=cbar_ax, orientation='horizontal', label='Distance to Central Point', norm=norm)

    # Adjust layout spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

############################### End Part 4 :  auxiliary functions ######################################### 


############################### Case study #############################
def case_study_preprocessing(case_study, cat):
    df_case_study = case_study.copy().dropna()
    df_case_study = df_case_study[df_case_study["general_categ"] == cat]
    df_case_study = df_case_study[(df_case_study['durationInSec'] < df_case_study.durationInSec.quantile(0.95))\
                                  & (df_case_study['num_clicks'] < df_case_study['num_clicks'].quantile(0.95)) ]
    df_case_study['time_per_click'] = df_case_study['durationInSec'] / df_case_study['num_clicks']
    return df_case_study