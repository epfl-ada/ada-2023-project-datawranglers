# Ada-2023-project-datawranglers
## Navigating the Maze of Mind: Analyzing Player Behavior and Cognitive Patterns in the Wikispeedia Game 🧠

### ABSTRACT :thought_balloon:

The human brain, with its complex networks and cognitive processes, remains a scientific mystery. In our quest to uncovering those details, we turn to an innovative approach: analyzing player behavior in the Wikispeedia game. TThe game where players move through Wikipedia articles to reach a goal helps us study how people process new information and make quick decisions, giving us a closer look into the most intuitive conenctions in their brain.

Our research focuses on analyzing these navigational choices and how they influence game outcomes. To make a more complete analysis we also move on to identify the critical factors that differentiate success from failure in the game setting other than user logic. By meticulously examining player paths and category selections, we gain insights into the cognitive patterns and strategies employed. 
We additionnaly employ network analysis techniques to map the connections between specific categories of information. This allows us to not only understand player behavior in this game context but it makes mapping the proximity of specific categories in the brain space possible.

### Research Questions :mag:

* By analyzing the path choices of users can we conclude a general rule of behaviour ? 

* Can we prove that the success of the gameplay is related to said behavior? Or is it rather other factors controling the outcome? 

* Are there some semantic links that players make between general categories?

* How does the human brain interpret and distinguish specific categories within a broad domain like science ?

* What methods can we employ to visually represent these semantic connections and understandings? 

### Methods :microscope:

**I- Preprocessing**

1) We constructed dataframes storing the links, their categories, the finished and the unfinished paths. We then create a directed graph of the links from the sources to the target. This graph is used in the pagerank algorithm associating a rank to each page based on its connectivity with other pages within the graph. 

2) We move to cleaning the paths to remove the back clicks and the deleted paths which are not relevant for our study since they do not provide any significant information on the player’s reflexion nor on the recipe of success. 

3) We uncover the category distribution on article giving us insights into how we should transform path of articles into paths of cateories to extract the most information about player reasoning.  


**II- Uncovering player reasoning**

In this segment, our focus leans towards studying the behavior of successful players within the game. This choice is predicated on the assumption that their decision-making processes might exhibit greater stability or consistency. 

Strategy: To explore the potential connection between a player's strategy and the pagerank of selected links, a list is derived. This list includes paths with links represented by their corresponding pageranks. From this data, line plots are generated to illustrate how the pagerank of chosen links evolves throughout the games. 

To ensure a comprehensive representation, this analysis is conducted on multiple path lengths. 

We then move on to examining semantic links: The categories are presented in ascending order of specificity. Initially, we will investigate the semantic connections among the general categories before delving into potential logical relationships within each of these categories.

We additionnaly generate various heat maps in order to detail the connectivity between two general categories. We deliberately nullify the value for identical categories due to it's lack of significance to our study. we apply a max-normalization technique to all other values. This crucial step neutralizes the impact of their frequency of occurrence, allowing the heat maps to underscore the inherent relationships between categories, independent of how often they appear.

The previous process is meticulously repeated for two specific segments: the part of the completed paths leading from their origin to the point of highest pagerank : The UP PATH, and the segment from this peak pagerank position to the ultimate destination : The DOWN PATH.

**III- Key for success**

The analysis made in the previous part showed that both finished and unfinshed paths have exactly the same pattern ; players initially navigate to an article with a high page rank score. Subsequently, they refine their strategy and direct their efforts towards their specific target article. This suggests that the difference between finished and unfinshed paths is not related to the starting point, since we always reach a hub starting from any article. the main difference lies in the final destination article, in this section we will try to analyse differences between the destinations of both finished and unfinished paths by comparing the distribution of their in-degrees,we will then analyze what influences the success or failure by building a success prediction model with the following features: target_rank : the page rank of the target page starting_rank the page rank of the starting page hub_rank: the page rank of the hub indegrees: number of ingoing edges num_of_games: number of played games.

**IV- Deeper dive into Cognitive Mappings**

In this final section, we will take a step back from the game difficulty and focus on analyzing further the cognitive connections between closely related categories. We will provide analysis for science-related categories, however note that this same procedure can be further generalized to other fields We focus on exploring the ‘Science’ category by considering completed paths where the destination is within the field of science. Each specific category encountered along these paths that is lying inside the general ‘Science’ category strengthens its link with the specific destination category. However, this reinforcement is proportional to the category's distance from the final destination. Specific categories closer to the final destination within the path receive higher similarity scores. This method results in constructing a matrix containing similarity scores between various categories solely within the realm of science.



### Proposed timeline :hourglass:

22/11/2023: Part 1 and 2

29/11/2023: Part 3 and 4

06/12/2023: Write Story and Create initial Website 

13/12/2023: Finalization of the Website

20/12/2023: Review

(Deadline: 22/12/2023)

### Organization within the team :busts_in_silhouette:

* Datastory: Everybody

* Visualizations: Veronika

* Part1: Leila

* Part2: Aymane

* Part3: Abdessalam

* Part4: Salim

* Website: Abdessalam, Aymane and Leila
