# ada-2023-project-datawranglers
Navigating the Maze of Mind: Analyzing Player Behavior and Cognitive Patterns in the Wikispeedia Game

ABSTRACT:

The human brain, with its complex networks and cognitive processes, remains a scientific mystery. In our quest to uncovering those details, we turn to an innovative approach: analyzing player behavior in the Wikispeedia game. The game where players navigate through Wikipedia articles with a target in mind, serves as a studying information processing and decision-making strategies. Our research analyzes these navigational choices influence game outcomes, we then move on to identify the critical factors that differentiate success from failure in reaching the designated targets. By meticulously examining player paths, article selections, and time taken for each decision, we gain insights into the cognitive patterns and strategies employed. Furthermore, we employ advanced statistical and network analysis techniques to map the connections between specific categories of information. This allows us to not only understand player behavior in the game context but also to draw broader implications about how humans process and link disparate pieces of information. Ultimately, this study aims to contribute to the broader understanding of cognitive patterns and information navigation in the brain's complex landscape."





Embark on a captivating journey into the depths of the Wikispeedia dataset as we discover the secrets behind winning or losing in this interesting game of navigation. Wikispeedia is a labyrinth of Wikipedia articles that challenges players to find optimal pathways from a source to a target article. This project serves as an expedition seeking to uncover the semantic and logical connections that players forge between different categories in the Wikispeedia game. Our objective is to gain a comprehensive understanding of how the human brain intuitively perceives and categorizes information, by studying the implicit relationships and patterns in player navigation choices, as well as uncover the principle factors that dictate triumph or defeat within this dynamic gaming environment.Additionally, the research dives into user behaviors during gameplay to extract valuable insights into effective navigation strategies across multiple article categories. This README serves as a guide to the dataset, methodologies employed and key insights.

Research questions:

As the most important rule in ADA is to be critical towards data, are the conclusions of West et al. (2009) correct? Can we prove that the players tend to go from specific links to general and back to specific to reach the target? Are there some semantic links that players make between general categories? Are there more targeted semantic links between the specific categories lying inside a general category? How does the human brain interpret and distinguish specific categories within a broad domain like science? Additionally, what methods can we employ to visually represent these semantic connections and understandings? What makes the difference between success and failure in this game? Is failure related to a wrong strategy or is it related to the difficulty of the game? Does it depend on the endpoints or the player experience?
I- Preprocessing:

In this preliminary step, we constructed dataframes storing the links, their categories, the finished and the unfinished paths. We, then, created a directed graph of the links from the sources to the target. This graph was, then, used in the pagerank algorithm affecting a rank to each page based on its interconnectedness with other pages within the graph. Paths were also cleaned to remove the back clicks and the deleted paths which are not relevant for our study since they do not provide any significant information on the player’s reflexion nor on the recipe of success. Moreover, in the process of discovering the data, a barplot showing the number of articles per category is drawn to understand the distribution of the categories and a pie chart is created to see the highest ranked categories.
II- Understanding a player’s reasoning:

In this segment, our focus leans towards studying the behavior of successful players within the game. This choice is predicated on the assumption that their decision-making processes might exhibit greater stability or consistency. Strategy: To explore the potential connection between a player's strategy and the pagerank of selected links, a list is derived. This list includes paths with links represented by their corresponding pageranks. From this data, line plots are generated to illustrate how the pagerank of chosen links evolves throughout the games. To ensure a comprehensive representation, this analysis is conducted on four path lengths,well distributed around the mean length. Semantic links: The categories are presented in ascending order of specificity. Initially, we will investigate the semantic connections among the general categories before delving into potential logical relationships within each of these categories.
III- Between general categories:

In order to observe the subconscious links that people make between general concepts, we generate different heat maps representing the weight of the edge between two general categories. The value between the same categories has been put to zero since it does not give any relevant information. The rest of the values has been max-normalized to remove the influence of their frequency of appearance. This normalization process ensures that the heatmaps emphasize intrinsic relationships between categories, irrespective of their occurrence frequencies. This was done separately for : The portion of the finished paths from their source to their highest pagerank link. The portion of the finished paths from their highest pagerank link to their final target.
IV- Key for success:

The analysis made in the previous part showed that both finished and unfinshed paths have exactly the same pattern ; players initially navigate to an article with a high page rank score. Subsequently, they refine their strategy and direct their efforts towards their specific target article. This suggests that the difference between finished and unfinshed paths is not related to the starting point, since we always reach a hub starting from any article. the main difference lies in the final destination article, in this section we will try to analyse differences between the destinations of both finished and unfinished paths by comparing the distribution of their in-degrees,we will then analyze what influences the success or failure by building a success prediction model with the following features: target_rank : the page rank of the target page starting_rank the page rank of the starting page hub_rank: the page rank of the hub indegrees: number of ingoing edges num_of_games: number of played games.
IV- Deeper dive into Cognitive Mappings

In this final section, we will take a step back from the game difficulty and focus on analyzing further the cognitive connections between closely related categories. We will provide analysis for science-related categories, however note that this same procedure can be further generalized to other fields We focus on exploring the ‘Science’ category by considering completed paths where the destination is within the field of science. Each specific category encountered along these paths that is lying inside the general ‘Science’ category strengthens its link with the specific destination category. However, this reinforcement is proportional to the category's distance from the final destination. Specific categories closer to the final destination within the path receive higher similarity scores. This method results in constructing a matrix containing similarity scores between various categories solely within the realm of science.
Proposed timeline

22/11/2023: Part 1

29/11/2023: Part 2

06/12/2023: Part 3

13/12/2023: Part 4

20/12/2023: Write the story and review

(deadline: 22/12/2023)
Organization within the team

Datastory: Everybody

Visualizations: Veronika

Part1: Leila

Part2: Aymane

Part3: Abdessalam

Part4: Salim
