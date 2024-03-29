# Data analysis - Student's Behavior

## 1. Problem Statement:

The task involves examining student actions to detect recurring trends and groupings that can offer valuable insights into their levels of involvement. By grasping these trends, educational institutions can customize interventions and tactics to enhance student engagement and academic achievement.

## 2. Introduction:

In the modern educational environment, comprehending student conduct is essential for educators and administrators to deliver impactful assistance and direction. Examining different facets of student involvement, such as active participation in classroom dialogues, utilization of resources, and engagement with notifications, can provide valuable perspectives into their educational journey and academic results.

| Variable                | Description                               |
|-------------------------|-------------------------------------------|
| gender                  | Gender of the student (M/F)              |
| NationalITy             | Nationality of the student              |
| PlaceofBirth            | Place of birth of the student            |
| StageID                 | Stage of education (e.g., lowerlevel)    |
| GradeID                 | Grade level                               |
| SectionID               | Section identifier                        |
| Topic                   | Topic of study                            |
| Semester                | Semester of the academic year            |
| Relation                | Relationship of the respondent to the student (e.g., father) |
| raisedhands             | Number of times the student raised their hand in class |
| VisITedResources        | Number of resources visited by the student |
| AnnouncementsView       | Number of announcements viewed by the student |
| Discussion              | Number of times the student participated in class discussions |
| ParentAnsweringSurvey   | Whether the parent answered the survey (Yes/No) |
| ParentschoolSatisfaction| Parent's satisfaction with the school (Good/Bad) |
| StudentAbsenceDays      | Number of days the student was absent from school |
| Class                   | Categorized class label (e.g., M: Medium) |

This dataset provides detailed information about student behavior, demographics, and academic performance. It includes variables such as gender, nationality, grade level, participation metrics (raised hands, resources visited, announcements viewed, discussions participated), parental involvement, student absence, and class categorization. 

The dataset appears to be comprehensive and suitable for exploring various aspects of student engagement and academic outcomes. Further analysis, including exploratory data analysis (EDA) and clustering techniques, can provide valuable insights into student behavior and help identify strategies to improve educational outcomes.

## Methodology 

## Exploratory Data Analysis (EDA) Approach

### Summary of EDA:

#### Distribution of Numeric Variables:
- Pair plots were utilized to visualize the distribution of numeric variables, enabling exploration of relationships between different numeric variables and identification of potential patterns or trends.

#### Distribution of Categorical Variables:
- Count plots were created to visualize the distribution of categorical variables such as gender, nationality, and class. This aided in understanding the frequency distribution of different categories within each variable.

#### Correlation Analysis:
- A correlation matrix was computed for numeric variables and visualized using a heatmap. This facilitated identification of correlations between different numeric variables, such as "raised hands" and "visited resources."

#### Cross-Tabulation:
- Cross-tabulation was performed between the 'NationalITy' and 'Class' variables to examine the distribution of student classes across different nationalities. This analysis provided insights into potential relationships between nationality and academic performance.

Overall, the EDA provided a comprehensive understanding of the dataset's characteristics, revealed potential relationships between variables, and uncovered insights to guide further analysis and decision-making processes. It offered valuable insights into student behavior and engagement levels, laying the groundwork for subsequent analyses such as clustering or predictive modeling.

## Machine Learning (ML) Approach:

#### Elbow Method and Silhouette Analysis
- **Elbow Method**: Determined the optimal number of clusters based on within-cluster sum of squares (WCSS), suggesting around 2 or 3 clusters.
- **Silhouette Analysis**: Evaluated cluster quality using silhouette scores, indicating that 2 clusters might be optimal based on the highest silhouette score.

#### K-means Clustering
- **Number of Clusters**: Proceeded with 2 clusters based on the elbow method and silhouette analysis.
- **Cluster Visualization**: Utilized scatter plots to visualize clusters based on 'raisedhands' and 'VisITedResources', with centroids marked for reference.
- **Cluster Profiles**: Visualized cluster profiles using scatter plots, highlighting mean values of 'raised hands' versus 'visited resources' for each cluster.

#### Cluster Characteristics
- **Cluster Summary**: Compared mean and median values of 'raised hands' and 'visited resources' for each cluster, with Cluster 0 showing lower values compared to Cluster 1.
- **Cross-Tabulation**: Examined distribution of clusters across different levels of 'raised hands', providing insight into cluster composition based on student behavior.

Overall, K-means clustering identified distinct student engagement levels based on 'raised hands' and 'visited resources', offering insights for tailored interventions to enhance student engagement and academic performance.

## Training Workflow

### Importing Required Libraries

```python
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

### Loading Data

The data is loaded from a CSV file  into a Pandas DataFrame named data.

```python
data = pd.read_csv('Studentsbehaviouranalysis.csv')
```

### Data Exploration

Initial exploration of the dataset is performed using head(), sample(), describe(), isnull().sum() and info() functions to understand its structure and contents

```python
data.head(5)
data.info()
data.sample()
print(data.describe())
print(data.isnull().sum())
```
### Feature Selection 

- **distribution of categorical variables**:
  
```python

categorical_cols = ['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID', 
                    'SectionID', 'Topic', 'Semester', 'Relation', 'ParentAnsweringSurvey',
                    'ParentschoolSatisfaction', 'StudentAbsenceDays', 'Class']
 ```

- **distribution of numerical variables**:

```python
numeric_cols = ['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']
```
  
## 6. EDA

### Visualize the distribution of numeric variables

<img src="https://github.com/jobinjoy12/Students-Behaviour-Analysis/assets/106422383/9255786b-d2d2-404c-9c58-15ed610322f7" alt="1 numeric" width="400">

### Visualize the distribution of categorical variables

<div style="display: flex;">
    <img src="https://github.com/jobinjoy12/Students-Behaviour-Analysis/assets/106422383/dee7296f-7c9f-4739-a796-687718febbb5" alt="2.1" width="500" style="float:left;">
    <img src="https://github.com/jobinjoy12/Students-Behaviour-Analysis/assets/106422383/1e2a5187-7b5f-4c78-9ed5-9f66581adfa0" alt="2.2" width="500" style="float:right;">
</div>
<div style="display: flex;">
    <img src="https://github.com/jobinjoy12/Students-Behaviour-Analysis/assets/106422383/b7b84bcd-6b43-417e-b639-f7e6da124194" alt="2.3" width="500" style="float:left;">
    <img src="https://github.com/jobinjoy12/Students-Behaviour-Analysis/assets/106422383/b084611c-29e8-4bab-a051-252d2acf351f" alt="2.4" width="500" style="float:right;">
</div>
<div style="display: flex;">
    <img src="https://github.com/jobinjoy12/Students-Behaviour-Analysis/assets/106422383/594fc5ba-91be-4c3e-a3a5-903854e1ebff" alt="2.5" width="500" style="float:left;">
    <img src="https://github.com/jobinjoy12/Students-Behaviour-Analysis/assets/106422383/5f1693ff-cd6d-40cc-85fb-2412df830413" alt="2.6" width="500" style="float:right;">
</div>
<div style="display: flex;">
    <img src="https://github.com/jobinjoy12/Students-Behaviour-Analysis/assets/106422383/16c3bf75-7cc2-4c63-aad8-999b1ab9093c" alt="2.7" width="500" style="float:left;">
    <img src="https://github.com/jobinjoy12/Students-Behaviour-Analysis/assets/106422383/dfb6d737-9689-43f8-ada6-be288a05436c" alt="2.8" width="500" style="float:right;">
</div>
<div style="display: flex;">
    <img src="https://github.com/jobinjoy12/Students-Behaviour-Analysis/assets/106422383/d3bcc9d2-3f9d-42d1-8792-f3168e1d3655" alt="2.9" width="500" style="float:left;">
    <img src="https://github.com/jobinjoy12/Students-Behaviour-Analysis/assets/106422383/e13f91dd-e6a3-4d5b-bb6e-5f5d4e5d862c" alt="2.10" width="500" style="float:right;">
</div>
<div style="display: flex;">
    <img src="https://github.com/jobinjoy12/Students-Behaviour-Analysis/assets/106422383/5d0aeb17-d7ef-4a27-ae1d-10292bff64aa" alt="2.11" width="500" style="float:left;">
    <img src="https://github.com/jobinjoy12/Students-Behaviour-Analysis/assets/106422383/304bec8f-ac29-4633-b855-683acc9e79ce" alt="2.12" width="500" style="float:right;">
</div>

<img src="https://github.com/jobinjoy12/Students-Behaviour-Analysis/assets/106422383/07f47111-f28c-4340-95d5-9ebefcc2dd35" alt="2.13" width="500">

### Correlation matrix

Compute the correlation matrix between numeric variables and visualize it using a heatmap. This will help identify any significant correlations between variables.

<img src="https://github.com/jobinjoy12/Students-Behaviour-Analysis/assets/106422383/6ca2518e-80f7-40a4-ae56-7f8aec212a25" alt="3 Correlation matrix" width="500">

### Cross-tabulation: 

Creating cross-tabulations between pairs of categorical variables to observe relationships between them.

| Class        |        |  H  |  L  |  M  |
|--------------|--------|-----|-----|-----|
| NationalITy  |        |     |     |     |
|--------------|--------|-----|-----|-----|
| Egypt        |        |  2  |  3  |  4  |
| Iran         |        |  0  |  2  |  4  |
| Iraq         |        | 14  |  0  |  8  |
| Jordan       |        | 53  | 37  | 82  |
| KW           |        | 36  | 68  | 75  |
| Lybia        |        |  0  |  6  |  0  |
| Morocco      |        |  1  |  1  |  2  |
| Palestine    |        | 12  |  0  | 16  |
| SaudiArabia  |        |  6  |  1  |  4  |
| Syria        |        |  2  |  2  |  3  |
| Tunis        |        |  3  |  4  |  5  |
| USA          |        |  3  |  1  |  2  |
| lebanon      |        |  9  |  2  |  6  |
| venzuela     |        |  1  |  0  |  0  |

## 7. Machine Learning Model to study segmentation: K-means clustering

### Finding the best value of k using elbow method

The Elbow Method is a heuristic technique used to determine the optimal number of clusters in a dataset. It works by plotting the within-cluster sum of squares (WCSS) against the number of clusters and identifying the point where the rate of decrease in WCSS slows down, forming an "elbow" shape. This point represents the optimal number of clusters. By using the Elbow Method, we can make an informed decision about the appropriate number of clusters to use in the K-means algorithm, thereby ensuring that the segmentation is meaningful and interpretable. 

<img src="https://github.com/jobinjoy12/Students-Behaviour-Analysis/assets/106422383/3e9f4c61-d1b5-46d8-8027-4b46423275f4" alt="elbow" width="500">

k=2 is the ideal value from this graph

### Using Silhouette Scores

<img src="https://github.com/jobinjoy12/Students-Behaviour-Analysis/assets/106422383/9ce7b23f-d2bf-43b2-bb20-fbb0ff80e43c" width="500">

Best Number of Clusters: 2

### Implementing K-means clustering

```python
# Perform k-means clustering
k = 2 # Number of clusters
kmeans = KMeans(n_clusters=k,n_init='auto', random_state=42)
kmeans.fit(X)
```

### Extracting labels and cluster centers

```python
# Get the cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Add the cluster labels to the DataFrame
data['Cluster'] = labels
```

### Visualizing the clustering using first two features

<img src="https://github.com/jobinjoy12/Students-Behaviour-Analysis/assets/106422383/8367a731-29c4-4332-a63d-70afb0acdfd2" alt="KMEANS1" width="500">
<img src="https://github.com/jobinjoy12/Students-Behaviour-Analysis/assets/106422383/eece4a7f-f24d-4769-865f-c038fbc99bff" alt="KMEANS2" width="500">


| Cluster | Raised Hands (mean) | Raised Hands (median) | Visited Resources (mean) | Visited Resources (median) |
|---------|----------------------|------------------------|--------------------------|----------------------------|
| 0       | 19.10                | 15.0                   | 22.35                    | 15.0                       |
| 1       | 67.94                | 72.0                   | 79.61                    | 82.0                       |

## Results

The analysis of clustering based on the features "raised hands" and "visited resources" revealed the presence of two distinct clusters. Here are the primary findings:

- **Cluster 0:** This cluster represents students with lower engagement levels. Both the mean and median values of "raised hands" and "visited resources" are comparatively low compared to Cluster 1.
- **Cluster 1:** This cluster represents students with higher engagement levels. The mean and median values of "raised hands" and "visited resources" are notably higher compared to Cluster 0.

## Conclusion

The clustering analysis provides valuable insights into student engagement levels in classroom activities. It delineates two distinct student groups:

1. **Low Engagement Group (Cluster 0):** These students exhibit lower participation in classroom activities, with infrequent hand-raising and resource utilization. Interventions targeting increased participation may benefit this group to enhance their learning experiences.

2. **High Engagement Group (Cluster 1):** Students in this group actively participate in classroom activities, frequently raising hands and utilizing educational resources. They represent students deeply engaged in the learning process, requiring strategies to sustain their engagement and academic success.

## Implications and Recommendations

- Educators can leverage clustering results to identify students needing additional support to boost their engagement.
- Tailored interventions can be designed for each cluster. For instance, specific strategies may be employed to enhance participation for students in the low engagement group.
- Classroom management practices can be adjusted based on clustering insights, fostering inclusive and engaging learning environments tailored to students' diverse needs.

In essence, the clustering analysis furnishes insights guiding interventions and decisions aimed at fostering student engagement and enriching the learning environment.




