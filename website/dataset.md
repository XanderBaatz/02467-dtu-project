# Dataset

We use the [EB-NeRD dataset](https://recsys.eb.dk/) which is a collection of over 125,000+ news articles published by Ekstra Bladet between 1993 and 2023. This dataset provides full article text, metadata, and annotations suitable for NLP tasks. 

Here is quick rundown of the columns found in the raw dataset:

| **Column**          | **Description**                                                                                 | **Type**    | **Example**                             |
|---------------------|-------------------------------------------------------------------------------------------------|-------------|-----------------------------------------|
| `article_id`        | Unique identifier for each article.                                                             | Integer     | `9803281`                               |
| `title`             | Title of the news article.                                                                       | Text        | `Dansk stjerne tvunget på minerydning`   |
| `subtitle`          | Subtitle or short summary of the article.                                                       | Text        | `Dj Aligator kom først til Danmark...`   |
| `last_modified_time`| Timestamp of the last modification to the article (UTC).                                         | Datetime    | `2023-06-29 06:20:33`                  |
| `premium`           | Indicates whether the content is premium (`True`/`False`).                                       | Boolean     | `False`                                 |
| `body`             | Full text content of the article.                                                               | Text        | `Tom Hanks skulle angiveligt have...`    |
| `published_time`    | Original publication timestamp (UTC).                                                           | Datetime    | `2006-05-04 11:03:12`                  |
| `image_ids`         | List of associated image ID(s) used in the article.                                              | List        | `[3518381]`                             |
| `article_type`      | Article category/type (e.g., default, opinion, etc.).                                            | Text        | `article_default`                       |
| `url`              | Link to the online article.                                                                     | URL         | `https://ekstrabladet.dk/...`           |
| `ner_clusters`      | Identified entities (e.g. person names/aliases) extracted using Named Entity Recognition (NER).                  | List        | `[Tom Hanks, SFGate.com, ...]`       |
| `entity_groups`     | Groups of entities detected (e.g., organizations, locations, persons).                          | List  | `[PER, ORG, ...]`         |
| `topics`            | Topics assigned to the article using topic modeling/classification.                             | List/Text   | `underholdning`, `kultur` (if available)|
| `category`          | Main content category (e.g., entertainment, news, sports).                                       | Text        | `underholdning`                         |
| `subcategory`       | More specific content subdivision (e.g., celebrities, politics, etc.).                          | Text        | `dkkendte`                              |
| `category_str`      | String representation of combined categories (category/subcategory).                             | Text        | `underholdning > dkkendte`              |
| `total_inviews`     | Internal views of the article (number of times viewed within the site/app).                     | Float/NaN   | `305134.0`                              |
| `total_pageviews`   | Total page views of the article.                                                               | Float/NaN   | `48926.0`                               |
| `total_read_time`   | Total read time accumulated for the article (in seconds).                                       | Float/NaN   | `1957682.0`                             |
| `sentiment_score`   | Sentiment polarity score of the article. | Float       | `0.888`                                 |
| `sentiment_label`   | Assigned sentiment category (`positive`, `neutral`, or `negative`).                             | Text        | `positive`                               |



## Overview

To get a high-level understanding of what kind of newspaper outlet Ekstra Bladet is, we can look at the proportion of articles from each category to see what they prefer to write about. Below is a pie chart of the top 7 categories:
![category-pie-chart](figures/category-pie-chart.svg)

From here we gather that 20% of the dataset is comprised of articles from `underholdning` (entertainment).

To get an idea of category popularity over the years, we plot a $\log$-normalized timeline of the number of articles per month, for each category:
![pdf](figures/monthly-article-count.svg)

The entertainment category is undoubtedly one of their most popular category, judging by the frequency of articles over the years. Entertainment is also a very broad category as it concatenates genres such as reality TV, Royal House coverage and sports etc. 



## Data Access and Licensing

Due to copyright restrictions, we cannot share the dataset directly. However, one can access the original dataset from the [EB-NeRD website](https://recsys.eb.dk/) after agreeing to their terms.

# Community Word Analysis by Gender

# Community Gender-Based Text Features

