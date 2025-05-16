
# Introduction 
How does the internet media speak about men and women?

This project explores 30 years worth of articles (1993–2023) to investigate patterns in how genders are spoken about in the media. By analyzing a large dataset of real internet articles, we examine whether the language, tone, and framing used about men and women differ in meaningful ways.

<!--- We use a combination of network analysis and natural language processing (NLP) techniques to uncover trends in sentiment, emotional expression, and the presence of biased language.

The goal is not just to identify obvious terms like "mom" or "dad", but to uncover subtle patterns in how people are talked about based on their gender. -->

<!--- A 30-year study of Danish news, sentiment and gender bias.


In this project, we explore how men and women are portrayed in the media, using a network of real news articles spanning 1993-2023. -->

## Explainer notebook:
All the methods, code and algorithms used for this report can be found here:

- [Link to explainer-notebook](https://github.com/XanderBaatz/02467-dtu-project/blob/main/explainer.ipynb)

## About the Project

The image of a public person is heavily influenced by how the media "paints" them. Whether it's a politician, celebrity, or athlete, being in the spotlight means being interpreted, judged, and framed.

We focus on gender inequality in media narratives. Even in modern societies like Denmark, men and women are often represented differently. This may affect how the public perceives them and breed existing biases.

Using a combination of Natural Language Processing (NLP) and Network Analysis, we analyze how people are mentioned, what tone is used (positive/negative), and generally how language is used across both genders.



### From Text to Network

We started with 125,000+ Danish news articles from the [EB-NeRD dataset](https://recsys.eb.dk/), published by Ekstra Bladet between 1993 and 2023. These articles were processed using various algorithms to identify:
- Who is mentioned in the article
- Which people appear together
- What is the sentiment of the article
- Whether offensive langauge is present
- The likely gender of each person

We built a social network where nodes are people and edges exist between persons co-mentioned in an article. Then we analyzed this network to understand patterns of visibility, tone, and bias.



### Findings

- Communities of people naturally appear in the media (think articles about reality, royal houses or entertainment.)
- Some communities show a large imbalance in how males and females are written about.
- Women are on average, described with a more positive sentiment, in contrast to men who are typically in articles with worse sentiment. Women in some communities are also more prone objectification by the words used about them in contrast to men. 



<!---
### Who's this for?

This project is aimed at anyone curious about:
- How biases can show up in media coverage
- The hidden passages in how we talk and write about public figures
- The intersection of language, society and data science
-->



```{tableofcontents}
```
