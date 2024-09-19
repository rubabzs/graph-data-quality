# Towards Semi-Supervised Data Quality Detection In Graphs

Graph databases have emerged as a powerful tool for representing and analyzing complex relationships in various domains, including social networks, healthcare, and financial systems. Despite their growing popularity, data quality issues such as node duplication, missing nodes or edges, incorrect formats, stale data, and misconfigured topology remain prevalent. While there are numerous libraries and approaches for addressing data quality in tabular data, graph-structured data pose unique challenges of their own. In this paper, we explore an automated approach for detecting data quality issues in graph structured data which focuses on both node attributes and relationships. Since data quality is often governed by pre-established rules and is highly context-dependent, our approach seeks to balance rule-based control with the automation potential of machine learning. We investigate the capabilities of graph convolutional networks (GCNs) and large language models (LLMs) at detecting data quality issues using a few-shot learning approach. We evaluate the data quality detection rates of these models on a graph dataset and compare their effectiveness and potential impact on improving data quality. Our results indicate that LLMs exhibit robust generalization capabilities from limited samples while GCNs offer distinct advantages in certain contexts.


## Citation

Please cite the [Graph Data Quality  paper](https://rubabzs.github.io/files/qdb.pdf) (QDB @ VLDB 2024) if you find it useful in your work:

~~~~
@InProceedings{sarfraz2024qdb,
  title = {Vizard: Improving Visual Data Literacy with Large Language Models},
  author = {Rubab Zahra Sarfraz},
  booktitle = {VLDB 2024 Workshop: 13th International Workshop on Quality In Databases (QDB 2024)},
  year = {2024},
  month = {August},
  url = "https://rubabzs.github.io/files/qdb.pdf"
  }
~~~~
