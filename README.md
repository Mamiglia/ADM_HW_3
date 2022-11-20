# Homework 3 - Search Engines

[![Button Icon]](https://nbviewer.org/github/Mamiglia/ADM_HW_3/blob/main/main.ipynb)

[Button Icon]: https://img.shields.io/badge/Notebook-EF2D5E?style=for-the-badge&logoColor=white

The purpose of this repository is to answer the assigned questions in the [Homework#3](https://github.com/lucamaiano/ADM/tree/master/2022/Homework_3) of the Algorithms for Data Mining course, 2022.

The data analysis is performed on data scraped from the [Atlas Obscura Website](atlasobscura.com/). The repository consists of the following files:

1. `main.ipynb`: a Jupyter Notebook that answers all the aforementioned questions, Research, Algorithmic, Command Line, Bonus, and can be viewed [here](https://nbviewer.org/github/Mamiglia/ADM_HW_3/blob/main/main.ipynb) (note that the map cannot be viewed on github).
2. `func` folder: a folder containing all the classes and functions used in the notebook, divided in:
  - `scraper.py`: functions for scraping and refining data from the original source.
  - `engines.py`: all the different implementations of the search engines.
  - `metrics.py`: functions useful for evaluating the search engines.
3. `CommandLine.sh`: a file that provides the answer to the Commmand Line question.
4. `refined_data` folder: all the TSVs files containig the data scraped from the webpages, also availiable in compact form in `places.tsv`.
5. `urls.txt`: the urls of the 7200 most popular webpages from Atlas Obscura.
6. `queries.csv`: a list of some randomized queries, and the results of the search on Atlas Obscura's search engine.
7. `RankingList.txt`: the output of question 7.
