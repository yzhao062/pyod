PyOD Impact
===========

Since its release in 2017, PyOD has become one of the most widely adopted anomaly detection libraries in the Python ecosystem. This page tracks external recognition: government and standards bodies, enterprise deployments, academic citations, books, courses, podcasts, and international tutorials.

For the full audit, see the `News & Media Coverage Audit <https://github.com/yzhao062/yzhao062.github.io/blob/main/news-coverage-audit.md>`_.


Government & Standards
----------------------

* **European Space Agency (ESA/ESOC)** implemented all 30 anomaly detection algorithms in the `OPS-SAT spacecraft telemetry benchmark <https://www.nature.com/articles/s41597-025-05035-3>`_ using PyOD 1.1.2. Published in *Nature Scientific Data* (2025).


Enterprise Deployments
----------------------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Company
     - Usage
   * - **Walmart**
     - Real-time pricing anomaly detection, 1M+ daily updates (KDD 2019 industry paper)
   * - **Databricks**
     - Kakapo framework integrating PyOD with MLflow and Hyperopt
   * - **Databricks**
     - Insider threat risk detection solution using PyOD
   * - **IQVIA**
     - Healthcare fraud detection on 123K+ pharmacy claims using PyOD and SUOD models
   * - **Ericsson**
     - Patent `WO2023166515A1 <https://patents.google.com/patent/WO2023166515A1>`_ cites PyOD (Zhao et al., JMLR 2019)
   * - **Altair AI Studio**
     - Industry whitepaper using PyOD's Isolation Forest for anomaly detection
   * - **Additional patents**
     - `EP4662606A1 <https://patents.google.com/patent/EP4662606A1>`_ (EU), `CN111666198A <https://patents.google.com/patent/CN111666198A>`_ (China) both cite PyOD


Books
-----

* **Outlier Detection in Python** by Brett Kennedy (Manning, 2024): chapters 6, 7, and 14 on PyOD.
* **Handbook of Anomaly Detection** by Chris Kuo (Columbia University): entire book built around PyOD.
* **Finding Ghosts in Your Data** by Kevin Feasel (Apress / O'Reilly): chapter 12 on COPOD.
* **Advanced Techniques for Anomaly Detection: Beyond the Basics** (Routledge / CRC Press, 2025).
* **Anomaly Detection: Recent Advances, AI and ML Perspectives** (IntechOpen, 2024).


Courses
-------

* **DataCamp** -- `Anomaly Detection in Python <https://www.datacamp.com/courses/anomaly-detection-in-python>`_: dedicated PyOD chapter; DataCamp's platform reports 19M+ learners.
* **Manning liveProject** -- `Using PyOD and Ensembles Methods <https://www.manning.com/liveproject/using-pyod-and-ensembles-methods>`_: hands-on project.
* **O'Reilly Video Edition** -- Outlier Detection in Python, dedicated PyOD chapters.
* **Udemy** -- multiple courses including *Anomaly Detection: ML, DL, AutoML* and *Certified Anomaly Detection & Outlier Analytics*.


Podcasts & Talks
----------------

* **Talk Python To Me #497** -- `Outlier Detection with Python <https://talkpython.fm/episodes/show/497/outlier-detection-with-python>`_.
* **Real Python Podcast #208** -- `Detecting Outliers and Visualizing With PyOD <https://realpython.com/podcasts/rpp/208/>`_.


Media Coverage
--------------

Articles and tutorials published by independent outlets:

* **Analytics Vidhya** -- `An Awesome Tutorial to Learn Outlier Detection in Python using PyOD <https://www.analyticsvidhya.com/blog/2019/02/outlier-detection-python-pyod/>`_
* **KDnuggets** -- `An Overview of Outlier Detection Methods from PyOD <https://www.kdnuggets.com/2019/06/overview-outlier-detection-methods-pyod.html>`_
* **KDnuggets** -- `Outlier Detection Methods Cheat Sheet <https://www.kdnuggets.com/2019/02/outlier-detection-methods-cheat-sheet.html>`_
* **Towards Data Science** -- `Introducing Anomaly Detection in Python with PyOD <https://towardsdatascience.com/introducing-anomaly-outlier-detection-in-python-with-pyod-40afcccee9ff/>`_
* **Towards Data Science** -- `Real-Time Anomaly Detection With Python <https://towardsdatascience.com/real-time-anomaly-detection-with-python-36e3455e84e2/>`_ (March 2025, PyOD + PySAD)
* **Towards Data Science** -- `Boosting Your Anomaly Detection With LLMs <https://towardsdatascience.com/boosting-your-anomaly-detection-with-llms/>`_ (September 2025, dedicated to PyOD 2's LLM-powered model selection)
* **Ericsson Blog** -- `How to make anomaly detection more accessible <https://www.ericsson.com/en/blog/2020/7/how-to-make-anomaly-detection-more-accessible>`_
* **Elder Research** -- `Business Insights Meet Analytics Skills in Anomaly Detection <https://www.elderresearch.com/blog/business-insights-meet-analytics-skills-in-anomaly-detection/>`_
* **Data Reply IT (Reply Group)** -- `Anomaly Detection made easy with PyOD <https://medium.com/data-reply-it-datatech/anomaly-detection-made-easy-with-pyod-960faf6da4e5>`_
* **Cake.ai** -- `Anomaly Detection Software: A Complete Guide <https://www.cake.ai/blog/open-source-anomaly-detection-tools>`_
* **Number Analytics** -- `Advanced Nonparametric Outlier Identification <https://www.numberanalytics.com/blog/advanced-nonparametric-outlier-identification>`_ (2025)
* **The Data Scientist** -- `Anomaly detection in Python using the PyOD library <https://thedatascientist.com/anomaly-detection-in-python-using-the-pyod-library/>`_
* **SmartDev** -- `Master AI Anomaly Detection: The Definitive Guide <https://smartdev.com/ai-anomaly-detection/>`_
* **Milvus / Zilliz** -- `Open-source libraries for anomaly detection <https://milvus.io/ai-quick-reference/what-are-opensource-libraries-for-anomaly-detection>`_


International Reach
-------------------

Beyond English, PyOD tutorials and translations exist in at least five non-English languages:

* **Chinese** -- 10+ tutorials across CSDN, Zhihu, 搜狐 (Sohu), 机器之心 (Jiqizhixin), 腾讯云开发者社区 (Tencent Cloud Developer), 智东西 (Zhidx), Bilibili. The community project `aidoczh.com <https://www.aidoczh.com>`_ maintains a full Chinese translation of PyOD documentation.
* **Japanese** -- 4+ tutorials including Qiita, Codemajin, DataPowerNow, Scutum, TRYETING, and ClassCat.
* **Korean** -- 3 tutorials (Tistory, JunPyoPark, DataNetworkAnalysis).
* **German** -- 5 sources including Hahn-Schickard / EmbedML, Konfuzio, Acervo Lima, KI Blog.
* **Spanish** -- Aprende Machine Learning and Medium tutorials.


Academic Follow-on Work
-----------------------

* `Text-ADBench <https://arxiv.org/abs/2507.12295>`_ (Jicong Fan et al., July 2025): external follow-on benchmark inspired by ADBench.
* COPOD and ECOD cited as "most efficacious" methods in digital forensics research (*CEUR-WS Vol-4092*).
* Two-phase Dual COPOD Method for ICS security (arXiv:2305.00982).
* Graph Diffusion Models for Anomaly Detection (Amazon Science, 2024): cites BOND and PyGOD.


Platforms
---------

* `Kaggle <https://www.kaggle.com/search?q=pyod>`_: 7+ dedicated public notebooks.
* `HelloGitHub <https://hellogithub.com>`_: featured open-source project.


Summary
-------

As of April 2026: **38+ million downloads** on `PyPI <https://pepy.tech/project/pyod>`_, **9K+ stars** on `GitHub <https://github.com/yzhao062/pyod>`_, one *Nature Scientific Data* citation (ESA OPS-SAT), 3+ dedicated books, 2 major podcasts, 4+ online courses, tutorials in 5 non-English languages, and 60+ third-party media articles. PyOD has also been cited in research from USC Viterbi, Amazon Science, Microsoft Research, and Lawrence Livermore National Laboratory.
