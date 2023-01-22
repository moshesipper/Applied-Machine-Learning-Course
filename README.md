# Course: Applied Machine Learning

**This course covers the applied/coding side of algorithmics in machine learning and deep learning, with a smidgen of evolutionary algorithms.**


***
[Course Contents](#contents)	

[Lesson Plan](#lessons)

[Cheat Sheets](#cheat)

[Algorithm Pros and Cons](#pros)

[Resources: Evolutionary Algorithms, Machine Learning, Deep Learning](#resources)


***

<a name="contents">**Course Contents**</a>

- What is machine learning (ML)?
- Basics of Python programming
- Applying ML: evaluation, dataset splits, cross-validation, performance measures, bias/variance tradeoff, visualization, confusion matrix, choosing estimators, hyperparameter tuning, statistics
- Supervised learning: models, features, objectives, model training, overfitting, regularization, classification, regression, gradient descent, k nearest neighbors, linear regression, logistic regression, decision tree, random forest, adaptive boosting, gradient boosting, support vector machine, naïve Bayes
- Dimensionality reduction: principal component analysis
- Unsupervised learning: hierarchical clustering, k-means, t-SNE ​
- Artificial neural networks: backpropagation, deep neural network, convolutional neural network
- Evolutionary algorithms: genetic algorithm, ​genetic programming

***	

<a name="lessons">**Lesson plan**</a>

​#1 Python, AI+ML Intro​

- [Python Programming](https://pythonbasics.org/) ([Python](https://www.python.org/downloads/windows/), [PyCharm](https://www.jetbrains.com/pycharm/download/#section=windows)), [Pandas](https://pythonbasics.org/what-is-pandas/), [NumPy](https://numpy.org/devdocs/user/absolute_beginners.html) / [NumPy](https://www.w3schools.com/python/numpy/default.asp) ([Numba](https://numba.pydata.org/)) ([np.dot](https://numpy.org/doc/stable/reference/generated/numpy.dot.html) vs [loop example](code/loop-vs-np.py))
- [Computing Machinery and Intelligence](https://www.cs.mcgill.ca/~dprecup/courses/AI/Materials/turing1950.pdf)
- [Machine Learning: history, applications, recent successes](https://data-psl.github.io/lectures2020/slides/01_machine_learning_successes)
- [Building an ML Model](https://media.licdn.com/dms/image/C4E22AQE-asPVy92oEw/feedshare-shrink_1280/0/1672175694811?e=1675296000&v=beta&t=T5fS_yiU8DZF28Vakg13meGwcg3u8DchAPNP9kXYy5s),
  [Data Science Landscape](https://media.licdn.com/dms/image/C4E22AQEchDYrmM8Qdg/feedshare-shrink_800/0/1672377766999?e=1675296000&v=beta&t=IQRzEXzma_xee0BzL4lRnhh-wTwEbKbHh_iZHB6vT8g)

#2 ML Intro, Simple Example, KNN, Cross-Validation

- [Introduction to machine learning](https://data-psl.github.io/lectures2020/slides/02_intro_to_machine_learning)
- [simple weather example](code/weather.py)
- [iris knn (map)](https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html#sphx-glr-auto-examples-neighbors-plot-classification-py)
- [kfold​](code/kfold.py)

#3 Scikit, Models, Decision Trees

- [Machine learning with scikit-learn](https://data-psl.github.io/lectures2020/slides/04_scikit_learn/#1)
- [Machine learning models](https://data-psl.github.io/lectures2020/slides/03_machine_learning_models/)
- [Boston dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#boston-dataset)
- ​[Decision trees](https://youtu.be/_L39rN6gz7Y)​

#4 Random Forest, Linear Regression Logistic Regression

- [Machine learning models](https://data-psl.github.io/lectures2020/slides/03_machine_learning_models/)
- [Random Forests](https://youtu.be/J4Wdy0Wc_xQ)
- [Linear Regression](https://youtu.be/PaFPbb66DxQ)
- [LinearReg.ipynb](https://colab.research.google.com/drive/1fCQjAiEce6hU0osLzCWo73hC9iYva3QK?usp=sharing)
- [Logistic Regression](https://youtu.be/yIYKR4sgzI8), [Logistic Regression](https://www.analyticsvidhya.com/blog/2021/07/an-introduction-to-logistic-regression/), [Cross-Entropy Loss](https://towardsdatascience.com/cross-entropy-loss-function-f38c4ec8643e)
- [Optimization of linear models](https://data-psl.github.io/lectures2020/slides/05_optimization_linear_models/)
- ​[Ridge vs. Lasso](https://www.statology.org/when-to-use-ridge-lasso-regression/)​

#5 AdaBoost, Gradient Boosting

- Summary: [Linear Regression](https://medium.com/analytics-vidhya/a-quick-summary-of-linear-regression-42d1dab85e3e), [Logistic Regression](https://www.analyticsvidhya.com/blog/2021/07/an-introduction-to-logistic-regression/), [LinVsLog](code/LinVsLog.py), [PolynomialFeatures](code/PolynomialFeatures.py)
- [Adaptive Boosting](https://youtu.be/LsK-xG1cLYA)​​
- ​[Gradient Boosting](https://youtu.be/3CC4N4z3GJc)
- [AddGBoost](https://www.sciencedirect.com/science/article/pii/S2666827021001225)​

#6 XGBoost, Comparing ML algos, Gradient Descent

- Reminder: [Adaboost](https://www.cs.bgu.ac.il/~sipper/adaboost.jpg), [Gradient boost](https://www.cs.bgu.ac.il/~sipper/gradboost.jpg)
- ​[XGBoost](https://youtu.be/OtD8wVaFm6E)
- [Comparing supervised learning algorithms](https://www.dataschool.io/comparing-supervised-learning-algorithms/)
- ​[How to find the best performing Machine Learning algorithm](https://medium.com/analytics-vidhya/how-to-find-the-best-performing-machine-learning-algorithm-dc4eb4ff34b6) ([load\_iris ](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)vs. [load\_wine](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html), plt.show(), [Lazy Predict](https://lazypredict.readthedocs.io/en/latest/readme.html))
- [Gradient Descent](https://youtu.be/sDv4f4s2SB8) ([Least Squares](https://www.mathsisfun.com/data/least-squares-regression.html), [Least Squares](https://textbooks.math.gatech.edu/ila/least-squares.html))
- [Stochastic Gradient Descent](https://youtu.be/vMh0zPT0tLI)

#7 Choosing Model, SVM, Bayes, Metrics

- [Choosing the right estimator](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html) ([one model doesn&#39;t fit all](https://media-exp1.licdn.com/dms/image/C5622AQHK0ySAmVHlvQ/feedshare-shrink_800/0/1652450701365?e=1655942400&amp;v=beta&amp;t=3KKtUu5AD30HD1n75tXUizL6UdgeDX8sBjH2UH20XfE)...)
- [Support Vector Machine](https://youtu.be/efR1C6CvhmE) [(](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)[sklearn.svm.SVC)](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html), [Plot different SVM classifiers in the iris dataset](https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html#sphx-glr-auto-examples-svm-plot-iris-svc-py)
- [Multinomial Naive Bayes](https://youtu.be/O2L2Uv9pdDA)
- Metrics: [sklearn.metrics](https://scikit-learn.org/stable/modules/classes.html?highlight=metrics#module-sklearn.metrics), [Confusion Matrix](https://youtu.be/Kdsp6soqA7o), [Sensitivity and Specificity,](https://youtu.be/vP06aMoz4v8) [ROC and AUC](https://youtu.be/4jRBRDbJemM) ([Online ROC Curve Calculator](https://www.rad.jhmi.edu/jeng/javarad/roc/JROCFITi.html)), ​[balanced accuracy](https://www.statology.org/balanced-accuracy/), [balanced accuracy](https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score), [various metrics from CM](https://en.wikipedia.org/wiki/Confusion_matrix)
- [Could machine learning fuel a reproducibility crisis in science?](https://www.nature.com/articles/d41586-022-02035-w)

#8 Metrics, PCA, t-SNE, Clustering

- [What companies think AI looks like...](https://media-exp1.licdn.com/dms/image/C4D22AQGHDRqNV6uPPg/feedshare-shrink_1280/0/1653029435934?e=1655942400&amp;v=beta&amp;t=PQytCPFFbbKE-AbHBLo3zsNbRcQ8nVqTqb6JbTc4kUs), [ML in practice](https://info.deeplearning.ai/the-batch-one-model-for-hundreds-of-tasks-recognizing-workplace-hazards-when-data-means-danger-vision-transformer-upgrade-1)
- [balanced accuracy](https://www.statology.org/balanced-accuracy/), [balanced accuracy](https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score), [various metrics from CM](https://en.wikipedia.org/wiki/Confusion_matrix)
- [11 Different Uses of Dimensionality Reduction](https://towardsdatascience.com/11-different-uses-of-dimensionality-reduction-4325d62b4fa6)
- ​[PCA](https://youtu.be/FgakZw6K1QQ), [PCA](https://www.sartorius.com/en/knowledge/science-snippets/what-is-principal-component-analysis-pca-and-how-it-is-used-507186), [PCA vs LR](https://starship-knowledge.com/pca-vs-linear-regression#PCA_vs_Linear_Regression_-_How_do_they_Differ), [pca.ipynb](https://colab.research.google.com/drive/1h6xLxKyEltPwsck-mJ5nQPFkMGYI8VOs?usp=sharing)
- ​[t-SNE](https://youtu.be/NEaUSP4YerM), [tsne.py](code/tsne.py)
- [Hierarchical clustering](https://youtu.be/7xHsRkOdVwo)
- [K-means clustering](https://youtu.be/4b5d3muPQmA), [kmeans.py](code/kmeans.py)

#9 Hyperparameter Tuning, p-vals, t-test, Permutation Test

- [Model Parameter vs. Hyperparameter](https://www.youtube.com/watch?v=Qcgav8NmPxY&t=1224s)
- [Hyperparameter tuning](https://towardsdatascience.com/hyperparameter-tuning-explained-d0ebb2ba1d35), [Hyperparameter tuning](https://towardsdatascience.com/hyperparameter-tuning-a-practical-guide-and-template-b3bf0504f095), [Optuna](https://optuna.org/), [optuna.py](code/optuna.py)
- [p-values](https://youtu.be/vemZtEM63GY), [how to calculate p-values](https://youtu.be/JQc3yx0-Q9E), [p-hacking](https://youtu.be/HDCOUXE3HMM)
- [Probability is not Likelihood](https://www.youtube.com/watch?v=pYxNSUDSFH4)
- [17 Statistical Hypothesis Tests in Python](https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/)
- [t-test](https://youtu.be/0Pd3dc1GcHc), ​[t-test](https://youtu.be/pTmLQvMM-1M) ([t-test vs p-value)](https://askanydifference.com/difference-between-t-test-and-p-value/), [scpipy ttest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)
- ([chi-square](https://youtu.be/ZjdBM7NO7bY))
- [permutation test](https://youtu.be/GmvpsJHGCxQ) ([AddGBoost ](https://www.sciencedirect.com/science/article/pii/S2666827021001225)+ [code](https://github.com/moshesipper/AddGBoost)), [permutation test](https://www.linkedin.com/feed/update/urn:li:activity:6934781784937107456/)

#10, #11 Neural Networks

- ​[Neural networks](https://youtube.com/playlist?list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1)
- [Neural Networks with À La Carte Selection of Activation Functions](https://drive.google.com/file/d/10wy_gFPoNLwQXHkHKAojYBxbKkB5W_da/view?usp=sharing)
- [PyTorch](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html), [PyTorch](https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e)
- [Convolution...](https://www.linkedin.com/posts/pascalbornet_artificialintelligence-ugcPost-6925288775740776448-0S-K/)
- [Growth of AI computing](https://twitter.com/pmddomingos/status/1535112033137401857), [AI move from Academia](https://twitter.com/GaryMarcus/status/1536150812795121664), [Artificial General Intelligence Is Not as Imminent as You Might Think](https://www.scientificamerican.com/article/artificial-general-intelligence-is-not-as-imminent-as-you-might-think1/)
- [Tinker With a Neural Network in Your Browser](https://playground.tensorflow.org/)

#12, #13 Evolutionary Algorithms

- [Evolutionary Computation](http://www.evolutionarycomputation.org/slides/)
- ​[Tiny GA](https://github.com/moshesipper/tiny_ga), [Tiny GP](https://github.com/moshesipper/tiny_gp)
- Genetic Programming (GP): [Koza](http://www.genetic-programming.com/c2003lecture1modified.ppt),[ Koza Tutorial,](http://www.genetic-programming.com/gecco2003tutorial.pdf) [Koza &amp; Poli](http://www.cs.bham.ac.uk/~wbl/biblio/cache/bin/cache.php?koza:2003:gpt,http___www.genetic-programming.com_jkpdf_burke2003tutorial.pdf,http://www.genetic-programming.com/jkpdf/burke2003tutorial.pdf), [Yoo, ](https://coinse.kaist.ac.kr/assets/files/teaching/cs454/cs454-slide09.pdf)[Herrmann](https://www.inf.ed.ac.uk/teaching/courses/nat/slides/nat09h.pdf), ​[Koza's vids](https://www.youtube.com/channel/UC9MEHhji3ODbE_e66EgFkew)
- ​[Multi-Objective Optimization](https://engineering.purdue.edu/~sudhoff/ee630/Lecture09.pdf)​
- [Schema theorem](https://engineering.purdue.edu/~sudhoff/ee630/Lecture03.pdf)
- [Linear GP](http://www.am.chalmers.se/~wolff/AI2/Lect05LGP.pdf)
- [Cartesian GP](http://cs.ijs.si/ppsn2014/files/slides/ppsn2014-tutorial3-miller.pdf)
- [Grammatical Evolution](https://web.archive.org/web/20110721124315/http:/www.grammaticalevolution.org/tutorial.pdf)
- [New Pathways in Coevolutionary Computation](https://drive.google.com/file/d/1fKymYCJPyd9rBmpEDgRPRe0GE7Yr3SuI/view?usp=sharing)
- [Novelty search](https://www.cs.ucf.edu/~gitars/cap6671-2010/Presentations/lehman_alife08.pdf)​​
- [Humies](https://www.human-competitive.org/)


***	

<a name="cheat">**Cheat Sheets**</a>
*   [Machine Learning Glossary](https://ml-cheatsheet.readthedocs.io/en/latest/index.html)
*	[Cheat Sheets for Machine Learning and Data Science](https://sites.google.com/view/datascience-cheat-sheets)
*	[The Illustrated Machine Learning Website](https://illustrated-machine-learning.github.io/)


***	

<a name="pros">**Algorithm Pros and Cons**</a>

- KN Neighbors \
   ✔ Simple, No training, No assumption about data, Easy to implement, New data can be added seamlessly, Only one hyperparameter \
   ✖ Doesn't work well in high dimensions, Sensitive to noisy data, missing values and outliers, Doesn't work well with large data sets —  cost of calculating distance is high, Needs feature scaling, Doesn't work well on imbalanced data, Doesn't deal well with missing values

- Decision Tree \
   ✔ Doesn't require standardization or normalization, Easy to implement, Can handle missing values, Automatic feature selection \
   ✖ High variance, Higher training time, Can become complex, Can easily overfit

- Random Forest \
   ✔ Left-out data can be used for testing, High accuracy, Provides feature importance estimates, Can handle missing values, Doesn't require feature scaling, Good performance on imbalanced datasets, Can handle large dataset, Outliers have little impact, Less overfitting \
   ✖ Less interpretable, More computational resources, Prediction time high

- Linear Regression \
   ✔ Simple, Interpretable, Easy to Implement \
   ✖ Assumes linear relationship between features, Sensitive to outliers

- Logistic Regression \
   ✔ Doesn’t assume linear relationship between independent and dependent variables, Output can be interpreted as probability, Robust to noise \
   ✖ Requires more data, Effective when linearly separable

- Lasso Regression (L1) \
   ✔ Prevents overfitting, Selects features by shrinking coefficients to zero \
   ✖ Selected features will be biased, Prediction can be worse than Ridge

- Ridge Regression (L2) \
   ✔ Prevents overfitting \
   ✖ Increases bias, Less interpretability 

- AdaBoost \
   ✔ Fast, Reduced bias, Little need to tune \
   ✖ Vulnerable to noise, Can overfit

- Gradient Boosting \
   ✔ Good performance \
   ✖ Harder to tune hyperparameters

- XGBoost \
   ✔ Less feature engineering required, Outliers have little impact, Can output feature importance, Handles large datasets, Good model performance, Less prone to overfitting \​
   ✖ Difficult to interpret, Harder to tune as there are numerous hyperparameters

- SVM \
   ✔ Performs well in higher dimensions, Excellent when classes are separable, Outliers have less impact \
   ✖ Slow, Poor performance with overlapping classes, Selecting appropriate kernel functions can be tricky

- Naïve Bayes \
   ✔ Fast, Simple, Requires less training data, Scalable, Insensitive to irrelevant features, Good performance with high-dimensional data \
   ✖ Assumes independence of features

- Deep Learning \
  ✔ Superb performance with unstructured data (images, video, audio, text) \
  ✖ (Very) long training time, Many hyperparameters, Prone to overfitting


***	

<a name="resources">**Resources: Evolutionary Algorithms, Machine Learning, Deep Learning**</a>

*Reads / Vids*

*   [Genetic and Evolutionary Algorithms and Programming](https://drive.google.com/file/d/0B6G3tbmMcpR4WVBTeDhKa3NtQjg/view?usp=sharing)
*   [גיא כתבי - אלגוריתמים אבולוציוניים](https://www.youtube.com/watch?v=XPx-a6MVne8&ab_channel=guykatabi) (YouTube) \[גיא בוגר הקורס שלי: _אלגוריתמים אבולוציוניים וחיים מלאכותיים_\]
*	[Choosing Representation, Mutation, and Crossover in Genetic Algorithms
](https://ieeexplore.ieee.org/document/9942691/interactive)
*   [Introduction to Evolutionary Computing](http://www.evolutionarycomputation.org/) (course/book slides)
*   [John Koza Genetic Programming](https://www.youtube.com/channel/UC9MEHhji3ODbE_e66EgFkew) (YouTube)
*   [Some reports in the popular press](https://www.moshesipper.com/publications.html)
*   [Why video games are essential for inventing artificial intelligence](https://togelius.blogspot.co.il/2016/01/why-video-games-are-essential-for.html)
*   [Biologic or “By Ole Logic”](http://www.moshesipper.com/blog/biologic-or-by-ole-logic)
*   [26 Top Machine Learning Interview Questions and Answers: Theory Edition](https://www.blog.confetti.ai/post/26-top-machine-learning-interview-questions-and-answers-theory)
*   [10 Popular Machine Learning Algorithms In A Nutshell](https://www.theinsaneapp.com/2021/11/machine-learning-algorithms-for-beginners.html)
*   [StatQuest with Josh Starmer](https://www.youtube.com/user/joshstarmer)
*   [Machine learning preparatory week @PSL](https://data-psl.github.io/lectures2020/)
*   [Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning/home/welcome) (coursera)
*   [ROC-AUC](https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/)
*   [Tinker With a Neural Network in Your Browser](https://playground.tensorflow.org/)
*   [Common Machine Learning Algorithms for Beginners](https://www.dezyre.com/article/common-machine-learning-algorithms-for-beginners/202)
*   ​[ML YouTube Courses](https://github.com/dair-ai/ML-YouTube-Courses)​
*	[Machine Learning Essentials for Biomedical Data Science: Introduction and ML Basics](https://www.youtube.com/watch?v=Qcgav8NmPxY&list=PLafPhSv1OSDfEqFsBnurxzJbcwZSJA8X4)
  

*Books*

*   M. Sipper, _[Evolved to Win](https://www.moshesipper.com/evolved-to-win.html)_, Lulu, 2011 (freely downloadable)
*   M. Sipper, _[Machine Nature: The Coming Age of Bio-Inspired Computing](https://www.moshesipper.com/machine-nature-the-coming-age-of-bio-inspired-computing.html)_, McGraw-Hill, New York, 2002
*   A.E. Eiben and J.E. Smith, [_Introduction to Evolutionary Computing_](http://www.cs.vu.nl/~gusz/ecbook/ecbook.html), Springer, 1st edition, 2003, Corr. 2nd printing, 2007
*   R. Poli, B. Langdon, & N. McPhee, [_A Field Guide to Genetic Programming_](http://www.gp-field-guide.org.uk/), 2008. (freely downloadable)
*   J. Koza, [_Genetic Programming: On the Programming of Computers by Means of Natural Selection_](http://www.genetic-programming.org/gpbook1toc.html), MIT Press, Cambridge, MA, 1992.
*   S. Luke, [_Essentials of Metaheuristics_](http://cs.gmu.edu/~sean/book/metaheuristics/), 2010. (freely downloadable)
*   A. Geron, [Hands On Machine Learning with Scikit Learn and TensorFlow](https://github.com/yanshengjia/ml-road/blob/master/resources/Hands%20On%20Machine%20Learning%20with%20Scikit%20Learn%20and%20TensorFlow.pdf)
*   G. James, D. Witten, T. Hastie, R. Tibshirani, [An Introduction to Statistical Learning](https://www.statlearning.com/), 2nd edition, 2021 (freely downloadable)
*   J. VanderPlas, [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
*   K. Reitz, [The Hitchhiker’s Guide to Python](https://docs.python-guide.org/)
*   M. Nielsen, [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html)
*   Z. Michalewicz & D.B. Fogel, [_How to Solve It: Modern Heuristics_](https://www.springer.com/computer/theoretical+computer+science/foundations+of+computations/book/978-3-540-22494-5), 2nd ed. Revised and Extended, 2004
*   Z. Michalewicz. [_Genetic Algorithms + Data Structures = Evolution Programs_](http://www.springeronline.com/sgw/cda/frontpage/0,10735,5-40109-22-1430991-0,00.html). Springer-Verlag, Berlin, 3rd edition, 1996
*   D. Floreano & C. Mattiussi, [_Bio-Inspired Artificial Intelligence: Theories, Methods, and Technologies_](http://baibook.epfl.ch/), MIT Press, 2008
*   A. Tettamanzi & M. Tomassini, [_Soft Computing: Integrating Evolutionary, Neural, and Fuzzy Systems_](https://www.springer.com/computer/theoretical+computer+science/book/978-3-540-42204-4), Springer-Verlag, Heidelberg, 2001
*   M. Mohri, A. Rostamizadeh, and A. Talwalka, [Foundations of Machine Learning](https://www.dropbox.com/s/4fij1xrclwjdu5y/Foundations%20of%20Machine%20Learning%2C%20Mohri%202012.pdf?dl=0), MIT Press, 2012 (freely downloadable)  
    

*Software*

*	[EC-KitY: Evolutionary Computation Tool Kit in Python with Seamless Machine Learning Integration](https://www.eckity.org/)
*   [gplearn: Genetic Programming in Python, with a scikit-learn inspired and compatible API](https://gplearn.readthedocs.io/en/stable/#)
*   [LEAP: Library for Evolutionary Algorithms in Python](https://github.com/AureumChaos/LEAP)
*   [DEAP: Distributed Evolutionary Algorithms in Python](https://deap.readthedocs.io/en/master/)
*   [Swarm Intelligence in Python (Genetic Algorithm, Particle Swarm Optimization, Simulated Annealing, Ant Colony Algorithm, Immune Algorithm, Artificial Fish Swarm Algorithm in Python)](https://github.com/guofei9987/scikit-opt)
*   [Scikit-learn: Machine Learning in Python](https://scikit-learn.org/stable/index.html)
*   [​Mlxtend (machine learning extensions)](https://rasbt.github.io/mlxtend/) 
*   [PyTorch (deep networks)](https://pytorch.org/)
*   [Best-of Machine Learning with Python​](https://github.com/ml-tooling/best-of-ml-python)
*   [Fundamental concepts of PyTorch through self-contained examples](https://github.com/jcjohnson/pytorch-examples)​
*   [Faster Python calculations with Numba](https://pythonspeed.com/articles/numba-faster-python)  
    

*Datasets*

*   [Tabular & cleaned (PMLB)](https://github.com/EpistasisLab/pmlb)
*   [By domain](https://www.datasetlist.com/)
*   ​[By application](https://github.com/awesomedata/awesome-public-datasets)
*   [Search engine](https://datasetsearch.research.google.com/)
*   [Kaggle competitions](https://www.kaggle.com/datasets)
*   [OpenML​](https://www.openml.org/)
*   [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php)
*   ​[Image Databases](https://homepages.inf.ed.ac.uk/rbf/CVonline/Imagedbase.htm)
*   [AWS Open Data Registry](https://registry.opendata.aws/)
*   ​[Wikipedia ML Datasets](https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research)
*   [The Big Bad NLP Database](https://datasets.quantumstat.com/)
*   ​[Datasets for Machine Learning and Deep Learning](https://sebastianraschka.com/blog/2021/ml-dl-datasets.html)
*   [Browse State-of-the-Art](https://paperswithcode.com/sota)  
    