# Equation Learner - Enhanced Training Convergence

This version of the Equation Learner (EQL), is based on the tensorflow implementation by Martius et al.:

https://github.com/martius-lab/EQL_Tensorflow

With a few structural changes, parameter tracking, saving of results in files aswell as an optional MLFlow implementation. Additionally there are scripts for DataFrame creation, inspecting resulting formulas and plotting the results in an interactive graph.

Details about the usage of the EQL can be seen at above stated github repository of the tensorflow implementation and reader wanting to know something about the EQL itself are referred to "Learning equations for extrapolation and control" by Martius et al. [1] and my master thesis "Exploring Symbolic Regression for hypothesis testing of
London-Dispersion corrections in theoretical molecular physics." [2], where the application of this version is shown.

[1] https://al.is.tuebingen.mpg.de/publications/sahoolampertmartius2018-eqldiv

[2] https://zenodo.org/records/4382219

# Dependencies

python>=3.5

tensorflow==1.14.0

graphviz (including binaries)

latex

mlfow>=1.10 (for optional mlflow implementation)

holoviews (for displaying an interactive graph)




 
