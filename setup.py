from distutils.core import setup

setup(
    name="settree",
    packages=['settree'],
    version="0.2.1",
    author="Roy Hirsch",
    license='MIT',
    description="A framework for learning tree-based models over sets",
    long_description='Set-Tree\nExtending decision trees to process sets\n\n'
                     'This is the official repository for the paper: "Trees with Attention for Set Prediction Tasks" (ICML21).\n'
                     'This repository contains a prototypical implementaion of Set-Tree and GBeST (Gradient Boosted Set-Tree) algorithms\n'
                     'The Set-Tree package can be downloaded from PIP: pip install settree\n'
                     'We also supply the code and datasets for reproducing our experimetns under exps folder.\n\n'
                     'In many machine learning applications, each record represents a set of items. A set is an unordered group of items,'
                     ' the number of items may differ between different sets. Problems comprised from sets of items are present in diverse fields,'
                     ' from particle physics and cosmology to statistics and computer graphics.'
                     ' In this work, we present a novel tree-based algorithm for processing sets.\n\n'
                     'Set-Tree model comprised from two components:\n'
                     'Set-compatible split creteria: we specifically support the familly of split creteria defined'
                     ' by the following equation and parametrized by alpha and beta.\n'
                     'Attention-Sets: a mechanism for allplying the split creteria to subsets of the input.'
                     ' The attention-sets are derived forn previous split-creteria and allows the model to learn more complex set-functions.',
    author_email='royhirsch@mail.tau.ac.il',
    url='https://github.com/TAU-MLwell/Set-Tree',
    download_url='https://github.com/TAU-MLwell/Set-Tree/archive/refs/tags/0.2.1.tar.gz',

    install_requires=['numpy>=1.19.2', 'scikit-learn>= 0.23.1', 'scipy>=pi1.5.2'],
    classifiers=["Programming Language :: Python :: 3",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent"],
)

