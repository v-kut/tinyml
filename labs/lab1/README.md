---
title: "Lab 1"
date: \today
author:
    - "Sam Kutsyn, 2581500"
    - "Volodymyr Kuchera, 2523181"
    - "EE 446"
geometry: margin=1in
fontsize: 11pt
documentclass: article
header-includes:
    - \usepackage{graphicx}
    - \usepackage{float}
    - \floatplacement{figure}{H}
    - \setkeys{Gin}{width=0.7\textwidth,height=0.3\textheight,keepaspectratio}
    - \usepackage{hyperref}
    - \usepackage{amsmath}
    - \usepackage{amsthm}
---

# Preamble

You can access the project at https://studio.edgeimpulse.com/studio/1044661.

# Task 1

![Training accuracy for the default model.](./assets/acc_train_1.png)

![Testing accuracy for the default model.](./assets/acc_test_1.png)

# Task 2

To improve the inference performance, we have switched the flatten layer for the dense layer and increased the training epochs to 120, which allowed us to reach 96% accuracy in testing.

![New model architecture.](./assets/new_model_arch.png)

![Training accuracy for the new model.](./assets/acc_train_2.png)

![Testing accuracy for the new model.](./assets/acc_test_2.png)
