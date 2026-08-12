---
title: "Lab 7"
date: \today
author:
    - "Sam Kutsyn, 2581500"
    - "EE 446"
geometry: margin=1in
fontsize: 11pt
documentclass: article
header-includes:
    - \usepackage{graphicx}
    - \usepackage{float}
    - \floatplacement{figure}{H}
    # - \setkeys{Gin}{width=0.7\textwidth,height=0.3\textheight,keepaspectratio}
    - \usepackage{hyperref}
    - \usepackage{amsmath}
    - \usepackage{amsthm}
---

## Submission

See submission files in [./submission](./submission) folder.

## Inference

![Screenshot of inference on the board.](./assets/screenshot.png)

## Fixes applied

1. Use Adam optimizer to improve convergence speed and model accuracy.
2. Use categorical cross entropy or multi-class classification.
3. Use `accuracy` as grading metric to evaluate model performance.
4. Update plot to use the new grading metric for correct visualization.
