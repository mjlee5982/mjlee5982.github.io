---
layout: distill
title: Rethinking Quantization for the Real World
description: This blog post highlights the fact that most existing quantization techniques have been developed and validated under clean, balanced benchmark settings. It critically examines whether these methods can be reliably applied in real-world environments, where data imbalance and custom evaluation metrics constraints are the norm. Furthermore, it proposes ideas for partially mitigating the impact of real-world data imbalance during the Mixed Precision Quantization (MPQ) process.
date: 2025-05-30
future: true
htmlwidgets: true
hidden: true

# Anonymize when submitting
authors:
  - name: Anonymous

# authors:
#  - name: 
#    url: " "
#    affiliations:
#      name: 

# must be the exact same name as your blogpost
bibliography: 2025-04-28-distill-example.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Background
    subsections:
    - name: Application Driven Machine Learning (ADML)
    - name: Long Tail Distribution (LTD)
    - name: Decoupling Representation and Classifier for Lonng-Tailed Recognition
    - name: Quantization and Mixed Precision Quantization (MPQ)
  - name: Insight from Experiments
    subsections:
    - name: Are Traditional Evaluation Methods Applicable?
    - name: Can Effecitve Training Methods Address LTD Issues?
    - name: Does LTD Impact Quantized Modes?
    - name: Can Quantization Process Mitigate REal-World LTD Issues?
  - name : Conclusion
    subsections:
    - name: What problem is this work trying to tackle?
    - name: That contributions did this work make, and what impact should this work have?
    - name: How new is this effort?
    - name: what are the limitations of this work?


# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---



