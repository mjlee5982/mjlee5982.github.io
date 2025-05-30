---
layout: distill
title: Rethinking Quantization for the Real World
description: This blog post highlights the fact that most existing quantization techniques have been developed and validated under clean, balanced benchmark settings. It critically examines whether these methods can be reliably applied in real-world environments, where data imbalance and custom evaluation metrics constraints are the norm. Furthermore, it proposes ideas for partially mitigating the impact of real-world data imbalance during the Mixed Precision Quantization (MPQ) process.
date: 2025-05-30
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
# authors:
#  - name: Anonymous

authors:
  - name: Myeongjun Lee
    url: "https://github.com/mjlee5982"
    affiliations: 
      name: POHANG POSTECH

# must be the exact same name as your blogpost
bibliography: 2025-05-30-rethinking-quantization-for-the-real-world.bib

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

## 1. Introduction 
This blog post offers a fresh perspective on traditional machine learning concepts and techniques by questioning whether quantization can be effectively applied to models trained on real-world, application-driven data, from the viewpoint of Application-Driven Machine Learning (ADML) <d-cite key="rolnick2024application"></d-cite>. To date, research on quantization techniques for CNN models has largely focused on a limited set of widely used benchmark datasets, such as CIFAR-10 and ImageNet, and have been evaluated using narrow metrics such as accuracy and compression rate. However, real world applications often involve datasets with very different characteristics and evaluation criteria. For example, many naturally occurring datasets exhibit long-tailed distributions, where a few classes dominate the data and the majority of classes appear infrequently. These imbalanced distributions often cause models to focus on high-frequency classes during training, leading to severely degraded performance on rare classes. When quantization is applied to models trained on such long-tailed data, it’s possible that the overall accuracy may appear stable, yet performance on tail classes may deteriorate significantly. This highlights a critical limitation in how quantized models are evaluated and interpreted, especially under real-world constraints.


## 2. Background

### 2.1 Application Driven Machine Learning (ADML)
Application-Driven Machine Learning (ADML) is an approach to machine learning research emphasizing the design of algorithms specifically tailored to address real-world problems, rather than optimizing performance solely on standardized benchmark datasets. Unlike traditional methods-driven research, which prioritizes generalized metrics like accuracy and loss on clean, well-structured datasets (e.g., CIFAR-10, ImageNet), ADML explicitly incorporates domain-specific considerations, such as custom evaluation criteria, data characteristics, and user-defined constraints <d-cite key="rolnick2024application"></d-cite>. This paradigm is crucial because it acknowledges that real-world tasks frequently differ substantially from benchmark scenarios. For instance, tasks in remote sensing, healthcare, or biodiversity monitoring often require models to consider computational constraints, domain knowledge, and specialized metrics like uncertainty quantification or cost-sensitive accuracy. Recognizing and addressing these practical considerations significantly enhances the applicability and effectiveness of machine learning solutions in real-world environments.


<div class="row mt-3">
    {% include figure.html path="assets/img/2025-05-30-rethinking-quantization-for-the-real-world/fig1.jpg" class="img-fluid" %}
</div>
<div class="caption">
    Figure 1: Methods-Driven ML vs Application-Driven ML 
</div>


### 2.2 Long Tail Distribution (LTD)
Long-tail distribution (LTD) describes a scenario where a small number of classes dominate the dataset (head classes), while the majority of classes have significantly fewer samples (tail classes) <d-cite key="kang2020decoupling"></d-cite>. Such distributions are ubiquitous in real-world datasets like iNaturalist (biodiversity data), ImageNet-LT, and Places-LT, where many categories are represented by very few examples. LTD poses substantial challenges for machine learning models, which tend to overfit on head classes while underperforming on tail classes due to insufficient exposure during training <d-cite key="suhee2022long"></d-cite>. Consequently, the overall accuracy metric often masks severe performance degradation on rare classes, making standard evaluation metrics unreliable indicators of true model effectiveness. Addressing LTD requires strategies such as data re-sampling, class-balanced losses, and specialized evaluation protocols that better reflect performance across all classes, ensuring equitable attention to rare but potentially critical categories.

<div class="row mt-3">
    {% include figure.html path="assets/img/2025-05-30-rethinking-quantization-for-the-real-world/fig2.jpg" class="img-fluid" %}
</div>
<div class="caption">
    Figure 2: Long Tail Distribution (LTD) 
</div>


### 2.3 Decoupling Representation and Classifier for Long-Tailed Recognition
The concept of "Decoupling Representation and Classifier" for long-tailed recognition involves separating the training procedure into two distinct phases: representation learning and classifier training <d-cite key="kang2020decoupling"></d-cite>. Traditionally, models learn representations and classifiers jointly, which can obscure the sources of performance improvements. The decoupling approach demonstrates that effective representation learning does not necessarily require complex class-balancing strategies. Instead, training representations with simple instance-balanced sampling can yield robust features applicable across both frequent and infrequent classes. Subsequently, a classifier is trained or adjusted separately using methods like class-balanced re-training, nearest class mean classification, or weight normalization. Experimental studies using this decoupling strategy have shown substantial improvements over traditional methods on benchmarks such as ImageNet-LT, iNaturalist, and Places-LT, clearly illustrating its potential to effectively address long-tailed challenges without resorting to overly complex methods.

<div class="row mt-3">
    {% include figure.html path="assets/img/2025-05-30-rethinking-quantization-for-the-real-world/fig3.jpg" class="img-fluid" %}
</div>
<div class="caption">
    Figure 3: Decoupling representation and classifier for long-tailed recognition <d-cite key="suhee2022long"></d-cite>.
    In Stage 1, feature extraction and the downstream task are simultaneously trained using an imbalanced dataset. In Stage 2, only the downstream task is separately trained.
</div>


### 2.4. Quantization & Mixed Precision Quantization (MPQ)
Quantization refers to reducing the numerical precision (bit-width) used to represent neural network weights and activations, aiming to decrease computational and memory demands, which are critical for deploying models in resource-constrained environments. Mixed Precision Quantization (MPQ) takes this concept further by assigning different bit-widths to various layers or components within a neural network, optimizing the trade-off between model accuracy and compression efficiency <d-cite key="wang2019haq"></d-cite> <d-cite key="tang2023mixed"></d-cite>. This approach leverages the observation that different layers have varying sensitivities to quantization for example, some layers can be heavily quantized without significant performance loss, whereas others require higher precision to maintain accuracy. By carefully choosing bit-width assignments based on layer importance or sensitivity, MPQ can achieve substantial reductions in memory and energy usage while maintaining acceptable accuracy, making it a highly effective method for efficient, real-world deployment of machine learning models.

## 3. Insight from Experiments
This section provides insights derived from a series of straightforward and clear experiments aimed at understanding how quantization, a key technique in efficient machine learning, behaves under real-world data imbalance conditions like long-tail distributions (LTD). We investigated characteristic challenges and potential solutions when traditional evaluation environments and metrics are applied to LTD.
The experimental setup was designed for ease of reproducibility and to provide a simple, intuitive sensitivity evaluation approach for setting layer-wise bit-widths in Mixed Precision Quantization (MPQ). We created a CIFAR10-LT dataset and used the VGG16 CNN model for experiments. 

<div class="row mt-3">
    {% include figure.html path="assets/img/2025-05-30-rethinking-quantization-for-the-real-world/fig4.png" class="img-fluid" %}
</div>
<div class="caption">
    Figure 4: CIFAR10-LT(Long Tailed Distribution). 
    The CIFAR10 dataset is a balanced dataset consisting of 10 classes with 5000 images per class. CIFAR10-LT is generated by intentionally creating an imbalance in the number of images per class to form a long-tail distribution. For instance, with an imbalance factor (imb_factor) of 0.01, the ratio between Class 0 and Class 9 is 100:1
</div>


### 3.1 Are Traditional Evaluation Methods Applicable? 
Given that traditional quantization techniques have been developed and validated on balanced and well-structured datasets, we questioned whether existing evaluation metrics remain effective when applied to models trained on LTD datasets.
Normalized overall accuracy, commonly used as a performance metric for CNN-based image classification tasks, proved inadequate for evaluating models trained on LTD datasets. We generated CIFAR10-LT datasets with imbalance factors of 0.1(the number of image gap between head and tail is 10 times), 0.02, and 0.01. In addition to the overall accuracy (Acc(Norm)), we separately measured accuracy for head classes (approximately 80% of data) and tail classes (remaining 20%). The results revealed that Acc(Norm) failed to reflect performance accurately for head (Acc(Head)) and tail (Acc(Tail)) classes, particularly as imbalance factors increased. This demonstrates the necessity of evaluating head and tail classes separately in LTD scenarios. 

<div class="row mt-3">
    {% include figure.html path="assets/img/2025-05-30-rethinking-quantization-for-the-real-world/fig5.jpg" class="img-fluid" %}
</div>
<div class="caption">
    Figure 5: Comparison of Normal Accuracy with Head & Tail Class Accuracy.
    As the imbalance factor increases, the performance gap between the head and tail classes becomes more significant. Normal Accuracy fails to accurately reflect this imbalanced performance.
</div>


### 3.2 Can Effective Training Methods Address LTD Issues? 
Several methodologies have been proposed to tackle LTD datasets commonly found in real-world scenarios. Prominent methods include data-level solutions like re-sampling to balance head and tail classes, cost-sensitive learning to adjust class-specific loss during training, and transfer learning methods to leverage information from head classes to enhance tail class performance.
The Decoupling Representation and Classifier method, presented at ICLR 2020, demonstrated substantial performance improvements on LTD datasets by training representations (feature extractors) using instance-balanced sampling and separately retraining classifiers using class-balanced sampling. Using this method, we confirmed substantial performance improvements on CIFAR10-LT with imbalance factors of 0.1, 0.02, and 0.01. The improvement was more significant with higher imbalance factors, effectively narrowing the performance gap between head and tail classes compared to baseline methods. However, despite notable improvements, data imbalance persisted, especially evident at an imbalance factor of 0.01, where the accuracy gap between head and tail exceeded 10%. This indicates that while techniques like Decoupling Learning substantially mitigate data imbalance, significant imbalance still poses challenges.

<div class="row mt-3">
    {% include figure.html path="assets/img/2025-05-30-rethinking-quantization-for-the-real-world/table1.jpg" class="img-fluid" %}
</div>
<div class="caption">
    Table 1: Performance Changes Due to Decoupling Learning
</div>

<div class="row mt-3">
    {% include figure.html path="assets/img/2025-05-30-rethinking-quantization-for-the-real-world/fig6.jpg" class="img-fluid" %}
</div>
<div class="caption">
    Figure 6: Accuracy Changes of Head and Tail Classes due to Decoupling Learning.
    Decoupling learning improves performance imbalance caused by LTD. However, a residual imbalance in model performance persists when the imbalance factor is above 0.01.
</div>

### 3.3 Does LTD Impact Quantized Models? 
Previous experiments revealed that CNN models trained on LTD datasets exhibited a bias toward head classes, leading to poorer tail class performance. While training methods such as Decoupling Learning improved performance, severe data imbalance inevitably resulted in notable performance disparities. This raised questions about how such disparities manifest in quantized models. We hypothesized and experimentally confirmed that tail class performance degradation becomes more pronounced relative to head class performance as bit precision decreases (from 32-bit to 8-bit, 6-bit, and 4-bit quantization). Additionally, models improved through Decoupling Learning showed smaller performance reductions in tail classes compared to baseline models at lower bit precisions. These findings clearly illustrate that quantization exacerbates data imbalance issues, particularly impacting tail class performance.

<div class="row mt-3">
    {% include figure.html path="assets/img/2025-05-30-rethinking-quantization-for-the-real-world/fig7.jpg" class="img-fluid" %}
</div>
<div class="caption">
    Figure 7: Performance Degradation of Head and Tail Classes in a Regular Model due to Quantization.
    When quantizing from 32-bit to 6-bit, head class performance decreases by 0.5%, while tail class performance decreases by 6.5%.
</div>

<div class="row mt-3">
    {% include figure.html path="assets/img/2025-05-30-rethinking-quantization-for-the-real-world/fig8.jpg" class="img-fluid" %}
</div>
<div class="caption">
    Figure 8: Performance Degradation of Head and Tail Classes in a Decoupling Learning Model due to Quantization.
    Quantizing from 32-bit to 6-bit causes a performance reduction of 2.7% in head classes and 6.6% in tail classes, further widening the performance gap between head and tail classes.
</div>



### 3.4 Can Quantization Processes Mitigate Real-World LTD Issues?
Considering the prevalence of LTD datasets across various real-world applications, and acknowledging that sampling and training techniques such as Decoupling Learning cannot entirely resolve performance disparities between head and tail classes, we explored whether the quantization process itself could further mitigate these disparities. Is there a simpler and more efficient way to reduce tail class performance degradation during quantization without extensive retraining?
Our experimental results offer a clear answer: Mixed Precision Quantization (MPQ) can indeed partially address tail class performance issues during quantization. We propose repurposing the existing Mixed Precision Quantization (MPQ) technique, originally designed to improve computational cost and energy efficiency, as a novel approach to address data imbalance issues that arise during the LTD based model compression process. In case of MPQ, to assign appropriate bit-widths to each layer, sensitivity analysis is required <d-cite key="yeji2025amc"></d-cite>. In our approach, we incorporate tail-class performance indicators into the analysis, guiding the model to reduce its overemphasis on head classes and instead improve the performance on tail classes, effectively driving compression in a more balanced direction. This approach significantly improved tail class performance without additional training overhead associated with post-training quantization (PTQ). Compared to uniform quantization, this imbalance-aware MPQ showed negligible overall accuracy degradation while effectively reducing the performance gap between head and tail classes during the quantization process.

<div class="row mt-3">
    {% include figure.html path="assets/img/2025-05-30-rethinking-quantization-for-the-real-world/table2.jpg" class="img-fluid" %}
</div>
<div class="caption">
    Table 2: MPQ Model Performance Based on Sensitivity Evaluation Methods.
    The performance difference between MPQ evaluated with Normal accuracy and MPQ evaluated with Tail class accuracy is negligible compared to the decoupling learning model (baseline).
</div>

<div class="row mt-3">
    {% include figure.html path="assets/img/2025-05-30-rethinking-quantization-for-the-real-world/fig9_new.jpg" class="img-fluid" %}
</div>
<div class="caption">
    Figure 9: Rank Changes Based on Tail Performance (Top 9).
    Rank changes between sensitivity evaluations based on Normal Accuracy and Tail class Accuracy were observed, suggesting that MPQ can be effectively tailored to enhance tail class performance.
</div>

<div class="row mt-3">
    {% include figure.html path="assets/img/2025-05-30-rethinking-quantization-for-the-real-world/fig10_new.jpg" class="img-fluid" %}
</div>
<div class="caption">
    Figure 10: Class-wise Performance of MPQ Applied Model for Tail Class Improvement.
    Performance reduction in head classes (Class 0~3) and performance improvement in tail classes (Class 4~9) confirm that MPQ effectively mitigates performance disparities caused by data imbalance.
</div>

<div class="row mt-3">
    {% include figure.html path="assets/img/2025-05-30-rethinking-quantization-for-the-real-world/fig11.jpg" class="img-fluid" %}
</div>
<div class="caption">
    Figure 11: Layer-wise Sensitivity Analysis Considering Data Imbalance.
</div>



## 4.Conclusion
### 4.1 What problem is this work trying to tackle?
This research seeks solutions from the perspective of Application-Driven Machine Learning to address Long Tail Distribution (LTD), a common real-world problem, specifically through the application of model quantization. We experimentally explored the effects of data imbalance in LTD datasets on the conventional Decoupling Learning method and model quantization. Furthermore, we propose a method to resolve data imbalance using Mixed Precision Quantization (MPQ) at the quantization stage, rather than relying solely on data sampling or model training.

### 4.2 That contributions did this work make, and what impact should this work have?
By exploring and proposing methods to overcome LTD problems at the quantization stage rather than during data sampling or model training, this work is expected to inspire future research aimed at addressing data imbalance issues during the model compression phase. Using CIFAR10-LT datasets and the VGG16 model, we conducted experiments that are relatively straightforward to reproduce and provided several significant results:
- We confirmed that traditional performance metrics, such as model accuracy, are insufficient for accurately reflecting performance and data imbalance in LTD datasets. It is essential to separately measure and evaluate accuracy for head and tail classes to assess genuine performance improvements.
- Although significant performance improvements were achieved using the representative LTD dataset training method, Decoupling Learning, experiments demonstrated persistent data imbalance impacts when class imbalance exceeded 100 times.
- The impact of LTD datasets is also evident in quantized models, with tail class performance degradation becoming significantly more pronounced relative to head classes as bit precision decreases.
- Finally, to address LTD issues at the quantization stage, we propose an MPQ method. By incorporating tail class performance metrics into the MPQ sensitivity evaluation for layer-wise bit-width settings, we experimentally demonstrated the feasibility of an LTD-aware MPQ method.

### 4.3 How new is this effort?
While existing research primarily focuses on data sampling and training methods to solve LTD dataset problems, the approach proposed in this blog is novel in addressing data imbalance at the model compression stage through MPQ application. Additionally, by experimentally demonstrating the impact of data imbalance on quantized models, this research underscores the need for considering real-world problems like LTD at the quantization stage, suggesting new avenues for future research.

### 4.4 what are the limitations of this work? 
Despite showing meaningful experimental results, the study is currently limited to the CIFAR10-LT dataset and a single VGG16 model. To generalize these findings more broadly, it will be necessary to expand experiments to include additional datasets and models.
Although we heuristically evaluated MPQ sensitivity in relatively simple models like VGG16, further research is needed to optimize the extensive search space required for determining appropriate bit-width settings in more complex models.



