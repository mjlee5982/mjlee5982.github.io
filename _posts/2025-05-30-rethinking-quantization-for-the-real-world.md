---
layout: distill
title: Rethinking Quantization for the Real World
description: This blog post highlights the fact that most existing quantization techniques have been developed and validated under clean, balanced benchmark settings. It critically examines whether these methods can be reliably applied in real-world environments, where data imbalance and custom evaluation metrics constraints are the norm. Furthermore, it proposes ideas for partially mitigating the impact of real-world data imbalance during the Mixed Precision Quantization (MPQ) process.
date: 2025-05-30
future: true
htmlwidgets: true
hidden: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Myeongjun Lee
    url: "https://github.com/mjlee5982"
    affiliations:
      name: POHANG, POSTECH

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
  - -name : Conclusion 


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
Note: please use the table of contents as defined in the front matter rather than the traditional markdown styling.

## 1. Introduction 
This blog post offers a fresh perspective on traditional machine learning concepts and techniques by questioning whether quantization can be effectively applied to models trained on real-world, application-driven data, from the viewpoint of Application-Driven Machine Learning (ADML). To date, research on quantization techniques for CNN models has largely focused on a limited set of widely used benchmark datasets, such as CIFAR-10 and ImageNet, and have been evaluated using narrow metrics such as accuracy and compression rate. However, real world applications often involve datasets with very different characteristics and evaluation criteria. For example, many naturally occurring datasets exhibit long-tailed distributions, where a few classes dominate the data and the majority of classes appear infrequently. These imbalanced distributions often cause models to focus on high-frequency classes during training, leading to severely degraded performance on rare classes. When quantization is applied to models trained on such long-tailed data, it’s possible that the overall accuracy may appear stable, yet performance on tail classes may deteriorate significantly. This highlights a critical limitation in how quantized models are evaluated and interpreted, especially under real-world constraints.


## 2. Background

### 2.1 Application Driven Machine Learning (ADML)
Application-Driven Machine Learning (ADML) is an approach to machine learning research emphasizing the design of algorithms specifically tailored to address real-world problems, rather than optimizing performance solely on standardized benchmark datasets. Unlike traditional methods-driven research, which prioritizes generalized metrics like accuracy and loss on clean, well-structured datasets (e.g., CIFAR-10, ImageNet), ADML explicitly incorporates domain-specific considerations, such as custom evaluation criteria, data characteristics, and user-defined constraints. This paradigm is crucial because it acknowledges that real-world tasks frequently differ substantially from benchmark scenarios. For instance, tasks in remote sensing, healthcare, or biodiversity monitoring often require models to consider computational constraints, domain knowledge, and specialized metrics like uncertainty quantification or cost-sensitive accuracy. Recognizing and addressing these practical considerations significantly enhances the applicability and effectiveness of machine learning solutions in real-world environments.


<div class="row mt-3">
    {% include figure.html path="assets/img/2025-05-30-rethinking-quantization-for-the-real-world/fig1.jpg" class="img-fluid" %}
</div>
<div class="caption">
    Figure 1: Methods-Driven ML vs Application-Driven ML 
</div>


### 2.2 Long Tail Distribution (LTD)
Long-tail distribution (LTD) describes a scenario where a small number of classes dominate the dataset (head classes), while the majority of classes have significantly fewer samples (tail classes). Such distributions are ubiquitous in real-world datasets like iNaturalist (biodiversity data), ImageNet-LT, and Places-LT, where many categories are represented by very few examples. LTD poses substantial challenges for machine learning models, which tend to overfit on head classes while underperforming on tail classes due to insufficient exposure during training. Consequently, the overall accuracy metric often masks severe performance degradation on rare classes, making standard evaluation metrics unreliable indicators of true model effectiveness. Addressing LTD requires strategies such as data re-sampling, class-balanced losses, and specialized evaluation protocols that better reflect performance across all classes, ensuring equitable attention to rare but potentially critical categories.

<div class="row mt-3">
    {% include figure.html path="assets/img/2025-05-30-rethinking-quantization-for-the-real-world/fig2.jpg" class="img-fluid" %}
</div>
<div class="caption">
    Figure 2: Long Tail Distribution (LTD) 
</div>


### 2.3 Decoupling Representation and Classifier for Long-Tailed Recognition
The concept of "Decoupling Representation and Classifier" for long-tailed recognition involves separating the training procedure into two distinct phases: representation learning and classifier training. Traditionally, models learn representations and classifiers jointly, which can obscure the sources of performance improvements. The decoupling approach demonstrates that effective representation learning does not necessarily require complex class-balancing strategies. Instead, training representations with simple instance-balanced sampling can yield robust features applicable across both frequent and infrequent classes. Subsequently, a classifier is trained or adjusted separately using methods like class-balanced re-training, nearest class mean classification, or weight normalization. Experimental studies using this decoupling strategy have shown substantial improvements over traditional methods on benchmarks such as ImageNet-LT, iNaturalist, and Places-LT, clearly illustrating its potential to effectively address long-tailed challenges without resorting to overly complex methods.

<div class="row mt-3">
    {% include figure.html path="assets/img/2025-05-30-rethinking-quantization-for-the-real-world/fig3.jpg" class="img-fluid" %}
</div>
<div class="caption">
    Figure 3: Decoupling representation and classifier for long-tailed recognition
    In Stage 1, feature extraction and the downstream task are simultaneously trained using an imbalanced dataset. In Stage 2, only the downstream task is separately trained.
</div>


### 2.4. Quantization & Mixed Precision Quantization (MPQ)
Quantization refers to reducing the numerical precision (bit-width) used to represent neural network weights and activations, aiming to decrease computational and memory demands, which are critical for deploying models in resource-constrained environments. Mixed Precision Quantization (MPQ) takes this concept further by assigning different bit-widths to various layers or components within a neural network, optimizing the trade-off between model accuracy and compression efficiency. This approach leverages the observation that different layers have varying sensitivities to quantization for example, some layers can be heavily quantized without significant performance loss, whereas others require higher precision to maintain accuracy. By carefully choosing bit-width assignments based on layer importance or sensitivity, MPQ can achieve substantial reductions in memory and energy usage while maintaining acceptable accuracy, making it a highly effective method for efficient, real-world deployment of machine learning models.

## 3. Insight from Experiments
This section provides insights derived from a series of straightforward and clear experiments aimed at understanding how quantization, a key technique in efficient machine learning, behaves under real-world data imbalance conditions like long-tail distributions (LTD). We investigated characteristic challenges and potential solutions when traditional evaluation environments and metrics are applied to LTD.
The experimental setup was designed for ease of reproducibility and to provide a simple, intuitive sensitivity evaluation approach for setting layer-wise bit-widths in Mixed Precision Quantization (MPQ). We created a CIFAR10-LT dataset and used the VGG16 CNN model for experiments. 

<div class="row mt-3">
    {% include figure.html path="assets/img/2025-05-30-rethinking-quantization-for-the-real-world/fig4.png" class="img-fluid" %}
</div>
<div class="caption">
    Figure 4: CIFAR10-LT(Long Tailed Distribution) 
    The CIFAR10 dataset is a balanced dataset consisting of 10 classes with 5000 images per class. CIFAR10-LT is generated by intentionally creating an imbalance in the number of images per class to form a long-tail distribution. For instance, with an imbalance factor (imb_factor) of 0.01, the ratio between Class 0 and Class 9 is 100:1
</div>


### 3.1 Are Traditional Evaluation Methods Applicable? 
Given that traditional quantization techniques have been developed and validated on balanced and well-structured datasets, we questioned whether existing evaluation metrics remain effective when applied to models trained on LTD datasets.
Standard accuracy, commonly used as a performance metric for CNN-based image classification tasks, proved inadequate for evaluating models trained on LTD datasets. We generated CIFAR10-LT datasets with imbalance factors of 10, 50, and 100. In addition to the overall accuracy (Acc(Norm)), we separately measured accuracy for head classes (approximately 80% of data) and tail classes (remaining 20%). The results revealed that Acc(Norm) failed to reflect performance accurately for head (Acc(Head)) and tail (Acc(Tail)) classes, particularly as imbalance factors increased. This demonstrates the necessity of evaluating head and tail classes separately in LTD scenarios. 

<div class="row mt-3">
    {% include figure.html path="assets/img/2025-05-30-rethinking-quantization-for-the-real-world/fig5.jpg" class="img-fluid" %}
</div>
<div class="caption">
    Figure 5: Comparison of Normal Accuracy with Head & Tail Class Accuracy
    As the imbalance factor increases, the performance gap between the head and tail classes becomes more significant. Normal Accuracy fails to accurately reflect this imbalanced performance.
</div>


### 3.2 Can Effective Training Methods Address LTD Issues? 
Several methodologies have been proposed to tackle LTD datasets commonly found in real-world scenarios. Prominent methods include data-level solutions like re-sampling to balance head and tail classes, cost-sensitive learning to adjust class-specific loss during training, and transfer learning methods to leverage information from head classes to enhance tail class performance.
The Decoupling Representation and Classifier method, presented at ICLR 2020, demonstrated substantial performance improvements on LTD datasets by training representations (feature extractors) using instance-balanced sampling and separately retraining classifiers using class-balanced sampling. Using this method, we confirmed substantial performance improvements on CIFAR10-LT with imbalance factors of 10, 50, and 100. The improvement was more significant with higher imbalance factors, effectively narrowing the performance gap between head and tail classes compared to baseline methods. However, despite notable improvements, data imbalance persisted, especially evident at an imbalance factor of 100, where the accuracy gap between head and tail exceeded 10%. This indicates that while techniques like Decoupling Learning substantially mitigate data imbalance, significant imbalance still poses challenges.

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
    Figure 6: Performance Changes of Head and Tail Classes due to Decoupling Learning
    Decoupling learning improves performance imbalance caused by LTD. However, a residual imbalance in model performance persists when the imbalance factor is above 0.01.
</div>

### 3.3 Does LTD Impact Quantized Models? 
Previous experiments revealed that CNN models trained on LTD datasets exhibited a bias toward head classes, leading to poorer tail class performance. While training methods such as Decoupling Learning improved performance, severe data imbalance inevitably resulted in notable performance disparities. This raised questions about how such disparities manifest in quantized models. We hypothesized and experimentally confirmed that tail class performance degradation becomes more pronounced relative to head class performance as bit precision decreases (from 32-bit to 8-bit, 6-bit, and 4-bit quantization). Additionally, models improved through Decoupling Learning showed smaller performance reductions in tail classes compared to baseline models at lower bit precisions. These findings clearly illustrate that quantization exacerbates data imbalance issues, particularly impacting tail class performance.

<div class="row mt-3">
    {% include figure.html path="assets/img/2025-05-30-rethinking-quantization-for-the-real-world/fig7.jpg" class="img-fluid" %}
</div>
<div class="caption">
    Figure 7: Performance Degradation of Head and Tail Classes in a Regular Model due to Quantization
    When quantizing from 32-bit to 6-bit, head class performance decreases by 0.5%, while tail class performance decreases by 6.5%.
</div>

<div class="row mt-3">
    {% include figure.html path="assets/img/2025-05-30-rethinking-quantization-for-the-real-world/fig8.jpg" class="img-fluid" %}
</div>
<div class="caption">
    Figure 8: Performance Degradation of Head and Tail Classes in a Decoupling Learning Model due to Quantization
    Quantizing from 32-bit to 6-bit causes a performance reduction of 2.7% in head classes and 6.6% in tail classes, further widening the performance gap between head and tail classes.
</div>



### 3.4 Can Quantization Processes Mitigate Real-World LTD Issues?
Considering the prevalence of LTD datasets across various real-world applications, and acknowledging that sampling and training techniques such as Decoupling Learning cannot entirely resolve performance disparities between head and tail classes, we explored whether the quantization process itself could further mitigate these disparities. Is there a simpler and more efficient way to reduce tail class performance degradation during quantization without extensive retraining?
Our experimental results offer a clear answer: Mixed Precision Quantization (MPQ) can indeed partially address tail class performance issues during quantization. MPQ evaluates bit-width sensitivity for each layer, allowing varied bit-width assignments based on sensitivity. By incorporating tail class performance metrics into sensitivity evaluations, we developed a data imbalance-aware MPQ. This approach significantly improved tail class performance without additional training overhead associated with post-training quantization (PTQ). Compared to uniform quantization, this imbalance-aware MPQ showed negligible overall accuracy degradation while effectively reducing the performance gap between head and tail classes during the quantization process.

<div class="row mt-3">
    {% include figure.html path="assets/img/2025-05-30-rethinking-quantization-for-the-real-world/table2.jpg" class="img-fluid" %}
</div>
<div class="caption">
    Table 2: MPQ Model Performance Based on Sensitivity Evaluation Methods
    The performance difference between MPQ evaluated with Normal accuracy and MPQ evaluated with Tail class accuracy is negligible compared to the decoupling learning model (baseline).
</div>

<div class="row mt-3">
    {% include figure.html path="assets/img/2025-05-30-rethinking-quantization-for-the-real-world/fig9.jpg" class="img-fluid" %}
</div>
<div class="caption">
    Figure 9: VGG16 Layer-wise Sensitivity Rank (Top 9) for MPQ Application
    Rank changes between sensitivity evaluations based on Normal Accuracy and Tail class Accuracy were observed, suggesting that MPQ can be effectively tailored to enhance tail class performance.
</div>

<div class="row mt-3">
    {% include figure.html path="assets/img/2025-05-30-rethinking-quantization-for-the-real-world/fig10.jpg" class="img-fluid" %}
</div>
<div class="caption">
    Figure 10: Class-wise Performance of MPQ Applied Model for Tail Class Improvement
    Performance reduction in head classes (Class 0~3) and performance improvement in tail classes (Class 4~9) confirm that MPQ effectively mitigates performance disparities caused by data imbalance.
</div>


## 4.Conclusion
### 4.1 What problem is this work trying to tackle?
This research seeks solutions from the perspective of Application-Driven Machine Learning to address Long Tail Distribution (LTD), a common real-world problem, specifically through the lens of model quantization. We experimentally explored the effects of data imbalance in LTD datasets on the conventional Decoupling Learning method and model quantization. Furthermore, we propose a method to resolve data imbalance using Mixed Precision Quantization (MPQ) at the quantization stage, rather than relying solely on data sampling or model training.

### 4.2 That contributions did this work make, and what impact should this work have?
By exploring and proposing methods to overcome LTD problems at the quantization stage rather than during data sampling or model training, this work is expected to inspire future research aimed at addressing data imbalance issues during the model compression phase. Using CIFAR10-LT datasets and the VGG16 model, we conducted experiments that are relatively straightforward to reproduce and provided several significant results:
•	We confirmed that traditional performance metrics, such as model accuracy, are insufficient for accurately reflecting performance and data imbalance in LTD datasets. It is essential to separately measure and evaluate accuracy for head and tail classes to assess genuine performance improvements.
•	Although significant performance improvements were achieved using the representative LTD dataset training method, Decoupling Learning, experiments demonstrated persistent data imbalance impacts when class imbalance exceeded 100 times.
•	The impact of LTD datasets is also evident in quantized models, with tail class performance degradation becoming significantly more pronounced relative to head classes as bit precision decreases.
•	Finally, to address LTD issues at the quantization stage, we propose an MPQ method. By incorporating tail class performance metrics into the MPQ sensitivity evaluation for layer-wise bit-width settings, we experimentally demonstrated the feasibility of an LTD-aware MPQ method.

### 4.3 How new is this effort?
While existing research primarily focuses on data sampling and training methods to solve LTD dataset problems, the approach proposed in this blog is novel in addressing data imbalance at the model compression stage through MPQ application. Additionally, by experimentally demonstrating the impact of data imbalance on quantized models, this research underscores the need for considering real-world problems like LTD at the quantization stage, suggesting new avenues for future research.

### 4.4 what are the limitations of this work? 
Despite showing meaningful experimental results, the study is currently limited to the CIFAR10-LT dataset and a single VGG16 model. To generalize these findings more broadly, it will be necessary to expand experiments to include additional datasets and models.


## Equations

This theme supports rendering beautiful math in inline and display modes using [MathJax 3](https://www.mathjax.org/) engine.
You just need to surround your math expression with `$$`, like `$$ E = mc^2 $$`.
If you leave it inside a paragraph, it will produce an inline expression, just like $$ E = mc^2 $$.

To use display mode, again surround your expression with `$$` and place it as a separate paragraph.
Here is an example:

$$
\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)
$$

Note that MathJax 3 is [a major re-write of MathJax](https://docs.mathjax.org/en/latest/upgrading/whats-new-3.0.html) 
that brought a significant improvement to the loading and rendering speed, which is now 
[on par with KaTeX](http://www.intmath.com/cg5/katex-mathjax-comparison.php).


## Images and Figures

Its generally a better idea to avoid linking to images hosted elsewhere - links can break and you
might face losing important information in your blog post.
To include images in your submission in this way, you must do something like the following:

```markdown
{% raw %}{% include figure.html path="assets/img/2025-04-28-distill-example/iclr.png" class="img-fluid" %}{% endraw %}
```

which results in the following image:

{% include figure.html path="assets/img/2025-04-28-distill-example/iclr.png" class="img-fluid" %}

To ensure that there are no namespace conflicts, you must save your asset to your unique directory
`/assets/img/2025-04-28-[SUBMISSION NAME]` within your submission.

Please avoid using the direct markdown method of embedding images; they may not be properly resized.
Some more complex ways to load images (note the different styles of the shapes/shadows):

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-distill-example/9.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-distill-example/7.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    A simple, elegant caption looks good between image rows, after each row, or doesn't have to be there at all.
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-distill-example/8.jpg" class="img-fluid z-depth-2" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-distill-example/10.jpg" class="img-fluid z-depth-2" %}
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-distill-example/11.jpg" class="img-fluid"  %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-distill-example/12.jpg" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-distill-example/7.jpg" class="img-fluid" %}
    </div>
</div>

### Interactive Figures

Here's how you could embed interactive figures that have been exported as HTML files.
Note that we will be using plotly for this demo, but anything built off of HTML should work
(**no extra javascript is allowed!**).
All that's required is for you to export your figure into HTML format, and make sure that the file
exists in the `assets/html/[SUBMISSION NAME]/` directory in this repository's root directory.
To embed it into any page, simply insert the following code anywhere into your page.

```markdown
{% raw %}{% include [FIGURE_NAME].html %}{% endraw %} 
```

For example, the following code can be used to generate the figure underneath it.

```python
import pandas as pd
import plotly.express as px

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/earthquakes-23k.csv')

fig = px.density_mapbox(
    df, lat='Latitude', lon='Longitude', z='Magnitude', radius=10,
    center=dict(lat=0, lon=180), zoom=0, mapbox_style="stamen-terrain")
fig.show()

fig.write_html('./assets/html/2025-04-28-distill-example/plotly_demo_1.html')
```

And then include it with the following:

```html
{% raw %}<div class="l-page">
  <iframe src="{{ 'assets/html/2025-04-28-distill-example/plotly_demo_1.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>{% endraw %}
```

Voila!

<div class="l-page">
  <iframe src="{{ 'assets/html/2025-04-28-distill-example/plotly_demo_1.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>

## Citations

Citations are then used in the article body with the `<d-cite>` tag.
The key attribute is a reference to the id provided in the bibliography.
The key attribute can take multiple ids, separated by commas.

The citation is presented inline like this: <d-cite key="gregor2015draw"></d-cite> (a number that displays more information on hover).
If you have an appendix, a bibliography is automatically created and populated in it.

Distill chose a numerical inline citation style to improve readability of citation dense articles and because many of the benefits of longer citations are obviated by displaying more information on hover.
However, we consider it good style to mention author last names if you discuss something at length and it fits into the flow well — the authors are human and it’s nice for them to have the community associate them with their work.

***

## Footnotes

Just wrap the text you would like to show up in a footnote in a `<d-footnote>` tag.
The number of the footnote will be automatically generated.<d-footnote>This will become a hoverable footnote.</d-footnote>

***

## Code Blocks

This theme implements a built-in Jekyll feature, the use of Rouge, for syntax highlighting.
It supports more than 100 languages.
This example is in C++.
All you have to do is wrap your code in a liquid tag:

{% raw  %}
{% highlight c++ linenos %}  <br/> code code code <br/> {% endhighlight %}
{% endraw %}

The keyword `linenos` triggers display of line numbers. You can try toggling it on or off yourself below:

{% highlight c++ %}

int main(int argc, char const \*argv[])
{
string myString;

    cout << "input a string: ";
    getline(cin, myString);
    int length = myString.length();

    char charArray = new char * [length];

    charArray = myString;
    for(int i = 0; i < length; ++i){
        cout << charArray[i] << " ";
    }

    return 0;
}

{% endhighlight %}

***

## Diagrams

This theme supports generating various diagrams from a text description using [jekyll-diagrams](https://github.com/zhustec/jekyll-diagrams){:target="\_blank"} plugin.
Below, we generate a few examples of such diagrams using languages such as [mermaid](https://mermaid-js.github.io/mermaid/){:target="\_blank"}, [plantuml](https://plantuml.com/){:target="\_blank"}, [vega-lite](https://vega.github.io/vega-lite/){:target="\_blank"}, etc.

**Note:** different diagram-generation packages require external dependencies to be installed on your machine.
Also, be mindful of that because of diagram generation the first time you build your Jekyll website after adding new diagrams will be SLOW.
For any other details, please refer to [jekyll-diagrams](https://github.com/zhustec/jekyll-diagrams){:target="\_blank"} README.

**Note:** This is not supported for local rendering! 

The diagram below was generated by the following code:

{% raw %}
```
{% mermaid %}
sequenceDiagram
    participant John
    participant Alice
    Alice->>John: Hello John, how are you?
    John-->>Alice: Great!
{% endmermaid %}
```
{% endraw %}

{% mermaid %}
sequenceDiagram
participant John
participant Alice
Alice->>John: Hello John, how are you?
John-->>Alice: Great!
{% endmermaid %}

***

## Tweets

An example of displaying a tweet:
{% twitter https://twitter.com/rubygems/status/518821243320287232 %}

An example of pulling from a timeline:
{% twitter https://twitter.com/jekyllrb maxwidth=500 limit=3 %}

For more details on using the plugin visit: [jekyll-twitter-plugin](https://github.com/rob-murray/jekyll-twitter-plugin)

***

## Blockquotes

<blockquote>
    We do not grow absolutely, chronologically. We grow sometimes in one dimension, and not in another, unevenly. We grow partially. We are relative. We are mature in one realm, childish in another.
    —Anais Nin
</blockquote>

***


## Layouts

The main text column is referred to as the body.
It is the assumed layout of any direct descendants of the `d-article` element.

<div class="fake-img l-body">
  <p>.l-body</p>
</div>

For images you want to display a little larger, try `.l-page`:

<div class="fake-img l-page">
  <p>.l-page</p>
</div>

All of these have an outset variant if you want to poke out from the body text a little bit.
For instance:

<div class="fake-img l-body-outset">
  <p>.l-body-outset</p>
</div>

<div class="fake-img l-page-outset">
  <p>.l-page-outset</p>
</div>

Occasionally you’ll want to use the full browser width.
For this, use `.l-screen`.
You can also inset the element a little from the edge of the browser by using the inset variant.

<div class="fake-img l-screen">
  <p>.l-screen</p>
</div>
<div class="fake-img l-screen-inset">
  <p>.l-screen-inset</p>
</div>

The final layout is for marginalia, asides, and footnotes.
It does not interrupt the normal flow of `.l-body`-sized text except on mobile screen sizes.

<div class="fake-img l-gutter">
  <p>.l-gutter</p>
</div>

***

## Other Typography?

Emphasis, aka italics, with *asterisks* (`*asterisks*`) or _underscores_ (`_underscores_`).

Strong emphasis, aka bold, with **asterisks** or __underscores__.

Combined emphasis with **asterisks and _underscores_**.

Strikethrough uses two tildes. ~~Scratch this.~~

1. First ordered list item
2. Another item
  * Unordered sub-list.
1. Actual numbers don't matter, just that it's a number
   1. Ordered sub-list
4. And another item.

   You can have properly indented paragraphs within list items. Notice the blank line above, and the leading spaces (at least one, but we'll use three here to also align the raw Markdown).

   To have a line break without a paragraph, you will need to use two trailing spaces.
   Note that this line is separate, but within the same paragraph.
   (This is contrary to the typical GFM line break behavior, where trailing spaces are not required.)

* Unordered lists can use asterisks
- Or minuses
+ Or pluses

[I'm an inline-style link](https://www.google.com)

[I'm an inline-style link with title](https://www.google.com "Google's Homepage")

[I'm a reference-style link][Arbitrary case-insensitive reference text]

[I'm a relative reference to a repository file](../blob/master/LICENSE)

[You can use numbers for reference-style link definitions][1]

Or leave it empty and use the [link text itself].

URLs and URLs in angle brackets will automatically get turned into links. 
http://www.example.com or <http://www.example.com> and sometimes 
example.com (but not on Github, for example).

Some text to show that the reference links can follow later.

[arbitrary case-insensitive reference text]: https://www.mozilla.org
[1]: http://slashdot.org
[link text itself]: http://www.reddit.com

Here's our logo (hover to see the title text):

Inline-style: 
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")

Reference-style: 
![alt text][logo]

[logo]: https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 2"

Inline `code` has `back-ticks around` it.

```javascript
var s = "JavaScript syntax highlighting";
alert(s);
```
 
```python
s = "Python syntax highlighting"
print(s)
```
 
```
No language indicated, so no syntax highlighting. 
But let's throw in a <b>tag</b>.
```

Colons can be used to align columns.

| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |

There must be at least 3 dashes separating each header cell.
The outer pipes (|) are optional, and you don't need to make the 
raw Markdown line up prettily. You can also use inline Markdown.

Markdown | Less | Pretty
--- | --- | ---
*Still* | `renders` | **nicely**
1 | 2 | 3

> Blockquotes are very handy in email to emulate reply text.
> This line is part of the same quote.

Quote break.

> This is a very long line that will still be quoted properly when it wraps. Oh boy let's keep writing to make sure this is long enough to actually wrap for everyone. Oh, you can *put* **Markdown** into a blockquote. 


Here's a line for us to start with.

This line is separated from the one above by two newlines, so it will be a *separate paragraph*.

This line is also a separate paragraph, but...
This line is only separated by a single newline, so it's a separate line in the *same paragraph*.
