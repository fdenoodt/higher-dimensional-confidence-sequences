---
layout: default
title: Project page
---

# Higher-Dimensional Confidence Sequences

On the Tightness and Computational Tractability of Higher-Dimensional Confidence Sequences

Fabian Denoodt, Sibylle Hess, Joaquin Vanschoren, Christian A. Naesseth

[Read the paper on arXiv (PDF)](https://arxiv.org/pdf/2610.03727)

## Tutorials

{% for post in site.posts %}
## [{{ post.title }}]({{ post.url | relative_url }})

{{ post.date | date: "%B %-d, %Y" }}

{{ post.excerpt | strip_html | truncatewords: 45 }}

{% endfor %}
