---
layout: default
title: Blog
---

# Blog

{% for post in site.posts %}
## [{{ post.title }}]({{ post.url | relative_url }})

{{ post.date | date: "%B %-d, %Y" }}

{{ post.excerpt | strip_html | truncatewords: 45 }}

{% endfor %}
