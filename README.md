# Website

This repository does not only contain the code for our website: [https://itsadeepizza.github.io/](https://itsadeepizza.github.io/), but it represents also the way it is deployed using github pages.


## Run locally

Install ruby and jekyll (you could refer to this [guide](https://docs.github.com/fr/pages/setting-up-a-github-pages-site-with-jekyll/testing-your-github-pages-site-locally-with-jekyll)) and from this directory run:

```bundle exec jekyll serve```


## Multilanguage
I make some modification to original theme for supporting some multilanguage features.
At the beginning of each post, list available languages and current languages, as follows:

```
lang: ita
languages: [ita, eng]
```

All language version need to share the same file name, but put in a folder with the language name (`_posts` 
for english, `ita_posts` for italian, etc.)

## Credits
![](https://i.imgur.com/zNBkzj1.png)
Do you like this website? It is created with [beautifuljekyll](https://beautifuljekyll.com), a very easy template to use and deploy on github pages.
