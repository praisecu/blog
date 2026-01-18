# RF-Style Technical Blog (Astro + MDX + KaTeX + Svelte)

This repository is a **static** technical blog designed for:
- **Math-heavy writing** (LaTeX via KaTeX)
- **Interactive figures** embedded inside posts (Svelte components)
- **GitHub Pages** deployment (no server)

Everything builds to static HTML/CSS/JS in `dist/` and is deployed via GitHub Actions.

---

## Local development

### Prerequisites
- Node.js **20+**
- npm

### Install & run
```bash
npm install
npm run dev
```

Astro will print a local URL (typically `http://localhost:4321`).

### Production build preview
```bash
npm run build
npm run preview
```

---

## Writing blog posts

Blog posts live in:

```
src/pages/blog/
```

Each post is an **MDX** file (`.mdx`), which allows:
- normal Markdown
- KaTeX math (`$...$`, `$$...$$`)
- importing and embedding interactive components

### Create a new post

1) Create a new file, e.g.
```
src/pages/blog/optical-flow-motion-flow.mdx
```

2) Add frontmatter at the top:

```mdx
---
layout: ../../layouts/PostLayout.astro
title: "Your Post Title"
author: "Your Name"
date: "January 18, 2026"
---
```

3) Write content using Markdown headings (`##`, `###`) to populate the left Table of Contents.

### Example skeleton

```mdx
---
layout: ../../layouts/PostLayout.astro
title: "Example Post"
author: "Your Name"
date: "January 18, 2026"
---

## Introduction
Write text here.

## Math
Inline math: $E = mc^2$

Block math:
$$
\dot{x} = f(x, u, t)
$$

## Conclusion
Wrap up.
```

---

## Editing the homepage list of posts

The homepage is currently a simple manual list:

```
src/pages/index.astro
```

Add your post to the `posts` array:

```js
const posts = [
  { title: "A Visual Introduction to Rectified Flows", href: "/blog/rectified-flow/" },
  { title: "Optical Flow vs Motion Flow", href: "/blog/optical-flow-motion-flow/" }
];
```

**Important:** In Astro, a page at
`src/pages/blog/my-post.mdx`
is served at:
`/blog/my-post/`

---

## Adding interactive figures inside a post

Interactive figures are implemented as components under:

```
src/components/
```

This blog uses **Svelte** components for interactive demos (Canvas/WebGL/D3/etc.).

### 1) Create a component

Example:
```
src/components/MyDemo.svelte
```

### 2) Import and embed it in an MDX post

```mdx
import MyDemo from "../../components/MyDemo.svelte";

<div class="figure-card">
  <MyDemo client:load />
  <div class="caption">
    <b>Figure 1:</b> Description of the interactive figure.
  </div>
</div>
```

Use `client:load` so the component runs in the browser (GitHub Pages is static, so all interactivity must be client-side).

### Styling for figures
Use the existing helper classes from `src/styles/global.css`:
- `.figure-card` for a bordered panel
- `.caption` for figure text

---

## Math / LaTeX support

KaTeX is enabled for all MDX posts.

- Inline math: `$ ... $`
- Display math: `$$ ... $$`

KaTeX CSS is imported globally in:
```
src/styles/global.css
```

---

## Layout and Table of Contents

The “paper-like” layout is defined in:
```
src/layouts/PostLayout.astro
```

The left Table of Contents is generated from headings via:
```
src/components/Toc.astro
```

To ensure headings appear in the TOC:
- use `##` and `###` headings
- avoid skipping heading levels

---

## Deploying to GitHub Pages

Deployment is done by GitHub Actions:
```
.github/workflows/deploy.yml
```

### One-time GitHub setup
In your repository:
- **Settings → Pages → Build and deployment**
- Set **Source** to **GitHub Actions**

### URL
Your site will be available at:
- `https://<USER>.github.io/<REPO>/` (project pages)

This repo is configured to handle the GitHub Pages base path automatically.

---

## Common issues

### “Site loads but CSS is missing”
This usually means the site base path is wrong. This repo sets `BASE_PATH=/<repo>` during the GitHub Actions build, so if you changed deployment style (custom domain / user pages), adjust `BASE_PATH` in the workflow and/or `astro.config.mjs`.

### “Action fails at npm ci”
`npm ci` requires `package-lock.json`. If you want deterministic CI:
- run `npm install` locally once
- commit `package-lock.json`
- then you can switch workflow install step to `npm ci`

---

## Recommended workflow for writing

1) Create/edit a post in `src/pages/blog/*.mdx`
2) Run `npm run dev`
3) Iterate until it looks right
4) Add the post link to `src/pages/index.astro`
5) Push to `main` to deploy

---

## License
Add your preferred license file if you plan to publish this publicly.
