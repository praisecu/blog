import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import svelte from "@astrojs/svelte";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

// GitHub Pages base-path handling:
// - For https://USER.github.io/REPO/ -> base should be "/REPO"
// - For custom domains or user pages root -> base should be "/"
const base = process.env.BASE_PATH ?? "/";

export default defineConfig({
  site: process.env.SITE_URL ?? "https://example.github.io",
  base,
  integrations: [
    svelte(),
    mdx({
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeKatex]
    })
  ],
  markdown: {
    syntaxHighlight: "shiki"
  }
});
