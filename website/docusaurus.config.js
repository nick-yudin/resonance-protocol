// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'RESONANCE',
  tagline: 'The clock stops. The resonance begins.',
  favicon: 'img/logo.svg',

  url: 'https://resonanceprotocol.org',
  baseUrl: '/',

  organizationName: 'rAI-Research-Collective', 
  projectName: 'resonance-protocol', 

  onBrokenLinks: 'warn', 
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  // --- ENABLE DIAGRAMS ---
  markdown: {
    mermaid: true,
  },
  themes: ['@docusaurus/theme-mermaid'],
  // -----------------------

  // --- ADD KATEX STYLESHEET ---
  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
      type: 'text/css',
      integrity: 'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
      crossorigin: 'anonymous',
    },
  ],
  // ---------------------------

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          path: '../docs', 
          sidebarPath: './sidebars.js',
          routeBasePath: 'docs',
          // --- ADD MATH PLUGINS ---
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
          // ------------------------
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      colorMode: {
        defaultMode: 'dark',
        disableSwitch: false,
        respectPrefersColorScheme: false,
      },
      navbar: {
        title: 'RESONANCE',
        logo: {
          alt: 'Resonance Protocol',
          src: 'img/logo.svg',
          srcDark: 'img/logo.svg', 
        },
        items: [
          {
            to: '/docs/manifesto',
            position: 'left',
            label: 'Manifesto',
          },
          {
            to: '/docs/specs/v1.0_current/spec-v1-final', 
            position: 'left',
            label: 'Specification',
          },
          {
            href: 'https://github.com/nick-yudin/resonance-protocol',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Protocol',
            items: [
              { label: 'Manifesto', to: '/docs/manifesto' },
              { label: 'Specification', to: '/docs/specs/v1.0_current/spec-v1-final' }, 
            ],
          },
          {
            title: 'Community',
            items: [
              { label: 'Twitter', href: 'https://twitter.com/rAI_stack' },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} rAI Research Collective.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
};

export default config;