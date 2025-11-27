// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'RESONANCE',
  tagline: 'The clock stops. The resonance begins.',
  favicon: 'img/logo.svg',

  url: 'https://resonanceprotocol.org',
  baseUrl: '/',

  organizationName: 'rAI-Research-Collective', 
  projectName: 'resonance-protocol', 

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          path: '../docs', 
          sidebarPath: './sidebars.js',
          routeBasePath: 'docs',
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
          // FIX: Direct links to documents instead of generic sidebar
          {
            to: '/docs/manifesto',
            position: 'left',
            label: 'Manifesto',
          },
          {
            to: '/docs/unified-spec', 
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
              { label: 'Specification', to: '/docs/unified-spec' },
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