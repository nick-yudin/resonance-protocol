/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  docsSidebar: [
    {
      type: 'category',
      label: 'Level 0: Philosophy',
      collapsed: false,
      items: [
        'intro/manifesto',
      ],
    },
    {
      type: 'category',
      label: 'Level 1: Specifications',
      collapsed: false,
      items: [
        'specs/v1.0_current/spec-v1-final',
      ],
    },
    {
      type: 'category',
      label: 'Research: Experimental Validation',
      collapsed: false,
      items: [
        'research/research-overview',
        {
          type: 'category',
          label: 'M2.5 Series: Data Efficiency',
          collapsed: true,
          items: [
            'research/m2-5-data-curation',
            'research/m2-5-curriculum',
          ],
        },
        'research/m2-6-compositional',
        'research/m3-series',
      ],
    },
  ],
};

module.exports = sidebars;
