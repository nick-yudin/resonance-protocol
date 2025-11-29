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
            // FIX: Corrected document ID to include hyphens, matching Docusaurus's internal metadata.
            'specs/v1.0_current/spec-v1-final', 
          ],
        },
      ],
    };

    module.exports = sidebars;