# RESONANCE Project Structure & Governance

**Documentation and Code Management Strategy**

This document outlines the repository organization for the RESONANCE project, along with formatting standards and language policies.

## 1. File Tree Structure

We utilize a "Monorepo" approach (everything in one place) for the initial phase.

```
/resonance-project
│
├── /docs                    <-- SINGLE SOURCE OF TRUTH (STRICTLY ENGLISH)
│   ├── /00_intro            <-- For newcomers
│   │   └── manifesto.md     <-- "Level 0" (Philosophy)
│   │   └── quick-start.md   <-- "How to run"
│   │
│   ├── /01_specs            <-- Technical Standards (Hardcore)
│   │   ├── v1.0_current     <-- Current Stable Version
│   │   │   └── resonance_unified_spec.md
│   │   └── v2.0_draft       <-- Future Drafts
│   │
│   └── /02_rfc              <-- "Request for Comments"
│       └── rfc_001_neuromorphic_chips.md
│
├── /website                 <-- SITE CODE (Frontend)
│   │   (Docusaurus/VitePress project)
│   └── /src/pages/index.js  <-- Landing Page
│
├── /reference_impl          <-- REFERENCE CODE
│   ├── /python_sdk          <-- SDK
│   └── /examples            <-- Usage Examples
│
└── README.md                <-- Repository Entry Point
```

## 2. Document Versioning

Documentation follows the same rules as software (Semantic Versioning).

* **Stable (v1.0):** Content located in the `specifications/v1.0` folder. This is an immutable contract.
* **Draft (Main Branch):** Ongoing work in the main branch.
* **Archived:** Old versions are moved to an archive.

**Rule:** The website always displays the **Stable** version by default.

## 3. Workflow Lifecycle

1. **Modification:** Edits are made to `.md` files in the `/docs` folder.
2. **Review:** Changes are reviewed for technical accuracy.
3. **Release:** CI/CD automatically updates https://resonanceprotocol.org.

## 4. Audiences and Styles

| Document Type | Location | Target Audience | Style |
| :--- | :--- | :--- | :--- |
| **Landing Page** | Website (Home) | Investors, Press | "Visionary", "Future", "Efficiency" |
| **Manifesto** | `/docs/00_intro` | Visionaries | Emotional, Philosophical |
| **Specification** | `/docs/01_specs` | Engineers | Dry, Precise Technical English |
| **Tutorial** | `/docs/tutorials` | Makers | Step-by-step |

## 5. Tooling

1. **Storage:** GitHub.
2. **Engine:** Docusaurus / VitePress.
3. **Hosting:** Vercel / GitHub Pages.

## 6. Identity Standards

The following details must be used in all public documents (Headers/Footers):

* **Author / Organization:** `rAI Research Collective`
* **Contact Email:** `1@resonanceprotocol.org`
* **Official Website:** `https://resonanceprotocol.org`
* **Twitter (X):** [`@rAI_stack`](https://twitter.com/rAI_stack)

**Document Header Example:**
```markdown
# Document Title
**Author:** rAI Research Collective
**Contact:** 1@resonanceprotocol.org
**Web:** [https://resonanceprotocol.org](https://resonanceprotocol.org)
```

## 7. Language Policy

We adhere to a **Strictly English** policy for all repository contents:

1. **Documentation & Code:**
   * All files in the `/docs` folder must be in **English**.
   * All website content must be in **English**.
   * Code comments and commit messages must be in **English**.
   * Repository `README.md` and `STRUCTURE.md` must be in **English**.
2. **Exceptions:** None.
