# AGENTS.md

- Downstream applications subclass `LibrarianAgent` and `SynthesisAgent`. Constructor
  arguments (`llm_client`, `tracer`, `literature_source`, `librarian`) and the underscore
  step methods (`_screen_query`, `_route`, `_retrieve`, `_output_formatting`,
  `_build_prompt`, `_summarize`) are an extension API: do not rename them or change
  their signatures without calling it out in the PR.
- Keep the package generic. Deployment-specific behaviour (providers, caches, chat
  formatting, request screening) belongs in the subclass; add an injectable dependency
  or a no-op hook here instead.
- Every push to `main` opens a pin bump downstream (`.github/workflows/bump-bio-agents.yml`).
- The skill reads `librarian/prompts/summarizer.md` via `skills/librarian/scripts/search.py`:
  check it still works when you change that prompt.
- PR descriptions: Why / What / commits in order / Testing.
