# Resolved references

DBLP-checked reference lines for both submissions. Numbering and order match each report's
existing "## References" section, so each line is a drop-in replacement for the line with the
same number. Generated from `references_dblp.bib`, which `dblp_bibtex.py` produced from
`references.bib`. Every resolved entry was compared by hand against our own title before being
accepted. Entries DBLP could not match are left exactly as the reports already have them.

### report_4page.md

1. Mazeika, M., Yin, X., Tamirisa, R., Lim, J., Lee, B. W., Ren, R., Phan, L., Mu, N., Khoja, A., Zhang, O., Hendrycks, D. (2025). *Utility Engineering. Analyzing and Controlling Emergent Value Systems in AIs.* Advances in Neural Information Processing Systems 38 (NeurIPS 2025).
2. nostalgebraist (2025). *the void.* LessWrong.
3. Long, R., Sebo, J., Butlin, P., Finlinson, K., Fish, K., Harding, J., Pfau, J., Sims, T., Birch, J., Chalmers, D. (2024). *Taking AI Welfare Seriously.* arXiv:2411.00986
4. Anthropic (2025). *Exploring Model Welfare.*
5. Lindsey, J. (2025). *Emergent Introspective Awareness in Large Language Models.* Transformer Circuits. arXiv:2601.01828

### report2.md

1. Naphade, A., Bhargav, S., Lim, S., Shah, M. (2026). *Me, Myself, and Pi. Evaluating and Explaining LLM Introspection.* arXiv:2603.20276
2. Lindsey, J. (2025). *Emergent Introspective Awareness in Large Language Models.* Transformer Circuits. arXiv:2601.01828
3. Ren, R. et al. (2025). *The MASK Benchmark. Disentangling Honesty From Accuracy in AI Systems.* arXiv:2503.03750
4. Mazeika, M. et al. (2025). *Utility Engineering. Analyzing and Controlling Emergent Value Systems in AIs.* Advances in Neural Information Processing Systems 38 (NeurIPS 2025).
5. Long, R., Sebo, J., Butlin, P., et al. (2024). *Taking AI Welfare Seriously.* arXiv:2411.00986
6. Anthropic (2025). *Exploring Model Welfare.*

### Resolution notes

These are working notes, not part of either reference list. Do not paste them into a report.

- **Mazeika et al.** is the only entry DBLP places at a real published venue, NeurIPS 2025
  (`dblp.org/rec/conf/nips/MazeikaYTLLRPMZ25`). The DBLP title is character-identical to ours, so
  the match is certain. One discrepancy worth a decision. The DBLP record for the NeurIPS version
  lists ten authors and does not include Adam Khoja, whom our reports list eleventh of eleven. We
  have left the author list as the reports already print it rather than dropping a name on DBLP's
  say-so. If the NeurIPS proceedings version really carries ten authors, drop Khoja, A. from both
  lines.
- **Long et al.** and **Ren et al.** resolved to CoRR only. Their DBLP arXiv identifiers,
  2411.00986 and 2503.03750, are identical to the ones the reports already cite, which confirms the
  match and confirms that no published venue is indexed. Their lines are unchanged. DBLP does supply
  the full sixteen-author list for the MASK benchmark if the "et al." is ever expanded.
- **Lindsey** needs a judgement call. DBLP has no Transformer Circuits record. It has a CoRR record,
  arXiv:2601.01828, with the identical title and the same single author, Jack Lindsey. That is the
  same work, but DBLP dates the arXiv posting to 2026 while our reports cite the 2025 Transformer
  Circuits publication. We kept the 2025 year and the Transformer Circuits venue as cited, and
  appended the arXiv identifier as an extra locator. If a single consistent year is preferred,
  choose one deliberately rather than letting the two forms drift between the two papers.
- **nostalgebraist**, **Anthropic** and **Naphade et al.** returned no confident DBLP match and are
  reproduced verbatim from the reports. The first two are not the kind of venue DBLP indexes. The
  third is a 2026 preprint, too recent to be indexed. No substitute was accepted for any of them.
