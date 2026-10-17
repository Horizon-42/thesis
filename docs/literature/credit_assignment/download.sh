#!/usr/bin/env bash
# Re-fetch every PDF indexed in README.md from its public source (URLs of 2026-10-17).
# arXiv versions are pinned to the version that was read. Ng et al. 1999 has no arXiv
# copy; it is fetched from Stuart Russell's page at Berkeley.
set -u
cd "$(dirname "$0")"
mkdir -p papers
UA="Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
get() { # get <destination> <url>
  if [ -s "$1" ]; then echo "have $1"; return; fi
  curl -sL -f --retry 2 -A "$UA" -o "$1" "$2" && echo "OK   $1" || { rm -f "$1"; echo "FAIL $1  $2"; }
}
get papers/Kazemnejad_2025_VinePPO_arXiv2410.01679v2.pdf "https://arxiv.org/pdf/2410.01679v2"
get papers/Ng_1999_Policy_Invariance_Reward_Shaping_ICML.pdf "https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf"
get papers/Ciosek_2018_Expected_Policy_Gradients_arXiv1706.05374v6.pdf "https://arxiv.org/pdf/1706.05374v6"
get papers/Ciosek_2018_Expected_Policy_Gradients_AAAI32.pdf "https://ojs.aaai.org/index.php/AAAI/article/download/11607/11466"
get papers/Ciosek_2020_Expected_Policy_Gradients_for_RL_JMLR_arXiv1801.03326v2.pdf "https://arxiv.org/pdf/1801.03326v2"
get papers/Foerster_2018_COMA_arXiv1705.08926v2.pdf "https://arxiv.org/pdf/1705.08926v2"
get papers/Foerster_2018_COMA_AAAI32.pdf "https://ojs.aaai.org/index.php/AAAI/article/download/11794/11653"
get papers/Guo_2025_Segment_Policy_Optimization_arXiv2505.23564v2.pdf "https://arxiv.org/pdf/2505.23564v2"
get papers/Hou_2025_TreeRL_arXiv2506.11902v1.pdf "https://arxiv.org/pdf/2506.11902v1"
