#!/usr/bin/env bash
# Re-fetch every PDF indexed in README.md from its public source (URLs of 2026-10-17).
# arXiv versions are pinned to the version that was read. Web pages and source files
# (Huang et al. blog, verl, TRL, transformers) are not fetched: README.md gives their URL,
# tag and the quoted text.
set -u
cd "$(dirname "$0")"
mkdir -p papers
UA="Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
get() { # get <destination> <url>
  if [ -s "$1" ]; then echo "have $1"; return; fi
  curl -sL -f --retry 2 -A "$UA" -o "$1" "$2" && echo "OK   $1" || { rm -f "$1"; echo "FAIL $1  $2"; }
}
get papers/Schulman_2017_Proximal_Policy_Optimization_arXiv1707.06347v2.pdf "https://arxiv.org/pdf/1707.06347v2"
get papers/Ye_2020_Mastering_Complex_Control_in_MOBA_Games_arXiv1912.09729v3.pdf "https://arxiv.org/pdf/1912.09729v3"
get papers/Ye_2020_Mastering_Complex_Control_in_MOBA_Games_AAAI34.pdf "https://ojs.aaai.org/index.php/AAAI/article/download/6144/6000"
get papers/Pascanu_2013_Difficulty_of_Training_RNNs_arXiv1211.5063v2.pdf "https://arxiv.org/pdf/1211.5063v2"
get papers/Engstrom_2020_Implementation_Matters_PPO_TRPO_arXiv2005.12729v1.pdf "https://arxiv.org/pdf/2005.12729v1"
get papers/Andrychowicz_2021_What_Matters_On-Policy_RL_arXiv2006.05990v1.pdf "https://arxiv.org/pdf/2006.05990v1"
get papers/Shao_2024_DeepSeekMath_GRPO_arXiv2402.03300v3.pdf "https://arxiv.org/pdf/2402.03300v3"
