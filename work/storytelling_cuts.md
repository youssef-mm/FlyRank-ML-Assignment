# Week 8: Tell the Story (ML-12)

This document contains the presentation materials, showcase demo outline, and shareable cuts for the **FlyRank Content Refresh Opportunity Scoring** research project.

---

## 1. Five-Minute Demo Outline (Week-8 Showcase)

**Title:** Beyond Fixed Staleness Rules: Prioritizing High-Exposure Content Refresh with Machine Learning  
**Presenter:** Youssef Mohamed  
**Track:** Applied Search Intelligence & Machine Learning  

### Minute-by-Minute Presentation Plan:

- **Minute 1: The Operational Bottleneck (Problem & Decision)**
  - *Context:* Enterprise digital publishers manage tens of thousands of articles (30,000 pages across 32 clients).
  - *Bottleneck:* Editorial teams can only realistically review 20–50 articles per week.
  - *Core Question:* Which specific underperforming pages should content teams refresh first to arrest traffic decay and reclaim lost rankings, versus pages to leave alone?

- **Minute 2: The Data Contract & Leakage Discipline (Methodology)**
  - *Data:* 30,000 pseudonymized content items across 32 clients. Telemetry encompasses 90-day search impressions, clicks, avg position, sessions, update staleness, and word count.
  - *Leakage Control:* Strict audit barred `trend_pct` and `trend_direction` from input features, avoiding the circular leakage trap that artificially inflates scores.
  - *Validation:* Grouped Client-Holdout split — models are trained on one set of clients and validated on entirely unseen clients to ensure true cross-domain generalization.

- **Minute 3: The Counter-Intuitive Finding (The Zombie Page Trap)**
  - *Visual:* Action Recommendation Mix (`work/figures/action_mix.png`) & Priority by Position Tier (`work/figures/priority_by_position.png`).
  - *Finding:* A naive heuristic rule (*"refresh content older than 180 days"*) completely breaks down in practice:
    - Pages unrefreshed for 91–180 days peak at **61.1% decline rate**.
    - But pages unrefreshed for 181+ days drop back to **47.1% decline rate**.
    - *Why?* Extreme stale pages are dormant "zombie" articles (median 15.5 impressions) that have already flatlined. An unconditioned fixed rule wastes editorial budget rewriting dead pages.

- **Minute 4: Honest Empirical Results vs Baseline (Proof)**
  - *Metric:* Precision@50 (directly mapping to editorial capacity of 50 weekly candidate reviews).
  - *Baseline Rule:* Precision@50 = **0.240** (only 12 of 50 recommended pages truly in decline).
  - *Random Forest Model:* Precision@50 = **0.740** (37 of 50 correctly flagged) with **ROC-AUC = 0.750**.
  - *Impact:* Triples the operational yield of the editorial team, preventing wasted rewrites on stable content.

- **Minute 5: The Content Action Playbook (Operational Impact)**
  - *Playbook Queues:* The 30,000 pages are triaged into actionable queues with human-interpretable reason codes:
    - **`refresh` (8,178 items):** High-volume declining assets on striking distance.
    - **`refresh_and_review_ctr` (6,657 items):** Page-1 rankings with critical CTR deficits (title/snippet decay).
    - **`refresh_and_review_engagement` (1,990 items):** Traffic with high bounce and low scroll rates.
    - **`monitor` (13,093 items):** Healthy, stable, or growing evergreen assets.

---

## 2. Two Shareable Cuts

### Cut 1: Short Social Post (LinkedIn / X / Technical Blog)

> **Most SEO teams rely on a simple heuristic:** *"If an article hasn't been updated in 6 months, schedule it for a refresh."*
>
> Testing this rule against 30,000 real content pages across 32 enterprise clients revealed why that rule quietly fails:
>
> 1️⃣ Content unrefreshed for 3–6 months experiences peak search traffic decline (**61.1%**).  
> 2️⃣ But content unrefreshed for over 6 months drops to **47.1%** decline. Why? They're dormant "zombie" pages (median 15.5 impressions) that have already bottomed out. Fixed rules waste editorial budget on zero-traffic content.  
>
> By framing Content Refresh as a machine learning ranking problem with strict client-holdout validation (training and testing on distinct clients), we built an opportunity-scoring model that lifts **Precision@50 from 24% (heuristic rule) to 74% (37 of 50 true declining pages)** with an **ROC-AUC of 0.750**.
>
> **The key takeaway:** Never prioritize content refresh by age alone. Jointly model freshness with search volume floors and ranking position tiers to protect high-exposure Page 1 assets.
>
> 📄 Read the research paper: https://youssef-mm.github.io/FlyRank-ML-Assignment/  
> 💻 Explore the code & notebooks: https://github.com/youssef-mm/FlyRank-ML-Assignment  
>
> `#MachineLearning` `#SEO` `#DataScience` `#InformationRetrieval` `#MLOps`

---

### Cut 2: Three-Sentence Employer-Facing Summary

> **"What I built, on what data, what it showed:"**
>
> 1. **What I built:** An end-to-end Content Refresh Opportunity Ranking pipeline that triages large-scale digital publishing portfolios into prioritized editorial action queues with transparent reason codes and confidence tiers.
> 2. **On what data:** Evaluated on 90-day multi-channel search and engagement telemetry (30,000 pages across 32 enterprise clients) with strict leakage auditing and client-holdout validation.
> 3. **What it showed:** Machine learning successfully captures non-linear decay dynamics where fixed age rules fail, lifting top-50 review precision from 24.0% to 74.0% (a 3.1x operational yield multiplier) and ensuring editorial teams triage high-exposure Page 1 decays first.
