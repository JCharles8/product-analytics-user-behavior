# Product Analytics: Unlocking User Behavior in a Productivity Tool

## Objective
To simulate and analyze user behavior in a SaaS productivity platform. The goal is to uncover usage patterns and identify product opportunities that improve retention and engagement.

## Dataset
- 500 simulated users
- Actions: login, create_page, invite_collab, add_tag, use_template, export, delete_page
- Tracked over 60 days from signup

## Key Analyses
- Funnel analysis: From login to key feature engagement
- Feature adoption and popularity
- Retention curves (7, 14, 30, 60 days)
- Power user behavior vs. casual user patterns

## Tools
- Python (Pandas, Matplotlib)
- SQL (optional for segment queries)
- Tableau (for visualization)
- Jupyter Notebook

## Results (Teaser)
- Users who create a page within 3 days of signup have 2x retention at 30 days
- Templates and tagging usage strongly correlate with long-term engagement

## Business Insights
- Onboarding should emphasize `create_page` and `use_template`
- Consider nudging tagging behavior to reduce churn

## View the Analysis
- 👉 [Jupyter Notebook](./notebooks/user_behavior_analysis.ipynb)
- 👉 [Slides Summary](./slides/product_insights_summary.pdf)

## Author
Jeff Charles — Analyst with a focus on behavioral insights, clarity-driven design, and user-first strategy.
