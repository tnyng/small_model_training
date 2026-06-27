# Round 4 Tricky-Score Audit Pack

- 42 new unique rows
- exact scores only: 21 rows with score 3, 21 rows with score 7
- 3 annotators per row
- 6 annotators, 21 rows each
- source data uses old step-3 branch outputs only
- excluded datasets: article_bias, bureaucracies, tweets_news, misinfo_cancer

## Per-combo counts

- `heuristic_ann_gpt41mini`: 14
- `llm_gpt5mini_ann_gpt41mini`: 14
- `llm_gptoss_ann_gpt41mini`: 14

## Per-dataset counts

- `misinfo`: 8
- `tweets`: 7
- `implicit_hate`: 6
- `gwsd`: 4
- `polite_stack`: 4
- `tweetscongress`: 4
- `yelp`: 4
- `rumoureval2019`: 2
- `tweets23`: 2
- `hatecot`: 1

## Score counts

- score `3`: 21
- score `7`: 21
