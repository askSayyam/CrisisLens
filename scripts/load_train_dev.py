import ast
import os
import json
import pandas as pd


dataset_path = 'train_dev_sets'

posts_path = os.path.join(dataset_path, 'posts.csv')
fact_checks_path = os.path.join(dataset_path, 'fact_checks.csv')
pairs_path = os.path.join(dataset_path, 'pairs.csv')
tasks_path = os.path.join(dataset_path, 'tasks.json')

for path in [posts_path, fact_checks_path, pairs_path, tasks_path]:
    assert os.path.isfile(path)

parse_col = lambda s: ast.literal_eval(s) if s else s

print("Loading fact-checks")
df_fact_checks = pd.read_csv(fact_checks_path).fillna('').set_index('fact_check_id')
for col in ['claim', 'instances', 'title']:
    df_fact_checks[col] = df_fact_checks[col].apply(parse_col)
print("Fact-checks loaded!")
print(df_fact_checks.info())
print()

print("Loading posts")
df_posts = pd.read_csv(posts_path).fillna('').set_index('post_id')
for col in ['instances', 'ocr', 'verdicts', 'text']:
    df_posts[col] = df_posts[col].apply(parse_col)
print("Posts loaded!")
print(df_posts.info())
print()


print("Loading pairs")
df_pairs = pd.read_csv(pairs_path)
print("Pairs loaded!")
print(df_pairs.info())
print()

with open(os.path.join(tasks_path), 'r') as f:
    tasks = json.load(f)

print("Example: Filtering train for 'eng' language.")
eng_post_ids = tasks["monolingual"]["eng"]["posts_train"]
eng_fact_check_ids = tasks["monolingual"]["eng"]["fact_checks"]

df_eng_posts = df_posts[df_posts.index.isin(eng_post_ids)]
df_eng_fact_checks = df_fact_checks[df_fact_checks.index.isin(eng_fact_check_ids)]
df_eng_pairs = df_pairs[df_pairs['post_id'].isin(eng_post_ids) & df_pairs['fact_check_id'].isin(eng_fact_check_ids)]

print(f"Training posts for 'eng': {df_eng_posts.shape[0]}")
print(f"Fact-checks for 'eng': {df_eng_fact_checks.shape[0]}")
print(f"Training pars for 'eng': {df_eng_pairs.shape[0]}")