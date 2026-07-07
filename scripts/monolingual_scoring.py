import json
import os
import numpy as np


def read_from_file(path):
    with open(path, "r") as f:
        data = json.load(f)   
    return data 

def get_post_id_to_lang_dict(tasks, split):
    monolingual = tasks["monolingual"]  
    post_id_to_lang_dict = {}
    for k, v in monolingual.items():
        if split == "train_dev":
            posts_train = v["posts_train"]
            posts_dev = v["posts_dev"]
            for pt in posts_train:
                if pt not in post_id_to_lang_dict:
                    post_id_to_lang_dict[str(pt)] = k
                else:
                    raise Exception("you have duplicated post ids in train")    
            for pd in posts_dev:
                if pd not in post_id_to_lang_dict:
                    post_id_to_lang_dict[str(pd)] = k 
                else:
                    raise Exception("you have duplicated post ids_in dev and train") 
        elif split == "test":
            posts_test = v["posts_test"]
            for pt in posts_test:
                if pt not in post_id_to_lang_dict:
                    post_id_to_lang_dict[str(pt)] = k
                else:
                    raise Exception("you have duplicated post ids in train") 
        else:
            raise Exception("invalid split")

    return post_id_to_lang_dict    


def calculate_success_at_k(truth, prediction, k, post_id_to_lang): 
    assert len(truth)==len(prediction), "The number of prediction is not correct"    
    # success_at_k = 0  
    success_by_lang = {} 

    for key, t in truth.items():
        if key not in prediction:
            raise Exception("Missing post id: " + str(key)) 
        key_lang = post_id_to_lang[key]  
        if key_lang not in success_by_lang:
            success_by_lang[key_lang] = [] 

        p = prediction[key]  
        assert len(p) >= 10, ("You should provide top 10 predictions for each example")
        if len(t) == 1:
            if t[0] in p[:k]: 
                success_by_lang[key_lang].append(1)
            else:
                success_by_lang[key_lang].append(0)

        elif len(t)>1:
            flag = 0
            for t_ in t:
                if t_ in p[:k]:
                    flag += 1
                    break 
            if flag > 0:
                success_by_lang[key_lang].append(1)
            else:
                success_by_lang[key_lang].append(0)

        else:
            raise Exception("Some examples have gold label less than 1")  

    average = {}
    for lang, success_at_k in success_by_lang.items():
        average[lang] = sum(success_at_k) / len(success_at_k)

    average_score =  np.mean(list(average.values()))
    average["avg"] = average_score

    return average 

# Change path to "train_dev_set" to evaluate predictions on train and dev sets
reference_dir = os.path.join(".", "test_set") 
# Change path to a folder containing your prediction json file(s)
prediction_dir = os.path.join(".", "baselines")  
score_dir = "."

print ("We are performing cross-lingual evaluation!")
print('Reading prediction') 
# Change the path to one of the baseline monolingual predictions:
# bm25_monolingual.json
# gtr-t5-large_monolingual.json
# paraphrase-multilingual-mpnet-base-v2_monolingual.json
# multilingual-e5-large_monolingual.json
# Or to your own predictions json file 
prediction = read_from_file(os.path.join(prediction_dir, 'bm25_monolingual.json'))

print ("Done loading predictions!")
truth = read_from_file(os.path.join(reference_dir, 'monolingual_reference.json'))  
print ("Done loading labels!")

print ("Loading language information!")
with open(os.path.join(reference_dir,"tasks.json"), 'r') as f:
    tasks = json.load(f)

# Change split to "train_dev" to evaluate predictions on train and dev sets
post_id_to_lang = get_post_id_to_lang_dict(tasks, split="test")

print('Checking Success@K') 
scores = calculate_success_at_k(truth, prediction, 10, post_id_to_lang)   

print('Scores:') 
print(scores)  

with open(os.path.join(score_dir, 'scores.json'), 'w') as score_file:
    score_file.write(json.dumps(scores))