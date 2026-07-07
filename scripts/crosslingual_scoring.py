import json
import os


def read_from_file(path):
    with open(path, "r") as f:
        data = json.load(f)   
    return data 

def calculate_success_at_k(truth, prediction, k):
    assert len(truth)==len(prediction), "The number of prediction is not correct" 
    success_at_k = 0 

    for key, t in truth.items(): 
        if key not in prediction:
            raise Exception("Missing post id: " + str(key))
        p = prediction[key]
        assert len(p) >= 10, ("You should provide top 10 predictions for each example")
        if len(t) == 1:
            if t[0] in p[:k]: 
                success_at_k += 1 
        elif len(t)>1:
            for t_ in t:
                if t_ in p[:k]:
                    success_at_k += 1  
                    break 
        else:
            raise Exception("Some examples have gold label less than 1")  

    return success_at_k/len(truth) 

# Change path to train_dev_set to evaluate predictions on dev set
reference_dir = os.path.join(".", "test_set") 
# Change path to a folder containing your prediction json file(s)
prediction_dir = os.path.join(".", "baselines")  
score_dir = "."

print ("We are performing cross-lingual evaluation!")
print('Reading prediction') 
# Change the path to one of the baseline crosslingual predictions:
# bm25_crosslingual.json
# gtr-t5-large_crosslingual.json
# paraphrase-multilingual-mpnet-base-v2_crosslingual.json
# multilingual-e5-large_crosslingual.json
# Or to your own predictions json file
prediction = read_from_file(os.path.join(prediction_dir, 'bm25_crosslingual.json')) 

print ("Done loading predictions!")
truth = read_from_file(os.path.join(reference_dir, 'crosslingual_reference.json'))  

print ("Done loading labels!")


print('Checking Success@K') 
success_at_10 = calculate_success_at_k(truth, prediction, 10)  

print('Scores:')
scores = {
    'avg': success_at_10 
} 
print(scores)

with open(os.path.join(score_dir, 'scores.json'), 'w') as score_file:
    score_file.write(json.dumps(scores))