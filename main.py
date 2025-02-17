# from experiment import run_experiment
import requests
import json
import pandas as pd
import os
from google.cloud import vision
import requests
import io
from tqdm import tqdm
from prepare_datasets import prepare_verite
import time
# prepare_Misalign, get_K_most_similar, extract_entities, prepare_CLIP_NESt, prepare_R_NESt
# from extract_features import extract_CLIP_features, extract_CLIP_twitter

# Scrape the images of VERITE and prepare the dataset
# prepare_verite(download_images=True)

PATH = "image-text-verification\VERITE"

# --- Google Vision API for Reverse Image Search ---
def detect_web(path):
    print('PATH ---', path)
    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.web_detection(image=image, max_results=15)
    if response.error.message:
        raise Exception(
            f"{response.error.message}\n"
            "For more info on error messages, check: https://cloud.google.com/apis/design/errors"
        )

    annotations = response.web_detection

    # Collect context data
    context_data = {
        "best_guess_labels": [],
        "web_entities": [],
        "pages_with_matching_images": []
        # "visually_similar_images": []
    }

    # Extracting best guess labels
    if annotations.best_guess_labels:
        for label in annotations.best_guess_labels:
            context_data["best_guess_labels"].append(label.label)

    # Pages with matching images
    if annotations.pages_with_matching_images:
        for page in annotations.pages_with_matching_images:
            context_data["pages_with_matching_images"].append(
                {
                    "url": page.url,
                    "page_title": page.page_title
                }
            )

    # Web entities (e.g., people, places, events)
    if annotations.web_entities:
        for entity in annotations.web_entities:
            context_data["web_entities"].append({
                "description": entity.description,
                "score": entity.score
            })

    # # Visually similar images
    # if annotations.visually_similar_images:
    #     for image in annotations.visually_similar_images:
    #         context_data["visually_similar_images"].append(image.url)


    return context_data

def set_credentials(api_key_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = api_key_path
    

def get_response(input_text):
    url = 'https://ws.gvlab.org/fablab/ura/llama/haystack/generate_stream'
    data = {"inputs": "<|start_header_id|>user<|end_header_id|>\n " + input_text + "\n<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n"}
    headers = {'Content-Type': 'application/json'}

    response = requests.post(url, json=data, headers=headers)

    if response.status_code == 200:
        lines = response.text.strip().split('\n')
        return lines
    else:
        print(f"Code: {response.status_code}, Response Text: {response.text}")
        return []

def parse_generated_text(lines):
    responses = []

    for raw_data in lines:
        if raw_data.startswith("data:"):
            json_data = json.loads(raw_data[5:])

            if "generated_text" in json_data and json_data["generated_text"]:
                return clean_response(json_data["generated_text"])
            elif "token" in json_data and "text" in json_data["token"]:
                responses.append(json_data["token"]["text"])

    complete_response = " ".join(responses).strip()
    return clean_response(complete_response)

def clean_response(response):
    if "----------" in response:
        response = response.split("----------")[0]
    return response.strip()

# input_text = f"""Please check if these two captions might have relationship with each other, according to the informations contained in each caption.\
# Caption 1 is the original caption. Caption 2 is the generated caption which contains some additional context.\
# You must output only "YES" if the captions can be related to each other based on all the context given, and "NO" otherwise.\
# Caption 1: {result["original_caption"]}\
# Caption 2: {result["generated_caption"]}\
# Result:
# """

# print(input_text)

# lines = get_response(input_text)

# if lines:
#     response_text = parse_generated_text(lines)
#     print(response_text)
# else:
#     print("Không nhận được phản hồi từ API.")

# create a function read the csv file and return the original caption and the generated caption
def create_verite_v2(verite_file_path):
    # read the csv file VERITE.csv
    verite = pd.read_csv(verite_file_path, index_col=0)

    # filter the dataset, keeps only labels with 'true' and 'out-of-context'
    verite = verite[verite['label'].isin(['true', 'out-of-context'])]
    # copy verite to verite_v2
    verite_v2 = verite.copy()
    # we need to create new columns for veritev2, storing belows:
    # 1. best_guess_labels
    # create a new column for best_guess_labels
    verite_v2['best_guess_labels'] = ''
    # 2. web entities
    # create a new column for web_description
    verite_v2['web_description'] = ''
    # 3. pages_with_matching_images
    # create a new column for page_title_matching_images
    verite_v2['page_title_matching_images'] = ''

    print('Filtering dataset done. Length of dataset:', len(verite))
    # ,caption,image_path,label
    # return the original caption and the generated caption
    for index, row in tqdm(verite.iterrows(), total=len(verite)):
        image_path = row['image_path']
        # use image_path to feed to the Google Vision API
        context_data = detect_web(PATH + "\\" + image_path)
        # print(context_data)
        # {'best_guess_labels': ['girl'], 'web_entities': [{'description': '2019–20 coronavirus pandemic', 'score': 0.5981999635696411}, {'description': 'COVID-19', 'score': 0.5831000208854675}, {'description': 'just', 'score': 0.5188999772071838}, {'description': 'China', 'score': 0.46274998784065247}, {'description': 'opening', 'score': 0.4388999938964844}, {'description': 'Pandemic', 'score': 0.4350000023841858}, {'description': '2020', 'score': 0.3813000023365021}, {'description': 'Quarantine', 'score': 0.34619998931884766}, {'description': '', 'score': 0.3084999918937683}, {'description': 'TikTok', 'score': 0.30000001192092896}, {'description': '2022', 'score': 0.2856999933719635}, {'description': 'Meme', 'score': 0.2831000089645386}], 'pages_with_matching_images': ['https://www.snopes.com/fact-check/china-covid-quarantine-video/']}
        
        # print(context_data)
        # add the context_data to the verite_v2
        verite_v2.loc[index, 'best_guess_labels'] = ', '.join(context_data['best_guess_labels'])
        # filter the web_entities, keep the ones with score > 0.3, and join them with comma
        verite_v2.loc[index, 'web_description'] = ', '.join([entity['description'] for entity in context_data['web_entities'] if entity['score'] > 0.3])
        # keep all page_title in  pages_with_matching_images, join them with comma
        verite_v2.loc[index, 'page_title_matching_images'] = ', '.join([page['page_title'] for page in context_data['pages_with_matching_images']])
    

    # save the verite_v2 to a new csv file
    verite_v2.to_csv(PATH + "\VERITE_v2.csv", index=False)

if __name__ == "__main__":
    # set_credentials("image-text-verification\cita-2025-key-gg.json")
    # create_verite_v2(PATH + "\VERITE.csv")
    
    # read the csv file VERITE_v2.csv
    verite_v2 = pd.read_csv(PATH + "\VERITE_v2.csv")
    # print the first 5 rows of the dataframe
    # filter out the rows where label is 'true'
    verite_v2_true = verite_v2[verite_v2['label'] == 'true']
    # filter out the rows where label is 'out-of-context'
    verite_v2_false = verite_v2[verite_v2['label'] == 'out-of-context']
    
    PROMPT = """
        You are a helpful assistant specializing in rewriting image captions for improved accuracy and informativeness.
        You will be provided with an image caption along with additional context in the following format:
        <context>
        LABEL: {label}  
        DESCRIPTION: {description}  
        RELATED PAGE TITLE: {page_title}  
        </context>  

        <caption>  
        {caption}  
        </caption>
        Your task is to integrate all of the key details from the context into the caption while maintaining clarity and relevance.
        Ensure the revised caption is more precise and informative.
        Output only the rewritten caption, without any extra text.
    """

    # create a new column for rewritten_caption
    verite_v2_true['rewritten_caption'] = ''
    
    # for each row in verite_v2_true, generate a new caption
    for index, row in tqdm(verite_v2_true.iterrows(), total=len(verite_v2_true)):
        label = row['best_guess_labels']
        description = row['web_description']
        page_title = row['page_title_matching_images']
        caption = row['caption']
        input_text = PROMPT.format(label=label, description=description, page_title=page_title, caption=caption)
        print('[INPUT] ', input_text)
        # get the response from the API
        lines = get_response(input_text)
        # parse the response
        response_text = parse_generated_text(lines)
        print('[OUTPUT] ', response_text)
        # add the response_text to the rewritten_caption column
        verite_v2_true.loc[index, 'rewritten_caption'] = response_text

        # add wait 3 seconds
        time.sleep(3)
    
    # save the verite_v2_true to a new csv file
    verite_v2_true.to_csv(PATH + "\VERITE_v2_true.csv", index=False)
    
    
    


# # Extract features with CLIP ViT-L/14 from VERITE, COSMOS, Twitter, VisualNews etc
# extract_CLIP_features(data_path='VERITE/', output_path='VERITE/VERITE_')
# extract_CLIP_features(data_path='COSMOS/', output_path='COSMOS/COSMOS_') 
# extract_CLIP_twitter(output_path='Twitter/', choose_clip_version="ViT-L/14", choose_gpu=0)
# extract_CLIP_features(data_path='VisualNews/origin/', output_path='VisualNews/') 
# extract_CLIP_features(data_path='Fakeddit/', output_path='Fakeddit/fd_original_') 
# extract_CLIP_features(data_path='VisualNews/MISALIGN', output_path='VisualNews/MISALIGN_', use_image=False) 

# # After extracting the Fakeddit we can create the Misalign dataset
# prepare_Misalign(CLIP_VERSION="ViT-L/14", choose_gpu=0)

# # Calculate the K most similar pairs from VisualNews. Necassary for creating CLIP-NESt. Can also be used to re-create CSt
# get_K_most_similar(K_most_similar = 20, CLIP_VERSION="ViT-L/14", choose_gpu=1)

# # Extract named entities from VisualNews pairs
# extract_entities()

# # Create the CLIP-NESt dataset and then extract its CLIP features
# prepare_CLIP_NESt()
# extract_CLIP_features(data_path='EntitySwaps_topic_clip', output_path='VisualNews/EntitySwaps_topic_clip', use_image=False) 

# # Create the R-NESt dataset and then extract its CLIP features
# prepare_R_NESt()
# extract_CLIP_features(data_path='EntitySwaps_topic_random', output_path='VisualNews/EntitySwaps_topic_random', use_image=False)


# # ### Table: 2 (Twitter)
# run_experiment(
#     dataset_methods_list = [
#         'Twitter_comparable', # Uses the evaluation protocol of previous works
#         'Twitter_corrected', # Uses a corrected evaluation protocol
#     ],
#     modality_options = [
#         ["images", "texts"],
#         ["images", "texts", "-attention"],        
#         ["texts"], 
#         ["images"]
#     ],
#     epochs=30,
#     seed_options = [0],
#     lr_options = [5e-5, 1e-5],
#     batch_size_options = [16],
#     tf_layers_options = [1, 4],
#     tf_head_options = [2, 8],
#     tf_dim_options = [128, 1024],
#     use_multiclass = False, 
#     balancing_method = None,
#     choose_gpu = 0, 
#     init_model_name = ''
# )

# # ### Tables: 3, 4 and parts of 6 (single binary datasets)
# run_experiment(
#     dataset_methods_list = [
#         'random_sampling_topic', # RSt
#         'clip_based_sampling_topic', # CSt
#         'news_clippings_txt2txt', # NC-t2t
#         'meir', 
#         'EntitySwaps_random_topic', # R-NESt
#         'EntitySwaps_CLIP_topic', # CLIP-NESt
#         'fakeddit_original',
#         'Misalign', 
#         'Misalign_D', # 'downsample' is automatically applied
#     ],
#     modality_options = [
#         ["images", "texts"],
#         ["images", "texts", "-attention"],        
#         ["texts"], 
#         ["images"]
#     ],
#     epochs=30,
#     seed_options = [0],
#     lr_options = [5e-5],
#     batch_size_options = [512],
#     tf_layers_options = [1, 4],
#     tf_head_options = [2, 8],
#     tf_dim_options = [128, 1024],
#     use_multiclass = False, 
#     balancing_method = None,
#     choose_gpu = 0, 
#     init_model_name = ''
# )

# # Table 5: Multiclass classification on VERITE
# run_experiment(
#     dataset_methods_list = [
#         'EntitySwaps_CLIP_topicXclip_based_sampling_topic',
#         'EntitySwaps_CLIP_topicXnews_clippings_txt2txt',     
#         'EntitySwaps_random_topicXclip_based_sampling_topic',
#         'EntitySwaps_random_topicXnews_clippings_txt2txt',     
#         'MisalignXclip_based_sampling_topic',
#         'MisalignXnews_clippings_txt2txt',  
#         'Misalign_DXclip_based_sampling_topic',
#         'Misalign_DXnews_clippings_txt2txt',
#         'EntitySwaps_random_topicXMisalign_DXnews_clippings_txt2txt',
#         'EntitySwaps_CLIP_topicXMisalign_DXnews_clippings_txt2txt',
#         'EntitySwaps_random_topicXMisalignXnews_clippings_txt2txt',
#         'EntitySwaps_CLIP_topicXMisalignXnews_clippings_txt2txt',        
#     ],
#     epochs=30,
#     use_multiclass = True,
#     balancing_method = 'downsample',
# )

# # Table 6: Ensemble datasets for binary classification on VERITE
# run_experiment(
#     dataset_methods_list = [
#         'EntitySwaps_CLIP_topicXnews_clippings_txt2txt',     
#         'EntitySwaps_random_topicXnews_clippings_txt2txt',     
#         'MisalignXnews_clippings_txt2txt',  
#         'Misalign_DXnews_clippings_txt2txt',
#         'EntitySwaps_random_topicXMisalign_DXnews_clippings_txt2txt',
#         'EntitySwaps_CLIP_topicXMisalign_DXnews_clippings_txt2txt',
#         'EntitySwaps_random_topicXMisalignXnews_clippings_txt2txt',
#         'EntitySwaps_CLIP_topicXMisalignXnews_clippings_txt2txt',        
#     ],
#     epochs=30,
#     use_multiclass = False,
#     balancing_method = 'downsample',
# )

