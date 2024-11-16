from dotenv import load_dotenv
import tiktoken
from functools import lru_cache
import os
import json

load_dotenv(os.path.join(os.path.expanduser('~'), '.gpt_ncats.env'))
load_dotenv(os.path.join(os.path.expanduser('~'), '.ncats_llama.env'))

LLM_MODEL = 'gpt-4o-mini'
TEMPERATURE = 0.1
MAX_TOKENS = 750
MAX_TOKEN_LIMIT = 6000


HIGH_LEVEL_SUMMARY_SYSTEM_PROMPT_WITHOUT_JSON_RESPONSE = '''
You are an expert in summarizing biomedical knowledge in max 1000 words. 

You will receive a dictionary in the following format:
{
  "query": "<a question of string datatype>",
  "results": [
    {
      "name": "<name of a biomedical concept>",
      "paths": [
        "<biological path 1>",
        "<biological path 2>",
        "<biological path 3>"
      ]
    }
  ]
}

"name" holds the <name of a biomedical concept> which forms an answer to the "query".
"paths" holds a list of biological paths which is a sequence of edges extracted from a knowledge graph. These paths form a mechanistic explanation or reasoning for how that biomedical concept suffices as an answer and relates to the query.

Your task is to provide a summarized answer to the given query in max 1000 words. Do NOT list the results, instead you are supposed to give a biologically sound summarization for these results in one paragraph. For that:

- Combine all the biological paths corresponding to all answers into a graph. Using that graph, provide a coherent textual summary that explains the mechanistic rationale for the given answers to the query.
- Summarize any shared paths or nodes by integrating them where applicable.
- Ensure that the summary is biologically accurate, by providing right explanations to the biomedical concepts such as genes or proteins.
- Give a conclusion paragraph in the last, followed by the string "End of Response"
- If no context is provided, report that there is no context available for summarization.
- Give your summarization in maximum three paragraphs.
- Do not give a big list of names of entities. Whenever you give the list of names, always separate the names with a comma (example: S-NITROSOGLUTATHIONE, alsterpaullone, AZD8186, XANTHOHUMOL).
- Do not give any dashes in the response  (like "-")


Example:
{
    "query": "what drugs may treat conditions related to Ehlers-Danlos syndrome, hypermobility type?"
    "results" : [
                {'name': 'cyclosporine',
                 'paths': ['cyclosporine-[causes increased activity of]-COL3A1-[gene associated with condition]-Ehlers-Danlos syndrome, hypermobility type',
                  'cyclosporine-[treats or applied or studied to treat]-gingival overgrowth-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'cyclosporine-[treats or applied or studied to treat]-Arthralgia-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'cyclosporine-[treats or applied or studied to treat]-xerophthalmia-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type']},
                {'name': 'citalopram',
                 'paths': ['citalopram-[in clinical trials for]-anxiety-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'citalopram-[treats]-depressive disorder-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'citalopram-[treats or applied or studied to treat]-depressive disorder-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'citalopram-[applied to treat]-depressive disorder-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'citalopram-[applied to treat]-anxiety-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'citalopram-[in clinical trials for]-depressive disorder-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type']},
                {'name': 'lidocaine',
                 'paths': ['lidocaine-[preventative for condition]-Chronic pain-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[treats or applied or studied to treat]-depressive disorder-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[in clinical trials for]-scoliosis-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[in clinical trials for]-Chronic pain-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[preventative for condition]-Myalgia-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[treats or applied or studied to treat]-cardiac rhythm disease-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[in clinical trials for]-cardiac rhythm disease-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[treats or applied or studied to treat]-migraine disorder-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[in clinical trials for]-osteoarthritis-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[in clinical trials for]-anxiety-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[in clinical trials for]-severe cutaneous adverse reaction-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[treats or applied or studied to treat]-Chronic pain-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[in clinical trials for]-Myalgia-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[in clinical trials for]-migraine disorder-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type',
                  'lidocaine-[preventative for condition]-cardiac rhythm disease-[phenotype of]-Ehlers-Danlos syndrome, hypermobility type']}
          ]
}

Response (not that this example response is given in three paragraphs):
Ehlers-Danlos syndrome, hypermobility type (hEDS) is a connective tissue disorder characterized by joint hypermobility, skin elasticity, and a predisposition to musculoskeletal pain and other associated phenotypes, including chronic pain, depressive disorders, anxiety, and various mild to severe complications. Several pharmacological interventions have shown potential in managing the symptoms related to this condition. Cyclophosphamide is recognized for its ability to modulate collagen synthesis, as indicated by its effect on increasing the activity of the COL3A1 gene, linked to hEDS. It has been studied for alleviating specific symptoms or phenotypes of hEDS, notably gingival overgrowth, arthralgia, and xerophthalmia. These indications suggest that the drug could play a role in managing some of the secondary issues often faced by patients with hEDS. Citalopram, a selective serotonin reuptake inhibitor (SSRI), has been found beneficial in addressing psychiatric manifestations often associated with hEDS, including anxiety and depressive disorders. It has been evaluated in clinical trials specifically aimed at treating these conditions as they are prevalent among patients, indicating that mental health management is crucial in the holistic treatment of hEDS. 

Lidocaine, a local anesthetic, has also been involved in treatment strategies for hEDS. It demonstrates capabilities in preventing and treating chronic pain, myalgia, and other neurogenic pain manifestations. Additionally, lidocaine has been investigated for its efficacy in managing cardiac rhythm disorders, migraines, and even anxiety, showcasing its multi-faceted potential in addressing various complications attributable to hEDS. It is noteworthy that the diverse range of phenotypes linked to hEDS creates a complex clinical picture, prompting the investigation of various medications to provide comprehensive symptom relief.

In conclusion, the drugs cyclosporine, citalopram, and lidocaine are among those that may provide therapeutic benefits for individuals with Ehlers-Danlos syndrome, hypermobility type, specifically targeting its varied symptoms including pain management, mental health disorders, and other related phenotypes. The integration of these treatments into clinical practice underscores the need for a tailored approach to individual patient care, reflecting the multifactorial nature of hEDS.
'''



@lru_cache(maxsize=None)
def count_tokens_cached(text, model):
    return count_tokens(text, model)

def get_token_limited_prompt(query, system_prompt):
    results = query["results"]
    tot_results = len(results)
    
    def create_prompt(num_results):
        limited_results = results[:num_results]
        prompt_data = {
            "query": query["query"],
            "results": limited_results
        }
        return json.dumps(prompt_data, indent=2)

    def is_within_limit(num_results):
        prompt = create_prompt(num_results)
        full_prompt = system_prompt + '\n' + prompt
        return count_tokens_cached(full_prompt, LLM_MODEL) <= MAX_TOKEN_LIMIT

    # Binary search
    left, right = 1, tot_results
    while left <= right:
        mid = (left + right) // 2
        if is_within_limit(mid):
            left = mid + 1
        else:
            right = mid - 1
    
    return create_prompt(right)

def count_tokens(text: str, model: str) -> int:    
    if 'gpt-4' in model.lower() or 'gpt-35-turbo' in model.lower() or 'gpt-3.5-turbo' in model.lower() or model == 'text-embedding-ada-002':
        encoding_name = 'cl100k_base'
    elif model == 'text-davinci-003':
        encoding_name = 'p50k_base'
    elif 'gpt-4o' in model:
        encoding_name = 'o200k_base'
    else:
        raise ValueError(f'{model} model not supported for encoding in count_tokens function')
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(text))
        return num_tokens
    except Exception as e:
        print(f'Unable to count tokens from input \"{text}\". Error: {e}')
        return 0