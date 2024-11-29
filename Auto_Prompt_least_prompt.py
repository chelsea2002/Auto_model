import pandas as pd
from prompt import easy_prompt_example_1_less,easy_prompt_example_2_less,complex_prompt_example_2_less,easy_prompt_gpt3_5,complex_prompt_example_1_less,complex_prompt_example_3_less,easy_prompt_example_3_less
from util.arg import main_args
import re
import os
from llm import LLM_generation

args = main_args()
DATASET_SCHEMA = f"./data/{args.dataset}/tables.json"
DATASET = f"./data/{args.dataset}/dev.json"
OUTPUT_FILE = "log/spider/gpt4o_s3c3_less/predicted.sql"





def load_data(DATASET):
    return pd.read_json(DATASET)



def easy_prompt_maker(test_sample_text,table_match,database,schema_links):
    Instruct = "NL2SQL task on schema linking information and corresponding table information for a given problem"
    fields = find_fields_SL_like(database,table_match)
    schema_links = generate_mappings(schema_links)
    if args.example == 1:
        easy_prompt = easy_prompt_example_1_less
    elif args.example == 2:
        easy_prompt = easy_prompt_example_2_less
    else:
        easy_prompt = easy_prompt_example_3_less
    if args.gpt == "gpt-4o-mini":
        prompt = Instruct + easy_prompt + f"### Input:{test_sample_text}"+f"\n### Database: {fields}" + f'\n###schema_links:{schema_links}'
    else:
        prompt = Instruct + easy_prompt_gpt3_5 + f"### Input:{test_sample_text}"+f"\n### Database: {fields}" + f'\n###schema_links:{schema_links}'
    return prompt

def complex_prompt_maker(test_sample_text,table_match,database,schema_links):
    Instruct = "NL2SQL task on schema linking information and corresponding table information for a given problem"
    fields = find_fields_SL_like(database,table_match)
    fields += "Foreign_keys = " + find_foreign_keys_SL_MYSQL_like(database,table_match) + '\n'
    schema_links = generate_mappings(schema_links)
    if args.example == 1:
        complex_prompt = complex_prompt_example_1_less
    elif args.example == 2:
        complex_prompt = complex_prompt_example_2_less
    else:
        complex_prompt = complex_prompt_example_3_less
    prompt = Instruct + complex_prompt + f"### Input:{test_sample_text}"+f"\n### Database: {fields}" + f'\n###schema_links:{schema_links}'
    return prompt
def find_foreign_keys_MYSQL_like(db_name):
  df = spider_foreign[spider_foreign['Database name'] == db_name]
  output = "["
  for index, row in df.iterrows():
    output += row['First Table Name'] + '.' + row['First Table Foreign Key'] + " = " + row['Second Table Name'] + '.' + row['Second Table Foreign Key'] + ','
  output= output[:-1] + "]"
  return output

def find_foreign_keys_SL_MYSQL_like(db_name,table_match):
  df = spider_foreign[spider_foreign['Database name'] == db_name]
  output = "["
  for index, row in df.iterrows():
      if row['First Table Name'] in table_match and row['Second Table Name'] in table_match:
        output += row['First Table Name'] + '.' + row['First Table Foreign Key'] + " = " + row['Second Table Name'] + '.' + row['Second Table Foreign Key'] + ','
  output= output[:-1] + "]"
  return output
def find_fields_MYSQL_like(db_name):
  df = spider_schema[spider_schema['Database name'] == db_name]
  df = df.groupby(' Table Name')
  output = ""
  for name, group in df:
    output += "Table " +name+ ', columns = ['
    for index, row in group.iterrows():
      output += row[" Field Name"]+','
    output = output[:-1]
    output += "]\n"
  return output

def find_fields_SL_like(db_name, table_match):
  df = spider_schema[spider_schema['Database name'] == db_name]
  df = df.groupby(' Table Name')
  output = ""
  for name, group in df:
    if name in table_match:
        output += "Table " +name+ ', columns = ['
        for index, row in group.iterrows():
            output += row[" Field Name"]+','
    output = output[:-1]
    output += "]\n"
  return output

def generate_mappings(schema_linkings):
    result = []
    for key, value in schema_linkings.items():
        if key == 'value':
            result.append(f'Values [{value}] will be used')
        else:
            for field_type, field_values in value.items():
                if field_values:
                    for field in field_values:
                        result.append(f'According "{key}" "{field_type}": [{field}] will be used')
    return result
def find_primary_keys_MYSQL_like(db_name):
  df = spider_primary[spider_primary['Database name'] == db_name]
  output = "["
  for index, row in df.iterrows():
    output += row['Table Name'] + '.' + row['Primary Key'] +','
  output = output[:-1]
  output += "]\n"
  return output


def creatiing_schema(DATASET_JSON):
    schema_df = pd.read_json(DATASET_JSON)
    schema_df = schema_df.drop(['column_names','table_names'], axis=1)
    schema = []
    f_keys = []
    p_keys = []
    for index, row in schema_df.iterrows():
        tables = row['table_names_original']
        col_names = row['column_names_original']
        col_types = row['column_types']
        foreign_keys = row['foreign_keys']
        primary_keys = row['primary_keys']
        for col, col_type in zip(col_names, col_types):
            index, col_name = col
            if index == -1:
                for table in tables:
                    schema.append([row['db_id'], table, '*', 'text'])
            else:
                schema.append([row['db_id'], tables[index], col_name, col_type])
        for primary_key in primary_keys:
            index, column = col_names[primary_key]
            p_keys.append([row['db_id'], tables[index], column])
        for foreign_key in foreign_keys:
            first, second = foreign_key
            first_index, first_column = col_names[first]
            second_index, second_column = col_names[second]
            f_keys.append([row['db_id'], tables[first_index], tables[second_index], first_column, second_column])
    spider_schema = pd.DataFrame(schema, columns=['Database name', ' Table Name', ' Field Name', ' Type'])
    spider_primary = pd.DataFrame(p_keys, columns=['Database name', 'Table Name', 'Primary Key'])
    spider_foreign = pd.DataFrame(f_keys,
                        columns=['Database name', 'First Table Name', 'Second Table Name', 'First Table Foreign Key',
                                 'Second Table Foreign Key'])
    return spider_schema,spider_primary,spider_foreign

def file_exists(file_path):
    return os.path.isfile(file_path)

if __name__ == '__main__':
    spider_schema,spider_primary,spider_foreign = creatiing_schema(DATASET_SCHEMA)
    val_df = load_data(DATASET)
    print(f"Number of data samples {val_df.shape[0]}")
    CODEX = []
    index = 0
    if file_exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            line_count = sum(1 for line in f)
    else:
        line_count = 0
    for index, row in val_df.iterrows():
        if index < line_count:
            continue
        print(f"-------------------------------------------index is {index}-------------------------------------------")
        classification = None
        while classification is None:
            table_match = set()
            for key in row['schema_linkings']:
                if key != 'value':
                    for ke,value in row['schema_linkings'][key].items():
                        if len(value) == 0:
                            pass
                        else:
                            if ke == 'table':
                                table_match.add(value[0])
                            else:
                                table_match.add(value[0].split('.')[0])
            if len(table_match) == 1:
                classification = 'simple'
            else:
                classification = 'complex'
        schema_links = row['schema_linkings']
        SQL = None
        if 'simple' in classification:
            while SQL is None:
                SQL = LLM_generation(easy_prompt_maker(row['question'],table_match, row['db_id'], schema_links))
                try:
                    SQL = re.search(r'```sql\s*(.*?)```', SQL, re.DOTALL).group(1).replace("\n", " ")
                except:
                    SQL = None
                print(SQL)

        else:
            SQL = LLM_generation(complex_prompt_maker(row['question'],table_match, row['db_id'], schema_links))

            try:
                SQL = re.search(r'```sql\s*(.*?)```', SQL, re.DOTALL).group(1).replace("\n", " ")
            except:
                SQL = None
            print(SQL)

        CODEX.append([row['question'],row['query'] ,SQL])
        if (index + 1) % 10 == 0 or index == len(val_df) - 1:
            df = pd.DataFrame(CODEX, columns=['NLQ', 'GOLD SQL', 'PREDICTED SQL'])
            results = df['PREDICTED SQL'].tolist()
            with open(OUTPUT_FILE, 'a') as f:
                for line in results:
                    f.write(f"{line}\n")
            CODEX = []