import pandas as pd
from prompts import easy_prompt,complex_prompt
from prompt import debug_prompt
from util.arg import main_args
import re
import os
from llm import LLM_generation
import sqlite3
from prompt import PromptMaker

args = main_args()
DATASET_SCHEMA = f"./data/{args.dataset}/tables.json"
DATASET = f"./data/{args.dataset}/dev.json"
OUTPUT_FILE = "log/spider/clear/predicted.sql"


def load_data(DATASET):
    return pd.read_json(DATASET)

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

def easy_prompt_maker(test_sample_text,table_match,database,schema_links,prompt_content,example):
    Instruct = "### NL2SQL task on schema linking information and corresponding table information for a given problem, and do notselect extra columns that are not explicitly in the schema_links.\n"
    fields = find_fields_SL_like(database, table_match)
    if example == 1:
        easy_prompt = easy_prompt_1
    elif example == 2:
        easy_prompt = easy_prompt_2
    else:
        easy_prompt = easy_prompt_3
    schema_links = generate_mappings(schema_links)
    prompt = Instruct + easy_prompt + f"### Input:{test_sample_text}"+f"\n### Database: {fields}" + prompt_content + f'\n###schema_links:{schema_links}'
    return prompt

def complex_prompt_maker(test_sample_text,table_match,database,schema_links,prompt_content,example):
    Instruct = "### NL2SQL task on schema linking information and corresponding table information for a given problem, and do notselect extra columns that are not explicitly in the schema_links.\n"
    fields = find_fields_SL_like(database,table_match)
    fields += "Foreign_keys = " + find_foreign_keys_SL_MYSQL_like(database,table_match) + '\n'
    schema_links = generate_mappings(schema_links)
    if example == 1:
        complex_prompt = complex_prompt_1
    elif example == 2:
        complex_prompt = complex_prompt_2
    else:
        complex_prompt = complex_prompt_3
    prompt = Instruct + complex_prompt + f"### Input:{test_sample_text}"+f"\n### Database: {fields}" + prompt_content + f'\n###schema_links:{schema_links}'
    return prompt

def debuger(test_sample_text,schema_links,SQLs):
    Instruct = '### Vote for the correct SQL statement from the following generated SQL queries based on the question and pattern link information. You must choose one\n'
    schema_links = generate_mappings(schema_links)
    prompt = Instruct + debug_prompt + '### Question: ' + test_sample_text + '\n### Schema_links:\n' + str(schema_links) +f'\n### Candidate SQL:{SQLs}'
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

def content_maker(table_match,content):
    pattern = r"/\*([\s\S]*?)\*/"
    content_prompt = ""
    matches = re.findall(pattern, content[args.content])
    for tab in table_match:
        i = content[0].index(tab)
        content_prompt += matches[i]
    return content_prompt



def file_exists(file_path):
    return os.path.isfile(file_path)

if __name__ == '__main__':
    prompt_maker = PromptMaker(args=args)
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
        contnent = prompt_maker.db_prompts[row['db_id']]
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
        prompt_content = ""
        if args.content != 0:
            prompt_content = content_maker(table_match, contnent)
        SQLs = []
        if 'simple' in classification:
            SQL_1_org = LLM_generation(easy_prompt_maker(row['question'],table_match, row['db_id'], schema_links,prompt_content,1))
            SQL_1 = re.search(r'```sql\s*(.*?)```', SQL_1_org, re.DOTALL).group(1).replace("\n", " ")
            SQLs.append(SQL_1)
            SQL_2_org = LLM_generation(easy_prompt_maker(row['question'], table_match, row['db_id'], schema_links, prompt_content, 2))
            SQL_2 = re.search(r'```sql\s*(.*?)```', SQL_2_org, re.DOTALL).group(1).replace("\n", " ")
            SQLs.append(SQL_2)
            SQL_3_org = LLM_generation(easy_prompt_maker(row['question'], table_match, row['db_id'], schema_links, prompt_content, 2))
            SQL_3 = re.search(r'```sql\s*(.*?)```', SQL_3_org, re.DOTALL).group(1).replace("\n", " ")
            SQLs.append(SQL_3)
        else:
            SQL_1_org = LLM_generation(
                complex_prompt_maker(row['question'], table_match, row['db_id'], schema_links, prompt_content, 1))
            SQL_1 = re.search(r'```sql\s*(.*?)```', SQL_1_org, re.DOTALL).group(1).replace("\n", " ")
            SQLs.append(SQL_1)
            SQL_2_org = LLM_generation(
                complex_prompt_maker(row['question'], table_match, row['db_id'], schema_links, prompt_content, 2))
            SQL_2 = re.search(r'```sql\s*(.*?)```', SQL_2_org, re.DOTALL).group(1).replace("\n", " ")
            SQLs.append(SQL_2)
            SQL_3_org = LLM_generation(
                complex_prompt_maker(row['question'], table_match, row['db_id'], schema_links, prompt_content, 2))
            SQL_3 = re.search(r'```sql\s*(.*?)```', SQL_3_org, re.DOTALL).group(1).replace("\n", " ")
            SQLs.append(SQL_3)
        db_path = os.path.join('data', args.dataset, 'database', row['db_id'], row['db_id'] + '.sqlite')
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        results = []
        for query in SQLs:
            try:
                cursor.execute(query)
                results.append(cursor.fetchone())
            except Exception as e:
                SQLs.remove(query)
        cursor.close()
        conn.close()
        print(SQLs)
        if all(result == results[0] for result in results):
            SQL = SQLs[0]
            print("三个语句的执行结果完全相同:")
        else:
            CH_SQL = LLM_generation(debuger(row['question'], schema_links, SQLs))
            SQL = re.search(r'```output\s*(.*?)```', CH_SQL, re.DOTALL).group(1).replace("\n", " ")
            print("三个语句的执行结果不相同:")
        CODEX.append([row['question'],row['query'] ,SQL])
        if (index + 1) % 10 == 0 or index == len(val_df) - 1:
            df = pd.DataFrame(CODEX, columns=['NLQ', 'GOLD SQL', 'PREDICTED SQL'])
            results = df['PREDICTED SQL'].tolist()
            with open(OUTPUT_FILE, 'a') as f:
                for line in results:
                    f.write(f"{line}\n")
            CODEX = []