classification_prompt = '''
You're an expert at categorizing NL2SQL problems.
The questions are divided into two categories according to the number of tables required
# Just use a table classified as simple
# The classification of using two or more tables is complex
# I will give you 2 examples and generate answers based on their formats.
# Input: "How many heads of the departments are older than 56 ?"  
### Database:
          Table department, columns = [*,Department_ID,Name,Creation,Ranking,Budget_in_Billions,Num_Employees]
          Table head, columns = [*,head_ID,name,born_state,age]
          Table management, columns = [*,department_ID,head_ID,temporary_acting]
          Foreign_keys = [head.head_ID = management.head_ID, department.Department_ID = management.department_ID]
### Table Analysis:
(1. To select which tables you must indicate what columns of the table are selected.)
2. Tips：Use the grammatical structure of the example to compose inferences,According to <sub-problems> AND <Database information>, so <reasoning>)

#### management
- Columns are not relevant to filtering department heads by age; therefore, it is unnecessary for this question.----pass
#### head
- According to “heads of the departments” and [head.age], so we need to use the head table, which contains age. -----choose
#### department
- No direct connection to age; the table is not required to answer the question.----pass


### Selection Table:(According to table analysis, Output the table marked as choose)
- head

# The final Result:(strict output format)
- classified: [SIMPLE] 
- tables_choosed: [head]

# Input: "What are the distinct creation years of the departments managed by a secretary born in state 'Alabama'?"  
### Divide the problem into sub-problems: 
- "the distinct creation years"
- "the departments managed "
- "by a secretary born in state 'Alabama'?"

### Table Analysis:
#### head
- According to "by a secretary born in state 'Alabama'?" AND [head.born_state] we need to use the head table, which contains born_state. -----choose
#### department
- According to "the distinct creation years" AND [department.Creation], It seems that we need to use the department table.which contains Creation.----choose
#### management
- According to "the departments managed " AND [management.department_ID], It seems that we need to use the management table.which contains department_ID.----choose

### Selection Table:
- head
- department
- management
# The final Result:
- classified: [COMPLEX] 
- tables_choosed: [head, department, management]

'''

schema_choose_prompt = '''
You are an expert in NL2SQL problem mode linking.
# I will give you examples and generate answers based on their formats.
# Input: "How many heads of the departments are older than 56 ?"  
### Database:
          Table department, columns = [*,Department_ID,Name,Creation,Ranking,Budget_in_Billions,Num_Employees]
          Table head, columns = [*,head_ID,name,born_state,age]
          Table management, columns = [*,department_ID,head_ID,temporary_acting]
          Foreign_keys = [head.head_ID = management.head_ID, department.Department_ID = management.department_ID]
### Question Analysis:(Break the problem down into a series of sub-problems)
- Q1: "How many heads of the departments"
- Q2: "are older than 56 ?"
### Table Analysis:
(Select which tables you need to use and rate them on a scale of 0-1)
#### head(Relevance: 1)
- Contains 'age' column which is crucial for the age filter
- Contains head_ID which links to management table
- Primary table for finding heads' information
#### management(Relevance: 0.5)
- Links departments to heads through foreign keys
- Serves as a junction table
### department (Relevance: 0.3)
- While referenced through management table
- Not directly needed for this query since we only need count and age filter

### Selection Table:(According to table analysis, Output the table with score greater than 0.6)
- head
### Clumns Analysis:(Select the columns that may be used and make predictions about their probabilities)
From head table:
- age(1),Directly required for the age filter condition (> 56)
- head_ID (0.4),identifying unique heads, but other methods can also be used to calculate the number
- name(0),Not needed for this query as we only need count

# The final Result:(strict output format, Output all non-zero columns)
```text
tables[head(1)],colums[head.age(1),head.head_ID(0.4),head.name(0)]
```
'''

schema_linkings_prompt = '''Enter the problem and database information to give its schema link status,
# Scoring criteria:
Schema_linkings_table: (The table need to use to solve this problem)
### Input: "How many heads of the departments are older than 56 ?"  
### Divide the focus of pattern linking according to the problem: "heads""departments""older than 56"
### Database:
          Table department, columns = [*,Department_ID,Name,Creation,Ranking,Budget_in_Billions,Num_Employees]
          Table head, columns = [*,head_ID,name,born_state,age]
          Table management, columns = [*,department_ID,head_ID,temporary_acting]
          Foreign_keys = [head.head_ID = management.head_ID, department.Department_ID = management.department_ID]
### ALL Mode Linking Situations:
#### head
- according to “How many heads of the departments” we need to use the head table, which contains age. -----10
#### department
- according to "heads of the departments", It seems that we need to use the department table. However, the department table itself does not provide any information needed to answer the query on the age of the principal.----5
#### management
- This table does not appear in the question.And the management table itself does not provide any information needed to answer the query on the age of the principal.----0

# Results
Select all tables with scores greater than 5 and sort them by score
1.head
2.department
# AND Then give which columns in the above table should be used to solve the problem. If the given table cannot solve the problem, discard it.
- head:According to "older than 56",We can use the age column in the head table to solve the problem---------------[head.age]
- department:This table does not have any columns related to the question------------- wrong
# The final Result:(strict output format)
Answer: [tables:[head],columns:[head.age]]

'''
