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

sql_prompt = '''For a given problem, the database content and schema links generate SQL queries
### Input: "How many heads of the departments are older than 56 ?" 
### Database:
          Table department, columns = [*,Department_ID,Name,Creation,Ranking,Budget_in_Billions,Num_Employees]
          Table head, columns = [*,head_ID,name,born_state,age]
          Table management, columns = [*,department_ID,head_ID,temporary_acting]
          Foreign_keys = [head.head_ID=management.head_ID,department.Department_ID=management.department_ID] 
###schema_links: [table:[head],columns:[head.age]](attention schema linking)
### Analyze:
 It does this by querying the head table and applying the age filter to return the desired result. The simplicity of the query reflects the straightforward nature of the question, focusing solely on the age condition without needing to join other tables.
### SQL:
```sql
SELECT count(*) FROM head WHERE age  >  56
```
For a given problem, the database content and schema links generate SQL queries
### Input: "Show the album names, ids and the number of tracks for each album."
### Database:
          Table Album, columns = [*,AlbumId,Title,ArtistId]
          Table Artist, columns = [*,ArtistId,Name,CustomerId,FirstName,LastName,Company,Address,City,State,Country,PostalCode,Phone,Fax,Email,SupportRepId]
          Table Customer, columns = [*,EmployeeId,LastName,FirstName,Title,ReportsTo,BirthDate,HireDate,Address,City,State,Country,PostalCode,Phone,Fax,Email]
          Table Track, columns = [*,TrackId,Name,AlbumId,MediaTypeId,GenreId,Composer,Milliseconds,Bytes,UnitPrice]
          Foreign_keys = [Album.ArtistId=Artist.ArtistId,Artist.SupportRepId=Customer.EmployeeId,Track.AlbumId=Album.AlbumId] 
###schema_links: [table:[Track,Album],columns:[Track.AlbumId,Album.Title]](attention schema linking)
### Analyze:
 It does this by querying the head table and applying the age filter to return the desired result. The simplicity of the query reflects the straightforward nature of the question, focusing solely on the age condition without needing to join other tables.
### SQL:
```sql
SELECT T1.Title ,  T2.AlbumID ,  COUNT(*) FROM ALBUM AS T1 JOIN TRACK AS T2 ON T1.AlbumId  =  T2.AlbumId GROUP BY T2.AlbumID
```

'''

easy_prompt_gpt3_5 = '''
### Input: "How many heads of the departments are older than 56 ?" 
### Database:   Table head, columns = [*,head_ID,name,born_state,age]
###schema_links: [According to "heads of the departments are older than", columns [head.age] will be used.Values [56] will be used.]
###Analyze
To answer this question, we need a query that selects the number of entries in the header table and applies the condition of age greater than 56.
###The SQL query(strict output format)
```sql
SELECT count(*) FROM head WHERE age  >  56;
```

### Input: "List the name, born state and age of the heads of departments ordered by age." 
### Database:   Table department, columns = [*,Department_ID,Name,Creation,Ranking,Budget_in_Billions,Num_Employees]
###schema_links: [According to "heads of departments" column: [head.born_state] will be used, According to "name , born state and age of the heads of"columns: [head.name] will be used, According to "age of the heads" column: [head.age] will be used]
###Analyze
To answer this question, we need a query that selects the number of entries in the department table and sorts them by age.
###The SQL query(strict output format)
```sql
SELECT name ,  born_state ,  age FROM head ORDER BY age;
```

'''

debug_prompt = '''
<EXAMPLE>
 1. **First Candidate**: 
   ```sql
   SELECT T2.Name, COUNT(T1.concert_ID) AS Number_of_Concerts 
   FROM stadium AS T2 
   LEFT JOIN concert AS T1 ON T2.Stadium_ID = T1.Stadium_ID 
   GROUP BY T2.Name;
   ```
   - This query uses a `LEFT JOIN`, which is appropriate for counting concerts per stadium, even if some stadiums have no concerts. 
   - The `GROUP BY` clause correctly groups by the stadium name.
   - The column alias for the count is appropriately defined.

2. **Second Candidate**: 
   ```sql
   SELECT T2.Name, COUNT(T1.concert_ID) AS number_of_concerts  
   FROM stadium AS T2 
   JOIN concert AS T1 ON T2.Stadium_ID = T1.Stadium_ID 
   GROUP BY T2.Name;
   ```
   - This query uses an `INNER JOIN`. This means if a stadium has no concerts, it won't appear in the results, which is not what we want when counting concerts per stadium.
   - The `GROUP BY` and alias are correct, but the join type is inappropriate for the requirement.

3. **Third Candidate**: 
   ```sql
   SELECT stadium.Name, COUNT(concert.concert_ID) AS number_of_concerts  
   FROM stadium  
   LEFT JOIN concert ON stadium.Stadium_ID = concert.Stadium_ID  
   GROUP BY stadium.Name;
   ```
   - This query also uses a `LEFT JOIN`, which is appropriate.
   - The syntax is correct, grouping by the stadium name and counting concerts.
   - The alias for the count adheres to the standard naming convention.

### Final Choose:
Given the analysis, the first and third SQL statements are correct due to their use of a `LEFT JOIN`. Among these, I will choose the **third candidate** as it directly uses the `stadium` and `concert` aliases without introducing additional aliasing (T2 and T1).

**Final Choose:** (strict output format)
```output
SELECT stadium.Name, COUNT(concert.concert_ID) AS number_of_concerts  
FROM stadium  
LEFT JOIN concert ON stadium.Stadium_ID = concert.Stadium_ID  
GROUP BY stadium.Name;
```
</EXAMPLE>
'''

schema_filter_prompt = '''
### Input: "How many stadiums do we have?" 
### schema_filter: (The earlier the table or column, the more likely it is to be used.)  
          Table stadium, columns = [Name,Capacity,Location]
          Table singer, columns = [Name,Singer_ID,Is_male]
          Table concert, columns = [Theme,concert_Name,Year]
### Foreign key information：
          Foreign_keys = [concert.Stadium_ID = stadium.Stadium_ID]
###Analyze
- The question "How many stadiums do we have?" implies counting entries in the stadium table.
- No additional filtering conditions are required since the question is general.
###The SQL query(strict output format)
```sql
SELECT COUNT(*) FROM stadium;
```
### Input: "Which flight numbers correspond to Southwest Airlines flights?"
### schema_filter:
          Table airlines, columns = [Airline,uid,Abbreviation]
          Table flights, columns = [Airline,FlightNo,DestAirport]
          Table airports, columns = [Country,CountryAbbrev,AirportName]
### Foreign key information：
          Foreign_keys = [flights.SourceAirport = airports.AirportCode, flights.DestAirport = airports.AirportCode]
###Analyze
- flights.Airline must match airlines.uid based on logical inference.
- Filter by Airline = 'Southwest Airlines'.
- Use the retrieved uid to filter flights in the flights table.
###The SQL query(strict output format)
```sql
SELECT T1.FlightNo FROM FLIGHTS AS T1 JOIN AIRLINES AS T2 ON T2.uid  =  T1.Airline WHERE T2.Airline  =  "Southwest Airlines";
```
'''

easy_prompt_example_1_less = '''
### Input: "How many heads of the departments are older than 56 ?" 
###Analyze
To answer the question, we need a query that:
1.Selects the count of entries in the head table.
2.Applies a condition that age is greater than 56.
###The SQL query(strict output format)
```sql
SELECT count(*) FROM head WHERE age  >  56;
```
'''

complex_prompt_example_1_less = '''
### Input: "What are the distinct creation years of the departments managed by a secretary born in state 'Alabama'?" 
###Analyze(Distinguish whether to use nested or JOIN queries)
- The query asks for information that needs to be merged from multiple tables, so a JOIN query is more appropriate.
###Connectivity and Filtering Relationships
- Use the department_ID and head_ID in the management table to connect to the department and head tables, respectively.
- The filter is head.born_state = 'Alabama'.
###The SQL query(strict output format)
```sql
SELECT DISTINCT T1.creation FROM department AS T1
JOIN management AS T2 ON T1.department_id  =  T2.department_id
JOIN head AS T3 ON T2.head_id  =  T3.head_id
WHERE T3.born_state  =  'Alabama';
```
'''

complex_prompt_example_2_less = '''
### Input: "What are the distinct creation years of the departments managed by a secretary born in state 'Alabama'?" 
###Analyze(Distinguish whether to use nested or JOIN queries)
- The query asks for information that needs to be merged from multiple tables, so a JOIN query is more appropriate.
###Connectivity and Filtering Relationships
- Use the department_ID and head_ID in the management table to connect to the department and head tables, respectively.
- The filter is head.born_state = 'Alabama'.
###The SQL query(strict output format)
```sql
SELECT DISTINCT T1.creation FROM department AS T1
JOIN management AS T2 ON T1.department_id  =  T2.department_id
JOIN head AS T3 ON T2.head_id  =  T3.head_id
WHERE T3.born_state  =  'Alabama';
```
###Input:"How many departments are led by heads who are not mentioned?"
###Analyze(Distinguish whether to use nested or JOIN queries)
- We just need to find the records in the management table that are missing head_ID and correlate them with the department table to calculate the number of departments that match the criteria.
###Connectivity and Filtering Relationships
- What needs to be counted is the number of eligible "sectors".
- The filter is "no mention of head" .
###The SQL query(strict output format)
```sql
SELECT count(*) FROM department 
WHERE department_id NOT IN (SELECT department_id FROM management);
```
'''

easy_prompt_example_2_less = '''
### Input: "How many heads of the departments are older than 56 ?" 
###Analyze
To answer the question, we need a query that:
1.Selects the count of entries in the head table.
2.Applies a condition that age is greater than 56.
###The SQL query(strict output format)
```sql
SELECT count(*) FROM head WHERE age  >  56;
```
### Input: "How many hosts does each nationality have? List the nationality and the count." 
###Analyze
To answer the question, we need a query that:
1.The query should group rows by the Nationality column to segregate the hosts based on their nationality.
2.Counts the hosts, for each nationality group, count the number of hosts.

###The SQL query(strict output format)
```sql
SELECT Nationality ,  COUNT(*) FROM HOST GROUP BY Nationality;
```
'''

easy_prompt_example_3_less = '''
### Input: "How many heads of the departments are older than 56 ?" 
###Analyze
To answer the question, we need a query that:
1.Selects the count of entries in the head table.
2.Applies a condition that age is greater than 56.
###The SQL query(strict output format)
```sql
SELECT count(*) FROM head WHERE age  >  56;
```
### Input: "How many hosts does each nationality have? List the nationality and the count." 
###Analyze
To answer the question, we need a query that:
1.The query should group rows by the Nationality column to segregate the hosts based on their nationality.
2.Counts the hosts, for each nationality group, count the number of hosts.

###The SQL query(strict output format)
```sql
SELECT Nationality ,  COUNT(*) FROM HOST GROUP BY Nationality;
```
### Input: "what is the highest point in the united states" 
### Analyze
To answer the question, we need a query that:
###The SQL query(strict output format)
- The question seeks the highest point in the United States, which is mapped to the column highest_point.
- Since the query is about the entire United States, aggregate across all states to find the record with the highest elevation (highest_elevation) and retrieve the corresponding highest_point.
```sql
SELECT highest_point FROM highlow WHERE highest_elevation  =  ( SELECT MAX ( highest_elevation ) FROM highlow );
```
'''

complex_prompt_example_3_less = '''
### Input: "What are the distinct creation years of the departments managed by a secretary born in state 'Alabama'?" 
###Analyze(Distinguish whether to use nested or JOIN queries)
- The query asks for information that needs to be merged from multiple tables, so a JOIN query is more appropriate.
###Connectivity and Filtering Relationships
- Use the department_ID and head_ID in the management table to connect to the department and head tables, respectively.
- The filter is head.born_state = 'Alabama'.
###The SQL query(strict output format)
```sql
SELECT DISTINCT T1.creation FROM department AS T1
JOIN management AS T2 ON T1.department_id  =  T2.department_id
JOIN head AS T3 ON T2.head_id  =  T3.head_id
WHERE T3.born_state  =  'Alabama';
```
###Input:"How many departments are led by heads who are not mentioned?"
###Analyze(Distinguish whether to use nested or JOIN queries)
- We just need to find the records in the management table that are missing head_ID and correlate them with the department table to calculate the number of departments that match the criteria.
###Connectivity and Filtering Relationships
- What needs to be counted is the number of eligible "sectors".
- The filter is "no mention of head" .
###The SQL query(strict output format)
```sql
SELECT count(*) FROM department 
WHERE department_id NOT IN (SELECT department_id FROM management);
```
###Input:"List the states where both the secretary of 'Treasury' department and the secretary of 'Homeland Security' were born."
###Analyze(Distinguish whether to use nested or JOIN queries)
- We need to combine data from multiple tables (department, management, head).
###Connectivity and Filtering Relationships
- Connect department to management using department.Department_ID = management.department_ID.
- Connect management to head using management.head_ID = head.head_ID.
- Filter for departments named "Treasury" and "Homeland Security" from the department table.
- Use these Department IDs in the management table to get the associated head_IDs.
- Finally, use the head_IDs in the head table to retrieve the born_state.
###The SQL query(strict output format)
```sql
SELECT h1.born_state FROM department d1 JOIN management m1 ON d1.Department_ID = m1.department_ID JOIN head h1 ON m1.head_ID = h1.head_ID WHERE d1.Name = 'Treasury' INTERSECT SELECT h2.born_state FROM department d2 JOIN management m2 ON d2.Department_ID = m2.department_ID JOIN head h2 ON m2.head_ID = h2.head_ID WHERE d2.Name = 'Homeland Security';
```
'''
complex_prompt_no_pf = '''
### Input: "What are the distinct creation years of the departments managed by a secretary born in state 'Alabama'?" 
### Database:   
          Table department, columns = [*,Department_ID,Name,Creation,Ranking,Budget_in_Billions,Num_Employees]
          Table head, columns = [*,head_ID,name,born_state,age]
          Table management, columns = [*,department_ID,head_ID,temporary_acting]
###schema_links: [According to "years", tables [management] will be used, According to "creation", columns [department.Creation] will be used,According to "born in state 'Alabama ' ?", columns [head.born_state] will be used,Values [Alabama] will be used.]
###Analyze(Distinguish whether to use nested or JOIN queries)
- The query asks for information that needs to be merged from multiple tables, so a JOIN query is more appropriate.
###Connectivity and Filtering Relationships
- Use the department_ID and head_ID in the management table to connect to the department and head tables, respectively.
- The filter is head.born_state = 'Alabama'.
###The SQL query(strict output format)
```sql
SELECT DISTINCT T1.creation FROM department AS T1
JOIN management AS T2 ON T1.department_id  =  T2.department_id
JOIN head AS T3 ON T2.head_id  =  T3.head_id
WHERE T3.born_state  =  'Alabama';
```
###Input:"How many departments are led by heads who are not mentioned?"
### Database:   
        Table department, columns = [*,Department_ID,Name,Creation,Ranking,Budget_in_Billions,Num_Employees]
        Table management, columns = [*,department_ID,head_ID,temporary_acting]
###schema_links: [According to "departments", columns [management.department_ID] will be used,According to "departments", columns [department.Department_ID] will be used]
###Analyze(Distinguish whether to use nested or JOIN queries)
- We just need to find the records in the management table that are missing head_ID and correlate them with the department table to calculate the number of departments that match the criteria.
###Connectivity and Filtering Relationships
- What needs to be counted is the number of eligible "sectors".
- The filter is "no mention of head" .
###The SQL query(strict output format)
```sql
SELECT count(*) FROM department 
WHERE department_id NOT IN (SELECT department_id FROM management);
```
'''

