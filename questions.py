import os
from langchain.chains import OntotextGraphDBQAChain
from langchain_community.graphs import OntotextGraphDBGraph
from langchain_google_genai import ChatGoogleGenerativeAI

# Step 1: Connect to your graph
graph = OntotextGraphDBGraph(
    query_endpoint="http://localhost:7200/repositories/sample1",
    local_file="D:/VolumeEStuff/sony/Untitled.ttl"
)

# Step 2: Initialize your LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Step 3: Simulate the question
question = "What are the polyhouses present?"

# Step 4: Manually write the SPARQL query (simulate what LangChain would generate)
sparql_query = """
PREFIX : <http://amrita.sony.org/terms#>
SELECT ?polyhouse WHERE {
  ?polyhouse a :Polyhouse .
}
"""

# Step 5: Run the SPARQL query and fetch the result directly
raw_result = graph.query(sparql_query)
print(raw_resultresult)

# Step 6: Parse the bindings to extract meaningful values
polyhouses = [
    binding["polyhouse"]["value"].split("#")[-1]  # Extracts "Polyhouse_P1" from URI
    for binding in raw_result["results"]["bindings"]
]

# Step 7: Create a human-readable intermediate prompt
intermediate_prompt = f"""
User Question: {question}
SPARQL Result: {', '.join(polyhouses)}
Please generate a natural language answer based on this result.
"""

print("Intermediate Prompt Sent to LLM:\n", intermediate_prompt)

# Step 8: Send to LLM manually
response = llm.invoke(intermediate_prompt)

print("\nAnswer:", response)



"""Questions"""

"""
BOT ONTOLOGY
"""
question="What are the polyhouses present?"
# question = "What are the grids present in the Polyhouse?"
# question = "What are the grids present in the Polyhouse 1?"
# question = "How many spaces are there in the polyhouse 2?"
# question = "How many elements are there in polyhouse 1?"
# question = "Are there cameras in the polyhouse 1?"
# question = "Are there any sensors in polyhouse p2?"
# question="What is the hierarchical structure of polyhouse?" #change label names
# question="What is the definition of edge node?"
# question="In which node(edge,end) is SHT25 2 connected?"

"""
SSN ONTOLOGY
"""
# question = "Did the cameras make any observation?"
# question="What type of observation does the sht25 sensor make?"
# question="What are all connected to Grid 1?"
# question="What are the times different observations were made?"
# question="What are the observations each sensor made?"
# question="Did any camera not make any observation in Polyhouse P1?" 
# question="Did any camera not make any observation?" 

"""
TOCO ONTOLOGY
"""
# question="What are the interfaces in Spresense?"
# question="Are there any wireless interfaces?"
# question="What are the interfaces in ESP32?"
# question="How many esp32 are connected to polyhouse 1?"
# question="can you tell me how the communication works in polyhouse 1 with the help of the interfaces?"
# question="Can you tell me how the data moves from the sensors?"
# question = "how are the sht25 sensors and central server connected?"
# question="What is the centralised server?"
# question="How the sensor data reaches the server?"
# question="what are the connections to centralised server?"
# question="can you verify if the sensor data from the end node goes to the edge node and then through the sim7600 goes to the server?"
# question="How is server connected to other devices?"




# question="In which node(edge or end) is SHT25 2 located in polyhouse 1?"
# question="Where is sht25 2 present in polyhouse 1?"


response = qa_chain.invoke({qa_chain.input_key: question})

print("\nAnswer:", response[qa_chain.output_key])