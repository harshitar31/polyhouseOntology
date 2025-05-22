import os
from langchain.chains import OntotextGraphDBQAChain
from langchain_community.graphs import OntotextGraphDBGraph
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GRAPHDB_USERNAME"] = "admin"
os.environ["GRAPHDB_PASSWORD"] = "admin"

graph = OntotextGraphDBGraph(
    query_endpoint="http://localhost:7200/repositories/sample1",
    local_file="D:/VolumeEStuff/sony/Untitled.ttl"
)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

qa_chain = OntotextGraphDBQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    allow_dangerous_requests=True
)



"""Questions"""

"""
BOT ONTOLOGY
"""
# question="What are the polyhouses present?"
# question = "What are the grids present in the Polyhouse?"
# question = "What are the grids present in the Polyhouse 1?"
# question = "How many spaces are there in the polyhouse 2?"
# question = "How many elements are there in polyhouse 1?"
# question = "Are there cameras in the polyhouse 1?"
# question = "Are there any sensors in polyhouse p2?"
# question="What is the hierarchical structure of polyhouse?" #change label names
# question="What is the definition of edge node?"

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
question="can you tell me how the communication works in polyhouse 1 with the help of the interfaces?"
# question="Can you tell me how the data moves from the sensors?"
# question = "how are the sht25 sensors and central server connected?"
# question="What is the centralised server?"
# question="what are the connections to central server?"
# question="How the sensor data reaches the server?"
# question="can you verify if the sensor data from the end node goes to the edge node and then through the sim7600 goes to the server?"







response = qa_chain.invoke({qa_chain.input_key: question})

print("\nAnswer:", response[qa_chain.output_key])