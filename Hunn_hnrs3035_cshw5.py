import argparse
import os
import chromadb
import json
from openai import OpenAI
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv(".env")

# establish OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load dataset
with open('dev-v2.0 (1).json', 'r') as f:
    data = json.load(f)

chroma_client = chromadb.PersistentClient(path="../data/my_chromadb")

# Establish embedding functions
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

# Get the collection
collection = chroma_client.get_collection(
    name="contexts",
    embedding_function=openai_ef
)

# Store the context from the dataset for the vector embeddings
context = []
for topic in data["data"]:
    for paragraph in topic["paragraphs"]:
        context.append(paragraph["context"])

# Store the questions and answers from the dataset
q_and_as = {}
for topic in data["data"]:
    for paragraph in topic["paragraphs"]:
        for qas in paragraph["qas"]:
            if not qas["is_impossible"]:
                question = qas["question"]
                q_id = qas["id"]
                ans = []
                for answer in qas["answers"]:
                    ans.append(answer["text"])
                q_and_as[(question, q_id)] = ans
            if len(q_and_as) >= 500:
                break
        if len(q_and_as) >= 500:
            break
    if len(q_and_as) >= 500:
        break

class AIAgent:
    """
    Class AIAgent implements an AI Agent that, based on given commands, can choose between three tools to perform
    RAG, Summarization, and Joke Telling.

    attribute:
        agents_data: stores the name and chat history of the current agent being used
    """

    agents_data = {}

    def __init__(self, name='DefaultAgent'):
        self.name = name
        self.chat_history = []
        if len(AIAgent.agents_data) > 0:
            self.load_agent_data()

    def load_agent_data(self):
        """
        load_agent_data simply retrieves the chat history of the current agent if it exists
        """

        if self.name in AIAgent.agents_data:
            self.chat_history = AIAgent.agents_data[self.name].get("chat_history", [])


    def select_tool(self, prompt: str):
        """
        select_tool, with gpt-4o-mini, chooses a tool (RAG, Summarization, or Joke Telling) to perform based on a user
        prompt

        :param prompt: the prompt given by the user
        """

        tool_selector_prompt = """Given a prompt from a user, formulate a response that selects one of three tools to use.
                                
                                Tools:
                                1. RAG: If you determine RAG, just respond with "RAG".
                                2. Summarize: If you determine Summarize, just respond with "Summarize".
                                3. Additional Tool: If you determine Additional Tool, just respond with "Tool".
                                
                                Guidelines:
                                1. If the user asks a question, you should select RAG.
                                2. If the user tells you to summarize something, you should select Summarize.
                                3. If the user asks for a joke, you should select Additional Tool.
                                4. You should only ever give one of these three answers.
                                
                                User Prompt:
                                {prompt}"""

        message = [
            {"role": "system", "content": tool_selector_prompt.format(prompt=prompt)},
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=20,
            messages=message,
        )

        if (response.choices[0].message.content == "RAG"):
            self.rag(prompt)
        elif response.choices[0].message.content == "Summarize":
            line = prompt.strip().split()
            filename = line[1]
            with open(filename, "r", encoding='utf-8') as f:
                file_contents = f.read()
            self.summarize(file_contents, prompt)
        elif response.choices[0].message.content == "Tool":
            self.joke(prompt)


    def joke(self, prompt: str):
        """
        joke combines a user prompt and a joke prompt and provides it to gpt-4o-mini to create a joke based on a
        specified topic. Then stores the prompt and response in the chat history

        :param prompt: user prompt
        """

        joke_prompt = """Craft the most hilarious joke you can think of pertaining to a certain topic the user will provide.
                        You are not to use basic jokes. Use raunchy language and unconventional joke styles. Also, make
                        them short (max 2 sentences). DONT mention ladders at all ever. 
        
                        User Prompt:
                        {prompt}"""

        message = [
            {"role": "system", "content": joke_prompt.format(prompt=prompt)},
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=100,
            messages=message,
            temperature=0.8
        )

        print(response.choices[0].message.content)
        self.chat_history.append({"prompt": prompt, "context":response.choices[0].message.content})


    def summarize(self, prompt: str, tool_prompt: str):
        """
        summarize uses a Chain of Density prompt in order to create a summary of a file provided by the user. Then
        stores the prompt and response in the chat history

        :param prompt: a str of the text
        :param tool_prompt: the name of the file provided
        :return:
        """

        cod_prompt = """Summarize the following article using Chain of Density.
        
                        Article: {article}"""


        message = [
            {"role": "system", "content": cod_prompt.format(article=prompt)},
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=1000,
            messages=message,
        )

        self.chat_history.append({"prompt": tool_prompt, 'context': response.choices[0].message.content})



    def rag(self, prompt: str):
        """
        rag performs a retrieval augmented generation search on the chromadb vector database created above based on a
        question provided by the user. If the prompt is not the first chat to the agent, it first gives the chat history
        to gpt-4o-mini and contextualizes the prompt along with the history. It will then perform normal RAG based on
        the contextualized output

        :param prompt: user's question
        """

        if len(self.chat_history) > 0:

            contextualized_prompt = """Given a chat history and the latest user question which might reference context in 
            the chat history, formulate a standalone question which can be understood without the chat history. Do NOT 
            answer the question, just reformulate if needed and otherwise return as is: 
        
            Chat History: 
            {history}
        
            Latest Question: 
            {question}"""

            message = [
                {"role": "system", "content": contextualized_prompt.format(history=self.chat_history, question=prompt)}
            ]

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=100,
                messages=message,
            )

            contextualized = response.choices[0].message.content

            system_prompt = """You are a genius AI assistant that answers questions concisely using data returned by a search engine.

                                Guidelines:
                                \t1. You will be provided with a question by the user, you must answer that question, and nothing else.
                                \t2. Your answer should come directly from the provided context from the search engine.
                                \t3. Do not make up any information not provided in the context.
                                \t4. If the provided question does not contain the answers, respond with 'I am sorry, but I am unable to answer that question.'
                                \t5. Be aware that some chunks in the context may be irrelevant, incomplete, and/or poorly formatted.

                                Here is the provided context:
                                {context}

                                Here is the question: {question}

                                Your response: """

            result = collection.query(query_texts=contextualized, n_results=5)

            message = [
                {"role": "system", "content": system_prompt.format(context=result['documents'], question=contextualized)}
            ]

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=100,
                messages=message,
            )

            print(response.choices[0].message.content)
            self.chat_history.append({'prompt': prompt, 'context': response.choices[0].message.content})
        else:
            system_prompt = """You are a genius AI assistant that answers questions concisely using data returned by a search engine.

                    Guidelines:
                    \t1. You will be provided with a question by the user, you must answer that question, and nothing else.
                    \t2. Your answer should come directly from the provided context from the search engine.
                    \t3. Do not make up any information not provided in the context.
                    \t4. If the provided question does not contain the answers, respond with 'I am sorry, but I am unable to answer that question.'
                    \t5. Be aware that some chunks in the context may be irrelevant, incomplete, and/or poorly formatted.

                    Here is the provided context:
                    {context}

                    Here is the question: {question}

                    Your response: """

            result = collection.query(query_texts=prompt, n_results=5)

            message = [
                {"role": "system", "content": system_prompt.format(context=result['documents'], question=prompt)}
            ]

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=100,
                messages=message,
            )

            print(response.choices[0].message.content)
            self.chat_history.append({'prompt': prompt, 'context': response.choices[0].message.content})

    def save_to_file(self, filename="agents.json"):
        """
        saves the agent's name along with full chat history to a file

        :param filename: file to store data
        """

        AIAgent.agents_data[self.name] = {'name': self.name, 'chat_history': self.chat_history}

        with open(filename, "w", encoding='utf-8') as f:
            json.dump(AIAgent.agents_data, f, indent=4)

    @classmethod
    def load_from_file(cls, agent_name: str):
        cls.load_all_agents()
        return cls(agent_name)

    @classmethod
    def load_all_agents(cls, filename="agents.json"):
        """
        Loads the agents_data with the data stored in the json file

        :param filename: file with data stored in it
        :return:
        """

        if os.path.exists(filename): # Check if the file exists
            with open(filename, "r", encoding='utf-8') as f:
                cls.agents_data = json.load(f)
                print(cls.agents_data)
        else:
            cls.agents_data = {}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Simple AI Agent",
                                     description="Simple AI Agent")
    parser.add_argument('s', type=str,
                        help='Specify which agent you want to use.')
    parser.add_argument('--p', type=str,
                        help='The prompt to be given to the AI Agent')
    parser.add_argument('--new', type=str,
                        help='Creates a new instance of an agent.')


    args = parser.parse_args()

    if args.new:
        agent = AIAgent(args.new)
        agent.save_to_file()
        print(f"Created new agent {args.s}. Now give it a query.")
    else:
        agent = AIAgent.load_from_file(args.s)
    if args.p:
        agent.select_tool(args.p)
        agent.save_to_file()