import os
import gradio as gr
import base64
from langchain_groq import ChatGroq
from groq import Groq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Set environment variables
groq_api_key = os.environ.get("GROQ_API_KEY")
tavily_api_key = os.environ.get("TAVILY_API_KEY")

# Initialize Groq client
client = Groq(api_key=groq_api_key)

# Initialize ChatGroq
llm = ChatGroq(model_name="llama-3.2-11b-vision-preview")

# Set up search tool
search = TavilySearchResults(max_results=2, tavily_api_key=tavily_api_key)
tools = [search]

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# OCR function
def ocr(image_path):
    base64_image = encode_image(image_path)
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is the full product name in the image? just give me the product name"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model="llama-3.2-11b-vision-preview",
    )

    return chat_completion.choices[0].message.content



# Set up agent
prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Product analysis function
def analyze_product(product_name):
    query = f"""
    Analyze the product: {product_name}.
    Search for its ingredients, nutritional information, environmental impact, and customer reviews.
    Provide ratings on a scale of 1-10 for:
    1. Health impact
    2. Environmental sustainability
    3. Social responsibility
    output only the rating seperately for all the three aspects out of 10
    also give a short brief for each rating and a health related adjective to product based on the ratings.
    """
    analysis = agent_executor.invoke({"input": query})
    return analysis

def consumewise_analysis(product_name):
    analysis = analyze_product(product_name)
    return analysis

# Main processing function
def process_inputs(image):
    if image is not None:
        chatgpt_output = ocr(image)
    else:
        chatgpt_output = "No image provided."
    
    result = consumewise_analysis(chatgpt_output)
    return result['output']

# Create Gradio interface
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Image(sources=["upload", "webcam"], type="filepath")
    ],
    outputs=[
        gr.Textbox(label="Product Review")
    ],
    title="sehat.ai",
    description="Upload or capture an image of a product for analysis"
)

# Launch the interface
iface.launch()