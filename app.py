import os, requests, folium, html, traceback
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import gradio as gr
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from typing import TypedDict, Annotated, List, Any


# Load secrets from environment (set in Hugging Face repo settings)
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY")
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

llm = ChatGroq(
    temperature=0,
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-70b-8192"
)

class PlannerState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], "conversation messages"]
    city: str
    interests: List[str]
    itinerary: str
    weather_info: str
    preferences: List[str]

def get_weather(city: str) -> str:
    try:
        data = requests.get(
            f"https://api.openweathermap.org/data/2.5/weather",
            params={"q": city, "appid": WEATHER_API_KEY, "units": "metric"}
        ).json()
        return f"{data['weather'][0]['description'].capitalize()}, {data['main']['temp']}°C"
    except:
        return "Unknown weather"

def create_itinerary(state: PlannerState) -> PlannerState:
    weather = get_weather(state["city"])
    prefs = state["preferences"]

    # Base prompt
    base_instructions = [
        "- travel time between stops" if "Show travel time" in prefs else "",
        "- transport suggestions" if "Include transport suggestions" in prefs else "",
        "- opening hours" if "Show opening hours" in prefs else "",
        "- cost estimation" if "Include cost estimates" in prefs else "",
        "- dietary accommodations" if "Dietary filter (veg/non-veg)" in prefs else "",
        "- kid-friendly notes" if "Only kid-safe places" in prefs else "",
        "- emergency info" if "Include emergency info" in prefs else ""
    ]

    # Clean empty lines
    filtered_instructions = [item for item in base_instructions if item]
    bullet_list = "\n".join(filtered_instructions)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a smart travel assistant."),
        ("human", f"""City: {state['city']}
Weather: {weather}
Interests: {', '.join(state['interests'])}
Preferences: {', '.join(prefs)}

Generate a day trip itinerary with:
{bullet_list}
Use bullet points.""")
    ])

    resp = llm.invoke(prompt.format_messages())
    return {**state, "messages": state["messages"] + [AIMessage(content=resp.content)], "itinerary": resp.content}

interface = gr.Interface(
    fn=run_planner,
    inputs=[
        gr.Textbox(label="City", placeholder="e.g. Mumbai", lines=1),
        gr.Textbox(label="Interests (comma-separated)", 
                 placeholder="e.g. food, culture, shopping", lines=1),
        gr.CheckboxGroup(
            label="Preferences",
            choices=[
                "Show travel time", "Include transport suggestions", 
                "Only kid-safe places", "Include cost estimates", 
                "Dietary filter (veg/non-veg)", "Show opening hours", 
                "Include emergency info"
            ]
        ),
        gr.Checkbox(label="Download itinerary as PDF")
    ],
    outputs=[
        gr.Markdown(label="Itinerary"),
        gr.HTML(label="Map"),
        gr.File(label="Download PDF")
    ],
    title="✈️ Smart Travel Itinerary Planner",
    description="Generate optimized travel plans with AI-powered recommendations",
    theme=gr.themes.Base(
        primary_hue="cyan",
        secondary_hue="gray",
        neutral_hue="slate",
    ),
    css="""
        :root {
            --dark-1: #0f172a;
            --dark-2: #1e293b;
            --dark-3: #334155;
            --accent: #06b6d4;
            --text: #e2e8f0;
            --text-light: #94a3b8;
        }
        
        .gradio-container {
            background: var(--dark-1) !important;
            color: var(--text) !important;
            font-family: 'Inter', system-ui, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            border-radius: 12px;
        }
        
        .gradio-header {
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--dark-3);
            margin-bottom: 1.5rem;
        }
        
        h1 {
            font-weight: 700;
            color: white !important;
            letter-spacing: -0.025em;
        }
        
        .gradio-description {
            color: var(--text-light) !important;
            font-size: 0.95rem;
        }
        
        .gr-input, .gr-textbox {
            background: var(--dark-2) !important;
            color: var(--text) !important;
            border: 1px solid var(--dark-3) !important;
            border-radius: 8px;
            padding: 12px 16px !important;
        }
        
        .gr-input:focus, .gr-textbox:focus {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 2px rgba(6, 182, 212, 0.2) !important;
            outline: none;
        }
        
        .gr-textbox::placeholder {
            color: var(--text-light) !important;
            opacity: 0.7;
        }
        
        .gr-button {
            background: var(--accent) !important;
            color: var(--dark-1) !important;
            border: none !important;
            border-radius: 8px;
            padding: 12px 24px !important;
            font-weight: 600 !important;
            transition: all 0.2s;
        }
        
        .gr-button:hover {
            background: #0891b2 !important;
            transform: translateY(-1px);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .gr-checkbox-group, .gr-checkbox {
            background: var(--dark-2) !important;
            border: 1px solid var(--dark-3) !important;
            border-radius: 8px;
            padding: 16px !important;
        }
        
        .gr-checkbox-group label, .gr-checkbox label {
            color: var(--text) !important;
        }
        
        .gr-checkbox-group .selected {
            background: var(--dark-3) !important;
        }
        
        .gr-markdown, .gr-html {
            background: var(--dark-2) !important;
            border: 1px solid var(--dark-3) !important;
            border-radius: 8px;
            padding: 20px !important;
        }
        
        .gr-file {
            background: var(--dark-2) !important;
            border: 1px solid var(--dark-3) !important;
            border-radius: 8px;
            padding: 12px !important;
        }
        
        .gr-label {
            font-weight: 600;
            color: var(--text) !important;
            margin-bottom: 8px;
            font-size: 0.95rem;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--dark-2);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--dark-3);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--accent);
        }
    """,
    allow_flagging="never"
)
interface.launch()
