# Smart_Travel_Itinerary
# AI-Powered Travel Itinerary Planner

This is a smart, weather-aware travel itinerary generator powered by LLM agents (Groq’s LLaMA-3-70B) using LangGraph and LangChain. It allows users to generate personalized day trip plans for 50+ Indian and global cities with rich contextual awareness and multiple filters.

---

## Features

- **Real-time Weather Integration** (via OpenWeatherMap API)
- **LLM-Powered Personalization** using LangGraph & Groq
- **8+ User-Selectable Filters**:
  - Travel time estimation  
  - Cost estimation per stop  
  - Opening hours  
  - Kid-safe locations  
  - Dietary preferences (Veg, Halal, etc.)  
  - Transport suggestions  
  - Emergency info  
  - Offline-ready PDF export
- **Interactive HTML Maps** (via Folium)
- **Offline Support** via auto-generated PDF itineraries
- **Public Deployment** on Hugging Face Spaces

---

## Tech Stack

- **LLM Model:** [Groq LLaMA-3-70B](https://groq.com/)
- **Frameworks:** LangChain, LangGraph, Gradio
- **APIs:** OpenWeatherMap API, Google Maps API
- **Visualization:** Folium (maps), ReportLab (PDF)
- **Languages:** Python
- **Deployment:** Hugging Face Spaces

---

## Demo

Try it live here 👉 [Hugging Face Space](https://huggingface.co/spaces/Dhanishtajaggi/smart_travel_itinerary)

---

## Installation

```bash
git clone https://github.com/your-username/ai-travel-itinerary-planner.git
cd ai-travel-itinerary-planner
pip install -r requirements.txt
