import os
import pandas as pd
import folium
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema.runnable import RunnableLambda
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Initialize OpenAI model
llm = ChatOpenAI(model="gpt-4o-mini")

# Define the Pydantic Model
class GaelicCheckResponse(BaseModel):
    """Pydantic model for the response from the LLM."""
    origin: str = Field(..., description="Whether the place name has one of the following origins: Scottish Gaelic, Norse, Brittonic, Scots or English. Use only this list. If you are not sure say 'Not sure'.")
    reason: str = Field(..., description="Short explanation of your reason for selecting the place name origin.")

# Create output parser
parser = PydanticOutputParser(pydantic_object=GaelicCheckResponse)

# Define the prompt
prompt = PromptTemplate(
    template="""
    You are an expert in toponymy and origin of Scottish place names.
    Please determine the origin of the following Scottish place names.
    I will provide the place name and the historic county (for additional context) . 
    
    Place name: {place_name}
    Historic country: {historic_county}
    
    Respond in JSON format following this schema:
    {format_instructions}

    """,
    input_variables=["place_name", "historic_county"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Define processing pipeline
origin_chain = (
    prompt
    | llm
    | parser
    | RunnableLambda(lambda response: response.model_dump())
)

def process_place_name(place_name, historic_county):
    """Runs the LLM pipeline for a given place name and county."""
    try:
        result = origin_chain.invoke({'place_name': place_name,
                                      'historic_county': historic_county})
        return result["origin"], result["reason"]
    except Exception as e:
        return "Error", str(e)
    
def save_map(df):
    """Saves a map with markers for each place."""
    map_center = [df["lat"].mean(), df["long"].mean()]
    m = folium.Map(location=map_center, zoom_start=5)

    # Function to assign colors
    def get_marker_color(language):
        return {
            "Scottish Gaelic": "green",
            "Norse": "red",
            "Scots": "blue",
            "English": "purple",
            "Brittonic": "black"
        }.get(language, "gray")  # Default to gray if unknown

    # Add Markers
    for _, row in df.iterrows():
        folium.Marker(
            location=[row["lat"], row["long"]],
            popup=f"{row['place23nm']} - {row['reason']}",
            icon=folium.Icon(color=get_marker_color(row["origin"]))
        ).add_to(m)

    m.save("output/topollm_map_test.html")

def main():
    """Loads data, processes it, and saves results."""
    
    # Load dataset
    input_file = "place_data/IPN_GB_2024.csv"
    output_file = "output/IPN_GB_2024_with_origin_test.csv"
    
    df = pd.read_csv(input_file, encoding="latin-1").drop_duplicates(subset=["placeid"])
    df = df[(df.ctry23nm == "Scotland") & (df.descnm == "LOC")]
    df["placesort"] = df["placesort"].str.title()

    # Sample n rows for processing
    df = df.sample(10)

    # Apply processing function with progress bar
    tqdm.pandas()
    df[['origin', 'reason']] = df[['placesort', 'cty23nm']].progress_apply(
        lambda x: pd.Series(process_place_name(x['placesort'], x['cty23nm'])), axis=1
        )

    # Save results
    df.to_csv(output_file, index=False)
    save_map(df)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    main()
