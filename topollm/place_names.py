import os
import pandas as pd
import folium
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema.runnable import RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
import logging

# Load environment variables
load_dotenv()

# Initialize Anthropic model
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# Define the Pydantic Model
class PlaceNameOriginResponse(BaseModel):
    """Pydantic model for the response from the LLM."""
    origin: Literal["Scottish Gaelic", "Norse", "Brittonic", "Scots", "English", "Unsure"] = Field(
        ..., 
        description="The linguistic origin of the place name. Must be one of the specified values."
    )
    confidence: Literal["High", "Medium", "Low"] = Field(
        ...,
        description="Your confidence level in this classification"
    )
    reason: str = Field(
        ..., 
        description="Short explanation including linguistic elements, etymology, and historical context that support your classification",
    )

# Create output parser
parser = PydanticOutputParser(pydantic_object=PlaceNameOriginResponse)

# Define the prompt
prompt = PromptTemplate(
    template="""
    You are a leading expert in Scottish toponymy with deep knowledge of Celtic, Germanic, and Romance linguistic influences on place names.
    
    Analyze this Scottish place name for its linguistic origin:
    
    Place Name: {place_name}
    Historic County: {historic_county}
    
    Consider these linguistic patterns:
    - Scottish Gaelic: prefixes like 'bal-', 'inver-', 'glen-', 'dun-', 'craig-'; suffixes like '-more', '-beg'
    - Norse: elements like '-by', '-thorp', '-wick', '-dale', '-fell', '-force'
    - Brittonic/Welsh: 'llan-', 'aber-', 'pen-', 'tre-', 'caer-'
    - Scots: Germanic/Anglo-Saxon elements, often phonetically altered
    - English: Standard English toponymic elements
    
    Historical context: Consider the geographic location, known settlement patterns, and linguistic layering in Scottish place names.
    
    Provide your analysis with:
    1. Your confidence level (High/Medium/Low)
    3. Detailed reasoning based on etymological evidence
    
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

def process_place_name(place_name: str, historic_county: str) -> tuple[str, str]:
    """Runs the LLM pipeline for a given place name and county.
    
    Args:
        place_name: The name of the place to analyze
        historic_county: The historic county for context
        
    Returns:
        Tuple of (origin, reason)
    """
    try:
        result = origin_chain.invoke({
            'place_name': place_name,
            'historic_county': historic_county
        })
        # Return additional fields for richer analysis
        return (result["origin"], result["reason"], 
                result.get("confidence", "Low"))
    except Exception as e:
        logging.error(f"Error processing {place_name}: {str(e)}")
        return "Error", str(e), "Low"
    
def save_map(df: pd.DataFrame, output_path: str = "output/topollm_map.html") -> None:
    """Saves a map with markers for each place.
    
    Args:
        df: DataFrame containing place data with lat, long, place23nm, reason, origin columns
        output_path: Path to save the map HTML file
    """
    if df.empty:
        logging.warning("DataFrame is empty, cannot create map")
        return
        
    map_center = [df["lat"].mean(), df["long"].mean()]
    m = folium.Map(location=map_center, zoom_start=6)

    # Color mapping for different origins
    ORIGIN_COLORS = {
        "Scottish Gaelic": "green",
        "Norse": "red",
        "Scots": "blue",
        "English": "purple",
        "Brittonic": "black",
        "Unsure": "orange",
        "Error": "gray"
    }

    def get_marker_color(origin: str) -> str:
        return ORIGIN_COLORS.get(origin, "gray")

    # Add markers with improved popups
    for _, row in df.iterrows():
        popup_text = f"<b>{row['place23nm']}</b><br>Origin: {row['origin']}<br>Reason: {row['reason']}"
        folium.Marker(
            location=[row["lat"], row["long"]],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color=get_marker_color(row["origin"]))
        ).add_to(m)

    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 150px; height: 180px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>Place Name Origins</b></p>
    '''
    for origin, color in ORIGIN_COLORS.items():
        legend_html += f'<p><i class="fa fa-circle" style="color:{color}"></i> {origin}</p>'
    legend_html += '</div>'
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m.save(output_path)
    logging.info(f"Map saved to {output_path}")

def main(sample_size: int = 10, input_file: str = "data/IPN_GB_2024.csv", 
         output_file: str = "output/IPN_GB_2024_origin.csv") -> None:
    """Loads data, processes it, and saves results.
    
    Args:
        sample_size: Number of places to process
        input_file: Path to input CSV file
        output_file: Path to output CSV file
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Load and filter dataset
        logging.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file, encoding="latin-1")
        
        # Remove duplicates and filter for Scottish locations
        df = df.drop_duplicates(subset=["placeid"])
        df = df[(df.ctry23nm == "Scotland") & (df.descnm == "LOC")]
        
        if df.empty:
            logging.error("No Scottish locations found in dataset")
            return
            
        # Clean place names
        df["placesort"] = df["placesort"].str.title()
        
        # Sample data for processing
        if len(df) > sample_size:
            df = df.sample(sample_size, random_state=42)  # Set seed for reproducibility
            logging.info(f"Sampling {sample_size} places from {len(df)} available")
        
        # Process place names with progress bar
        logging.info("Processing place names...")
        tqdm.pandas(desc="Analyzing origins")
        df[['origin', 'reason', 'confidence']] = df[['placesort', 'cty23nm']].progress_apply(
            lambda row: pd.Series(process_place_name(row['placesort'], row['cty23nm'])), 
            axis=1
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save results
        df.to_csv(output_file, index=False)
        save_map(df)
        
        # Print summary statistics
        origin_counts = df['origin'].value_counts()
        logging.info(f"Processing complete. Origin distribution:")
        for origin, count in origin_counts.items():
            logging.info(f"  {origin}: {count}")
            
        logging.info(f"Processed data saved to {output_file}")
        
    except FileNotFoundError:
        logging.error(f"Input file {input_file} not found")
    except Exception as e:
        logging.error(f"Error in main processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
