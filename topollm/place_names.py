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
import asyncio
from tqdm.asyncio import tqdm as atqdm
import time
import random
import click

# Load environment variables
load_dotenv()

# Initialize Anthropic model
llm = ChatAnthropic(model="claude-sonnet-4-20250514")

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

class BatchPlaceNameResponse(BaseModel):
    """Pydantic model for batch responses from the LLM."""
    analyses: list[PlaceNameOriginResponse] = Field(
        ...,
        description="List of place name analyses in the same order as input"
    )

# Create output parsers
parser = PydanticOutputParser(pydantic_object=PlaceNameOriginResponse)
batch_parser = PydanticOutputParser(pydantic_object=BatchPlaceNameResponse)

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

# Define batch prompt for multiple place names
batch_prompt = PromptTemplate(
    template="""
    You are a leading expert in Scottish toponymy with deep knowledge of Celtic, Germanic, and Romance linguistic influences on place names.
    
    Analyze these Scottish place names for their linguistic origins. For each place name, consider:
    
    Place Names and Counties:
    {place_list}
    
    Linguistic patterns to consider:
    - Scottish Gaelic: prefixes like 'bal-', 'inver-', 'glen-', 'dun-', 'craig-'; suffixes like '-more', '-beg'
    - Norse: elements like '-by', '-thorp', '-wick', '-dale', '-fell', '-force'
    - Brittonic/Welsh: 'llan-', 'aber-', 'pen-', 'tre-', 'caer-'
    - Scots: Germanic/Anglo-Saxon elements, often phonetically altered
    - English: Standard English toponymic elements
    
    Historical context: Consider geographic location, settlement patterns, and linguistic layering.
    
    Provide analysis for each place name in the exact same order as listed above.
    
    {format_instructions}
    """,
    input_variables=["place_list"],
    partial_variables={"format_instructions": batch_parser.get_format_instructions()},
)

# Define batch processing pipeline
batch_chain = (
    batch_prompt
    | llm
    | batch_parser
    | RunnableLambda(lambda response: response.model_dump())
)

def process_place_name(place_name: str, historic_county: str) -> tuple[str, str, str]:
    """Runs the LLM pipeline for a given place name and county.
    
    Args:
        place_name: The name of the place to analyze
        historic_county: The historic county for context
        
    Returns:
        Tuple of (origin, reason, confidence)
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

async def process_place_name_async(place_name: str, historic_county: str, semaphore: asyncio.Semaphore) -> tuple[str, str, str]:
    """Async version of process_place_name with concurrency control.
    
    Args:
        place_name: The name of the place to analyze
        historic_county: The historic county for context
        semaphore: Semaphore to limit concurrent requests
        
    Returns:
        Tuple of (origin, reason, confidence)
    """
    async with semaphore:
        try:
            # Run the sync chain in an executor to make it async
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: origin_chain.invoke({
                'place_name': place_name,
                'historic_county': historic_county
            }))
            return (result["origin"], result["reason"], 
                    result.get("confidence", "Low"))
        except Exception as e:
            logging.error(f"Error processing {place_name}: {str(e)}")
            return "Error", str(e), "Low"

def process_batch_with_retry(places_batch: list[tuple[str, str]], max_retries: int = 3) -> list[tuple[str, str, str]]:
    """Process a batch of place names with exponential backoff for rate limiting.
    
    Args:
        places_batch: List of (place_name, historic_county) tuples
        max_retries: Maximum number of retry attempts
        
    Returns:
        List of tuples containing (origin, reason, confidence)
    """
    for attempt in range(max_retries + 1):
        try:
            # Format place list for prompt
            place_list = "\n".join([f"- {name} (County: {county})" for name, county in places_batch])
            
            result = batch_chain.invoke({"place_list": place_list})
            analyses = result["analyses"]
            
            # Convert to expected format
            return [(analysis["origin"], analysis["reason"], analysis["confidence"]) 
                    for analysis in analyses]
                    
        except Exception as e:
            error_msg = str(e).lower()
            
            # Check if it's a rate limit error
            if "rate" in error_msg or "429" in error_msg or "limit" in error_msg:
                if attempt < max_retries:
                    # Exponential backoff with jitter
                    delay = (2 ** attempt) + random.uniform(0, 1)
                    logging.warning(f"Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1})")
                    time.sleep(delay)
                    continue
                else:
                    logging.error(f"Rate limit exceeded after {max_retries} retries")
            else:
                logging.error(f"Error processing batch: {str(e)}")
                break
    
    # Return error results for the entire batch
    return [("Error", str(e), "Low")] * len(places_batch)

async def process_batch_async(places_batch: list[tuple[str, str]], semaphore: asyncio.Semaphore) -> list[tuple[str, str, str]]:
    """Process a batch of place names asynchronously with rate limiting.
    
    Args:
        places_batch: List of (place_name, historic_county) tuples
        semaphore: Semaphore to limit concurrent requests
        
    Returns:
        List of tuples containing (origin, reason, confidence)
    """
    async with semaphore:
        # Add small delay between requests to be respectful to API
        await asyncio.sleep(random.uniform(0.1, 0.3))
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, process_batch_with_retry, places_batch)

async def process_places_async(df: pd.DataFrame, max_concurrent: int = 3, batch_size: int = 3) -> list[tuple[str, str, str]]:
    """Process all place names asynchronously with batching and concurrency control.
    
    Args:
        df: DataFrame with place names to process
        max_concurrent: Maximum number of concurrent API requests
        batch_size: Number of place names to process per API call
        
    Returns:
        List of tuples containing (origin, reason, confidence)
    """
    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create batches of place names
    places = [(row['placesort'], row['cty23nm']) for _, row in df.iterrows()]
    batches = [places[i:i + batch_size] for i in range(0, len(places), batch_size)]
    
    # Create tasks for all batches
    tasks = [process_batch_async(batch, semaphore) for batch in batches]
    
    # Run all tasks concurrently with progress bar
    batch_results = await atqdm.gather(*tasks, desc="Analyzing batches")
    
    # Flatten results
    results = []
    for batch_result in batch_results:
        results.extend(batch_result)
    
    return results
    
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

    # Add legend with proper sizing for all items
    legend_html = '''
    <div style="position: fixed; 
                bottom: 20px; left: 20px; width: 220px; height: 260px; 
                background-color: white; border: 2px solid #333; border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2); z-index:9999; 
                font-family: Arial, sans-serif; font-size: 14px; padding: 15px;
                overflow: hidden;">
    <h4 style="margin: 0 0 15px 0; color: #333; font-size: 16px; text-align: center; font-weight: bold;">Place Name Origins</h4>
    '''
    
    # Ensure consistent ordering and include all origins
    ordered_origins = [
        ("Scottish Gaelic", "green"),
        ("Norse", "red"), 
        ("Scots", "blue"),
        ("English", "purple"),
        ("Brittonic", "black"),
        ("Unsure", "orange"),
        ("Error", "gray")
    ]
    
    for origin, color in ordered_origins:
        legend_html += f'''
        <div style="margin: 6px 0; display: flex; align-items: center; line-height: 1.2;">
            <i class="fa fa-circle" style="color:{color}; margin-right: 10px; font-size: 12px; min-width: 12px;"></i>
            <span style="color: #333; font-size: 13px;">{origin}</span>
        </div>'''
    
    legend_html += '</div>'
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m.save(output_path)
    logging.info(f"Map saved to {output_path}")

def main(sample_size: int = 1000, input_file: str = "data/IPN_GB_2024.csv", 
         output_file: str = "output/IPN_GB_2024_origin_sonnet.csv") -> None:
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
            df = df.sample(sample_size, random_state=42).reset_index(drop=True)  # Reset index after sampling
            logging.info(f"Sampling {sample_size} places from {len(df)} available")
        
        # Process place names with async concurrency
        logging.info("Processing place names with concurrent requests...")
        results = asyncio.run(process_places_async(df))
        
        # Assign results back to df - make sure indices align
        df.reset_index(drop=True, inplace=True)
        results_df = pd.DataFrame(results, columns=['origin', 'reason', 'confidence'])
        df[['origin', 'reason', 'confidence']] = results_df[['origin', 'reason', 'confidence']]
        
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

def create_map_from_csv(csv_file: str = "output/IPN_GB_2024_origin_sonnet4.csv", 
                       output_path: str = "output/topollm_map.html") -> None:
    """Create map HTML from existing CSV file with origin data.
    
    Args:
        csv_file: Path to CSV file with origin, reason, confidence columns
        output_path: Path to save the map HTML file
    """
    try:
        # Load the CSV file
        logging.info(f"Loading data from {csv_file}")
        df = pd.read_csv(csv_file)
        
        # Check if required columns exist
        required_cols = ['lat', 'long', 'place23nm', 'origin', 'reason']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Filter out rows with missing coordinates or invalid data
        df = df.dropna(subset=['lat', 'long', 'origin', 'reason'])
        df = df[(df['origin'] != '') & (df['reason'] != '')]
        
        if df.empty:
            logging.error("No valid data found in CSV file")
            return
            
        logging.info(f"Creating map with {len(df)} places")
        
        # Create and save the map
        save_map(df, output_path)
        
        # Print summary statistics
        origin_counts = df['origin'].value_counts()
        logging.info(f"Map creation complete. Origin distribution:")
        for origin, count in origin_counts.items():
            logging.info(f"  {origin}: {count}")
            
        logging.info(f"Map saved to {output_path}")
        
    except FileNotFoundError:
        logging.error(f"CSV file {csv_file} not found")
    except Exception as e:
        logging.error(f"Error creating map from CSV: {str(e)}")
        raise

@click.group()
def cli():
    """TopoLLM - Scottish place name linguistic origin analysis."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@cli.command()
@click.option('--sample-size', default=1000, help='Number of places to process')
@click.option('--input-file', default='data/IPN_GB_2024.csv', help='Path to input CSV file')
@click.option('--output-file', default='output/IPN_GB_2024_origin_sonnet4.csv', help='Path to output CSV file')
def analyse(sample_size, input_file, output_file):
    """Analyze place names and save results."""
    main(sample_size, input_file, output_file)

@cli.command()
@click.option('--csv-file', default='output/IPN_GB_2024_origin_sonnet4.csv', help='Path to CSV file with origin data')
@click.option('--output-path', default='output/topollm_map.html', help='Path to save the map HTML file')
def create_map(csv_file, output_path):
    """Create interactive map from existing CSV file."""
    create_map_from_csv(csv_file, output_path)

if __name__ == "__main__":
    cli()
