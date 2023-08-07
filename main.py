import os
import re
import json
import numpy as np
import pandas as pd
import asyncio
from secret_key import openai_key
os.environ["OPEN_API_KEY"] = openai_key

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback


from typing import List, Optional, Union
from pydantic import BaseModel, Field, validator, ValidationError

from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number
from kor import extract_from_documents, from_pydantic


# Define constants
model = "gpt-3.5-turbo"
filename = "data/authorize_doc/StarlinkGen2_FCC-22-91A1.txt"

# ---------------------------------------------------

class OrbitEnv(BaseModel):
    const_name: str = Field(
        description="The satellite constellation name for which the company applied to deploy or operate",
    )
    orbit_type: str = Field(
        description="The orbit type into which the satellites will be launched"
    )
    application: str = Field(
        description="The application or services that the satellites would provide"
    )
    date_50: str = Field(
        description="The date when the company is order to deploy and operate half of its satellites"
    )
    date_100: str = Field(
        description="The date when the company is order to deploy and operate all of its remaining satellites."
    )
    total_sat_const: int = Field(
        description="The concluding total number of satellites that the company has been authorized to deploy and operate for the constellation"
    )
    altitude: Optional[List[float]]= Field(
        description="The granted altitudes of the satellites that the company has been authorized to deploy"
    )
    #can add to description: that the inclination would be in degree - maybe delete the words altitude
    #The granted inclination of the satellites that the company has been authorized to deploy, in degree
    inclination: Optional[List[float]] = Field(
        description="The granted inclination of the satellites that the company has been authorized to deploy, respective to the altitudes"
    )
    number_orb_plane: Optional[List[int]] = Field(
        description="The number of orbital planes, respective to the altitudes and inclination, that the company has been authorized to deploy"
    )
    total_sat_per_orb_plane: Optional[List[int]]= Field(
        description="The specific count of satellites located in each individual orbital plane. This count refers to the total number of satellites within one orbital plane, and it can vary from plane to plane based on the altitude and inclination, and if not mentioned in text, 'total_sat_per_alt_incl' divide by 'number_orb_plane' will give this value"
    )
    total_sat_per_alt_incl: Optional[List[int]] = Field(
        description="The total number of satellites at a specific altitude and inclination across all orbital planes sharing these characteristics. This count represents the overall number of satellites with the specified altitude and inclination parameters, and if not mentioned in the text, the multiplication of 'number_orb_plane' and 'total_sat_per_orb_plane' will give this value"
    )
    orbit_shape: Optional[str] = Field(
        description="The shape of the orbital plane whether its circular, elliptical or are not mention in the document"
    )
    operational_lifetime : Optional[str] = Field(
        description="The operational lifetime of the satellite in the constellation in years"
    )

    @validator("const_name", "orbit_type", "application")
    def validate_name(cls, v):
        if not re.match("^[a-zA-Z\s().,-]*$", v):
            raise ValueError("The field can only contain alphabetic characters, spaces, parentheses, periods, commas and hyphen.")
        return v
    
    @validator("total_sat_const", "number_orb_plane", "total_sat_per_orb_plane", "total_sat_per_alt_incl", "operational_lifetime")
    def validate_whole_number(cls, v):
        if isinstance(v, list):
            if not all(isinstance(i, int) for i in v):
                raise ValueError("All elements of the list must be whole numbers.")
        elif v is not None and not isinstance(v, int):
            raise ValueError("The field must be a whole number.")
        return v

    @validator("altitude", "inclination")
    def validate_number(cls, v):
        if isinstance(v, list):
            if not all(isinstance(i, (int, float)) for i in v):
                raise ValueError("All elements of the list must be numbers (integer or decimal).")
        elif v is not None and not isinstance(v, (int, float)):
            raise ValueError("The field must be a number (integer or decimal).")
        return v

    @validator("orbit_shape")
    def validate_orbit_shape(cls, v):
        if not re.match("^[a-zA-Z\s]*$", v):
            raise ValueError("orbit_shape can only contain alphabetic characters and spaces.")
        return v

# ---------------------------------------------------

schema, extraction_validator = from_pydantic(
    OrbitEnv,
    description="Extract the Orbital Environment information of a Satellite Constellation from the authorized document. Include details such as the company name, orbit type, application, dates for 50 percent and 100 percent satellite launches, total number of authorized satellites, altitude, inclination, number of orbital planes, number of satellites per plane, and orbit shape",
    examples=[
        (
            """In this Order and Authorization, we grant, to the extent set forth below, the request of Kuiper Systems LLC (Kuiper or Amazon) to deploy a non-geostationary satellite orbit (NGSO) system to provide service using certain Fixed-Satellite Service (FSS).
                Operating 3,372 satellites in 102 orbital planes at altitudes of 590 km, 610 km, and 630 km in a circular orbit.
                At 590 km, 30 orbital planes with 28 satellites per plane for a total of 840 satellites at inclination of 33 degree.
                At 610 km, 42 orbital planes with 36 satellites per plane for a total of 1512 satellites at inclination of 42 degree.
                At 630 km, 30 orbital planes with 34 satellites per plane for a total of 1020 satellite at inclination of 51.9 degree.
                The constellation are require to launch and operate 50 percent of its satellites no later than July 30, 2026, and Kuiper must launch the remaining space stations necessary to complete its authorized service constellation, place them in their assigned orbits, and operate each of them in accordance with the authorization no later than July 30, 2029.""",
                
            {"const_name": "Kuiper System LLC", "orbit_type": "non-geostationary satellite orbit (NGSO)", "application": "Fixed-Satellite Service (FSS)", "date_50": "July 30, 2026", "date_100": "July 30, 2029", "total_sat_const": 3372, "altitude": [590, 610, 630],  "inclination": [33, 42, 51.9], "number_orb_plane": [30, 42, 30], "total_sat_per_orb_plane": [28, 36, 34], "total_sat_per_alt_incl": [840, 1512, 1020], "orbit_shape": "circular"}
        ),
        (
            "Iridium must launch 50 percent of the maximum number of proposed space stations, place them in the assigned orbits, and operate them in accordance with this grant no later than November 12,2028, and must launch the remaining space stations necessary to complete its authorized service constellation, place them in their assigned orbits, and operate them in accordance with the authorization no later than May 16,2030.",
            {"const_name": "Iridium","date_50":"November 12,2028,","date_100":"May 16,2030"}
        ),
        (
            "They must launch 50 percent of the maximum number of proposed space stations, place them in the assigned orbits, and operate them in accordance with this grant of U.S. market access no later than December 31,1989, and must launch the remaining space stations necessary to complete its authorized service constellation, place them in their assigned orbits, and operate them in accordance with the grant of U.S. market access no later than December 21,1997.",
            {"date_50":"December 31,1989","date_100":"November 21,1997"}
        ),
        (
            "In this Order and Declaratory Ruling, we grant in part and defer in part the petition for declaratory ruling of WorldVu Satellites Limited (OneWeb) for modification of its grant of U.S. market access for a its satellite constellation authorized by the United Kingdom. As modified, the constellation will operate with four fewer satellites, reduced from 720 to 716 satellites.",
            {"const_name": "WorldVu Satellites Limited (OneWeb)", "total_sat_const": 716}
        ),
        (
            "They sought Commission approval for a non-geostationary satellite orbit (NGSO) system to provide fixed-satellite service (FSS) in the United States.",
            {"orbit_type": "non-geostationay satellite orbit (NGSO)", "application": "fixed-satellite service (FSS)"}
        ),
        (
            """The proposed Telesat system is set to feature a robust constellation of 124 satellites.
            A set of six orbital planes, each inclined at 99.5 degrees, will host nine satellites per plane at an approximate altitude of 1,000 kilometers.
            Additionally, seven more orbital planes, each tilted at 37.4 degrees, will carry another group of satellites, with each plane accommodating ten satellites at a higher altitude of approximately 1,248 kilometers.
            It's noteworthy that all satellites will occupy a circular orbit, ensuring systematic and efficient coverage.""",
            {"const_name": "Telesat", "total_sat_const": 124, "altitude": [1000, 1248], "inclination": [99.5, 37.4], "number_orb_plane": [6, 7], "total_sat_per_orb_plane": [9, 10], "total_sat_per_alt_incl": [54, 70], "orbit_shape": "circular"}
        ),
        #different between total_sat_per_orb_plane and total_sat_per_alt_incl
        (
            "20 orbital planes with 28 satellites per plane for a total of 560 satellites at inclination of 33 degree will be placed at an altitude approximately 800 km.",
            {"altitude": 800, "inclination": 33, "number_orb_plane": 20, "total_sat_per_orb_plane": 28, "total_sat_per_alt_incl": 560}
        ),
        #total_sat_per_alt_incl = number_orb_plane x total_sat_per_orb_plane
        (
            "8 orbital plane containing 15 satellites each which are inclined at 56 degree with altitude of 700 kilometers",
            {"altitude": 700, "inclination": 56, "number_orb_plane": 8, "total_sat_per_orb_plane": 15, "total_sat_per_alt_incl": 120}
        ),
        #total_sat_per_orb_plane = total_sat_per_alt_incl x number_orb_plane
        (
            "72 of the satellites will be distributed equally and place at 6 orbital planes, which are inclined 99.5 degrees, satellites will be at an approximate altitude of 1,000 kilometers",
            {"altitude": 1000, "inclination": 99.5, "number_orb_plane": 6, "total_sat_per_orb_plane": 12, "total_sat_per_alt_incl": 72}
        ),
        #operational_lifetime
        (
            "The operational lifetime for the satellite in the constellation in 10 years",
            {"operational_lifetime": 10}
        ),

    ],
    many=True,
)

# ---------------------------------------------------

# Define document
def import_document(filename: str) -> Optional[str]:
    encodings = ['utf-8', 'ISO-8859-1', 'utf-16', 'ascii', 'cp1252']
    for enc in encodings:
        try:
            with open(filename, 'r', encoding=enc) as file:
                document_text = file.read()
            return document_text
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            return None
        except Exception as e:
            print(f"Error occurred while importing the document: {e}")
            return None
    print(f"Error: Could not decode file with any of the tried encodings: {encodings}")
    return None

# ---------------------------------------------------
def extract_data(document: str, llm: ChatOpenAI, chain, split_docs) -> dict:
    loop = asyncio.get_event_loop()
    document_extraction_results = loop.run_until_complete(
        extract_from_documents(
            chain, split_docs, max_concurrency=5, use_uid=False, return_exceptions=True
        )
    )

    # Handling potential exceptions
    for result in document_extraction_results:
        if isinstance(result, Exception):
            print(f"Error encountered: {result}")

    print(f"Total Tokens: {get_openai_callback.total_tokens}")
    print(f"Prompt Tokens: {get_openai_callback.prompt_tokens}")
    print(f"Completion Tokens: {get_openai_callback.completion_tokens}")
    print(f"Successful Requests: {get_openai_callback.successful_requests}")
    print(f"Total Cost (USD): ${get_openai_callback.total_cost}")

    return document_extraction_results

# ---------------------------------------------------

def generate_dataframe(document_extraction_results: dict) -> pd.DataFrame:
    # Prepare an empty list to store all OrbitEnv data
    data = []

    for record in json_data:
        # Check if the record is a dictionary. If not, print an error and continue to the next record
        if not isinstance(record, dict):
            print(f"Error encountered: {record}")
            continue
        
        orbitenv_list = record.get('data', {}).get('orbitenv', [])
        for orbitenv in orbitenv_list:
            data.append([
                orbitenv.get('const_name', ''),
                orbitenv.get('orbit_type', ''),
                orbitenv.get('application', ''),
                orbitenv.get('date_50', ''),
                orbitenv.get('date_100', ''),
                orbitenv.get('total_sat_const', ''),
                orbitenv.get('altitude', '') or '',
                orbitenv.get('inclination', '') or '',
                orbitenv.get('number_orb_plane', '') or '',
                orbitenv.get('total_sat_per_orb_plane', '') or '',
                orbitenv.get('total_sat_per_alt_incl', '') or '',
                orbitenv.get('orbit_shape', ''),
                orbitenv.get('operational_lifetime', '')
            ])

    # Convert the list into a DataFrame
    df = pd.DataFrame(data, columns=['constellationName', 'orbitType', 'application','date50', 'date100', 'totalSatelliteNumber', 'altitudes','inclination', 'numberOrbPlane', 'totalSatellitePerOrbPlane','totalSatellitePerAltIncl', 'orbShape', 'operationalLifetime'])

    # Replace various values with None
    df.replace(['','-',0,'Null', 'null', 'Not Mentioned', 'Not mentioned', 'not mentioned', 'unknown', 'Unknown','N/A'], None, inplace=True)
    
    return df
# ---------------------------------------------------

def find_most_frequent(df: pd.DataFrame) -> dict:
    most_frequent_dict = {}
    for column in df.columns:
        column_without_none = df[column].dropna()
        if not column_without_none.empty:
            mode = column_without_none.mode()
            if len(mode) > 1:
                most_frequent_dict[column] = {"message": "Multiple modes found", "modes": mode.tolist()}
            else:
                most_frequent_dict[column] = mode[0]
        else:
            most_frequent_dict[column] = None
    return most_frequent_dict

# ---------------------------------------------------

def convert_to_json(data: dict) -> str:
    try:
        json_data = json.dumps(data, default=convert)
    except TypeError:
        return json.dumps({"error": "Failed to serialize data"})
    return json_data
# ---------------------------------------------------

def save_to_file(data: str, model: str) -> None:
    # Check if directory exists, if not create it
    directory = "output"
    if not os.path.exists(directory):
        os.makedirs(directory)
    result = json.loads(data)
    name = result.get('constellationName', {}).get('modes', [None])[0] if isinstance(result.get('constellationName', {}), dict) else result.get('constellationName', None)

    if name is not None:
        name = re.sub(r'\W+', '_', name)
        filename = f'output/{name}_{model}_data.json'
        with open(filename, 'w+') as txt_file:
            txt_file.write(data)

# ---------------------------------------------------

def main(model, filename):
    # Set up the environment and load the document
    document = import_document(filename)
    if document is None:
        print("Failed to load the document.")
        return

    # Setup for extraction chain and document splitting
    llm = ChatOpenAI(
        model_name=model,
        temperature=0,
        request_timeout=120,
        openai_api_key= openai_key 
    )
    doc = Document(page_content=document)
    split_docs = RecursiveCharacterTextSplitter().split_documents([doc])
    chain = create_extraction_chain(
        llm,
        schema,
        encoder_or_encoder_class="json",
        validator=extraction_validator,
        input_formatter="triple_quotes",
    )

    # Extract data
    document_extraction_results = extract_data(document, llm, chain, split_docs)

    # Generate DataFrame and find most frequent values
    df = generate_dataframe(document_extraction_results)
    result = find_most_frequent(df)

    # Save the result to file
    json_data = convert_to_json(result)
    save_to_file(json_data, model)

if __name__ == "__main__":
    model = "gpt-3.5-turbo"
    filename = "data/authorize_doc/StarlinkGen2_FCC-22-91A1.txt"
    main(model, filename)