# Information Extraction
 Leveraging Language Models for Information Extraction of Entities in Orbital Environment Evolution
## Abstract

### Entities and Description
| No. | Entities | Column 3 Header |
| --------------- | --------------- | --------------- |
| 1    | constellationName    | The satellite constellation name for which the company applied to deploy or operate    |
| 2   | dateRelease    | The date the document release     |
| 3   | date50    | The date when the company is order to launch and operate 50 percent of its satellites     |
| 4   | date100   | The date when the company is order to completely launch and operate all of its remaining satellites     |
| 5    | totalSatelliteNumber     | The concluding total number of satellites that the company has been authorized to deploy and operate for the constellation     |
| 6    | altitudes    | The granted altitudes of the satellites that the company has been authorized to deploy    |
| 7    | inclination     | The granted inclination of the satellites that the company has been authorized to deploy, respective to the altitudes     |
| 8    | numberOrbPlane     | The number of orbital planes, respective to the altitudes and inclination, that the company has been authorized to deploy     |
| 9    | totalSatellitePerOrbPlane     | The specific count of satellites located in each individual orbital plane. This count refers to the total number of satellites within one orbital plane, and it can vary from plane to plane based on the altitudes and inclination    |
| 10   | totalSatellitePerAltIncl   | The total number of satellites at a specific altitude and inclination across all orbital planes sharing these characteristics. This count represents the overall number of satellites with the specified altitude and inclination parameters   |
| 11   | operationalLifetime   | The operational lifetime of the satellite in the constellation in years    |


## Dependencies
huggingface-hub==0.16.4 <br>
kor==0.13.0<br>
langchain==0.0.245<br>
numpy==1.24.4<br>
openai==0.27.8<br>
pandas==1.5.3<br>
pydantic==1.10.11<br>
regex==2023.6.3<br>
scikit-learn==1.3.0<br>
simpletransformers==0.64.3<br>
spacy==3.6.0<br>
transformers==4.31.0


## Methodology Steps
![Alt text](data/image/image.png)

## Methods
### In-Context Learning
1. [Notebook](https://github.com/AlinZohari/InformationExtraction/blob/main/001_InContext_Learning.ipynb)

2. [Output - XLSX Dataframe](https://github.com/AlinZohari/InformationExtraction/blob/main/001_InContext_Learning.ipynb)

3. [Output - JSON Starlink Gen2](https://github.com/AlinZohari/InformationExtraction/blob/main/output/Gen2_Starlink_gpt-3.5-turbo_data.json)

### Embedding with Retrieval QA model
1. [Notebook](https://github.com/AlinZohari/InformationExtraction/blob/main/002_Embedding.ipynb)

2. [Output - XLSX Dataframe](https://github.com/AlinZohari/InformationExtraction/blob/main/output/embedding_gen2_results.xlsx)


### Fine Tuning QA model
1. [Notebook](https://github.com/AlinZohari/InformationExtraction/blob/main/003_SQuAD_TuneQAmodel.ipynb)

2. [Output - XLSX Dataframe](https://github.com/AlinZohari/InformationExtraction/blob/main/output/tuned_gen2.xlsx)

3. [Output -Tuned Model (Link Google Drive)](https://drive.google.com/drive/folders/16RcXBTkrHYmvM0Xu4-X0b8mMT-LyJHN8?usp=sharing)

## Results
![Alt text](data/image/image-1.png)