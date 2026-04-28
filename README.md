# Risk Masters

# Complete Project Roadmap: Cotton Yield & Risk Prediction for Azerbaijan

## 1. Problem Definition

The core objective is to determine if we can predict crop-relevant environmental conditions based on historical patterns and to assess the reliability of such data. The system focuses on modeling weather-driven conditions—such as heat stress, water availability, and seasonal patterns—that directly influence crop productivity. It aims to provide "explainable AI" where low yield predictions are linked to specific agronomic risks.


## 2. Data Sources & Integration
The project integrates two primary data streams to build a comprehensive agricultural dataset:


   - Cotton Production Data: Covers 29 districts over 25 years (2000–2024), providing annual yield in tonnes.  

   -Open-Meteo API : Provides long-term daily weather data without requiring an API key, ensuring high reproducibility. 

         -Archive API: Used for 24+ years of daily historical data (temp, rain, humidity, etc.). 

   - Station Mapping: Since not every district has a weather station, districts were geographically mapped to the nearest of 5 major stations: Ganja, Sabirabad,        Lankaran, Shamkir, and Nakhchivan.


## 3. Selected Cities & Variables

### Regional Focus

The project focuses on 15 districts located within the central lowland agricultural region of Azerbaijan, including Beylagan, Imishli, Saatli, Sabirabad, and Barda.

- These regions share similar agricultural practices, reducing unrelated variability in the model.  

- They exhibit minor climatic differences, which helps the model learn useful distinctions.  

- They are ideal for studying heat stress and irrigation-based farming.  

### Weather Variables

Variables were chosen to represent four key agricultural dimensions:  


- **Temperature:** Mean, max, and min (drivers of plant development and heat stress).  


- **Water Availability:** Total precipitation and soil moisture (top layer).  


- **Solar Energy:** Sunshine duration and shortwave radiation (needed for photosynthesis).  


- **Atmospheric Interaction:** Evapotranspiration (ET₀), relative humidity, and wind speed (impacts water loss and drought stress).


## 4. Feature Engineering
Raw daily weather data was transformed into meaningful agricultural features by aggregating data according to the four biological growth stages of cotton:

- **Planting (March–April):** Focuses on temperature and moisture levels required for successful germination.

- **Growing (May–August):** Tracks heat accumulation (GDD), drought stress, and extreme heat days.

- **Boll Forming (August–September):** Monitors for extreme heat that can damage developing bolls.

- **Harvest (September–November):** Evaluates rainfall and humidity levels that could interfere with picking conditions.

- **Core Metric - GDD (Growing Degree Days):** This is the standard measure of heat energy accumulated by the crop, calculated with a base of 15.5°C. It is a much more accurate predictor of plant development than raw temperature alone.


## 3. Team Division & Daily Schedule

### Roles and Responsibilities

| Member Role | Primary Focus |
|-------------|--------------|
| Ahmadova Esli | Data Engineer – API integration, data cleaning pipeline, and infrastructure |
| Dirayeva Narmin | Feature Engineer – Agro-logic implementation, GDD calculations, and risk label development |
| Gasimova Khaver | ML Engineer – Training 4 risk classifiers and the primary XGBoost yield model |
| Aliyeva Ulviyye | Data Analyst – Exploratory Data Analysis (EDA), visualization, and final reporting |


### Project Timeline (10-Day Plan)

The project follows a structured 10-day roadmap to ensure internal readiness:

| Day | Date | Brief | Focus |
|-----|------|-------|-------|
| 1 | 20 Apr | Kick-Off | Repo setup, API exploration, and project planning |
| 2 | 21 Apr | Data Ingestion | Building the ingestion module and fetching 24 years of historical data |
| 3 | 22 Apr | Database Design | Schema design and data validation queries |
| 4 | 23 Apr | Feature Engineering | Cleaning pipeline and transforming weather data into agro-features |
| 5 | 24 Apr | Automation | Orchestration, incremental loading, and quality gates |
| 6 | 27 Apr | EDA | Descriptive statistics and cross-city comparisons |
| 7 | 28 Apr | Statistical Analysis | Hypothesis testing and final feature selection |
| 8 | 29 Apr | Modeling | Training and evaluating 5 total models |
| 9 | 30 Apr | Dress Rehearsal | Full timed run-through and feedback loop |
| 10 | 01 May | Internal Readiness | Repo freeze and generation of the final PDF report |


## 4. Machine Learning Architecture

We developed 5 models in total to power the predictions:

- 4 Risk Classifiers (Random Forest): Predict the probability (0–100%) of risk for each growth stage.

- 1 Yield Regressor (XGBoost): Uses all weather features and risk scores to predict the final tonnage.


## 5. Project Risks & Limitations

### Technical & Data Constraints


- **Refined Dataset Size:**  After data cleaning and focusing on the core 15 agricultural districts, the dataset consists of 375 high-quality observations ($15 \text{ districts} \times 25 \text{ years}$). While this ensures better consistency, the smaller sample size requires the use of robust models (like Random Forest) to prevent overfitting.
- **Variable Scope:** The model relies exclusively on weather and climate data. It does not account for non-climatic factors such as:
    - Soil nutrient composition (Nitrogen, Phosphorus, Potassium).
    - Specific irrigation volumes or timing per district.
    - Fertilizer types and pesticide application schedules.
- **Station Spatial Density:** To represent the 15 districts, we utilize data from 5 primary weather stations. This creates a risk of "spatial generalization," where localized micro-climates within a district might not be fully captured by a station located in a neighboring area.


### Agronomic Risks (Environmental Stress)

The model is specifically designed to identify the following physical threats to cotton production:

- **Heat Stress Anomaly:** Extreme temperatures (>35°C) during the Boll Forming stage, which is the leading cause of yield loss in the central lowlands.

- **Hydrological Stress:** High evapotranspiration (ET₀) combined with low soil moisture, which increases the crop's dependency on artificial irrigation.

- **Harvest Quality Risk:**  Unseasonable rainfall or humidity (>70%) during the Harvest stage (Sept–Nov) that can degrade the fiber quality and make picking difficult.

