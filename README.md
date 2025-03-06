## Final Exercise - The Path to Become an AI Engineer

This project implements a recipe recommendation system with multilingual support and image generation capabilities. The system uses hybrid search (dense and sparse embeddings) to find relevant recipes and provides translations in Portuguese and Spanish.

### Prerequisites

- Python 3.10+
- Docker and Docker Compose

### Setup

1. Clone the repository

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start Qdrant vector database:
```bash
docker-compose up -d
```

5. Install Ollama for Mistral:
```bash
curl -fsSL https://ollama.com/install.sh | sh

ollama serve &
```

6. Download Mistral model:
```bash
ollama pull mistral
```

7. Create a `.env` file with the following variables:
```env
ENVIRONMENT=dev
COLLECTION_NAME=recipes
QDRANT_HOST=localhost
QDRANT_PORT=6333
ENCODER_ID=9999
```

when `ENVIRONMENT` is set to `dev` we only create the embeddings with a fraction of the dataset, to improve speed and create faster feedback loop.

### Running the Project

1. First, compute and store the embeddings:
```bash
python -m src.compute_embeddings
```

2. Run the main application with your food query:
```bash
python -m src.app --query "Your food request here"
```

Example:
```bash
python -m src.app --query "I want a healthy breakfast with eggs and vegetables"
```

### Architecture and Methodology

The project consists of several key components:

#### 1. Query Preprocessing
- Uses Mistral for query understanding
- Extracts keywords and restrictions
- Simplifies user queries for better search results

#### 2. Recipe Search and Recommendation
- Uses a hybrid search approach combining:
  - Dense embeddings (Sentence Transformers)
  - Sparse embeddings (BM25)
- Handles dietary restrictions through penalty scoring to select best candidate recipes
- At the end it uses Mistral for selecting the best fitting recipe

#### 3. Translation System
- Implements multilingual support using NLLB-200
- Translates recipes to Portuguese and Spanish
- Maintains formatting and structure across translations

#### 4. Image Generation
- Uses Stable Diffusion for recipe image generation
- Implements prompt engineering for food photography
- Saves in the `static` folder

### Models Used

1. **Text Embeddings**
   - Model: `all-MiniLM-L6-v2`
   - Purpose: Dense vector embeddings for semantic search

2. **Translation**
   - Model: `facebook/nllb-200-distilled-600M`
   - Purpose: High-quality multilingual translation

3. **Image Generation**
   - Model: `runwayml/stable-diffusion-v1-5`
   - Purpose: Generate appetizing food images

4. **Query Processing**
   - Model: Mistral (via Ollama)
   - Purpose: Natural language understanding

### Results
##### 1.
Running the command:

```bash
python -m src.app --query "I want a sugared meal that does not contain too much sugar and that I can share with my husband"
```

We got:

```json
{
  "original": {
    "title": "Sugar-Free Jello Salad",
    "ingredients": [
      "1 pkg. sugarless raspberry jello",
      "1 c. boiling water",
      "1 c. light diet cottage cheese",
      "1/2 c. light mayonnaise",
      "1/2 c. chopped nuts",
      "1 c. unsweetened crushed pineapple",
      "1 c. skim milk"
    ],
    "directions": [
      "Dissolve jello in hot water.",
      "Mix other ingredients together. Pour into jello.",
      "Chill.",
      "This is very good for diabetics."
    ],
    "NER": [
      "sugarless raspberry jello",
      "boiling water",
      "light diet cottage cheese",
      "light mayonnaise",
      "nuts",
      "pineapple",
      "milk"
    ],
    "image_url": "static/Sugar-Free Jello Salad_20250307_180530.png"
  },
  "pt": {
    "title": "Salada de gelatina sem açúcar",
    "ingredients": [
      "1 pkg. gelatina de framboesa sem açúcar",
      "1 c. água fervente",
      "1 c. queijo cottage dieteto leve",
      "1/2 c. maionese leve",
      "1/2 c. Nozes picadas",
      "1 c. Ananas trituradas não adoçadas",
      "1 c. leite desnatado"
    ],
    "directions": [
      "Dissolver gelatina em água quente.",
      "Misture outros ingredientes e coloque-os em gelatina.",
      "- Relaxa. - Não.",
      "Isto é muito bom para os diabéticos."
    ]
  },
  "es": {
    "title": "Salada de gelatina sin azúcar",
    "ingredients": [
      "1 pkg. gelatina de frambuesa sin azúcar",
      "1 c. agua hirviendo",
      "1 c. Queso de casita de dieta ligera",
      "1/2 c. mayonesa ligera",
      "1/2 c. nueces cortadas",
      "1 c. Ananas trituradas sin azúcar",
      "1 c. Leche desnatada"
    ],
    "directions": [
      "Disolver la gelatina en agua caliente.",
      "Mezcla otros ingredientes y vertiente en gelatina.",
      "- ¿Qué quieres?",
      "Esto es muy bueno para los diabéticos."
    ]
  },
  "image_path": "static/Sugar-Free Jello Salad_20250307_180530.png"
}
```

##### Generated Recipe Image

<img src="static/Sugar-Free Jello Salad_20250307_180530.png" alt="Sugar-Free Jello Salad" width="300" />

##### 2.

Running: 

```bash
python -m src.app --query "Give me an high protein health breakfast option"
```
```JSON
{
  "original": {
    "title": "High Protein Breakfast Quinoa",
    "ingredients": [
      "1/2 cup quinoa",
      "1 cup milk",
      "1 very ripe banana, mashed",
      "1/2 teaspoon cinnamon",
      "2 egg whites",
      "1 scoop vanilla protein powder (optional)"
    ],
    "directions": [
      "1. Combine quinoa, milk, mashed banana, cinnamon and egg whites in a pan on medium heat.",
      "2. Increase heat to medium high and, stirring constantly, allow mixture to thicken (approximately 10 minutes)",
      "3. Reduce heat to low and slowly stir in protein powder",
      "4. Divide quinoa into two bowls and enjoy"
    ],
    "NER": [
      "quinoa",
      "milk",
      "very ripe banana",
      "cinnamon",
      "egg whites"
    ],
    "image_url": "static/High Protein Breakfast Quinoa_20250307_185226.png"
  },
  "pt": {
    "title": "Quinoa de pequeno-almoço alto em proteínas",
    "ingredients": [
      "1/2 xícara de quinoa",
      "1 xícara de leite",
      "1 banana muito madura, puré",
      "1/2 colher de chá de canela",
      "2 brancos de ovo",
      "1 colher de pó de proteína de baunilha (facultativo)"
    ],
    "directions": [
      "1. Combinar quinoa, leite, purê de banana, canela e brancos de ovo numa panela a calor médio.",
      "2. Aumentar o calor a um nível médio e, agitando constantemente, permitir que a mistura se espesse (aproximadamente 10 minutos)",
      "3. Reduzir o calor para baixo e agitar lentamente o pó de proteína",
      "4. Divida a quinoa em duas tigelas e desfrute"
    ]
  },
  "es": {
    "title": "Quinoa para el desayuno alto en proteínas",
    "ingredients": [
      "1/2 taza de quinoa",
      "1 taza de leche",
      "1 plátano muy maduro, puré",
      "1/2 cucharadita de canela",
      "2 blancos de huevo",
      "1 cuchara de proteína de vainilla en polvo (opcional)"
    ],
    "directions": [
      "1. Combinar quinoa, leche, plátano, canela y blancos de huevo en una sartén a medio calor.",
      "2. Aumentar el calor a una altura media y, agitando constantemente, permitir que la mezcla se espese (aproximadamente 10 minutos)",
      "3. Reducir el calor a un nivel bajo y mezclar lentamente en polvo de proteínas",
      "4. Dividir la quinoa en dos tazones y disfrutar"
    ]
  },
  "image_path": "static/High Protein Breakfast Quinoa_20250307_185226.png"
}
````

##### Generated Recipe Image

<img src="static/High Protein Breakfast Quinoa_20250307_185226.png" alt="High Protein Breakfast Quinoa" width="300" />

##### 3.

Running:

`````bash
python -m src.app --query "Give me an high protein health breakfast option"
`````

We got:

`````JSON
{
  "original": {
    "title": "Veggie Sandwiches (Vegan)",
    "ingredients": [
      "4 slices multi-grain bread",
      "3/4 cup hummus",
      "4 -6 leaves romaine lettuce",
      "4 -6 slices tomatoes",
      "1/2 English cucumber, sliced",
      "10 baby carrots, shredded or thinly sliced",
      "1/2 small red onion, sliced (optional)",
      "1 large avocado, sliced"
    ],
    "directions": [
      "Add a thick layer of hummus to each slice of bread.",
      "Layer on additional ingredients (use half for each sandwich).",
      "Cut each sandwich in half and serve.",
      "NOTE: To avoid a soggy sandwich, if you are not eating this immediately, I like to wrap the lettuce, tomato, and cucumber in plastic wrap and the rest of the sandwich in its own plastic wrap. Blot your veggies with a napkin or paper towel and put it all together just before eating."
    ],
    "NER": [
      "multi-grain bread",
      "tomatoes",
      "cucumber",
      "baby carrots",
      "red onion",
      "avocado"
    ]
  },
  "pt": {
    "title": "Sandwiches Veggie (veganos)",
    "ingredients": [
      "4 fatias de pão multicorpo",
      "3/4 de xícara de hummus",
      "4 - 6 folhas de alface romana",
      "4 a 6 fatias de tomates",
      "1/2 pepino inglês, cortado",
      "10 cenouras de bebê, em triturados ou em fatias finas",
      "1/2 cebola vermelha pequena, cortada (facultativa)",
      "1 grande abacate, cortado em fatias"
    ],
    "directions": [
      "Adicione uma espessa camada de hummus a cada fatia de pão.",
      "Layer sobre ingredientes adicionais (utilizar metade para cada sanduíche).",
      "Corte cada sanduíche em duas e serve.",
      "NOTA: Para evitar um sanduíche molhado, se você não está comendo isso imediatamente, eu gosto de envolver a alface, tomate e pepino em plástico e o resto do sanduíche em sua própria embalagem de plástico."
    ]
  },
  "es": {
    "title": "Sandwiches vegetales (veganos)",
    "ingredients": [
      "4 rebanadas de pan multicereal",
      "3/4 tazas de hummus",
      "4 - 6 hojas de lechuga romana",
      "4 a 6 rebanas de tomates",
      "1/2 pepino inglés, cortado en rodajas",
      "10 zanahorias para bebés, en triturados o cortadas en pequeñas rebanas",
      "1/2 cebolla roja pequeña, cortada (opcional)",
      "1 aguacate grande, cortado en rebanas"
    ],
    "directions": [
      "Añade una gruesa capa de hummus a cada rebanada de pan.",
      "Capas sobre ingredientes adicionales (utilice la mitad para cada sándwich).",
      "Cortar cada sándwich a la mitad y servir.",
      "NOTA: Para evitar un sándwich húmedo, si no lo comes de inmediato, me gusta envuelver la lechuga, el tomate y el pepino en un revestimiento de plástico y el resto del sándwich en su propio revestimiento de plástico."
    ]
  },
  "image_path": "static/Veggie Sandwiches Vegan_20250307_192243.png"
}
``````
##### Generated Recipe Image

<img src="static/Veggie Sandwiches Vegan_20250307_192243.png" alt="Veggie Sandwiches (Vegan)" width="300" />

### Possible Improvements and Future Work

#### 1. Translation Enhancement
While all requirements have been met, each use case has significant room for improvement. One of the main areas for enhancement is the translation system, which often struggles with:
- Recipe-specific expressions and abbreviations (e.g., "pkg.")
- Context-aware translations
- Technical cooking terminology

**Proposed Solution**: Implement a hybrid approach combining LLM with the translation model, where the LLM would provide context interpretation before translation.

#### 2. Image Generation Refinement
The current image generation system has several limitations:
- Occasional hallucinations in generated images
- Inconsistent representation of recipes
- Limited exploration of different models and prompts

**Future Work**: 
- Test additional stable diffusion models
- Develop more sophisticated prompting techniques
- Implement better quality control mechanisms

#### 3. Recommendation Engine Optimization
While the recommendation engine produces satisfactory results in terms of accuracy, several improvements could be made:
- Performance optimization for larger scale deployment
- Enhanced query processing
- Better handling of dietary restrictions
- Improved scoring system for recipe matching

The current implementation provides good results for small to medium datasets, but would require significant optimization for production-scale deployment.

These improvements would significantly enhance the overall user experience and system reliability while maintaining the current functionality that successfully meets all initial requirements.  
